use cudarc::driver::{CudaContext, CudaFunction, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::mpsc::{self, sync_channel};
use std::thread;

// ========= 配置 =========
const WIDTH: u32 = 3840;
const HEIGHT: u32 = 2160;
const FPS: u32 = 60;
const SECONDS: u32 = 40;
const TOTAL_FRAMES: u32 = FPS * SECONDS;

const CENTER_RE: f64 = -0.743643887037158704752191506114774;
const CENTER_IM: f64 = 0.131825904205311970493132056385139;
const START_SCALE: f64 = 3.5;
const END_SCALE: f64 = 1.0e-14;

const BASE_ITER: u32 = 256;
const MAX_ITER_CAP: u32 = 12288;
const SAMPLES_PER_PIXEL: u32 = 16;

// 精度切换阈值（有足够安全余量）
const F32_THRESH: f64 = 1.0e-3;
const F64_THRESH: f64 = 1.0e-9;
// ========================

const KERNEL_SRC: &str = include_str!("kernel.cu");

#[derive(Clone, Copy, Debug)]
enum Prec {
    F32,
    F64,
    DD,
}
fn pick_prec(scale: f64) -> Prec {
    if scale > F32_THRESH {
        Prec::F32
    } else if scale > F64_THRESH {
        Prec::F64
    } else {
        Prec::DD
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    println!("CUDA 上下文已创建");

    println!("NVRTC 编译 kernel ...");
    let opts = CompileOptions {
        arch: Some("compute_89"),
        use_fast_math: Some(true),
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(KERNEL_SRC, opts)?;
    let module = ctx.load_module(ptx)?;
    let f_f32: CudaFunction = module.load_function("mandelbrot_f32")?;
    let f_f64: CudaFunction = module.load_function("mandelbrot_f64")?;
    let f_dd: CudaFunction = module.load_function("mandelbrot_dd")?;
    println!("✅ 三档 kernel 均已编译 (f32 / f64 / DD)");

    let n_pixels = (WIDTH * HEIGHT) as usize;
    let mut d_out = stream.alloc_zeros::<u32>(n_pixels)?;

    // ---- ffmpeg ----
    let mut ffmpeg = Command::new("ffmpeg")
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-stats",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgba",
            "-video_size",
            &format!("{}x{}", WIDTH, HEIGHT),
            "-framerate",
            &FPS.to_string(),
            "-i",
            "-",
            "-c:v",
            "av1_nvenc",
            "-preset",
            "p7",
            "-tune",
            "hq",
            "-rc",
            "vbr",
            "-cq",
            "15",
            "-b:v",
            "0",
            "-spatial-aq",
            "1",
            "-temporal-aq",
            "1",
            "-bf",
            "3",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "output.mp4",
        ])
        .stdin(Stdio::piped())
        .spawn()?;
    let ffmpeg_in = ffmpeg.stdin.take().unwrap();

    // ---- writer 线程 + 缓冲池（GPU 算下一帧时，它写上一帧到 pipe）----
    let (frame_tx, frame_rx) = sync_channel::<Vec<u32>>(3);
    let (free_tx, free_rx) = mpsc::channel::<Vec<u32>>();
    for _ in 0..4 {
        free_tx.send(vec![0u32; n_pixels]).unwrap();
    }
    let free_tx_w = free_tx.clone();
    let writer = thread::spawn(move || {
        let mut fin = ffmpeg_in;
        while let Ok(buf) = frame_rx.recv() {
            let bytes: &[u8] = bytemuck::cast_slice(&buf);
            if fin.write_all(bytes).is_err() {
                break;
            }
            let _ = free_tx_w.send(buf);
        }
    });

    let cfg = LaunchConfig {
        grid_dim: (WIDTH.div_ceil(16), HEIGHT.div_ceil(16), 1),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    };

    let ratio = END_SCALE / START_SCALE;
    let scale_at = |f: u32| START_SCALE * ratio.powf(f as f64 / (TOTAL_FRAMES - 1) as f64);

    let t0 = std::time::Instant::now();
    let mut cnt = [0u32; 3]; // 各档帧数统计

    for frame in 0..TOTAL_FRAMES {
        let s0 = scale_at(frame);
        let s1 = scale_at((frame + 1).min(TOTAL_FRAMES - 1));

        let zoom_log2 = (START_SCALE / s0).log2().max(0.0);
        let max_iter = ((BASE_ITER as f64 + zoom_log2 * 220.0) as u32).min(MAX_ITER_CAP);

        let prec = pick_prec(s0);
        cnt[prec as usize] += 1;
        let func: &CudaFunction = match prec {
            Prec::F32 => &f_f32,
            Prec::F64 => &f_f64,
            Prec::DD => &f_dd,
        };

        // 局部变量（launch_builder 要存引用）
        let (w, h) = (WIDTH, HEIGHT);
        let (cre_hi, cre_lo) = (CENTER_RE, 0.0_f64);
        let (cim_hi, cim_lo) = (CENTER_IM, 0.0_f64);
        let (s0_hi, s0_lo, s1_hi, s1_lo) = (s0, 0.0_f64, s1, 0.0_f64);
        let (mi, sp, fs) = (max_iter, SAMPLES_PER_PIXEL, frame);

        let mut lb = stream.launch_builder(func);
        lb.arg(&mut d_out);
        lb.arg(&w);
        lb.arg(&h);
        lb.arg(&cre_hi);
        lb.arg(&cre_lo);
        lb.arg(&cim_hi);
        lb.arg(&cim_lo);
        lb.arg(&s0_hi);
        lb.arg(&s0_lo);
        lb.arg(&s1_hi);
        lb.arg(&s1_lo);
        lb.arg(&mi);
        lb.arg(&sp);
        lb.arg(&fs);
        unsafe {
            lb.launch(cfg)?;
        }

        // 从缓冲池取空 buffer，D2H，送 writer
        let mut host = free_rx.recv().unwrap();
        stream.memcpy_dtoh(&d_out, &mut host)?;
        stream.synchronize()?;
        frame_tx.send(host).unwrap();

        if frame % 15 == 0 || frame + 1 == TOTAL_FRAMES {
            let el = t0.elapsed().as_secs_f64();
            let fps = (frame + 1) as f64 / el.max(0.001);
            let eta = (TOTAL_FRAMES - frame - 1) as f64 / fps.max(0.001);
            println!(
                "f{:4}/{}  [{:>3?}]  {:5.1} fps  iter={:5}  scale={:.2e}  ETA {:5.1}s",
                frame + 1,
                TOTAL_FRAMES,
                prec,
                fps,
                max_iter,
                s0,
                eta
            );
        }
    }

    // 结束
    drop(frame_tx);
    writer.join().unwrap();
    let status = ffmpeg.wait()?;
    println!(
        "\n完成  用时 {:.1}s  ffmpeg={:?}\n精度分布: f32={}  f64={}  DD={}",
        t0.elapsed().as_secs_f64(),
        status.code(),
        cnt[0],
        cnt[1],
        cnt[2]
    );
    Ok(())
}

