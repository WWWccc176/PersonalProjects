use std::io::Write;
use std::process::{Command, Stdio};

// ========= 配置 =========
const WIDTH: u32 = 3840;
const HEIGHT: u32 = 2160;
const FPS: u32 = 60;
const SECONDS: u32 = 40;
const TOTAL_FRAMES: u32 = FPS * SECONDS; // 2400

const CENTER_RE: f64 = -0.743643887037158704752191506114774;
const CENTER_IM: f64 = 0.131825904205311970493132056385139;

const START_SCALE: f64 = 3.5;
const END_SCALE: f64 = 1.0e-12;

const BASE_ITER: u32 = 256;
const MAX_ITER_CAP: u32 = 16384;
// ========================

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    center_re: [f32; 2],
    center_im: [f32; 2],
    scale: [f32; 2],
    dims: [u32; 2],
    max_iter: u32,
    _pad: [u32; 3],
}

fn split_df(x: f64) -> [f32; 2] {
    let hi = x as f32;
    let lo = (x - hi as f64) as f32;
    [hi, lo]
}

fn main() {
    pollster::block_on(run());
}

async fn run() {
    // ---- wgpu 24 新 API ----
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .expect("找不到 GPU 适配器");
    println!("🖥  GPU: {}", adapter.get_info().name);

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None, // 第二个参数是 trace_path，传 None 即可
        )
        .await
        .expect("请求 device 失败");

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("mandelbrot"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
    });

    let pixel_bytes = (WIDTH as u64) * (HEIGHT as u64) * 4;

    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output"),
        size: pixel_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    const N_STAGING: usize = 3;
    let staging: Vec<wgpu::Buffer> = (0..N_STAGING)
        .map(|i| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("staging{}", i)),
                size: pixel_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        })
        .collect();

    let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("uniforms"),
        size: std::mem::size_of::<Uniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buf.as_entire_binding(),
            },
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    // ---- wgpu 24 新 API：entry_point 由 Option<&str> 改为 Option<&str>，
    //       但部分中间版本曾短暂成为 &str。保险写法用 Some。
    //       若仍报错，把下面那行改成  entry_point: "main",
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("mandel-pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // 启动 ffmpeg（NVENC HEVC）
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
        .spawn()
        .expect("启动 ffmpeg 失败，请确认已安装且支持 hevc_nvenc");
    let mut ffmpeg_in = ffmpeg.stdin.take().unwrap();

    let ratio = END_SCALE / START_SCALE;
    let t0 = std::time::Instant::now();

    for frame in 0..TOTAL_FRAMES {
        let t = frame as f64 / (TOTAL_FRAMES - 1) as f64;
        let scale = START_SCALE * ratio.powf(t);

        let zoom_log2 = (START_SCALE / scale).log2().max(0.0);
        let max_iter = ((BASE_ITER as f64 + zoom_log2 * 150.0) as u32).min(MAX_ITER_CAP);

        let uniforms = Uniforms {
            center_re: split_df(CENTER_RE),
            center_im: split_df(CENTER_IM),
            scale: split_df(scale),
            dims: [WIDTH, HEIGHT],
            max_iter,
            _pad: [0; 3],
        };
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let stg = &staging[(frame as usize) % N_STAGING];

        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_pipeline(&pipeline);
            cp.set_bind_group(0, &bind_group, &[]);
            cp.dispatch_workgroups(WIDTH.div_ceil(16), HEIGHT.div_ceil(16), 1);
        }
        enc.copy_buffer_to_buffer(&output_buf, 0, stg, 0, pixel_bytes);
        queue.submit(Some(enc.finish()));

        // map & 写 pipe
        let slice = stg.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r); // 改用 let _ = 避免 panic
        });
        device.poll(wgpu::Maintain::Wait); // 新 API，没有 unwrap()
        rx.recv().unwrap().expect("map 失败");
        {
            let data = slice.get_mapped_range();
            ffmpeg_in.write_all(&data).expect("写 ffmpeg pipe 失败");
        }
        stg.unmap();

        if frame % 30 == 0 || frame + 1 == TOTAL_FRAMES {
            let el = t0.elapsed().as_secs_f64();
            let fps = (frame + 1) as f64 / el;
            let eta = (TOTAL_FRAMES - frame - 1) as f64 / fps;
            println!(
                "frame {:4}/{}  {:6.1} fps  iter={:4}  scale={:.3e}  ETA {:.1}s",
                frame + 1,
                TOTAL_FRAMES,
                fps,
                max_iter,
                scale,
                eta
            );
        }
    }

    drop(ffmpeg_in);
    let status = ffmpeg.wait().expect("等待 ffmpeg 失败");
    println!(
        "\n✅ 完成，用时 {:.1}s   ffmpeg 状态 {:?}",
        t0.elapsed().as_secs_f64(),
        status.code()
    );
}
