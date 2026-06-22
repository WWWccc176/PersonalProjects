// ========== Double-Double 算术 ==========
__device__ __forceinline__ double2 mkdd(double x, double y) { double2 r={x,y}; return r; }

__device__ __forceinline__ double2 two_sum(double a, double b) {
    double s = a + b; double bb = s - a;
    return mkdd(s, (a - (s - bb)) + (b - bb));
}
__device__ __forceinline__ double2 quick_two_sum(double a, double b) {
    double s = a + b; return mkdd(s, b - (s - a));
}
__device__ __forceinline__ double2 two_prod(double a, double b) {
    double p = a * b; return mkdd(p, fma(a, b, -p));
}
__device__ __forceinline__ double2 dd_add(double2 a, double2 b) {
    double2 s = two_sum(a.x, b.x);
    double2 t = two_sum(a.y, b.y);
    double2 v = quick_two_sum(s.x, s.y + t.x);
    return quick_two_sum(v.x, v.y + t.y);
}
__device__ __forceinline__ double2 dd_sub(double2 a, double2 b) {
    return dd_add(a, mkdd(-b.x, -b.y));
}
__device__ __forceinline__ double2 dd_mul(double2 a, double2 b) {
    double2 p = two_prod(a.x, b.x);
    p.y += a.x * b.y + a.y * b.x;
    return quick_two_sum(p.x, p.y);
}
__device__ __forceinline__ double2 dd_sqr(double2 a) {
    double2 p = two_prod(a.x, a.x);
    p.y += 2.0 * a.x * a.y;
    return quick_two_sum(p.x, p.y);
}
__device__ __forceinline__ double2 dd_mul_d(double2 a, double b) {
    double2 p = two_prod(a.x, b);
    p.y += a.y * b;
    return quick_two_sum(p.x, p.y);
}
__device__ __forceinline__ double2 dd_scale2(double2 a) { return mkdd(a.x*2.0, a.y*2.0); }

// ========== 通用工具 ==========
__device__ __forceinline__ float3 mix3(float3 a, float3 b, float k) {
    return make_float3(a.x+(b.x-a.x)*k, a.y+(b.y-a.y)*k, a.z+(b.z-a.z)*k);
}
__device__ float3 palette(float t_in) {
    float t = powf(fminf(fmaxf(t_in, 0.f), 1.f), 0.35f);
    float3 c0={0.00f,0.00f,0.00f}, c1={0.25f,0.07f,0.00f}, c2={0.70f,0.30f,0.05f},
           c3={1.00f,0.67f,0.25f}, c4={1.00f,0.95f,0.80f};
    if (t < 0.15f) return mix3(c0,c1, t           /0.15f);
    if (t < 0.45f) return mix3(c1,c2,(t-0.15f)   /0.30f);
    if (t < 0.75f) return mix3(c2,c3,(t-0.45f)   /0.30f);
                   return mix3(c3,c4,(t-0.75f)   /0.25f);
}
__device__ float halton(unsigned int i, unsigned int b) {
    float f=1.f, r=0.f;
    while (i>0u) { f/=(float)b; r += f*(float)(i%b); i/=b; }
    return r;
}
__device__ __forceinline__ unsigned int pack_rgba(float3 c) {
    float r=fminf(fmaxf(c.x,0.f),1.f), g=fminf(fmaxf(c.y,0.f),1.f), b=fminf(fmaxf(c.z,0.f),1.f);
    return (unsigned)(r*255.f) | ((unsigned)(g*255.f)<<8) | ((unsigned)(b*255.f)<<16) | (0xFFu<<24);
}

// 主心形 + 周期2球体快速内部检测
__device__ __forceinline__ bool in_main_body_d(double x, double y) {
    double xm=x-0.25, y2=y*y, q=xm*xm+y2;
    if (q*(q+xm) < 0.25*y2) return true;
    double xp=x+1.0;
    return (xp*xp + y2) < 0.0625;
}
__device__ __forceinline__ bool in_main_body_f(float x, float y) {
    float xm=x-0.25f, y2=y*y, q=xm*xm+y2;
    if (q*(q+xm) < 0.25f*y2) return true;
    float xp=x+1.f;
    return (xp*xp + y2) < 0.0625f;
}

// ========== 三种精度的 sample ==========
__device__ float3 sample_f32(float cre, float cim, unsigned int max_iter) {
    if (in_main_body_f(cre, cim)) return make_float3(0.f,0.f,0.f);
    float zre=0.f, zim=0.f, mag=0.f;
    unsigned int iter=0;
    #pragma unroll 1
    for (; iter<max_iter; iter++) {
        float zre2=zre*zre, zim2=zim*zim;
        mag = zre2+zim2;
        if (mag > 256.f) break;
        float nim = 2.f*zre*zim + cim;
        zre = zre2 - zim2 + cre;
        zim = nim;
    }
    if (iter >= max_iter) return make_float3(0.f,0.f,0.f);
    float log_zn = logf(mag)*0.5f;
    float nu = logf(log_zn/0.6931472f)/0.6931472f;
    return palette(((float)iter + 1.f - nu)/(float)max_iter);
}

__device__ float3 sample_f64(double cre, double cim, unsigned int max_iter) {
    if (in_main_body_d(cre, cim)) return make_float3(0.f,0.f,0.f);
    double zre=0.0, zim=0.0, mag=0.0;
    unsigned int iter=0;
    #pragma unroll 1
    for (; iter<max_iter; iter++) {
        double zre2=zre*zre, zim2=zim*zim;
        mag = zre2+zim2;
        if (mag > 256.0) break;
        double nim = 2.0*zre*zim + cim;
        zre = zre2 - zim2 + cre;
        zim = nim;
    }
    if (iter >= max_iter) return make_float3(0.f,0.f,0.f);
    float log_zn = (float)(log(mag)*0.5);
    float nu = logf(log_zn/0.6931472f)/0.6931472f;
    return palette(((float)iter + 1.f - nu)/(float)max_iter);
}

__device__ float3 sample_dd(double2 cre, double2 cim, unsigned int max_iter) {
    if (in_main_body_d(cre.x, cim.x)) return make_float3(0.f,0.f,0.f);
    double2 zre=mkdd(0.0,0.0), zim=mkdd(0.0,0.0);
    double zref_re = 0.0, zref_im = 0.0;
    unsigned int stride = 8;
    unsigned int next_update = 8;
    unsigned int iter=0;
    double mag=0.0;
    #pragma unroll 1
    for (; iter<max_iter; iter++) {
        double2 zre2 = dd_sqr(zre);
        double2 zim2 = dd_sqr(zim);
        mag = zre2.x + zim2.x;
        if (mag > 256.0) break;
        double2 new_re = dd_add(dd_sub(zre2, zim2), cre);
        double2 rziz   = dd_mul(zre, zim);
        double2 new_im = dd_add(dd_scale2(rziz), cim);
        zre = new_re; zim = new_im;
        // 每 8 步才做一次周期检测（warp 内齐步，无分歧）
        if ((iter & 3u) == 3u) {
            double dre = zre.x - zref_re;
            double dim = zim.x - zref_im;
            if (dre*dre + dim*dim < 1e-28) {
                return make_float3(0.f,0.f,0.f);
            }
            if (iter >= next_update) {
                zref_re = zre.x; zref_im = zim.x;
                stride *= 2;
                next_update = iter + stride;
            }
        }
    }
    if (iter >= max_iter) return make_float3(0.f,0.f,0.f);
    float log_zn = (float)(log(mag)*0.5);
    float nu = logf(log_zn/0.6931472f)/0.6931472f;
    return palette(((float)iter + 1.f - nu)/(float)max_iter);
}

// ========== 三个 kernel 入口（同一签名，方便 Rust 统一调度） ==========
#define COMMON_HEADER()                                                        \
    unsigned int px = blockIdx.x*blockDim.x + threadIdx.x;                     \
    unsigned int py = blockIdx.y*blockDim.y + threadIdx.y;                     \
    if (px>=W || py>=H) return;                                                \
    float aspect = (float)W/(float)H;                                          \
    float3 acc = make_float3(0.f,0.f,0.f);

extern "C" __global__ void mandelbrot_f32(
    unsigned int* output, unsigned int W, unsigned int H,
    double cre_hi, double cre_lo, double cim_hi, double cim_lo,
    double s0_hi, double s0_lo, double s1_hi, double s1_lo,
    unsigned int max_iter, unsigned int samples, unsigned int frame_seed)
{
    COMMON_HEADER();
    float cre=(float)cre_hi, cim=(float)cim_hi;
    float s0=(float)s0_hi, s1=(float)s1_hi;
    for (unsigned int k=0; k<samples; k++) {
        unsigned int idx = frame_seed*samples + k + 1u;
        float jx=halton(idx,2u), jy=halton(idx,3u);
        float jt=((float)k + halton(idx,5u))/(float)samples;
        float u=((float)px+jx)/(float)W - 0.5f;
        float v=((float)py+jy)/(float)H - 0.5f;
        float scale = s0 + (s1 - s0)*jt;
        float3 col = sample_f32(cre + scale*u, cim + scale*(v/aspect), max_iter);
        acc.x+=col.x; acc.y+=col.y; acc.z+=col.z;
    }
    float inv=1.f/(float)samples;
    output[py*W+px] = pack_rgba(make_float3(acc.x*inv, acc.y*inv, acc.z*inv));
}

extern "C" __global__ void mandelbrot_f64(
    unsigned int* output, unsigned int W, unsigned int H,
    double cre_hi, double cre_lo, double cim_hi, double cim_lo,
    double s0_hi, double s0_lo, double s1_hi, double s1_lo,
    unsigned int max_iter, unsigned int samples, unsigned int frame_seed)
{
    COMMON_HEADER();
    for (unsigned int k=0; k<samples; k++) {
        unsigned int idx = frame_seed*samples + k + 1u;
        float jx=halton(idx,2u), jy=halton(idx,3u);
        float jt=((float)k + halton(idx,5u))/(float)samples;
        double u = ((double)px + (double)jx)/(double)W - 0.5;
        double v = ((double)py + (double)jy)/(double)H - 0.5;
        double scale = cre_hi; // dummy to silence unused; actually:
        scale = s0_hi + (s1_hi - s0_hi)*(double)jt;
        float3 col = sample_f64(cre_hi + scale*u, cim_hi + scale*(v/(double)aspect), max_iter);
        acc.x+=col.x; acc.y+=col.y; acc.z+=col.z;
    }
    float inv=1.f/(float)samples;
    output[py*W+px] = pack_rgba(make_float3(acc.x*inv, acc.y*inv, acc.z*inv));
}

extern "C" __global__ void mandelbrot_dd(
    unsigned int* output, unsigned int W, unsigned int H,
    double cre_hi, double cre_lo, double cim_hi, double cim_lo,
    double s0_hi, double s0_lo, double s1_hi, double s1_lo,
    unsigned int max_iter, unsigned int samples, unsigned int frame_seed)
{
    COMMON_HEADER();
    double2 cre = mkdd(cre_hi, cre_lo);
    double2 cim = mkdd(cim_hi, cim_lo);
    double2 s0  = mkdd(s0_hi,  s0_lo);
    double2 s1  = mkdd(s1_hi,  s1_lo);
    for (unsigned int k=0; k<samples; k++) {
        unsigned int idx = frame_seed*samples + k + 1u;
        float jx=halton(idx,2u), jy=halton(idx,3u);
        float jt=((float)k + halton(idx,5u))/(float)samples;
        float u=((float)px+jx)/(float)W - 0.5f;
        float v=((float)py+jy)/(float)H - 0.5f;
        double2 diff  = dd_sub(s1, s0);
        double2 scale = dd_add(s0, dd_mul_d(diff, (double)jt));
        double2 dx = dd_mul_d(scale, (double)u);
        double2 dy = dd_mul_d(scale, (double)(v/aspect));
        float3 col = sample_dd(dd_add(cre, dx), dd_add(cim, dy), max_iter);
        acc.x+=col.x; acc.y+=col.y; acc.z+=col.z;
    }
    float inv=1.f/(float)samples;
    output[py*W+px] = pack_rgba(make_float3(acc.x*inv, acc.y*inv, acc.z*inv));
}
