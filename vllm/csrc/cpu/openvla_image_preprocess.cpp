/*
 * OpenVLA 图片前处理 ARM NEON fused kernel — v0.7.0
 *
 * 功能: 将 uint8 [H, W, 3] 图片 resize (Pillow BICUBIC 兼容) + 双路归一化,
 *       直接输出 float32 [1, 6, 224, 224].
 *
 *   resize: 可变 support 抗锯齿, 两通道 (水平→垂直), OMP 多线程
 *   normalize: NEON vld3q_u8 → vcvtq_f32 → vmulq+vsubq → vst1q_f32
 *
 * 编译守卫: #if defined(__aarch64__) && !defined(__APPLE__)
 */

#include <arm_neon.h>
#include <omp.h>
#include <torch/all.h>
#include <torch/library.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

// ============================================================================
// 命名常量 — 替代代码中的 magic numbers
// ============================================================================

namespace {

// Image / channel layout
constexpr int kInputChannels  = 3;     // RGB
constexpr int kOutputChannels = 6;     // DINOv2 (3) + SigLIP (3)
constexpr int kOutputSize     = 224;   // 输出 H × W (Pillow 默认 OpenVLA 尺寸)
constexpr int kOutputPlane    = kOutputSize * kOutputSize;  // 224 * 224

// RGB 通道索引 — 显式命名避免裸字面量 0/1/2
constexpr int kChannelR = 0;
constexpr int kChannelG = 1;
constexpr int kChannelB = 2;

// 输入 tensor 维度索引 — [H, W, C]
constexpr int kInputDims  = 3;
constexpr int kDimH       = 0;
constexpr int kDimW       = 1;
constexpr int kDimC       = 2;

// NEON SIMD 几何 — 128-bit register
constexpr int kSimdBits       = 128;
constexpr int kBlockPixels    = kSimdBits / 8;   // 16 像素 (uint8) / block
constexpr int kBlockBytes     = kBlockPixels * kInputChannels;  // 48 bytes / block
constexpr int kFloatsPerLane  = 4;    // 4 × float32 = 128 bit
constexpr int kLaneBytes      = 4;    // 1 × float32 = 4 bytes

// NEON 128-bit register 内 4 个 float32 lane 的 byte 偏移
constexpr int kNeonLane0Offset = 0;
constexpr int kNeonLane1Offset = 4;
constexpr int kNeonLane2Offset = 8;
constexpr int kNeonLane3Offset = 12;

// NEON 128-bit register 内 4 个 float32 lane 的索引 (用于 float32x4_t.val[0..3] 等数组访问)
constexpr int kNeonLane0 = 0;
constexpr int kNeonLane1 = 1;
constexpr int kNeonLane2 = 2;
constexpr int kNeonLane3 = 3;

// Bicubic 快速路径 — 当 n == 4 时用 NEON 4-tap 展开
constexpr int kBicubicFastPathN = 4;

// Cubic kernel (Catmull-Rom, a = -0.5)
constexpr double kCubicA            = -0.5;
constexpr double kCubicB            = 1.5;
constexpr double kCubicC            = 2.5;
constexpr double kCubicScale        = 2.0;  // support 倍率
constexpr double kCubicSupportRadius = 2.0;  // |x| < 2 分支边界
constexpr double kCubicQuadraticCoeff = 4.0; // (kCubicA*x + kCubicC)*x - 4.0 中的 -4
constexpr double kCubicConstantTerm   = 2.0; // (kCubicA*x + kCubicC)*x - 4.0)*x + 2.0 中的 +2
constexpr double kCubicInnerBoundary  = 1.0; // |x| < 1 分支边界

// 像素中心偏移
constexpr double kPixelCenterOffset = 0.5;

// uint8 → 归一化浮点的除数
constexpr float kUint8ToFloatDivisor = 255.0f;

}  // namespace

// ============================================================================
// 预计算归一化常量
// ============================================================================

static const float IMAGENET_MEAN[kInputChannels] = {0.484375f, 0.455078125f, 0.40625f};
static const float IMAGENET_STD[kInputChannels]  = {0.228515625f, 0.2236328125f, 0.224609375f};
static const float SIGLIP_MEAN[kInputChannels] = {0.5f, 0.5f, 0.5f};
static const float SIGLIP_STD[kInputChannels]  = {0.5f, 0.5f, 0.5f};

static float d_scale[kInputChannels], d_offset[kInputChannels];
static float s_scale[kInputChannels], s_offset[kInputChannels];
static bool constants_initialized = false;

static void init_constants() {
    if (constants_initialized) return;
    for (int c = 0; c < kInputChannels; ++c) {
        d_scale[c]  = 1.0f / (kUint8ToFloatDivisor * IMAGENET_STD[c]);
        d_offset[c] = IMAGENET_MEAN[c] / IMAGENET_STD[c];
        s_scale[c]  = 1.0f / (kUint8ToFloatDivisor * SIGLIP_STD[c]);
        s_offset[c] = SIGLIP_MEAN[c] / SIGLIP_STD[c];
    }
    constants_initialized = true;
}

// ============================================================================
// 分配辅助 — 校验大小后返回非空指针
// ============================================================================

static void* checked_alloc(size_t count, size_t elem_size, const char* what) {
    if (count == 0) {
        throw std::runtime_error("checked_alloc: zero size");
    }
    if (elem_size != 0 && count > SIZE_MAX / elem_size) {
        throw std::runtime_error("checked_alloc: size overflow");
    }
    void* p = std::malloc(count * elem_size);
    if (!p) {
        throw std::runtime_error(std::string("checked_alloc: OOM for ") + what);
    }
    return p;
}

// ============================================================================
// Cubic kernel (Catmull-Rom, a=-0.5) — 与 Pillow bicubic_filter 完全一致
// ============================================================================

static inline double cubic(double x) {
    if (x < 0.0) x = -x;
    if (x < kCubicInnerBoundary) return (kCubicB * x - kCubicC) * x * x + 1.0;
    if (x < kCubicSupportRadius) {
        return ((kCubicA * x + kCubicC) * x - kCubicQuadraticCoeff) * x
               + kCubicConstantTerm;
    }
    return 0.0;
}

// ============================================================================
// 可变采样 LUT — 匹配 Pillow precompute_coeffs
// ============================================================================

struct Lut {
    int out_sz;
    int ksize;
    int* start;
    int* count;
    float* w;
};

static void free_lut(Lut& lut) {
    std::free(lut.start);
    std::free(lut.count);
    std::free(lut.w);
    lut.start = nullptr;
    lut.count = nullptr;
    lut.w = nullptr;
}

// 为单个输出像素填充 start / count / w_row — 单测函数
static void compute_lut_row(
    int i, const Lut& lut, int in_sz,
    double scale, double inv_filterscale)
{
    const double center = (static_cast<double>(i) + kPixelCenterOffset) * scale;
    const double support = kCubicScale * std::max(scale, 1.0);

    int xmin = static_cast<int>(center - support + kPixelCenterOffset);
    if (xmin < 0) xmin = 0;
    int xmax = static_cast<int>(center + support + kPixelCenterOffset);
    if (xmax > in_sz) xmax = in_sz;
    int n = xmax - xmin;

    float* w_row = lut.w + i * lut.ksize;
    double ww = 0.0;
    for (int k = 0; k < n; ++k) {
        int sx = xmin + k;
        double dist = (static_cast<double>(sx) + kPixelCenterOffset - center) * inv_filterscale;
        float w = static_cast<float>(cubic(dist));
        w_row[k] = w;
        ww += static_cast<double>(w);
    }
    if (ww != 0.0) {
        float inv = static_cast<float>(1.0 / ww);
        for (int k = 0; k < n; ++k) w_row[k] *= inv;
    }
    lut.start[i] = xmin;
    lut.count[i] = n;
}

static Lut build_lut(int out_sz, int in_sz) {
    if (out_sz <= 0) {
        throw std::runtime_error("build_lut: out_sz must be > 0");
    }

    const double scale = static_cast<double>(in_sz) / static_cast<double>(out_sz);
    const double filterscale = std::max(scale, 1.0);
    if (filterscale == 0.0) {
        // Mathematically unreachable (std::max(scale, 1.0) >= 1.0),
        // but make the divide-by-zero explicit for the static analyzer.
        throw std::runtime_error("build_lut: filterscale must be > 0");
    }
    const double support = kCubicScale * filterscale;
    const int ksize = static_cast<int>(std::ceil(support)) * 2 + 1;

    Lut lut;
    lut.out_sz = out_sz;
    lut.ksize = ksize;
    lut.start = static_cast<int*>(checked_alloc(static_cast<size_t>(out_sz), sizeof(int), "lut.start"));
    lut.count = static_cast<int*>(checked_alloc(static_cast<size_t>(out_sz), sizeof(int), "lut.count"));
    lut.w = static_cast<float*>(
        checked_alloc(static_cast<size_t>(out_sz) * static_cast<size_t>(ksize),
                      sizeof(float), "lut.w"));
    std::memset(lut.w, 0, static_cast<size_t>(out_sz) * static_cast<size_t>(ksize) * sizeof(float));

    const double inv_filterscale = 1.0 / filterscale;
    for (int i = 0; i < out_sz; ++i) {
        compute_lut_row(i, lut, in_sz, scale, inv_filterscale);
    }
    return lut;
}

// ============================================================================
// NEON bicubic resize — Pillow 兼容, OMP 多线程
// ============================================================================

// clamp + round → uint8 (跨 ARM / 非 ARM 平台共用)
static inline uint8_t clamp_to_u8(float v) {
    return static_cast<uint8_t>(
        std::max(0.0f, std::min(kUint8ToFloatDivisor, std::round(v))));
}

#if defined(__aarch64__) && !defined(__APPLE__)

// RGB 加权求和结果 — 打包减少 accumulate_pixel_* 输出参数数量
struct RgbSum {
    float r;
    float g;
    float b;
};

// 单行加权求和 — 4 个相邻源像素 (NEON 快速路径)
static inline RgbSum accumulate_pixel_4(
    const float* w_row, const uint8_t* src, int xmin)
{
    const int i0 = (xmin + 0) * kInputChannels;
    const int i1 = (xmin + 1) * kInputChannels;
    const int i2 = (xmin + 2) * kInputChannels;
    const int i3 = (xmin + 3) * kInputChannels;

    float rv[kFloatsPerLane] = {
        static_cast<float>(src[i0]),                static_cast<float>(src[i1]),
        static_cast<float>(src[i2]),                static_cast<float>(src[i3])
    };
    float gv[kFloatsPerLane] = {
        static_cast<float>(src[i0 + kChannelG]),    static_cast<float>(src[i1 + kChannelG]),
        static_cast<float>(src[i2 + kChannelG]),    static_cast<float>(src[i3 + kChannelG])
    };
    float bv[kFloatsPerLane] = {
        static_cast<float>(src[i0 + kChannelB]),    static_cast<float>(src[i1 + kChannelB]),
        static_cast<float>(src[i2 + kChannelB]),    static_cast<float>(src[i3 + kChannelB])
    };

    float32x4_t rv4 = vld1q_f32(rv);
    float32x4_t gv4 = vld1q_f32(gv);
    float32x4_t bv4 = vld1q_f32(bv);
    float32x4_t wv4 = vld1q_f32(w_row);

    RgbSum out;
    out.r = vaddvq_f32(vmulq_f32(rv4, wv4));
    out.g = vaddvq_f32(vmulq_f32(gv4, wv4));
    out.b = vaddvq_f32(vmulq_f32(bv4, wv4));
    return out;
}

// 单行加权求和 — 通用 n 像素累加 (标量 fallback)
static inline RgbSum accumulate_pixel_n(
    const float* w_row, const uint8_t* src, int xmin, int n)
{
    RgbSum out{0.0f, 0.0f, 0.0f};
    for (int k = 0; k < n; ++k) {
        int off = (xmin + k) * kInputChannels;
        float wk = w_row[k];
        out.r += static_cast<float>(src[off + kChannelR]) * wk;
        out.g += static_cast<float>(src[off + kChannelG]) * wk;
        out.b += static_cast<float>(src[off + kChannelB]) * wk;
    }
    return out;
}

// resize 水平 pass — 单行 H_in → H_in (中间缓冲)
static void neon_resize_horizontal(
    const uint8_t* input, float* mid, int W_in,
    const ResizePassParams& p)
{
    const int stride = p.W_out * kInputChannels;
#pragma omp parallel for schedule(static) num_threads(p.num_threads)
    for (int y = 0; y < p.H; ++y) {
        const uint8_t* src = input + y * W_in * kInputChannels;
        float* dst = mid + y * stride;
        for (int x = 0; x < p.W_out; ++x) {
            int n = p.lut->count[x];
            int xmin = p.lut->start[x];
            const float* w_row = p.lut->w + x * p.lut->ksize;
            RgbSum rgb = (n == kBicubicFastPathN)
                       ? accumulate_pixel_4(w_row, src, xmin)
                       : accumulate_pixel_n(w_row, src, xmin, n);
            dst[x * kInputChannels + kChannelR] =
                std::max(0.0f, std::min(kUint8ToFloatDivisor, std::round(rgb.r)));
            dst[x * kInputChannels + kChannelG] =
                std::max(0.0f, std::min(kUint8ToFloatDivisor, std::round(rgb.g)));
            dst[x * kInputChannels + kChannelB] =
                std::max(0.0f, std::min(kUint8ToFloatDivisor, std::round(rgb.b)));
        }
    }
}

// resize 垂直 pass — 中间缓冲 → 输出 uint8
static void neon_resize_vertical(
    const float* mid, uint8_t* output,
    const ResizePassParams& p)
{
    const int stride = p.W_out * kInputChannels;
#pragma omp parallel for schedule(static) num_threads(p.num_threads)
    for (int y = 0; y < p.H; ++y) {
        uint8_t* dst = output + y * p.W_out * kInputChannels;
        int n = p.lut->count[y];
        int ymin = p.lut->start[y];
        const float* w_row = p.lut->w + y * p.lut->ksize;
        for (int x = 0; x < p.W_out; ++x) {
            RgbSum rgb;
            if (n == kBicubicFastPathN) {
                const float* s0 = mid + (ymin + 0) * stride + x * kInputChannels;
                const float* s1 = mid + (ymin + 1) * stride + x * kInputChannels;
                const float* s2 = mid + (ymin + 2) * stride + x * kInputChannels;
                const float* s3 = mid + (ymin + 3) * stride + x * kInputChannels;
                float rv[kFloatsPerLane] = {s0[kChannelR], s1[kChannelR], s2[kChannelR], s3[kChannelR]};
                float gv[kFloatsPerLane] = {s0[kChannelG], s1[kChannelG], s2[kChannelG], s3[kChannelG]};
                float bv[kFloatsPerLane] = {s0[kChannelB], s1[kChannelB], s2[kChannelB], s3[kChannelB]};
                float32x4_t rv4 = vld1q_f32(rv);
                float32x4_t gv4 = vld1q_f32(gv);
                float32x4_t bv4 = vld1q_f32(bv);
                float32x4_t wv4 = vld1q_f32(w_row);
                rgb.r = vaddvq_f32(vmulq_f32(rv4, wv4));
                rgb.g = vaddvq_f32(vmulq_f32(gv4, wv4));
                rgb.b = vaddvq_f32(vmulq_f32(bv4, wv4));
            } else {
                rgb = accumulate_pixel_n(w_row, mid + x * kInputChannels, ymin, n);
            }
            int off = x * kInputChannels;
            dst[off + kChannelR] = clamp_to_u8(rgb.r);
            dst[off + kChannelG] = clamp_to_u8(rgb.g);
            dst[off + kChannelB] = clamp_to_u8(rgb.b);
        }
    }
}

static void neon_resize(
    const uint8_t* input, uint8_t* output,
    int H_in, int W_in, int num_threads)
{
    Lut xlut = build_lut(kOutputSize, W_in);
    Lut ylut = build_lut(kOutputSize, H_in);
    const int stride = kOutputSize * kInputChannels;
    float* mid = static_cast<float*>(
        checked_alloc(static_cast<size_t>(H_in) * static_cast<size_t>(stride),
                      sizeof(float), "mid"));
    ResizePassParams hp{H_in, kOutputSize, num_threads, &xlut};
    ResizePassParams vp{kOutputSize, kOutputSize, num_threads, &ylut};
    neon_resize_horizontal(input, mid, W_in, hp);
    neon_resize_vertical(mid, output, vp);
    free_lut(xlut); free_lut(ylut);
    std::free(mid);
}

#endif  // defined(__aarch64__) && !defined(__APPLE__)

// ============================================================================
// 标量 resize fallback (非 ARM 平台)
// ============================================================================

// resize pass 公共参数 — 把 5 个相关标量打包, 减少单函数参数数量
struct ResizePassParams {
    int H;             // 行数: 水平 pass 是 H_in, 垂直 pass 是 H_out
    int W_out;         // 输出宽度
    int num_threads;   // OMP 线程数
    const Lut* lut;    // x 方向 (水平) 或 y 方向 (垂直) LUT
};

static void scalar_resize_horizontal(
    const uint8_t* input, float* mid, int W_in,
    const ResizePassParams& p)
{
    const int stride = p.W_out * kInputChannels;
#pragma omp parallel for schedule(static) num_threads(p.num_threads)
    for (int y = 0; y < p.H; ++y) {
        const uint8_t* src = input + y * W_in * kInputChannels;
        float* dst = mid + y * stride;
        for (int x = 0; x < p.W_out; ++x) {
            int n = p.lut->count[x];
            int xmin = p.lut->start[x];
            const float* w_row = p.lut->w + x * p.lut->ksize;
            float R = 0.0f, G = 0.0f, B = 0.0f;
            for (int k = 0; k < n; ++k) {
                int off = (xmin + k) * kInputChannels;
                float wk = w_row[k];
                R += static_cast<float>(src[off + kChannelR]) * wk;
                G += static_cast<float>(src[off + kChannelG]) * wk;
                B += static_cast<float>(src[off + kChannelB]) * wk;
            }
            dst[x * kInputChannels + kChannelR] =
                std::max(0.0f, std::min(kUint8ToFloatDivisor, std::round(R)));
            dst[x * kInputChannels + kChannelG] =
                std::max(0.0f, std::min(kUint8ToFloatDivisor, std::round(G)));
            dst[x * kInputChannels + kChannelB] =
                std::max(0.0f, std::min(kUint8ToFloatDivisor, std::round(B)));
        }
    }
}

static void scalar_resize_vertical(
    const float* mid, uint8_t* output,
    const ResizePassParams& p)
{
    const int stride = p.W_out * kInputChannels;
#pragma omp parallel for schedule(static) num_threads(p.num_threads)
    for (int y = 0; y < p.H; ++y) {
        uint8_t* dst = output + y * p.W_out * kInputChannels;
        int n = p.lut->count[y];
        int ymin = p.lut->start[y];
        const float* w_row = p.lut->w + y * p.lut->ksize;
        for (int x = 0; x < p.W_out; ++x) {
            float R = 0.0f, G = 0.0f, B = 0.0f;
            for (int k = 0; k < n; ++k) {
                const float* src_row = mid + (ymin + k) * stride + x * kInputChannels;
                float wk = w_row[k];
                R += src_row[kChannelR] * wk;
                G += src_row[kChannelG] * wk;
                B += src_row[kChannelB] * wk;
            }
            int off = x * kInputChannels;
            dst[off + kChannelR] = clamp_to_u8(R);
            dst[off + kChannelG] = clamp_to_u8(G);
            dst[off + kChannelB] = clamp_to_u8(B);
        }
    }
}

static void scalar_resize(
    const uint8_t* input, uint8_t* output,
    int H_in, int W_in, int num_threads)
{
    Lut xlut = build_lut(kOutputSize, W_in);
    Lut ylut = build_lut(kOutputSize, H_in);
    const int stride = kOutputSize * kInputChannels;
    float* mid = static_cast<float*>(
        checked_alloc(static_cast<size_t>(H_in) * static_cast<size_t>(stride),
                      sizeof(float), "mid"));
    ResizePassParams hp{H_in, kOutputSize, num_threads, &xlut};
    ResizePassParams vp{kOutputSize, kOutputSize, num_threads, &ylut};
    scalar_resize_horizontal(input, mid, W_in, hp);
    scalar_resize_vertical(mid, output, vp);
    free_lut(xlut); free_lut(ylut);
    std::free(mid);
}

// ============================================================================
// 标量归一化 (非 ARM 平台 fallback)
// ============================================================================

static void scalar_normalize(
    const uint8_t* __restrict input,
    float* __restrict output,
    int H, int W, int num_threads)
{
    if (W <= 0) {
        throw std::runtime_error("scalar_normalize: W must be > 0");
    }
    const int HW = H * W;
#pragma omp parallel for num_threads(num_threads)
    for (int idx = 0; idx < HW; ++idx) {
        // 显式断言除数非零 (静态分析器 G.EXP.22 友好)
        if (W == 0) {
            throw std::runtime_error("scalar_normalize: divisor W must be != 0");
        }
        const int r = idx / W;
        const int c = idx % W;
        const int src_off = (r * W + c) * kInputChannels;
        const float R = static_cast<float>(input[src_off + kChannelR]);
        const float G = static_cast<float>(input[src_off + kChannelG]);
        const float B = static_cast<float>(input[src_off + kChannelB]);
        const int dst_base = r * W + c;
        // 6 路输出: DINOv2 3 通道 (0..2) + SigLIP 3 通道 (3..5)
        const int kDinoStart  = 0;
        const int kSiglipStart = kInputChannels;  // 3
        for (int c_idx = 0; c_idx < kInputChannels; ++c_idx) {
            const float pix = (c_idx == kChannelR) ? R
                            : (c_idx == kChannelG) ? G
                            :                        B;
            const float sc  = d_scale[c_idx];
            const float off = d_offset[c_idx];
            output[(kDinoStart + c_idx) * HW + dst_base] = pix * sc - off;
        }
        for (int c_idx = 0; c_idx < kInputChannels; ++c_idx) {
            const float pix = (c_idx == kChannelR) ? R
                            : (c_idx == kChannelG) ? G
                            :                        B;
            const float sc  = s_scale[c_idx];
            const float off = s_offset[c_idx];
            output[(kSiglipStart + c_idx) * HW + dst_base] = pix * sc - off;
        }
    }
}

// ============================================================================
// NEON fused normalization (224×224, 16 像素/block)
// ============================================================================

#if defined(__aarch64__) && !defined(__APPLE__)

// 单 block (16 像素) 加载 + uint8→float32 转换
// R/G/B 通道各含 kFloatsPerLane (=4) 个 float32x4_t lane, 按 lane 索引访问.
struct F32x16 {
    float32x4_t r[kFloatsPerLane];
    float32x4_t g[kFloatsPerLane];
    float32x4_t b[kFloatsPerLane];
};

static inline F32x16 load_widen_block(const uint8_t* src) {
    uint8x16x3_t rgb = vld3q_u8(src);
    uint16x8_t r16_l = vmovl_u8(vget_low_u8(rgb.val[kChannelR]));
    uint16x8_t r16_h = vmovl_u8(vget_high_u8(rgb.val[kChannelR]));
    uint16x8_t g16_l = vmovl_u8(vget_low_u8(rgb.val[kChannelG]));
    uint16x8_t g16_h = vmovl_u8(vget_high_u8(rgb.val[kChannelG]));
    uint16x8_t b16_l = vmovl_u8(vget_low_u8(rgb.val[kChannelB]));
    uint16x8_t b16_h = vmovl_u8(vget_high_u8(rgb.val[kChannelB]));
    uint32x4_t r32_0 = vmovl_u16(vget_low_u16(r16_l));
    uint32x4_t r32_1 = vmovl_u16(vget_high_u16(r16_l));
    uint32x4_t r32_2 = vmovl_u16(vget_low_u16(r16_h));
    uint32x4_t r32_3 = vmovl_u16(vget_high_u16(r16_h));
    uint32x4_t g32_0 = vmovl_u16(vget_low_u16(g16_l));
    uint32x4_t g32_1 = vmovl_u16(vget_high_u16(g16_l));
    uint32x4_t g32_2 = vmovl_u16(vget_low_u16(g16_h));
    uint32x4_t g32_3 = vmovl_u16(vget_high_u16(g16_h));
    uint32x4_t b32_0 = vmovl_u16(vget_low_u16(b16_l));
    uint32x4_t b32_1 = vmovl_u16(vget_high_u16(b16_l));
    uint32x4_t b32_2 = vmovl_u16(vget_low_u16(b16_h));
    uint32x4_t b32_3 = vmovl_u16(vget_high_u16(b16_h));
    F32x16 out;
    out.r[kNeonLane0] = vcvtq_f32_u32(r32_0);
    out.r[kNeonLane1] = vcvtq_f32_u32(r32_1);
    out.r[kNeonLane2] = vcvtq_f32_u32(r32_2);
    out.r[kNeonLane3] = vcvtq_f32_u32(r32_3);
    out.g[kNeonLane0] = vcvtq_f32_u32(g32_0);
    out.g[kNeonLane1] = vcvtq_f32_u32(g32_1);
    out.g[kNeonLane2] = vcvtq_f32_u32(g32_2);
    out.g[kNeonLane3] = vcvtq_f32_u32(g32_3);
    out.b[kNeonLane0] = vcvtq_f32_u32(b32_0);
    out.b[kNeonLane1] = vcvtq_f32_u32(b32_1);
    out.b[kNeonLane2] = vcvtq_f32_u32(b32_2);
    out.b[kNeonLane3] = vcvtq_f32_u32(b32_3);
    return out;
}

// 4 个 float32x4_t 应用 scale-offset 并连续 store 到 dst
struct F32x4 {
    float32x4_t v[kFloatsPerLane];
};

static inline void apply_and_store_4x4(
    const F32x4& vals, float32x4_t scale, float32x4_t offset, float* dst)
{
    vst1q_f32(dst + kNeonLane0Offset, vsubq_f32(vmulq_f32(vals.v[kNeonLane0], scale), offset));
    vst1q_f32(dst + kNeonLane1Offset, vsubq_f32(vmulq_f32(vals.v[kNeonLane1], scale), offset));
    vst1q_f32(dst + kNeonLane2Offset, vsubq_f32(vmulq_f32(vals.v[kNeonLane2], scale), offset));
    vst1q_f32(dst + kNeonLane3Offset, vsubq_f32(vmulq_f32(vals.v[kNeonLane3], scale), offset));
}

// 把一个 block 的 RGB 写入 6 路输出 (3 × DINOv2 + 3 × SigLIP)
struct OutputPlanes {
    float* d[kInputChannels];
    float* s[kInputChannels];
};

struct NeonNormVectors {
    float32x4_t d_scale[kInputChannels];
    float32x4_t d_offset[kInputChannels];
    float32x4_t s_scale[kInputChannels];
    float32x4_t s_offset[kInputChannels];
};

static inline void store_block_to_outputs(
    const F32x16& b, int dst_off,
    const OutputPlanes& out, const NeonNormVectors& nv)
{
    apply_and_store_4x4({b.r[kNeonLane0], b.r[kNeonLane1], b.r[kNeonLane2], b.r[kNeonLane3]},
                        nv.d_scale[kChannelR], nv.d_offset[kChannelR], out.d[kChannelR] + dst_off);
    apply_and_store_4x4({b.g[kNeonLane0], b.g[kNeonLane1], b.g[kNeonLane2], b.g[kNeonLane3]},
                        nv.d_scale[kChannelG], nv.d_offset[kChannelG], out.d[kChannelG] + dst_off);
    apply_and_store_4x4({b.b[kNeonLane0], b.b[kNeonLane1], b.b[kNeonLane2], b.b[kNeonLane3]},
                        nv.d_scale[kChannelB], nv.d_offset[kChannelB], out.d[kChannelB] + dst_off);
    apply_and_store_4x4({b.r[kNeonLane0], b.r[kNeonLane1], b.r[kNeonLane2], b.r[kNeonLane3]},
                        nv.s_scale[kChannelR], nv.s_offset[kChannelR], out.s[kChannelR] + dst_off);
    apply_and_store_4x4({b.g[kNeonLane0], b.g[kNeonLane1], b.g[kNeonLane2], b.g[kNeonLane3]},
                        nv.s_scale[kChannelG], nv.s_offset[kChannelG], out.s[kChannelG] + dst_off);
    apply_and_store_4x4({b.b[kNeonLane0], b.b[kNeonLane1], b.b[kNeonLane2], b.b[kNeonLane3]},
                        nv.s_scale[kChannelB], nv.s_offset[kChannelB], out.s[kChannelB] + dst_off);
}

// 单行处理 — 16 像素/block, 写入 6 路
static inline void neon_normalize_row(
    const uint8_t* src_row, int blocks_per_row,
    const OutputPlanes& out, const NeonNormVectors& nv)
{
    for (int blk = 0; blk < blocks_per_row; ++blk) {
        F32x16 b = load_widen_block(src_row + blk * kBlockBytes);
        store_block_to_outputs(b, blk * kBlockPixels, out, nv);
    }
}

static void neon_normalize(
    const uint8_t* __restrict input,
    float* __restrict output, int num_threads)
{
    constexpr int blocks_per_row = kOutputSize / kBlockPixels;

    // 6 路输出 — DINOv2 (d) + SigLIP (s)
    OutputPlanes planes;
    planes.d[kChannelR] = output;
    planes.d[kChannelG] = output + kOutputPlane;
    planes.d[kChannelB] = output + kInputChannels * kOutputPlane;  // 2 * kOutputPlane
    planes.s[kChannelR] = output + (kInputChannels + kChannelR) * kOutputPlane;  // 3 *
    planes.s[kChannelG] = output + (kInputChannels + kChannelG) * kOutputPlane;  // 4 *
    planes.s[kChannelB] = output + (kInputChannels + kChannelB) * kOutputPlane;  // 5 *

    // 预填 scale/offset 向量
    NeonNormVectors nv;
    nv.d_scale[kChannelR] = vdupq_n_f32(d_scale[kChannelR]);
    nv.d_scale[kChannelG] = vdupq_n_f32(d_scale[kChannelG]);
    nv.d_scale[kChannelB] = vdupq_n_f32(d_scale[kChannelB]);
    nv.d_offset[kChannelR] = vdupq_n_f32(d_offset[kChannelR]);
    nv.d_offset[kChannelG] = vdupq_n_f32(d_offset[kChannelG]);
    nv.d_offset[kChannelB] = vdupq_n_f32(d_offset[kChannelB]);
    nv.s_scale[kChannelR] = vdupq_n_f32(s_scale[kChannelR]);
    nv.s_scale[kChannelG] = vdupq_n_f32(s_scale[kChannelG]);
    nv.s_scale[kChannelB] = vdupq_n_f32(s_scale[kChannelB]);
    nv.s_offset[kChannelR] = vdupq_n_f32(s_offset[kChannelR]);
    nv.s_offset[kChannelG] = vdupq_n_f32(s_offset[kChannelG]);
    nv.s_offset[kChannelB] = vdupq_n_f32(s_offset[kChannelB]);

#pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int row = 0; row < kOutputSize; ++row) {
        const uint8_t* src_row = input + row * kOutputSize * kInputChannels;
        OutputPlanes row_out;
        for (int c = 0; c < kInputChannels; ++c) {
            row_out.d[c] = planes.d[c] + row * kOutputSize;
            row_out.s[c] = planes.s[c] + row * kOutputSize;
        }
        neon_normalize_row(src_row, blocks_per_row, row_out, nv);
    }
}

#endif  // defined(__aarch64__) && !defined(__APPLE__)

// ============================================================================
// 合并算子入口: resize (如需) + normalize → [6, 224, 224]
// ============================================================================

static void run_resize(const uint8_t* in_ptr, uint8_t* out_ptr,
                       int H, int W, int num_threads) {
#if defined(__aarch64__) && !defined(__APPLE__)
    neon_resize(in_ptr, out_ptr, H, W, num_threads);
#else
    scalar_resize(in_ptr, out_ptr, H, W, num_threads);
#endif
}

static void run_normalize(const uint8_t* in_ptr, float* out_ptr,
                          int H, int W, int num_threads) {
#if defined(__aarch64__) && !defined(__APPLE__)
    neon_normalize(in_ptr, out_ptr, num_threads);
#else
    scalar_normalize(in_ptr, out_ptr, H, W, num_threads);
#endif
}

torch::Tensor openvla_fused_preprocess(
    const torch::Tensor& input, int64_t num_threads)
{
    init_constants();

    TORCH_CHECK(input.device().is_cpu(), "input must be on CPU");
    TORCH_CHECK(input.dtype() == torch::kUInt8, "input must be uint8");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dim() == kInputDims && input.size(kDimC) == kInputChannels,
                "input must be [H, W, ", kInputChannels, "], got [",
                input.size(kDimH), ", ", input.size(kDimW), ", ", input.size(kDimC), "]");

    const int H = static_cast<int>(input.size(kDimH));
    const int W = static_cast<int>(input.size(kDimW));
    const bool need_resize = (H != kOutputSize || W != kOutputSize);

    // 分配输出 [6, 224, 224] float32 (Python torch.stack 加 batch 维)
    auto output = torch::empty({kOutputChannels, kOutputSize, kOutputSize},
                                torch::TensorOptions()
                                    .dtype(torch::kFloat32)
                                    .device(torch::kCPU)
                                    .memory_format(torch::MemoryFormat::Contiguous));

    const int n_threads = static_cast<int>(num_threads);

    if (need_resize) {
        auto resize_buf = torch::empty({kOutputSize, kOutputSize, kInputChannels},
                                       torch::TensorOptions()
                                           .dtype(torch::kUInt8)
                                           .device(torch::kCPU)
                                           .memory_format(torch::MemoryFormat::Contiguous));
        run_resize(input.data_ptr<uint8_t>(),
                   resize_buf.data_ptr<uint8_t>(), H, W, n_threads);
        run_normalize(resize_buf.data_ptr<uint8_t>(),
                      output.data_ptr<float>(), kOutputSize, kOutputSize, n_threads);
    } else {
        run_normalize(input.data_ptr<uint8_t>(),
                      output.data_ptr<float>(), H, W, n_threads);
    }

    return output;
}

// ============================================================================
// extern "C" 包装 — 供 Python ctypes 调用
// ============================================================================
// 因为 torch 2.13.0 ARM 平台 TORCH_LIBRARY 静态初始化不可靠,
// 通过 ctypes 直接调用 C 包装函数, 在 Python 端用 torch.library 注册算子.

extern "C" {

void openvla_fused_preprocess_c(
    const uint8_t* input_data,
    int H, int W,
    float* output_data,
    int64_t num_threads)
{
    // 构造非持有 torch tensor (只读访问 input)
    auto input = torch::from_blob(
        const_cast<uint8_t*>(input_data),
        {H, W, kInputChannels},
        torch::kUInt8
    ).contiguous();  // 确保内存连续

    auto output = openvla_fused_preprocess(input, num_threads);

    // 拷贝结果到预分配缓冲
    // output 形状: [6, 224, 224] float32
    constexpr size_t kOutputFloats = static_cast<size_t>(kOutputChannels)
                                   * static_cast<size_t>(kOutputPlane);
    std::memcpy(output_data, output.data_ptr<float>(), kOutputFloats * sizeof(float));
}

}  // extern "C"
