/*
 * OpenVLA 合并前处理 — 共享库 (.so), 供 Python ctypes 直接调用
 *
 * 编译:
 *   g++ -std=c++14 -march=armv8.2-a+fp16+dotprod -fopenmp -O3 \
 *       -shared -fPIC -o libpreprocess.so lib_preprocess.cpp
 */

#include <arm_neon.h>
#include <omp.h>

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
constexpr int kOutputSize     = 224;   // 输出 H × W
constexpr int kOutputPlane    = kOutputSize * kOutputSize;

// RGB 通道索引
constexpr int kChannelR = 0;
constexpr int kChannelG = 1;
constexpr int kChannelB = 2;

// NEON SIMD 几何
constexpr int kBlockPixels    = 16;    // 16 像素 (uint8) / block
constexpr int kBlockBytes     = kBlockPixels * kInputChannels;  // 48 bytes
constexpr int kBlocksPerRow   = kOutputSize / kBlockPixels;  // 224 / 16 = 14
constexpr int kFloatsPerLane  = 4;     // 4 × float32 = 128 bit
constexpr int kLaneBytes      = 4;     // 1 × float32 = 4 bytes

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

// Bicubic 快速路径 — 当 n == 4 时用 NEON 4-tap 展开 (此文件无 fast path, 保留常量供其他调用方)
constexpr int kBicubicFastPathN = 4;

// Cubic kernel (Catmull-Rom, a = -0.5)
constexpr double kCubicA              = -0.5;
constexpr double kCubicB              = 1.5;
constexpr double kCubicC              = 2.5;
constexpr double kCubicScale          = 2.0;
constexpr double kCubicSupportRadius  = 2.0;
constexpr double kCubicQuadraticCoeff = 4.0;
constexpr double kCubicConstantTerm   = 2.0;
constexpr double kCubicInnerBoundary  = 1.0;
constexpr double kPixelCenterOffset   = 0.5;

// uint8 → 归一化浮点的除数
constexpr float kUint8ToFloatDivisor = 255.0f;

}  // namespace

// ============================================================================
// 预计算归一化常量
// ============================================================================

static const float D_SCALE[kInputChannels] = {
    1.0f / (kUint8ToFloatDivisor * 0.228515625f),
    1.0f / (kUint8ToFloatDivisor * 0.2236328125f),
    1.0f / (kUint8ToFloatDivisor * 0.224609375f)};
static const float D_OFFSET[kInputChannels] = {
    0.484375f / 0.228515625f,
    0.455078125f / 0.2236328125f,
    0.40625f / 0.224609375f};
static const float S_SCALE[kInputChannels] = {
    1.0f / (kUint8ToFloatDivisor * 0.5f),
    1.0f / (kUint8ToFloatDivisor * 0.5f),
    1.0f / (kUint8ToFloatDivisor * 0.5f)};
static const float S_OFFSET[kInputChannels] = {1.0f, 1.0f, 1.0f};

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
// Cubic + LUT
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

struct Lut {
    int out_sz, ksize;
    int* start;
    int* count;
    float* w;
};

// 单个输出像素的 LUT 计算
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

static void free_lut(Lut& lut) {
    std::free(lut.start);
    std::free(lut.count);
    std::free(lut.w);
    lut.start = nullptr;
    lut.count = nullptr;
    lut.w = nullptr;
}

// ============================================================================
// Bicubic resize — 水平 + 垂直 pass
// ============================================================================

// resize pass 公共参数 — 把 5 个相关标量打包, 减少单函数参数数量
struct ResizePassParams {
    int H;             // 行数: 水平 pass 是 H_in, 垂直 pass 是 H_out
    int W_out;         // 输出宽度
    int num_threads;   // OMP 线程数
    const Lut* lut;    // x 方向 (水平) 或 y 方向 (垂直) LUT
};

static void resize_horizontal(
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

static inline uint8_t clamp_to_u8(float v) {
    return static_cast<uint8_t>(
        std::max(0.0f, std::min(kUint8ToFloatDivisor, std::round(v))));
}

static void resize_vertical(
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

static void bicubic_resize(
    const uint8_t* input, uint8_t* resized, int H_in, int W_in, int num_threads)
{
    Lut xlut = build_lut(kOutputSize, W_in);
    Lut ylut = build_lut(kOutputSize, H_in);
    const int stride = kOutputSize * kInputChannels;
    float* mid = static_cast<float*>(
        checked_alloc(static_cast<size_t>(H_in) * static_cast<size_t>(stride),
                      sizeof(float), "mid"));
    ResizePassParams hp{H_in, kOutputSize, num_threads, &xlut};
    ResizePassParams vp{kOutputSize, kOutputSize, num_threads, &ylut};
    resize_horizontal(input, mid, W_in, hp);
    resize_vertical(mid, resized, vp);
    free_lut(xlut);
    free_lut(ylut);
    std::free(mid);
}

// ============================================================================
// NEON normalize — 16 像素/block, 6 路输出
// ============================================================================

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
    F32x16 out;
    out.r[kNeonLane0] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(r16_l)));
    out.r[kNeonLane1] = vcvtq_f32_u32(vmovl_u16(vget_high_u16(r16_l)));
    out.r[kNeonLane2] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(r16_h)));
    out.r[kNeonLane3] = vcvtq_f32_u32(vmovl_u16(vget_high_u16(r16_h)));
    out.g[kNeonLane0] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(g16_l)));
    out.g[kNeonLane1] = vcvtq_f32_u32(vmovl_u16(vget_high_u16(g16_l)));
    out.g[kNeonLane2] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(g16_h)));
    out.g[kNeonLane3] = vcvtq_f32_u32(vmovl_u16(vget_high_u16(g16_h)));
    out.b[kNeonLane0] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(b16_l)));
    out.b[kNeonLane1] = vcvtq_f32_u32(vmovl_u16(vget_high_u16(b16_l)));
    out.b[kNeonLane2] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(b16_h)));
    out.b[kNeonLane3] = vcvtq_f32_u32(vmovl_u16(vget_high_u16(b16_h)));
    return out;
}

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

static void neon_normalize(
    const uint8_t* resized, float* output, int num_threads)
{
    // 6 路输出 — DINOv2 (d) + SigLIP (s)
    OutputPlanes planes;
    planes.d[kChannelR] = output;
    planes.d[kChannelG] = output + kOutputPlane;
    planes.d[kChannelB] = output + kInputChannels * kOutputPlane;  // 2 * kOutputPlane
    planes.s[kChannelR] = output + (kInputChannels + kChannelR) * kOutputPlane;  // 3 *
    planes.s[kChannelG] = output + (kInputChannels + kChannelG) * kOutputPlane;  // 4 *
    planes.s[kChannelB] = output + (kInputChannels + kChannelB) * kOutputPlane;  // 5 *

    NeonNormVectors nv;
    nv.d_scale[kChannelR] = vdupq_n_f32(D_SCALE[kChannelR]);
    nv.d_scale[kChannelG] = vdupq_n_f32(D_SCALE[kChannelG]);
    nv.d_scale[kChannelB] = vdupq_n_f32(D_SCALE[kChannelB]);
    nv.d_offset[kChannelR] = vdupq_n_f32(D_OFFSET[kChannelR]);
    nv.d_offset[kChannelG] = vdupq_n_f32(D_OFFSET[kChannelG]);
    nv.d_offset[kChannelB] = vdupq_n_f32(D_OFFSET[kChannelB]);
    nv.s_scale[kChannelR] = vdupq_n_f32(S_SCALE[kChannelR]);
    nv.s_scale[kChannelG] = vdupq_n_f32(S_SCALE[kChannelG]);
    nv.s_scale[kChannelB] = vdupq_n_f32(S_SCALE[kChannelB]);
    nv.s_offset[kChannelR] = vdupq_n_f32(S_OFFSET[kChannelR]);
    nv.s_offset[kChannelG] = vdupq_n_f32(S_OFFSET[kChannelG]);
    nv.s_offset[kChannelB] = vdupq_n_f32(S_OFFSET[kChannelB]);

#pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int row = 0; row < kOutputSize; ++row) {
        const uint8_t* src_row = resized + row * kOutputSize * kInputChannels;
        OutputPlanes row_out;
        for (int c = 0; c < kInputChannels; ++c) {
            row_out.d[c] = planes.d[c] + row * kOutputSize;
            row_out.s[c] = planes.s[c] + row * kOutputSize;
        }
        for (int blk = 0; blk < kBlocksPerRow; ++blk) {
            F32x16 b = load_widen_block(src_row + blk * kBlockBytes);
            store_block_to_outputs(b, blk * kBlockPixels, row_out, nv);
        }
    }
}

// ============================================================================
// 合并前处理 — ctypes 入口
// ============================================================================

extern "C" void combined_preprocess(
    const uint8_t* input,
    float* output,
    int H_in, int W_in, int num_threads)
{
    constexpr size_t kResizedBytes = static_cast<size_t>(kOutputPlane) * kInputChannels;
    uint8_t* resized = static_cast<uint8_t*>(checked_alloc(kResizedBytes, 1, "resized"));

    bicubic_resize(input, resized, H_in, W_in, num_threads);
    neon_normalize(resized, output, num_threads);

    std::free(resized);
}
