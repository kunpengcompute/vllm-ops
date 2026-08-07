/*
 * OpenVLA 图片前处理 ARM NEON fused kernel — v0.6.0
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
#include <cstring>

// ============================================================================
// 预计算归一化常量
// ============================================================================

static const float IMAGENET_MEAN[3] = {0.484375f, 0.455078125f, 0.40625f};
static const float IMAGENET_STD[3]  = {0.228515625f, 0.2236328125f, 0.224609375f};
static const float SIGLIP_MEAN[3] = {0.5f, 0.5f, 0.5f};
static const float SIGLIP_STD[3]  = {0.5f, 0.5f, 0.5f};

static float d_scale[3], d_offset[3];
static float s_scale[3], s_offset[3];
static bool constants_initialized = false;

static void init_constants() {
    if (constants_initialized) return;
    for (int c = 0; c < 3; c++) {
        d_scale[c]  = 1.0f / (255.0f * IMAGENET_STD[c]);
        d_offset[c] = IMAGENET_MEAN[c] / IMAGENET_STD[c];
        s_scale[c]  = 1.0f / (255.0f * SIGLIP_STD[c]);
        s_offset[c] = SIGLIP_MEAN[c] / SIGLIP_STD[c];
    }
    constants_initialized = true;
}

// ============================================================================
// Cubic kernel (Catmull-Rom, a=-0.5) — 与 Pillow bicubic_filter 完全一致
// ============================================================================

static inline double cubic(double x) {
    if (x < 0.0) x = -x;
    if (x < 1.0) return (1.5 * x - 2.5) * x * x + 1.0;
    if (x < 2.0) return ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0;
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

static Lut build_lut(int out_sz, int in_sz) {
    double scale = (double)in_sz / (double)out_sz;
    double filterscale = (scale > 1.0) ? scale : 1.0;
    double support = 2.0 * filterscale;
    int ksize = (int)ceil(support) * 2 + 1;

    Lut lut;
    lut.out_sz = out_sz;
    lut.ksize = ksize;
    lut.start = new int[out_sz];
    lut.count = new int[out_sz];
    lut.w = new float[out_sz * ksize]();

    double inv_filterscale = 1.0 / filterscale;

    for (int i = 0; i < out_sz; i++) {
        double center = ((double)i + 0.5) * scale;

        int xmin = (int)(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = (int)(center + support + 0.5);
        if (xmax > in_sz) xmax = in_sz;
        int n = xmax - xmin;

        float* w_row = lut.w + i * ksize;
        double ww = 0.0;
        for (int k = 0; k < n; k++) {
            int sx = xmin + k;
            double dist = ((double)sx + 0.5 - center) * inv_filterscale;
            float w = (float)cubic(dist);
            w_row[k] = w;
            ww += (double)w;
        }
        if (ww != 0.0) {
            float inv = (float)(1.0 / ww);
            for (int k = 0; k < n; k++) w_row[k] *= inv;
        }
        lut.start[i] = xmin;
        lut.count[i] = n;
    }
    return lut;
}

static void free_lut(Lut& lut) {
    delete[] lut.start;
    delete[] lut.count;
    delete[] lut.w;
}

// ============================================================================
// NEON bicubic resize — Pillow 兼容, OMP 多线程
// ============================================================================

#if defined(__aarch64__) && !defined(__APPLE__)

static void neon_resize(
    const uint8_t* input, uint8_t* output,
    int H_in, int W_in, int num_threads)
{
    constexpr int H_out = 224, W_out = 224;
    Lut xlut = build_lut(W_out, W_in);
    Lut ylut = build_lut(H_out, H_in);
    const int STRIDE = W_out * 3;
    float* mid = new float[H_in * STRIDE];

    // Horizontal pass
#pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int y = 0; y < H_in; y++) {
        const uint8_t* src = input + y * W_in * 3;
        float* dst = mid + y * STRIDE;

        for (int x = 0; x < W_out; x++) {
            int n = xlut.count[x];
            int xmin = xlut.start[x];
            const float* w_row = xlut.w + x * xlut.ksize;

            if (n == 4) {
                int i0 = (xmin + 0) * 3, i1 = (xmin + 1) * 3;
                int i2 = (xmin + 2) * 3, i3 = (xmin + 3) * 3;

                float rv[4] = {(float)src[i0],     (float)src[i1],
                               (float)src[i2],     (float)src[i3]};
                float gv[4] = {(float)src[i0 + 1], (float)src[i1 + 1],
                               (float)src[i2 + 1], (float)src[i3 + 1]};
                float bv[4] = {(float)src[i0 + 2], (float)src[i1 + 2],
                               (float)src[i2 + 2], (float)src[i3 + 2]};

                float32x4_t rv4 = vld1q_f32(rv), gv4 = vld1q_f32(gv);
                float32x4_t bv4 = vld1q_f32(bv), wv4 = vld1q_f32(w_row);

                float R = vaddvq_f32(vmulq_f32(rv4, wv4));
                float G = vaddvq_f32(vmulq_f32(gv4, wv4));
                float B = vaddvq_f32(vmulq_f32(bv4, wv4));
                dst[x * 3 + 0] = std::max(0.0f, std::min(255.0f, roundf(R)));
                dst[x * 3 + 1] = std::max(0.0f, std::min(255.0f, roundf(G)));
                dst[x * 3 + 2] = std::max(0.0f, std::min(255.0f, roundf(B)));
            } else {
                float R = 0.0f, G = 0.0f, B = 0.0f;
                for (int k = 0; k < n; k++) {
                    int off = (xmin + k) * 3;
                    float wk = w_row[k];
                    R += (float)src[off + 0] * wk;
                    G += (float)src[off + 1] * wk;
                    B += (float)src[off + 2] * wk;
                }
                dst[x * 3 + 0] = std::max(0.0f, std::min(255.0f, roundf(R)));
                dst[x * 3 + 1] = std::max(0.0f, std::min(255.0f, roundf(G)));
                dst[x * 3 + 2] = std::max(0.0f, std::min(255.0f, roundf(B)));
            }
        }
    }

    // Vertical pass
#pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int y = 0; y < H_out; y++) {
        uint8_t* dst = output + y * W_out * 3;
        int n = ylut.count[y];
        int ymin = ylut.start[y];
        const float* w_row = ylut.w + y * ylut.ksize;

        for (int x = 0; x < W_out; x++) {
            float R, G, B;

            if (n == 4) {
                const float* s0 = mid + (ymin + 0) * STRIDE + x * 3;
                const float* s1 = mid + (ymin + 1) * STRIDE + x * 3;
                const float* s2 = mid + (ymin + 2) * STRIDE + x * 3;
                const float* s3 = mid + (ymin + 3) * STRIDE + x * 3;

                float rv[4] = {s0[0], s1[0], s2[0], s3[0]};
                float gv[4] = {s0[1], s1[1], s2[1], s3[1]};
                float bv[4] = {s0[2], s1[2], s2[2], s3[2]};

                float32x4_t rv4 = vld1q_f32(rv), gv4 = vld1q_f32(gv);
                float32x4_t bv4 = vld1q_f32(bv), wv4 = vld1q_f32(w_row);

                R = vaddvq_f32(vmulq_f32(rv4, wv4));
                G = vaddvq_f32(vmulq_f32(gv4, wv4));
                B = vaddvq_f32(vmulq_f32(bv4, wv4));
            } else {
                R = 0.0f; G = 0.0f; B = 0.0f;
                for (int k = 0; k < n; k++) {
                    const float* src_row = mid + (ymin + k) * STRIDE + x * 3;
                    float wk = w_row[k];
                    R += src_row[0] * wk;
                    G += src_row[1] * wk;
                    B += src_row[2] * wk;
                }
            }

            int off = x * 3;
            dst[off + 0] = (uint8_t)std::max(0.0f, std::min(255.0f, roundf(R)));
            dst[off + 1] = (uint8_t)std::max(0.0f, std::min(255.0f, roundf(G)));
            dst[off + 2] = (uint8_t)std::max(0.0f, std::min(255.0f, roundf(B)));
        }
    }

    free_lut(xlut); free_lut(ylut);
    delete[] mid;
}

#endif  // defined(__aarch64__) && !defined(__APPLE__)

// ============================================================================
// 标量 resize fallback (非 ARM 平台)
// ============================================================================

static void scalar_resize(
    const uint8_t* input, uint8_t* output,
    int H_in, int W_in, int num_threads)
{
    constexpr int H_out = 224, W_out = 224;
    Lut xlut = build_lut(W_out, W_in);
    Lut ylut = build_lut(H_out, H_in);
    const int STRIDE = W_out * 3;
    float* mid = new float[H_in * STRIDE];

    // Horizontal pass (标量, OMP 并行)
#pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int y = 0; y < H_in; y++) {
        const uint8_t* src = input + y * W_in * 3;
        float* dst = mid + y * STRIDE;

        for (int x = 0; x < W_out; x++) {
            int n = xlut.count[x];
            int xmin = xlut.start[x];
            const float* w_row = xlut.w + x * xlut.ksize;

            float R = 0.0f, G = 0.0f, B = 0.0f;
            for (int k = 0; k < n; k++) {
                int off = (xmin + k) * 3;
                float wk = w_row[k];
                R += (float)src[off + 0] * wk;
                G += (float)src[off + 1] * wk;
                B += (float)src[off + 2] * wk;
            }
            dst[x * 3 + 0] = std::max(0.0f, std::min(255.0f, roundf(R)));
            dst[x * 3 + 1] = std::max(0.0f, std::min(255.0f, roundf(G)));
            dst[x * 3 + 2] = std::max(0.0f, std::min(255.0f, roundf(B)));
        }
    }

    // Vertical pass
#pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int y = 0; y < H_out; y++) {
        uint8_t* dst = output + y * W_out * 3;
        int n = ylut.count[y];
        int ymin = ylut.start[y];
        const float* w_row = ylut.w + y * ylut.ksize;

        for (int x = 0; x < W_out; x++) {
            float R = 0.0f, G = 0.0f, B = 0.0f;
            for (int k = 0; k < n; k++) {
                const float* src_row = mid + (ymin + k) * STRIDE + x * 3;
                float wk = w_row[k];
                R += src_row[0] * wk;
                G += src_row[1] * wk;
                B += src_row[2] * wk;
            }
            int off = x * 3;
            dst[off + 0] = (uint8_t)std::max(0.0f, std::min(255.0f, roundf(R)));
            dst[off + 1] = (uint8_t)std::max(0.0f, std::min(255.0f, roundf(G)));
            dst[off + 2] = (uint8_t)std::max(0.0f, std::min(255.0f, roundf(B)));
        }
    }

    free_lut(xlut); free_lut(ylut);
    delete[] mid;
}

// ============================================================================
// 标量归一化 (非 ARM 平台 fallback)
// ============================================================================

static void scalar_normalize(
    const uint8_t* __restrict input,
    float* __restrict output,
    int H, int W, int num_threads)
{
    const int HW = H * W;
#pragma omp parallel for num_threads(num_threads)
    for (int idx = 0; idx < HW; idx++) {
        int r = idx / W, c = idx % W;
        int src_off = (r * W + c) * 3;
        float R = (float)input[src_off + 0];
        float G = (float)input[src_off + 1];
        float B = (float)input[src_off + 2];
        int dst_base = r * W + c;

        output[0 * HW + dst_base] = R * d_scale[0] - d_offset[0];
        output[1 * HW + dst_base] = G * d_scale[1] - d_offset[1];
        output[2 * HW + dst_base] = B * d_scale[2] - d_offset[2];
        output[3 * HW + dst_base] = R * s_scale[0] - s_offset[0];
        output[4 * HW + dst_base] = G * s_scale[1] - s_offset[1];
        output[5 * HW + dst_base] = B * s_scale[2] - s_offset[2];
    }
}

// ============================================================================
// NEON fused normalization (224×224, 16 像素/block)
// ============================================================================

#if defined(__aarch64__) && !defined(__APPLE__)

static void neon_normalize(
    const uint8_t* __restrict input,
    float* __restrict output, int num_threads)
{
    constexpr int H = 224, W = 224, HW = H * W;
    constexpr int BLOCKS_PER_ROW = W / 16;

    const float32x4_t d_s0 = vdupq_n_f32(d_scale[0]);
    const float32x4_t d_s1 = vdupq_n_f32(d_scale[1]);
    const float32x4_t d_s2 = vdupq_n_f32(d_scale[2]);
    const float32x4_t d_o0 = vdupq_n_f32(d_offset[0]);
    const float32x4_t d_o1 = vdupq_n_f32(d_offset[1]);
    const float32x4_t d_o2 = vdupq_n_f32(d_offset[2]);
    const float32x4_t s_s0 = vdupq_n_f32(s_scale[0]);
    const float32x4_t s_s1 = vdupq_n_f32(s_scale[1]);
    const float32x4_t s_s2 = vdupq_n_f32(s_scale[2]);
    const float32x4_t s_o0 = vdupq_n_f32(s_offset[0]);
    const float32x4_t s_o1 = vdupq_n_f32(s_offset[1]);
    const float32x4_t s_o2 = vdupq_n_f32(s_offset[2]);

    float* __restrict out_d0 = output;
    float* __restrict out_d1 = output + HW;
    float* __restrict out_d2 = output + 2 * HW;
    float* __restrict out_s0 = output + 3 * HW;
    float* __restrict out_s1 = output + 4 * HW;
    float* __restrict out_s2 = output + 5 * HW;

#pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int row = 0; row < H; row++) {
        const uint8_t* __restrict src_row = input + row * W * 3;
        float* __restrict dst_row_d0 = out_d0 + row * W;
        float* __restrict dst_row_d1 = out_d1 + row * W;
        float* __restrict dst_row_d2 = out_d2 + row * W;
        float* __restrict dst_row_s0 = out_s0 + row * W;
        float* __restrict dst_row_s1 = out_s1 + row * W;
        float* __restrict dst_row_s2 = out_s2 + row * W;

        for (int blk = 0; blk < BLOCKS_PER_ROW; blk++) {
            uint8x16x3_t rgb = vld3q_u8(src_row + blk * 48);

            uint16x8_t r16_l = vmovl_u8(vget_low_u8(rgb.val[0]));
            uint16x8_t r16_h = vmovl_u8(vget_high_u8(rgb.val[0]));
            uint16x8_t g16_l = vmovl_u8(vget_low_u8(rgb.val[1]));
            uint16x8_t g16_h = vmovl_u8(vget_high_u8(rgb.val[1]));
            uint16x8_t b16_l = vmovl_u8(vget_low_u8(rgb.val[2]));
            uint16x8_t b16_h = vmovl_u8(vget_high_u8(rgb.val[2]));

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

            float32x4_t rf0 = vcvtq_f32_u32(r32_0);
            float32x4_t rf1 = vcvtq_f32_u32(r32_1);
            float32x4_t rf2 = vcvtq_f32_u32(r32_2);
            float32x4_t rf3 = vcvtq_f32_u32(r32_3);
            float32x4_t gf0 = vcvtq_f32_u32(g32_0);
            float32x4_t gf1 = vcvtq_f32_u32(g32_1);
            float32x4_t gf2 = vcvtq_f32_u32(g32_2);
            float32x4_t gf3 = vcvtq_f32_u32(g32_3);
            float32x4_t bf0 = vcvtq_f32_u32(b32_0);
            float32x4_t bf1 = vcvtq_f32_u32(b32_1);
            float32x4_t bf2 = vcvtq_f32_u32(b32_2);
            float32x4_t bf3 = vcvtq_f32_u32(b32_3);

            float32x4_t dd0_0 = vsubq_f32(vmulq_f32(rf0, d_s0), d_o0);
            float32x4_t dd0_1 = vsubq_f32(vmulq_f32(rf1, d_s0), d_o0);
            float32x4_t dd0_2 = vsubq_f32(vmulq_f32(rf2, d_s0), d_o0);
            float32x4_t dd0_3 = vsubq_f32(vmulq_f32(rf3, d_s0), d_o0);
            float32x4_t dd1_0 = vsubq_f32(vmulq_f32(gf0, d_s1), d_o1);
            float32x4_t dd1_1 = vsubq_f32(vmulq_f32(gf1, d_s1), d_o1);
            float32x4_t dd1_2 = vsubq_f32(vmulq_f32(gf2, d_s1), d_o1);
            float32x4_t dd1_3 = vsubq_f32(vmulq_f32(gf3, d_s1), d_o1);
            float32x4_t dd2_0 = vsubq_f32(vmulq_f32(bf0, d_s2), d_o2);
            float32x4_t dd2_1 = vsubq_f32(vmulq_f32(bf1, d_s2), d_o2);
            float32x4_t dd2_2 = vsubq_f32(vmulq_f32(bf2, d_s2), d_o2);
            float32x4_t dd2_3 = vsubq_f32(vmulq_f32(bf3, d_s2), d_o2);
            float32x4_t sd0_0 = vsubq_f32(vmulq_f32(rf0, s_s0), s_o0);
            float32x4_t sd0_1 = vsubq_f32(vmulq_f32(rf1, s_s0), s_o0);
            float32x4_t sd0_2 = vsubq_f32(vmulq_f32(rf2, s_s0), s_o0);
            float32x4_t sd0_3 = vsubq_f32(vmulq_f32(rf3, s_s0), s_o0);
            float32x4_t sd1_0 = vsubq_f32(vmulq_f32(gf0, s_s1), s_o1);
            float32x4_t sd1_1 = vsubq_f32(vmulq_f32(gf1, s_s1), s_o1);
            float32x4_t sd1_2 = vsubq_f32(vmulq_f32(gf2, s_s1), s_o1);
            float32x4_t sd1_3 = vsubq_f32(vmulq_f32(gf3, s_s1), s_o1);
            float32x4_t sd2_0 = vsubq_f32(vmulq_f32(bf0, s_s2), s_o2);
            float32x4_t sd2_1 = vsubq_f32(vmulq_f32(bf1, s_s2), s_o2);
            float32x4_t sd2_2 = vsubq_f32(vmulq_f32(bf2, s_s2), s_o2);
            float32x4_t sd2_3 = vsubq_f32(vmulq_f32(bf3, s_s2), s_o2);

            int dst_off = blk * 16;

            vst1q_f32(dst_row_d0 + dst_off, dd0_0);
            vst1q_f32(dst_row_d0 + dst_off + 4, dd0_1);
            vst1q_f32(dst_row_d0 + dst_off + 8, dd0_2);
            vst1q_f32(dst_row_d0 + dst_off + 12, dd0_3);
            vst1q_f32(dst_row_d1 + dst_off, dd1_0);
            vst1q_f32(dst_row_d1 + dst_off + 4, dd1_1);
            vst1q_f32(dst_row_d1 + dst_off + 8, dd1_2);
            vst1q_f32(dst_row_d1 + dst_off + 12, dd1_3);
            vst1q_f32(dst_row_d2 + dst_off, dd2_0);
            vst1q_f32(dst_row_d2 + dst_off + 4, dd2_1);
            vst1q_f32(dst_row_d2 + dst_off + 8, dd2_2);
            vst1q_f32(dst_row_d2 + dst_off + 12, dd2_3);
            vst1q_f32(dst_row_s0 + dst_off, sd0_0);
            vst1q_f32(dst_row_s0 + dst_off + 4, sd0_1);
            vst1q_f32(dst_row_s0 + dst_off + 8, sd0_2);
            vst1q_f32(dst_row_s0 + dst_off + 12, sd0_3);
            vst1q_f32(dst_row_s1 + dst_off, sd1_0);
            vst1q_f32(dst_row_s1 + dst_off + 4, sd1_1);
            vst1q_f32(dst_row_s1 + dst_off + 8, sd1_2);
            vst1q_f32(dst_row_s1 + dst_off + 12, sd1_3);
            vst1q_f32(dst_row_s2 + dst_off, sd2_0);
            vst1q_f32(dst_row_s2 + dst_off + 4, sd2_1);
            vst1q_f32(dst_row_s2 + dst_off + 8, sd2_2);
            vst1q_f32(dst_row_s2 + dst_off + 12, sd2_3);
        }
    }
}

#endif  // defined(__aarch64__) && !defined(__APPLE__)

// ============================================================================
// 合并算子入口: resize (如需) + normalize → [1, 6, 224, 224]
// ============================================================================

torch::Tensor openvla_fused_preprocess(
    const torch::Tensor& input, int64_t num_threads)
{
    init_constants();

    TORCH_CHECK(input.device().is_cpu(), "input must be on CPU");
    TORCH_CHECK(input.dtype() == torch::kUInt8, "input must be uint8");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dim() == 3 && input.size(2) == 3,
                "input must be [H, W, 3], got [",
                input.size(0), ", ", input.size(1), ", ", input.size(2), "]");

    int H = (int)input.size(0);
    int W = (int)input.size(1);
    bool need_resize = (H != 224 || W != 224);

    // 分配输出 [6, 224, 224] float32 (Python torch.stack 加 batch 维)
    auto output = torch::empty({6, 224, 224},
                                torch::TensorOptions()
                                    .dtype(torch::kFloat32)
                                    .device(torch::kCPU)
                                    .memory_format(torch::MemoryFormat::Contiguous));

    const uint8_t* resize_output_ptr;
    const uint8_t* normalize_input_ptr;

    // 中间 resize 缓冲 (仅 resize 时分配)
    torch::Tensor resize_buf;
    if (need_resize) {
        resize_buf = torch::empty({224, 224, 3},
                                   torch::TensorOptions()
                                       .dtype(torch::kUInt8)
                                       .device(torch::kCPU)
                                       .memory_format(torch::MemoryFormat::Contiguous));
#if defined(__aarch64__) && !defined(__APPLE__)
        neon_resize(input.data_ptr<uint8_t>(),
                    resize_buf.data_ptr<uint8_t>(),
                    H, W, (int)num_threads);
#else
        scalar_resize(input.data_ptr<uint8_t>(),
                      resize_buf.data_ptr<uint8_t>(),
                      H, W, (int)num_threads);
#endif
        normalize_input_ptr = resize_buf.data_ptr<uint8_t>();
    } else {
        normalize_input_ptr = input.data_ptr<uint8_t>();
    }

    // 归一化: uint8 [224,224,3] → float32 [1,6,224,224]
    float* output_ptr = output.data_ptr<float>();
#if defined(__aarch64__) && !defined(__APPLE__)
    neon_normalize(normalize_input_ptr, output_ptr, (int)num_threads);
#else
    scalar_normalize(normalize_input_ptr, output_ptr, 224, 224, (int)num_threads);
#endif

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
        {H, W, 3},
        torch::kUInt8
    ).contiguous();  // 确保内存连续

    auto output = openvla_fused_preprocess(input, num_threads);

    // 拷贝结果到预分配缓冲
    // output 形状: [6, 224, 224] float32
    std::memcpy(output_data, output.data_ptr<float>(), 6 * 224 * 224 * sizeof(float));
}

}  // extern "C"
