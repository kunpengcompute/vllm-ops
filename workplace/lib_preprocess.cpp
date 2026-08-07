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
#include <cstring>

// ============================================================================
// 常量
// ============================================================================

static const float D_SCALE[3] = {
    1.0f / (255.0f * 0.228515625f),
    1.0f / (255.0f * 0.2236328125f),
    1.0f / (255.0f * 0.224609375f)};
static const float D_OFFSET[3] = {
    0.484375f / 0.228515625f,
    0.455078125f / 0.2236328125f,
    0.40625f / 0.224609375f};
static const float S_SCALE[3] = {
    1.0f / (255.0f * 0.5f), 1.0f / (255.0f * 0.5f), 1.0f / (255.0f * 0.5f)};
static const float S_OFFSET[3] = {1.0f, 1.0f, 1.0f};

// ============================================================================
// Cubic + LUT
// ============================================================================

static inline double cubic(double x) {
    if (x < 0.0) x = -x;
    if (x < 1.0) return (1.5 * x - 2.5) * x * x + 1.0;
    if (x < 2.0) return ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0;
    return 0.0;
}

struct Lut {
    int out_sz, ksize;
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
    delete[] lut.start; delete[] lut.count; delete[] lut.w;
}

// ============================================================================
// 合并前处理 — ctypes 入口
// ============================================================================

extern "C" void combined_preprocess(
    const uint8_t* input,
    float* output,
    int H_in, int W_in, int num_threads)
{
    constexpr int H_out = 224, W_out = 224;
    constexpr int HW = H_out * W_out;
    const int STRIDE = W_out * 3;

    // --- Step 1: bicubic resize → uint8 [224, 224, 3] ---
    uint8_t* resized = new uint8_t[HW * 3];

    Lut xlut = build_lut(W_out, W_in);
    Lut ylut = build_lut(H_out, H_in);
    float* mid = new float[H_in * STRIDE];

    // Horizontal pass
#pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int y = 0; y < H_in; y++) {
        const uint8_t* src = input + y * W_in * 3;
        float* dst = mid + y * STRIDE;
        for (int x = 0; x < W_out; x++) {
            int n = xlut.count[x], xmin = xlut.start[x];
            const float* w_row = xlut.w + x * xlut.ksize;
            float R = 0, G = 0, B = 0;
            for (int k = 0; k < n; k++) {
                int off = (xmin + k) * 3; float wk = w_row[k];
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
        uint8_t* dst = resized + y * W_out * 3;
        int n = ylut.count[y], ymin = ylut.start[y];
        const float* w_row = ylut.w + y * ylut.ksize;
        for (int x = 0; x < W_out; x++) {
            float R = 0, G = 0, B = 0;
            for (int k = 0; k < n; k++) {
                const float* src_row = mid + (ymin + k) * STRIDE + x * 3;
                float wk = w_row[k];
                R += src_row[0] * wk; G += src_row[1] * wk; B += src_row[2] * wk;
            }
            int off = x * 3;
            dst[off + 0] = (uint8_t)std::max(0.0f, std::min(255.0f, roundf(R)));
            dst[off + 1] = (uint8_t)std::max(0.0f, std::min(255.0f, roundf(G)));
            dst[off + 2] = (uint8_t)std::max(0.0f, std::min(255.0f, roundf(B)));
        }
    }

    free_lut(xlut); free_lut(ylut);
    delete[] mid;

    // --- Step 2: NEON normalize → float32 [6, 224, 224] ---
    const float32x4_t d_s0 = vdupq_n_f32(D_SCALE[0]);
    const float32x4_t d_s1 = vdupq_n_f32(D_SCALE[1]);
    const float32x4_t d_s2 = vdupq_n_f32(D_SCALE[2]);
    const float32x4_t d_o0 = vdupq_n_f32(D_OFFSET[0]);
    const float32x4_t d_o1 = vdupq_n_f32(D_OFFSET[1]);
    const float32x4_t d_o2 = vdupq_n_f32(D_OFFSET[2]);
    const float32x4_t s_s0 = vdupq_n_f32(S_SCALE[0]);
    const float32x4_t s_s1 = vdupq_n_f32(S_SCALE[1]);
    const float32x4_t s_s2 = vdupq_n_f32(S_SCALE[2]);
    const float32x4_t s_o0 = vdupq_n_f32(S_OFFSET[0]);
    const float32x4_t s_o1 = vdupq_n_f32(S_OFFSET[1]);
    const float32x4_t s_o2 = vdupq_n_f32(S_OFFSET[2]);

    float* out_d0 = output;
    float* out_d1 = output + HW;
    float* out_d2 = output + 2 * HW;
    float* out_s0 = output + 3 * HW;
    float* out_s1 = output + 4 * HW;
    float* out_s2 = output + 5 * HW;

#pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int row = 0; row < H_out; row++) {
        const uint8_t* src_row = resized + row * W_out * 3;
        float* dst_d0 = out_d0 + row * W_out;
        float* dst_d1 = out_d1 + row * W_out;
        float* dst_d2 = out_d2 + row * W_out;
        float* dst_s0 = out_s0 + row * W_out;
        float* dst_s1 = out_s1 + row * W_out;
        float* dst_s2 = out_s2 + row * W_out;

        for (int blk = 0; blk < 14; blk++) {
            uint8x16x3_t rgb = vld3q_u8(src_row + blk * 48);

            uint16x8_t r16_l = vmovl_u8(vget_low_u8(rgb.val[0]));
            uint16x8_t r16_h = vmovl_u8(vget_high_u8(rgb.val[0]));
            uint16x8_t g16_l = vmovl_u8(vget_low_u8(rgb.val[1]));
            uint16x8_t g16_h = vmovl_u8(vget_high_u8(rgb.val[1]));
            uint16x8_t b16_l = vmovl_u8(vget_low_u8(rgb.val[2]));
            uint16x8_t b16_h = vmovl_u8(vget_high_u8(rgb.val[2]));

            float32x4_t rf0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(r16_l)));
            float32x4_t rf1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(r16_l)));
            float32x4_t rf2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(r16_h)));
            float32x4_t rf3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(r16_h)));
            float32x4_t gf0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(g16_l)));
            float32x4_t gf1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(g16_l)));
            float32x4_t gf2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(g16_h)));
            float32x4_t gf3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(g16_h)));
            float32x4_t bf0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(b16_l)));
            float32x4_t bf1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(b16_l)));
            float32x4_t bf2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(b16_h)));
            float32x4_t bf3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(b16_h)));

            int off = blk * 16;
            vst1q_f32(dst_d0 + off, vsubq_f32(vmulq_f32(rf0, d_s0), d_o0));
            vst1q_f32(dst_d0 + off + 4, vsubq_f32(vmulq_f32(rf1, d_s0), d_o0));
            vst1q_f32(dst_d0 + off + 8, vsubq_f32(vmulq_f32(rf2, d_s0), d_o0));
            vst1q_f32(dst_d0 + off + 12, vsubq_f32(vmulq_f32(rf3, d_s0), d_o0));
            vst1q_f32(dst_d1 + off, vsubq_f32(vmulq_f32(gf0, d_s1), d_o1));
            vst1q_f32(dst_d1 + off + 4, vsubq_f32(vmulq_f32(gf1, d_s1), d_o1));
            vst1q_f32(dst_d1 + off + 8, vsubq_f32(vmulq_f32(gf2, d_s1), d_o1));
            vst1q_f32(dst_d1 + off + 12, vsubq_f32(vmulq_f32(gf3, d_s1), d_o1));
            vst1q_f32(dst_d2 + off, vsubq_f32(vmulq_f32(bf0, d_s2), d_o2));
            vst1q_f32(dst_d2 + off + 4, vsubq_f32(vmulq_f32(bf1, d_s2), d_o2));
            vst1q_f32(dst_d2 + off + 8, vsubq_f32(vmulq_f32(bf2, d_s2), d_o2));
            vst1q_f32(dst_d2 + off + 12, vsubq_f32(vmulq_f32(bf3, d_s2), d_o2));
            vst1q_f32(dst_s0 + off, vsubq_f32(vmulq_f32(rf0, s_s0), s_o0));
            vst1q_f32(dst_s0 + off + 4, vsubq_f32(vmulq_f32(rf1, s_s0), s_o0));
            vst1q_f32(dst_s0 + off + 8, vsubq_f32(vmulq_f32(rf2, s_s0), s_o0));
            vst1q_f32(dst_s0 + off + 12, vsubq_f32(vmulq_f32(rf3, s_s0), s_o0));
            vst1q_f32(dst_s1 + off, vsubq_f32(vmulq_f32(gf0, s_s1), s_o1));
            vst1q_f32(dst_s1 + off + 4, vsubq_f32(vmulq_f32(gf1, s_s1), s_o1));
            vst1q_f32(dst_s1 + off + 8, vsubq_f32(vmulq_f32(gf2, s_s1), s_o1));
            vst1q_f32(dst_s1 + off + 12, vsubq_f32(vmulq_f32(gf3, s_s1), s_o1));
            vst1q_f32(dst_s2 + off, vsubq_f32(vmulq_f32(bf0, s_s2), s_o2));
            vst1q_f32(dst_s2 + off + 4, vsubq_f32(vmulq_f32(bf1, s_s2), s_o2));
            vst1q_f32(dst_s2 + off + 8, vsubq_f32(vmulq_f32(bf2, s_s2), s_o2));
            vst1q_f32(dst_s2 + off + 12, vsubq_f32(vmulq_f32(bf3, s_s2), s_o2));
        }
    }
    delete[] resized;
}
