#include <stdint.h>
#include <string.h>
#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif
#include "quantize.h"
#include "instruct.h"

void ggml_fp32_to_fp16_row(const float * x, ggml_fp16_t * y, int64_t n)
{
    for (int64_t i = 0; i < n; i++) {
        y[i] = GGML_FP32_TO_FP16(x[i]);
    }
}

void ggml_fp16_to_fp32_row(const ggml_fp16_t * x, float * y, int64_t n)
{
    for (int64_t i = 0; i < n; i++) {
        y[i] = GGML_FP16_TO_FP32(x[i]);
    }
}

#define GGML_F16_SVE_STEP 64
#define GGML_F16_SVE_ARR 4
#define GGML_F16_SVE_EPR 16
                      
void ggml_vec_dot_f16(int n, float * __restrict__ s, size_t bs, ggml_fp16_t * __restrict__ x, size_t bx, ggml_fp16_t * __restrict__ y, size_t by, int nrc) {
    double sumf = 0.0;

#if defined(__ARM_FEATURE_SVE)
    svbool_t pre = svptrue_b16();

    const int np = (n & ~(GGML_F16_SVE_STEP - 1));

    svfloat16_t sum = {svdup_f16(0.0f)};

    svfloat16_t ax;    //需要4个256，即4个ax
    svfloat16_t ay;

    for (int i = 0; i < np; i += GGML_F16_SVE_STEP) {
        for (int j = 0; j < GGML_F16_SVE_ARR; j++) {    //4个256，即64个f16
            ax = svld1_f16(pre, x + i + j * GGML_F16_SVE_EPR);
            ay = svld1_f16(pre, y + i + j * GGML_F16_SVE_EPR);

            sum = svmla_f16_z(pre, sum, ax, ay);
        }
    }

    /* 合并 */
    sumf = svaddv_f16(pre, sum);
    /* 剩余的维度 */
    for (int i = np; i < n; ++i) {
        sumf += x[i] * y[i];
    }

#elif defined(GGML_SIMD)
    const int np = (n & ~(GGML_F16_STEP - 1));

    GGML_F16_VEC sum[GGML_F16_ARR] = { GGML_F16_VEC_ZERO };

    GGML_F16_VEC ax[GGML_F16_ARR];
    GGML_F16_VEC ay[GGML_F16_ARR];

    for (int i = 0; i < np; i += GGML_F16_STEP) {
        for (int j = 0; j < GGML_F16_ARR; j++) {
            ax[j] = GGML_F16_VEC_LOAD(x + i + j*GGML_F16_EPR, j);
            ay[j] = GGML_F16_VEC_LOAD(y + i + j*GGML_F16_EPR, j);

            sum[j] = GGML_F16_VEC_FMA(sum[j], ax[j], ay[j]);
        }
    }

    // reduce sum0..sum3 to sum0
    GGML_F16_VEC_REDUCE(sumf, sum);

    // leftovers
    for (int i = np; i < n; ++i) {
        sumf += (ggml_float)(GGML_FP16_TO_FP32(x[i])*GGML_FP16_TO_FP32(y[i]));
        //sumf += x[i] * y[i];
    }

#else
    for (int i = 0; i < n; ++i) {
        sumf += (ggml_float)(GGML_FP16_TO_FP32(x[i])*GGML_FP16_TO_FP32(y[i]));
    }
#endif

    *s = sumf;
}

void dequantize_row_q4_0(const block_q4_0 * __restrict__ src, float * __restrict__ dst, int64_t k)
{
    static const int qk = QK4_0;
    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(src[i].d);

        for (int j = 0; j < qk / 2; ++j) {
            const int x0 = (src[i].qs[j] & 0x0F) - 8;
            const int x1 = (src[i].qs[j] >> 4) - 8;

            dst[i*qk + j + 0   ] = x0*d;
            dst[i*qk + j + qk/2] = x1*d;
        }
    }
}

void dequantize_row_q8_0(const block_q8_0 *__restrict__ x, float *__restrict__ y, int64_t k)
{
    static const int qk = QK8_0;

    const int nb = k / qk;
    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int j = 0; j < qk; ++j) {
            y[i*qk + j] = x[i].qs[j]*d;
        }
    }
}

void dequantize_row_q2_K(const block_q2_K *__restrict__ x, float *__restrict__ y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);
        const float min = GGML_FP16_TO_FP32(x[i].dmin);

        const uint8_t * q = x[i].qs;

        int is = 0;
        float dl, ml;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {

                uint8_t sc = x[i].scales[is++];
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;

                sc = x[i].scales[is++];
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l+16] >> shift) & 3)) - ml;

                shift += 2;
            }
            q += 32;
        }
    }
}

void dequantize_row_q8_K(const block_q8_K *__restrict__ x, float *__restrict__ y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < QK_K; ++j) {
            *y++ = x[i].d * x[i].qs[j];
        }
    }
}

void quantize_row_q4_0(const float *__restrict__ x, block_q4_0 *__restrict__ y, int64_t k)
{
    static const int qk = QK4_0;
    const int nb = k / qk;

#if defined(__ARM_NEON)
    for (int i = 0; i < nb; i++) {
        // 加载32个float到8个NEON寄存器
        float32x4_t src[8];
        for (int j = 0; j < 8; j++) {
            src[j] = vld1q_f32(x + i*qk + 4*j);
        }

        // 计算最大值和最小值
        float32x4_t maxv = src[0];
        float32x4_t minv = src[0];
        for (int j = 1; j < 8; j++) {
            maxv = vmaxq_f32(maxv, src[j]);
            minv = vminq_f32(minv, src[j]);
        }
        
        float max_val = vmaxvq_f32(maxv);
        float min_val = vminvq_f32(minv);
        
        // 确定绝对值最大值及其符号
        float amax = fabsf(max_val) > fabsf(min_val) ? fabsf(max_val) : fabsf(min_val);
        float selected_val = fabsf(max_val) > fabsf(min_val) ? max_val : min_val;

        // 计算量化参数
        const float d = selected_val / -8;
        const float id = 1.0f / d;

        y[i].d = GGML_FP32_TO_FP16(d);

        // 准备常量和结果寄存器
        const float32x4_t vid = vdupq_n_f32(id);
        const float32x4_t v8_5 = vdupq_n_f32(8.5f);
        uint8x16_t result;

        // 处理前16个元素 (0-15)
        int32x4_t i0 = vcvtq_s32_f32(vaddq_f32(vmulq_f32(src[0], vid), v8_5));
        int32x4_t i1 = vcvtq_s32_f32(vaddq_f32(vmulq_f32(src[1], vid), v8_5));
        int32x4_t i2 = vcvtq_s32_f32(vaddq_f32(vmulq_f32(src[2], vid), v8_5));
        int32x4_t i3 = vcvtq_s32_f32(vaddq_f32(vmulq_f32(src[3], vid), v8_5));
        
        // 处理后16个元素 (16-31)
        int32x4_t i4 = vcvtq_s32_f32(vaddq_f32(vmulq_f32(src[4], vid), v8_5));
        int32x4_t i5 = vcvtq_s32_f32(vaddq_f32(vmulq_f32(src[5], vid), v8_5));
        int32x4_t i6 = vcvtq_s32_f32(vaddq_f32(vmulq_f32(src[6], vid), v8_5));
        int32x4_t i7 = vcvtq_s32_f32(vaddq_f32(vmulq_f32(src[7], vid), v8_5));

        // 将结果饱和到[0, 15]范围并打包
        uint8x8_t low = vqmovun_s16(vcombine_s16(
            vqmovn_s32(vmaxq_s32(vminq_s32(i0, vdupq_n_s32(15)), vdupq_n_s32(0))),
            vqmovn_s32(vmaxq_s32(vminq_s32(i1, vdupq_n_s32(15)), vdupq_n_s32(0)))
        ));
        uint8x8_t high = vqmovun_s16(vcombine_s16(
            vqmovn_s32(vmaxq_s32(vminq_s32(i2, vdupq_n_s32(15)), vdupq_n_s32(0))),
            vqmovn_s32(vmaxq_s32(vminq_s32(i3, vdupq_n_s32(15)), vdupq_n_s32(0)))
        ));
        result = vcombine_u8(low, high);

        low = vqmovun_s16(vcombine_s16(
            vqmovn_s32(vmaxq_s32(vminq_s32(i4, vdupq_n_s32(15)), vdupq_n_s32(0))),
            vqmovn_s32(vmaxq_s32(vminq_s32(i5, vdupq_n_s32(15)), vdupq_n_s32(0)))
        ));
        high = vqmovun_s16(vcombine_s16(
            vqmovn_s32(vmaxq_s32(vminq_s32(i6, vdupq_n_s32(15)), vdupq_n_s32(0))),
            vqmovn_s32(vmaxq_s32(vminq_s32(i7, vdupq_n_s32(15)), vdupq_n_s32(0)))
        ));
        uint8x16_t high_part = vcombine_u8(low, high);

        // 合并高低4位数据
        result = vorrq_u8(result, vshlq_n_u8(high_part, 4));

        // 存储结果
        vst1q_u8(y[i].qs, result);
    }
#else
    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max
        float max  = 0.0f;

        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                max  = v;
            }
        }

        const float d  = max / -8;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < qk/2; ++j) {
            const float x0 = x[i*qk + 0    + j]*id;
            const float x1 = x[i*qk + qk/2 + j]*id;

            const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
            const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

            y[i].qs[j]  = xi0;
            y[i].qs[j] |= xi1 << 4;
        }
    }
#endif
}

void quantize_row_q8_0(const float *__restrict__ x, block_q8_0 *__restrict__ vy, int64_t k)
{
    const int nb = k / QK8_0;
    block_q8_0 *__restrict__ y = vy;

#if defined(__ARM_NEON)
    for (int i = 0; i < nb; i++) {
        float32x4_t srcv [8];
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        for (int j = 0; j < 8; j++) srcv[j]  = vld1q_f32(x + i*32 + 4*j);
        for (int j = 0; j < 8; j++) asrcv[j] = vabsq_f32(srcv[j]);

        for (int j = 0; j < 4; j++) amaxv[2*j] = vmaxq_f32(asrcv[2*j], asrcv[2*j+1]);
        for (int j = 0; j < 2; j++) amaxv[4*j] = vmaxq_f32(amaxv[4*j], amaxv[4*j+2]);
        for (int j = 0; j < 1; j++) amaxv[8*j] = vmaxq_f32(amaxv[8*j], amaxv[8*j+4]);

        const float amax = vmaxvq_f32(amaxv[0]);

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < 8; j++) {
            const float32x4_t v  = vmulq_n_f32(srcv[j], id);
            const int32x4_t   vi = vcvtnq_s32_f32(v);

            y[i].qs[4*j + 0] = vgetq_lane_s32(vi, 0);
            y[i].qs[4*j + 1] = vgetq_lane_s32(vi, 1);
            y[i].qs[4*j + 2] = vgetq_lane_s32(vi, 2);
            y[i].qs[4*j + 3] = vgetq_lane_s32(vi, 3);
        }
    }
#else
    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < QK8_0; j++) {
            const float v = x[i*QK8_0 + j];
            amax = MAX(amax, fabsf(v));
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < QK8_0; ++j) {
            const float x0 = x[i*QK8_0 + j]*id;

            y[i].qs[j] = roundf(x0);
        }
    }
#endif
}

void quantize_row_q2_K(const float *__restrict__ x, block_q2_K *__restrict__ y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    uint8_t L[QK_K];
    uint8_t Laux[16];
    float   weights[16];
    float mins[QK_K/16];
    float scales[QK_K/16];

    const float q4scale = 15.f;

    for (int i = 0; i < nb; i++) {
        float max_scale = 0; // as we are deducting the min, scales are always positive
        float max_min = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            for (int l = 0; l < 16; ++l) weights[l] = fabsf(x[16*j + l]);
            scales[j] = make_qkx2_quants(16, 3, x + 16*j, weights, L + 16*j, &mins[j], Laux, -0.5f, 0.1f, 15, true);
            float scale = scales[j];
            if (scale > max_scale) {
                max_scale = scale;
            }
            float min = mins[j];
            if (min > max_min) {
                max_min = min;
            }
        }

        if (max_scale > 0) {
            float iscale = q4scale/max_scale;
            for (int j = 0; j < QK_K/16; ++j) {
                int l = nearest_int(iscale*scales[j]);
                y[i].scales[j] = l;
            }
            y[i].d = GGML_FP32_TO_FP16(max_scale/q4scale);
        } else {
            for (int j = 0; j < QK_K/16; ++j) y[i].scales[j] = 0;
            y[i].d = GGML_FP32_TO_FP16(0.f);
        }
        if (max_min > 0) {
            float iscale = q4scale/max_min;
            for (int j = 0; j < QK_K/16; ++j) {
                int l = nearest_int(iscale*mins[j]);
                y[i].scales[j] |= (l << 4);
            }
            y[i].dmin = GGML_FP32_TO_FP16(max_min/q4scale);
        } else {
            y[i].dmin = GGML_FP32_TO_FP16(0.f);
        }
        for (int j = 0; j < QK_K/16; ++j) {
            const float d = GGML_FP16_TO_FP32(y[i].d) * (y[i].scales[j] & 0xF);
            if (!d) continue;
            const float dm = GGML_FP16_TO_FP32(y[i].dmin) * (y[i].scales[j] >> 4);
            for (int ii = 0; ii < 16; ++ii) {
                int l = nearest_int((x[16*j + ii] + dm)/d);
                l = MAX(0, MIN(3, l));
                L[16*j + ii] = l;
            }
        }

        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                y[i].qs[j/4 + l] = L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
            }
        }

        x += QK_K;
    }
}

void quantize_row_q8_K(const float *__restrict__ x, block_q8_K *__restrict__ y, int64_t k)  {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        float max = 0;
        float amax = 0;
        for (int j = 0; j < QK_K; ++j) {
            float ax = fabsf(x[j]);
            if (ax > amax) {
                amax = ax; max = x[j];
            }
        }
        if (!amax) {
            y[i].d = 0;
            memset(y[i].qs, 0, QK_K);
            x += QK_K;
            continue;
        }
        //const float iscale = -128.f/max;
        // We need this change for IQ2_XXS, else the AVX implementation becomes very awkward
        const float iscale = -127.f/max;
        for (int j = 0; j < QK_K; ++j) {
            int v = nearest_int(iscale*x[j]);
            y[i].qs[j] = MIN(127, v);
        }
        for (int j = 0; j < QK_K/16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) {
                sum += y[i].qs[j*16 + ii];
            }
            y[i].bsums[j] = sum;
        }
        y[i].d = 1/iscale;
        x += QK_K;
    }
}

void ggml_vec_dot_q4_0_q8_0(int n, float * __restrict__ s, size_t bs, const void * __restrict__ vx,
                            size_t bx, const void * __restrict__ vy, size_t by, int nrc)
{
    const int qk = QK8_0;
    const int nb = n / qk;

    (void)(nrc);
    (void)(bx);
    (void)(by);
    (void)(bs);

    const block_q4_0 * __restrict__ x = static_cast<const block_q4_0 *>(vx);
    const block_q8_0 * __restrict__ y = static_cast<const block_q8_0 *>(vy);

#if defined(__ARM_FEATURE_MATMUL_INT8)
    if (nrc == 2) {
        const block_q4_0 * __restrict__ vx0 = x;
        const block_q4_0 * __restrict__ vx1 = (const block_q4_0 *) ((const uint8_t*)x + bx);

        const block_q8_0 * __restrict__ vy0 = y;
        const block_q8_0 * __restrict__ vy1 = (const block_q8_0 *) ((const uint8_t*)y + by);

        float32x4_t sumv0 = vdupq_n_f32(0.0f);

        for (int i = 0; i < nb; i++) {
            const block_q4_0 * __restrict__ b_x0 = &vx0[i];
            const block_q4_0 * __restrict__ b_x1 = &vx1[i];
            const block_q8_0 * __restrict__ b_y0 = &vy0[i];
            const block_q8_0 * __restrict__ b_y1 = &vy1[i];

            const uint8x16_t m4b = vdupq_n_u8(0x0F);
            const int8x16_t  s8b = vdupq_n_s8(0x8);

            const uint8x16_t v0_0 = vld1q_u8(b_x0->qs);
            const uint8x16_t v0_1 = vld1q_u8(b_x1->qs);

            // 4-bit -> 8-bit
            const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8  (v0_0, m4b));
            const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
            const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8  (v0_1, m4b));
            const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

            // sub 8
            const int8x16_t x0_l = vsubq_s8(v0_0l, s8b);
            const int8x16_t x0_h = vsubq_s8(v0_0h, s8b);
            const int8x16_t x1_l = vsubq_s8(v0_1l, s8b);
            const int8x16_t x1_h = vsubq_s8(v0_1h, s8b);

            // load y
            const int8x16_t y0_l = vld1q_s8(b_y0->qs);
            const int8x16_t y0_h = vld1q_s8(b_y0->qs + 16);
            const int8x16_t y1_l = vld1q_s8(b_y1->qs);
            const int8x16_t y1_h = vld1q_s8(b_y1->qs + 16);

            float32x4_t scale = {GGML_FP16_TO_FP32(b_x0->d)*GGML_FP16_TO_FP32(b_y0->d),
                                 GGML_FP16_TO_FP32(b_x0->d)*GGML_FP16_TO_FP32(b_y1->d),
                                 GGML_FP16_TO_FP32(b_x1->d)*GGML_FP16_TO_FP32(b_y0->d),
                                 GGML_FP16_TO_FP32(b_x1->d)*GGML_FP16_TO_FP32(b_y1->d)};

            int8x16_t l0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));
            int8x16_t l1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));

            int8x16_t l2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));
            int8x16_t l3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));

            int8x16_t r0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));
            int8x16_t r1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));

            int8x16_t r2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));
            int8x16_t r3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));

            sumv0 = vmlaq_f32(sumv0,(vcvtq_f32_s32(vmmlaq_s32((vmmlaq_s32((vmmlaq_s32((vmmlaq_s32(vdupq_n_s32(0), l0, r0)),
                                                                                l1, r1)), l2, r2)), l3, r3))), scale);
        }
        float32x4_t sumv1 = vextq_f32(sumv0, sumv0, 2);
        float32x4_t sumv2 = vzip1q_f32(sumv0, sumv1);

        vst1_f32(s, vget_low_f32(sumv2));
        vst1_f32(s + bs, vget_high_f32(sumv2));
        return;
    }
#endif

#if defined(__ARM_FEATURE_SVE)
    float sum[4] = {0.0};
    int64_t x0_sum, x1_sum, x0_sum_1, x1_sum_1;
    svbool_t pre = svptrue_b8();
    svbool_t pre32 = svptrue_b32();
    svint32_t x0_32_0 = svdup_s32(0);
    svint32_t x1_32_0 = svdup_s32(0);

    for (int i = 0; i < nb; i += 4) {
        __builtin_prefetch((char *)&x[i], 0, 3);
        __builtin_prefetch((char *)&y[i], 0, 3);
        const block_q4_0 *__restrict__ x0 = &x[i + 0];
        const block_q4_0 *__restrict__ x1 = &x[i + 1];
        const block_q4_0 *__restrict__ x2 = &x[i + 2];
        const block_q4_0 *__restrict__ x3 = &x[i + 3];

        const block_q8_0 *__restrict__ y0 = &y[i + 0];
        const block_q8_0 *__restrict__ y1 = &y[i + 1];
        const block_q8_0 *__restrict__ y2 = &y[i + 2];
        const block_q8_0 *__restrict__ y3 = &y[i + 3];

        svuint8_t x_merge = svld1(pre, x0->qs);
        memcpy((char *)&x_merge + 16, x1->qs, 16);
        /* 低4位-8 */
        svint8_t x_low4_0 = svsub_z(pre, svreinterpret_s8(svand_z(pre, x_merge, 0xf)), 8);
        svint8_t x_high4_0 = svsub_z(pre, svreinterpret_s8(svlsr_z(pre, x_merge, 4)), 8);
        /* 交叉存储 */
        svint8_t x_tmp = svext(x_high4_0, x_low4_0, 16);
        svint8_t x_low4 = svext(x_tmp, x_high4_0, 16);
        svint8_t x_high4 = svext(x_low4_0, x_tmp, 16);
        /* 提取y */
        svint8_t y0_merge = svld1(pre, y0->qs);
        svint8_t y1_merge = svld1(pre, y1->qs);

        svuint8_t x_merge1 = svld1(pre, x2->qs);
        memcpy((char *)&x_merge1 + 16, x3->qs, 16);
        /* 低4位-8 */
        svint8_t x_low4_2 = svsub_z(pre, svreinterpret_s8(svand_z(pre, x_merge1, 0xf)), 8);
        svint8_t x_high4_2 = svsub_z(pre, svreinterpret_s8(svlsr_z(pre, x_merge1, 4)), 8);
        x_tmp = svext(x_high4_2, x_low4_2, 16);
        svint8_t x_low4_1 = svext(x_tmp, x_high4_2, 16);
        svint8_t x_high4_1 = svext(x_low4_2, x_tmp, 16);
        /* 提取y */
        svint8_t y0_merge_1 = svld1(pre, y2->qs);
        svint8_t y1_merge_1 = svld1(pre, y3->qs);

        svint32_t x0_mul32 = svdot(x0_32_0, x_low4, y0_merge);
        svint32_t x1_mul32 = svdot(x1_32_0, x_high4, y1_merge);
        svint32_t x0_mul32_1 = svdot(x0_32_0, x_low4_1, y0_merge_1);
        svint32_t x1_mul32_1 = svdot(x1_32_0, x_high4_1, y1_merge_1);

        x0_sum = svaddv(pre32, x0_mul32);
        x1_sum = svaddv(pre32, x1_mul32);
        x0_sum_1 = svaddv(pre32, x0_mul32_1);
        x1_sum_1 = svaddv(pre32, x1_mul32_1);

        sum[0] += x0_sum * x0->d * y0->d;
        sum[1] += x1_sum * x1->d * y1->d;
        sum[2] += x0_sum_1 * x2->d * y2->d;
        sum[3] += x1_sum_1 * x3->d * y3->d;
    }
    *s = sum[0] + sum[1] + sum[2] + sum[3];
#elif defined(__ARM_NEON)
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);

    for (int i = 0; i < nb; i += 2) {
        const block_q4_0 * __restrict__ x0 = &x[i + 0];
        const block_q4_0 * __restrict__ x1 = &x[i + 1];
        const block_q8_0 * __restrict__ y0 = &y[i + 0];
        const block_q8_0 * __restrict__ y1 = &y[i + 1];

        const uint8x16_t m4b = vdupq_n_u8(0x0F);
        const int8x16_t  s8b = vdupq_n_s8(0x8);

        const uint8x16_t v0_0 = vld1q_u8(x0->qs);
        const uint8x16_t v0_1 = vld1q_u8(x1->qs);

        // 4-bit -> 8-bit
        const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8  (v0_0, m4b));
        const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
        const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8  (v0_1, m4b));
        const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

        // sub 8
        const int8x16_t v0_0ls = vsubq_s8(v0_0l, s8b);
        const int8x16_t v0_0hs = vsubq_s8(v0_0h, s8b);
        const int8x16_t v0_1ls = vsubq_s8(v0_1l, s8b);
        const int8x16_t v0_1hs = vsubq_s8(v0_1h, s8b);

        // load y
        const int8x16_t v1_0l = vld1q_s8(y0->qs);
        const int8x16_t v1_0h = vld1q_s8(y0->qs + 16);
        const int8x16_t v1_1l = vld1q_s8(y1->qs);
        const int8x16_t v1_1h = vld1q_s8(y1->qs + 16);

        // dot product into int32x4_t
        const int32x4_t p_0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), v0_0ls, v1_0l), v0_0hs, v1_0h);
        const int32x4_t p_1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), v0_1ls, v1_1l), v0_1hs, v1_1h);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), GGML_FP16_TO_FP32(x0->d)*GGML_FP16_TO_FP32(y0->d));
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), GGML_FP16_TO_FP32(x1->d)*GGML_FP16_TO_FP32(y1->d));
    }

    *s = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
#else
    // scalar
    float sumf = 0.0;

    for (int i = 0; i < nb; i++) {
        int sumi = 0;

        for (int j = 0; j < qk/2; ++j) {
            const int v0 = (x[i].qs[j] & 0x0F) - 8;
            const int v1 = (x[i].qs[j] >>   4) - 8;

            sumi += (v0 * y[i].qs[j]) + (v1 * y[i].qs[j + qk/2]);
        }

        sumf += sumi*GGML_FP16_TO_FP32(x[i].d)*GGML_FP16_TO_FP32(y[i].d);
    }

    *s = sumf;

#endif
}

void ggml_vec_dot_q8_0_q8_0(int n, float *__restrict__ s, size_t bs, const void *__restrict__ vx, size_t bx, const void *__restrict__ vy, size_t by, int nrc)
{
    const int qk = QK8_0;
    const int nb = n / qk;

#if defined(__ARM_FEATURE_MATMUL_INT8)
    assert((nrc == 2) || (nrc == 1) || (nrc == 16));
#else
    assert(nrc == 1);
#endif

    const block_q8_0 *__restrict__ x = static_cast<const block_q8_0 *>(vx);
    const block_q8_0 *__restrict__ y = static_cast<const block_q8_0 *>(vy);

#if defined(__ARM_FEATURE_MATMUL_INT8)
    if (nrc == 2) {
        const block_q8_0 * __restrict__ vx0 = x;
        const block_q8_0 * __restrict__ vx1 = (const block_q8_0 *) ((const uint8_t*)x + bx);
        const block_q8_0 * __restrict__ vy0 = y;
        const block_q8_0 * __restrict__ vy1 = (const block_q8_0 *) ((const uint8_t*)y + by);

        float32x4_t sumv0 = vdupq_n_f32(0.0f);

        for (int i = 0; i < nb; i++) {
            const block_q8_0 * __restrict__ b_x0 = &vx0[i];
            const block_q8_0 * __restrict__ b_y0 = &vy0[i];

            const block_q8_0 * __restrict__ b_x1 = &vx1[i];
            const block_q8_0 * __restrict__ b_y1 = &vy1[i];

            const int8x16_t x0_l = vld1q_s8(b_x0->qs);
            const int8x16_t x0_h = vld1q_s8(b_x0->qs + 16);
            const int8x16_t x1_l = vld1q_s8(b_x1->qs);
            const int8x16_t x1_h = vld1q_s8(b_x1->qs + 16);

            // load y
            const int8x16_t y0_l = vld1q_s8(b_y0->qs);
            const int8x16_t y0_h = vld1q_s8(b_y0->qs + 16);
            const int8x16_t y1_l = vld1q_s8(b_y1->qs);
            const int8x16_t y1_h = vld1q_s8(b_y1->qs + 16);

            float32_t _scale[4] = {GGML_FP16_TO_FP32(b_x0->d)*GGML_FP16_TO_FP32(b_y0->d),
                                   GGML_FP16_TO_FP32(b_x0->d)*GGML_FP16_TO_FP32(b_y1->d),
                                   GGML_FP16_TO_FP32(b_x1->d)*GGML_FP16_TO_FP32(b_y0->d),
                                   GGML_FP16_TO_FP32(b_x1->d)*GGML_FP16_TO_FP32(b_y1->d)};
            float32x4_t scale = vld1q_f32(_scale);

            int8x16_t l0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));
            int8x16_t l1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));

            int8x16_t l2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));
            int8x16_t l3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));

            int8x16_t r0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));
            int8x16_t r1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));

            int8x16_t r2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));
            int8x16_t r3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));

            sumv0 = vmlaq_f32(sumv0,(vcvtq_f32_s32(vmmlaq_s32((vmmlaq_s32((vmmlaq_s32((vmmlaq_s32(vdupq_n_s32(0), l0, r0)),
                                                                                       l1, r1)), l2, r2)), l3, r3))), scale);
        }
        float32x4_t sumv1 = vextq_f32(sumv0, sumv0, 2);
        float32x4_t sumv2 = vzip1q_f32(sumv0, sumv1);

        vst1_f32(s, vget_low_f32(sumv2));
        vst1_f32(s + bs, vget_high_f32(sumv2));
        return;
    }
#endif

    int ib = 0;
    float sumf = 0;

#if defined(__ARM_FEATURE_SVE)
    svfloat32_t sumv0 = svdup_n_f32(0.0f);
    svfloat32_t sumv1 = svdup_n_f32(0.0f);

    const int vector_length = ggml_cpu_get_sve_cnt() * 8;

    //VLA Implemenation for SVE
    switch (vector_length) {
        case 128:
            {
                // predicate for activating lanes for 16 Int8 elements
                const svbool_t ph16 = svptrue_pat_b8 (SV_VL16);
                const svbool_t pl16 = svptrue_pat_b32(SV_VL4);

                for (; ib + 1 < nb; ib += 2) {
                    const block_q8_0 *__restrict__ x0 = &x[ib + 0];
                    const block_q8_0 *__restrict__ x1 = &x[ib + 1];
                    const block_q8_0 *__restrict__ y0 = &y[ib + 0];
                    const block_q8_0 *__restrict__ y1 = &y[ib + 1];

                    // load x
                    const svint8_t qx0_0 = svld1_s8(ph16, x0->qs);
                    const svint8_t qx0_1 = svld1_s8(ph16, x0->qs+16);
                    const svint8_t qx1_0 = svld1_s8(ph16, x1->qs);
                    const svint8_t qx1_1 = svld1_s8(ph16, x1->qs+16);

                    // load y
                    const svint8_t qy0_0 = svld1_s8(ph16, y0->qs);
                    const svint8_t qy0_1 = svld1_s8(ph16, y0->qs+16);
                    const svint8_t qy1_0 = svld1_s8(ph16, y1->qs);
                    const svint8_t qy1_1 = svld1_s8(ph16, y1->qs+16);

                    sumv0 = svmla_n_f32_x(pl16, sumv0, svcvt_f32_s32_x(pl16, svadd_x(pl16,
                                    svdot_s32(svdup_n_s32(0), qx0_0, qy0_0),
                                    svdot_s32(svdup_n_s32(0), qx0_1, qy0_1))), GGML_FP16_TO_FP32(x0->d)*GGML_FP16_TO_FP32(y0->d));
                    sumv1 = svmla_n_f32_x(pl16, sumv1, svcvt_f32_s32_x(pl16, svadd_x(pl16,
                                    svdot_s32(svdup_n_s32(0), qx1_0, qy1_0),
                                    svdot_s32(svdup_n_s32(0), qx1_1, qy1_1))), GGML_FP16_TO_FP32(x1->d)*GGML_FP16_TO_FP32(y1->d));
                }

                sumf = svaddv_f32(pl16, svadd_f32_x(pl16, sumv0, sumv1));
            } break;
        case 256:
            {
                //printf("sve256");
                for (; ib + 1 < nb; ib += 2) {
                    const block_q8_0 *__restrict__ x0 = &x[ib + 0];
                    const block_q8_0 *__restrict__ x1 = &x[ib + 1];
                    const block_q8_0 *__restrict__ y0 = &y[ib + 0];
                    const block_q8_0 *__restrict__ y1 = &y[ib + 1];

                    // load x
                    const svint8_t qx0 = svld1_s8(svptrue_b8(), x0->qs);
                    const svint8_t qx1 = svld1_s8(svptrue_b8(), x1->qs);

                    // load y
                    const svint8_t qy0 = svld1_s8(svptrue_b8(), y0->qs);
                    const svint8_t qy1 = svld1_s8(svptrue_b8(), y1->qs);

                    sumv0 = svmla_n_f32_x(svptrue_b32(), sumv0, svcvt_f32_s32_x(svptrue_b32(),
                                svdot_s32(svdup_n_s32(0), qx0, qy0)), GGML_FP16_TO_FP32(x0->d)*GGML_FP16_TO_FP32(y0->d));
                    sumv1 = svmla_n_f32_x(svptrue_b32(), sumv1, svcvt_f32_s32_x(svptrue_b32(),
                                svdot_s32(svdup_n_s32(0), qx1, qy1)), GGML_FP16_TO_FP32(x1->d)*GGML_FP16_TO_FP32(y1->d));
                }

                sumf = svaddv_f32(svptrue_b32(), svadd_f32_x(svptrue_b32(), sumv0, sumv1));
            } break;
        case 512:
            {
                // predicate for activating high 256 bit
                const svbool_t ph32 = svptrue_pat_b8(SV_VL32);
                // predicate for activating low 256 bit
                const svbool_t pl32 = svnot_b_z(svptrue_b8(), ph32);

                // predicate for activating high lanes for 8 float32 elements
                const svbool_t ph8 = svptrue_pat_b32(SV_VL8);
                // predicate for activating low lanes for 8 float32 elements
                const svbool_t pl8 = svnot_b_z(svptrue_b32(), ph8);

                svfloat32_t sumv00 = svdup_n_f32(0.0f);

                for (; ib + 1 < nb; ib += 2) {
                    const block_q8_0 *__restrict__ x0 = &x[ib + 0];
                    const block_q8_0 *__restrict__ x1 = &x[ib + 1];
                    const block_q8_0 *__restrict__ y0 = &y[ib + 0];
                    const block_q8_0 *__restrict__ y1 = &y[ib + 1];

                    //load 32 int8_t in first half of vector and put another 32 int8_t in second vector lower bits
                    // and add them to make one 64 element vector
                    // load x
                    const svint8_t qx_32 = svld1_s8(ph32, x0->qs);
                          svint8_t qx_64 = svld1_s8(pl32, x0->qs + 2);

                    qx_64 = svadd_s8_x(svptrue_b8(), qx_32, qx_64);

                    // load y
                    const svint8_t qy_32 = svld1_s8(ph32, y0->qs);
                          svint8_t qy_64 = svld1_s8(pl32, y0->qs + 2);

                    qy_64 = svadd_s8_x(svptrue_b8(), qy_32, qy_64);

                    // scale creation
                    const float32_t deq1 = GGML_FP16_TO_FP32(x0->d)*GGML_FP16_TO_FP32(y0->d);
                    const float32_t deq2 = GGML_FP16_TO_FP32(x1->d)*GGML_FP16_TO_FP32(y1->d);

                    // duplicate deq1 in first half of vector and deq2 in second half of vector
                    const svfloat32_t temp = svdup_f32_m(svdup_f32_z(ph8, deq1), pl8, deq2);

                    const svfloat32_t sumvt = svcvt_f32_s32_x(svptrue_b32(), svdot_s32(svdup_n_s32(0), qx_64, qy_64));

                    sumv00 = svmla_f32_m(svptrue_b32(), sumv00, sumvt, temp);
                }

                sumf = svaddv_f32(svptrue_b32(), sumv00);
                break;
            }
        default:
            assert(false && "Unsupported vector length");
            break;
    }
#elif defined(__ARM_NEON)
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);

    for (; ib + 1 < nb; ib += 2) {
        const block_q8_0 *__restrict__ x0 = &x[ib + 0];
        const block_q8_0 *__restrict__ x1 = &x[ib + 1];
        const block_q8_0 *__restrict__ y0 = &y[ib + 0];
        const block_q8_0 *__restrict__ y1 = &y[ib + 1];

        const int8x16_t x0_0 = vld1q_s8(x0->qs);
        const int8x16_t x0_1 = vld1q_s8(x0->qs + 16);
        const int8x16_t x1_0 = vld1q_s8(x1->qs);
        const int8x16_t x1_1 = vld1q_s8(x1->qs + 16);

        // load y
        const int8x16_t y0_0 = vld1q_s8(y0->qs);
        const int8x16_t y0_1 = vld1q_s8(y0->qs + 16);
        const int8x16_t y1_0 = vld1q_s8(y1->qs);
        const int8x16_t y1_1 = vld1q_s8(y1->qs + 16);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(
                        vdotq_s32(vdupq_n_s32(0), x0_0, y0_0),
                        vdotq_s32(vdupq_n_s32(0), x0_1, y0_1))), GGML_FP16_TO_FP32(x0->d)*GGML_FP16_TO_FP32(y0->d));

        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(
                        vdotq_s32(vdupq_n_s32(0), x1_0, y1_0),
                        vdotq_s32(vdupq_n_s32(0), x1_1, y1_1))), GGML_FP16_TO_FP32(x1->d)*GGML_FP16_TO_FP32(y1->d));
    }

    sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
#else
    /* 不加速 */
    for (; ib < nb; ++ib) {
        int sumi = 0;

        for (int j = 0; j < qk; j++) {
            sumi += x[ib].qs[j] * y[ib].qs[j];
        }

        sumf += sumi * (GGML_FP16_TO_FP32(x[ib].d) * GGML_FP16_TO_FP32(y[ib].d));
    }
#endif
    *s = sumf;
}

void ggml_vec_dot_q2_K_q8_K(int n, float * __restrict__ s, size_t bs, const void * __restrict__ vx, size_t bx, const void * __restrict__ vy, size_t by, int nrc) {
    assert(nrc == 1);

    const block_q2_K * __restrict__ x = static_cast<const block_q2_K *>(vx);
    const block_q8_K * __restrict__ y = static_cast<const block_q8_K *>(vy);

    const int nb = n / QK_K;

#ifdef __ARM_FEATURE_SVE
    const int vector_length = svcntb()*8;
    const svuint8_t m3s = svdup_n_u8(0x3);
    const svuint32_t m4s = svdup_n_u32(0xF);
    const svint32_t vzero_sv = svdup_n_s32(0);
    svfloat32_t acc_sum = svdup_n_f32(0);
    svbool_t pred_s32 = svptrue_pat_b32(SV_VL4);

    switch (vector_length) {
        case 128:
            for (int i = 0; i < nb; ++i) {
                const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
                svfloat32_t d_broad = svdup_n_f32((float32_t)d);
                const float dmin = -y[i].d * GGML_FP16_TO_FP32(x[i].dmin);
                svfloat32_t dmin_broad = svdup_n_f32((float32_t)dmin);

                const uint8_t * __restrict__ q2 = x[i].qs;
                const int8_t  * __restrict__ q8_sv = y[i].qs;
                const uint8_t * __restrict__ sc = x[i].scales;

                svuint32_t mins_and_scales_sve = svld1ub_u32(svptrue_b32(), sc);
                const svint32_t mins_sv_1 = svreinterpret_s32_u32(svlsr_n_u32_x(svptrue_b32(), mins_and_scales_sve, 4));

                mins_and_scales_sve = svld1ub_u32(svptrue_b32(), sc+4);
                const svint32_t mins_sv_2 = svreinterpret_s32_u32(svlsr_n_u32_x(svptrue_b32(), mins_and_scales_sve, 4));

                svint32_t q8sums_sv_1 = svld1sh_s32(svptrue_b32(), y[i].bsums);
                svint32_t q8sums_sv_2 = svld1sh_s32(svptrue_b32(), y[i].bsums+4);

                const svint32_t s0 = svadd_s32_x(svptrue_b32(), svmul_s32_x(svptrue_b32(), mins_sv_1, q8sums_sv_1), svmul_s32_x(svptrue_b32(), mins_sv_2, q8sums_sv_2));

                mins_and_scales_sve = svld1ub_u32(svptrue_b32(), sc+8);
                const svint32_t mins_sv_3 = svreinterpret_s32_u32(svlsr_n_u32_x(svptrue_b32(), mins_and_scales_sve, 4));

                mins_and_scales_sve = svld1ub_u32(svptrue_b32(), sc+12);
                const svint32_t mins_sv_4 = svreinterpret_s32_u32(svlsr_n_u32_x(svptrue_b32(), mins_and_scales_sve, 4));

                q8sums_sv_1 = svld1sh_s32(svptrue_b32(), y[i].bsums+8);
                q8sums_sv_2 = svld1sh_s32(svptrue_b32(), y[i].bsums+12);

                svint32_t s1 = svadd_s32_x(svptrue_b32(), svmul_s32_x(svptrue_b32(), mins_sv_3, q8sums_sv_1), svmul_s32_x(svptrue_b32(), mins_sv_4, q8sums_sv_2));

                svfloat32_t temp = svcvt_f32_s32_x(svptrue_b32(), svadd_s32_x(svptrue_b32(), s0, s1));

                acc_sum = svmla_f32_m(svptrue_b32(), acc_sum, temp, dmin_broad);

                svint32_t sumi1 = svdup_n_s32(0);

                {
                    const svuint8_t q2bits_1 = svld1_u8(svptrue_b8(), q2);
                    svint8_t q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), q2bits_1, m3s));
                    svint8_t q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv); q8_sv += 16;
                    const svint32_t scales_sv = svreinterpret_s32_u32(svand_u32_m(svptrue_b32(), svld1ub_u32(svptrue_b32(), sc), m4s));

                    sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv, 0));

                    const svuint8_t q2bits_3 = svld1_u8(svptrue_b8(), q2+16);
                    q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), q2bits_3, m3s));
                    q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv); q8_sv += 16;

                    sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv, 1));

                    q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_1, 2), m3s));
                    q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv); q8_sv += 16;

                    sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv, 2));

                    q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_3, 2), m3s));
                    q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv); q8_sv += 16;

                    sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv, 3));


                    const svint32_t scales_sv_1 = svreinterpret_s32_u32(svand_u32_m(svptrue_b32(), svld1ub_u32(svptrue_b32(), sc+4), m4s));

                    q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_1, 4), m3s));
                    q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv); q8_sv += 16;

                    sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_1, 0));

                    q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_3, 4), m3s));
                    q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv); q8_sv += 16;

                    sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_1, 1));

                    q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_1, 6), m3s));
                    q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv); q8_sv += 16;

                    sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_1, 2));

                    q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_3, 6), m3s));
                    q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv); q8_sv += 16;

                    sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_1, 3));

                    //-------------------------------

                    q2 += 32;
                    const svint32_t scales_sv_2 = svreinterpret_s32_u32(svand_u32_m(svptrue_b32(), svld1ub_u32(svptrue_b32(), sc+8), m4s));
                    const svuint8_t q2bits_2 = svld1_u8(svptrue_b8(), q2);

                    q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), q2bits_2, m3s));
                    q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv); q8_sv += 16;

                    sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_2, 0));

                    const svuint8_t q2bits_4 = svld1_u8(svptrue_b8(), q2+16);
                    q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), q2bits_4, m3s));
                    q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv); q8_sv += 16;

                    sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_2, 1));


                    q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_2, 2), m3s));
                    q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv); q8_sv += 16;

                    sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_2, 2));

                    q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_4, 2), m3s));
                    q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv); q8_sv += 16;

                    sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_2, 3));


                    const svint32_t scales_sv_3 = svreinterpret_s32_u32(svand_u32_m(svptrue_b32(), svld1ub_u32(svptrue_b32(), sc+12), m4s));

                    q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_2, 4), m3s));
                    q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv); q8_sv += 16;

                    sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_3, 0));

                    q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_4, 4), m3s));
                    q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv); q8_sv += 16;

                    sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_3, 1));



                    q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_2, 6), m3s));
                    q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv); q8_sv += 16;

                    sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_3, 2));

                    q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_4, 6), m3s));
                    q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv); q8_sv += 16;

                    sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_3, 3));
                }
                acc_sum = svmla_f32_m(svptrue_b32(), acc_sum, svcvt_f32_s32_x(svptrue_b32(), sumi1), d_broad);
            }
            *s = svaddv_f32(svptrue_b32(), acc_sum);
            break;

        case 256:
        case 512:
            for (int i = 0; i < nb; ++i) {
                const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
                svfloat32_t d_broad = svdup_n_f32((float32_t)d);
                const float dmin = -y[i].d * GGML_FP16_TO_FP32(x[i].dmin);
                svfloat32_t dmin_broad = svdup_n_f32((float32_t)dmin);

                const uint8_t * __restrict__ q2 = x[i].qs;
                const int8_t  * __restrict__ q8_sv = y[i].qs;
                const uint8_t * __restrict__ sc = x[i].scales;

                const svuint32_t mins_and_scales_sve = svld1ub_u32(svptrue_pat_b32(SV_VL8), sc); sc += 8;
                const svint32_t scales_sv = svreinterpret_s32_u32(svand_u32_m(svptrue_pat_b32(SV_VL8), mins_and_scales_sve, m4s));
                const svint32_t mins_sv_1 = svreinterpret_s32_u32(svlsr_n_u32_x(svptrue_pat_b32(SV_VL8), mins_and_scales_sve, 4));
                svint32_t q8sums_sv_1 = svld1sh_s32(svptrue_pat_b32(SV_VL8), y[i].bsums);

                const svuint32_t mins_and_scales_sve_1 = svld1ub_u32(svptrue_pat_b32(SV_VL8), sc);
                const svint32_t scales_sv_1 = svreinterpret_s32_u32(svand_u32_m(svptrue_pat_b32(SV_VL8), mins_and_scales_sve_1, m4s));
                const svint32_t mins_sv_2 = svreinterpret_s32_u32(svlsr_n_u32_x(svptrue_pat_b32(SV_VL8), mins_and_scales_sve_1, 4));

                svint32_t q8sums_sv_2 = svld1sh_s32(svptrue_pat_b32(SV_VL8), y[i].bsums+8);

                svfloat32_t temp = svcvt_f32_s32_x(svptrue_pat_b32(SV_VL8), svadd_s32_x(svptrue_pat_b32(SV_VL8), svmul_s32_x(svptrue_pat_b32(SV_VL8), mins_sv_1, q8sums_sv_1), svmul_s32_x(svptrue_pat_b32(SV_VL8), mins_sv_2, q8sums_sv_2)));

                acc_sum = svmla_f32_m(svptrue_pat_b32(SV_VL8), acc_sum, temp, dmin_broad);

                svint32_t sumi1 = svdup_n_s32(0);

                {
                    const svuint8_t q2bits_1 = svld1_u8(svptrue_pat_b8(SV_VL32), q2);
                    svint8_t q2bytes_sv = svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32), q2bits_1, m3s));
                    svint8_t q8bytes_sv = svld1_s8(svptrue_pat_b8(SV_VL32), q8_sv); q8_sv += 32;

                    svint32_t scale_1 = svsel(pred_s32, svdup_lane_s32(scales_sv, 0), svdup_lane_s32(scales_sv, 1));
                    sumi1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), scale_1);

                    q2bytes_sv = svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32), svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), q2bits_1, 2), m3s));
                    q8bytes_sv = svld1_s8(svptrue_pat_b8(SV_VL32), q8_sv); q8_sv += 32;

                    svint32_t scale_2 = svsel(pred_s32, svdup_lane_s32(scales_sv, 2), svdup_lane_s32(scales_sv, 3));
                    sumi1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), sumi1, svdot_s32(svdup_n_s32(0), q2bytes_sv, q8bytes_sv), scale_2);

                    q2bytes_sv = svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32), svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), q2bits_1, 4), m3s));
                    q8bytes_sv = svld1_s8(svptrue_pat_b8(SV_VL32), q8_sv); q8_sv += 32;

                    scale_1 = svsel(pred_s32, svdup_lane_s32(scales_sv, 4), svdup_lane_s32(scales_sv, 5));
                    sumi1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), scale_1);

                    q2bytes_sv = svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32), svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), q2bits_1, 6), m3s));
                    q8bytes_sv = svld1_s8(svptrue_pat_b8(SV_VL32), q8_sv); q8_sv += 32;

                    scale_2 = svsel(pred_s32, svdup_lane_s32(scales_sv, 6), svdup_lane_s32(scales_sv, 7));
                    sumi1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), scale_2);

                    q2 += 32;

                    const svuint8_t q2bits_2 = svld1_u8(svptrue_pat_b8(SV_VL32), q2);
                    q2bytes_sv = svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32), q2bits_2, m3s));
                    q8bytes_sv = svld1_s8(svptrue_pat_b8(SV_VL32), q8_sv); q8_sv += 32;

                    scale_1 = svsel(pred_s32, svdup_lane_s32(scales_sv_1, 0), svdup_lane_s32(scales_sv_1, 1));
                    sumi1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), scale_1);

                    q2bytes_sv = svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32), svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), q2bits_2, 2), m3s));
                    q8bytes_sv = svld1_s8(svptrue_pat_b8(SV_VL32), q8_sv); q8_sv += 32;

                    scale_2 = svsel(pred_s32, svdup_lane_s32(scales_sv_1, 2), svdup_lane_s32(scales_sv_1, 3));
                    sumi1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), scale_2);

                    q2bytes_sv = svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32), svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), q2bits_2, 4), m3s));
                    q8bytes_sv = svld1_s8(svptrue_pat_b8(SV_VL32), q8_sv); q8_sv += 32;

                    scale_1 = svsel(pred_s32, svdup_lane_s32(scales_sv_1, 4), svdup_lane_s32(scales_sv_1, 5));
                    sumi1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), scale_1);

                    q2bytes_sv = svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32), svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), q2bits_2, 6), m3s));
                    q8bytes_sv = svld1_s8(svptrue_pat_b8(SV_VL32), q8_sv); q8_sv += 32;

                    scale_2 = svsel(pred_s32, svdup_lane_s32(scales_sv_1, 6), svdup_lane_s32(scales_sv_1, 7));
                    sumi1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), scale_2);
                }
                acc_sum = svmla_f32_m(svptrue_pat_b32(SV_VL8), acc_sum, svcvt_f32_s32_x(svptrue_pat_b32(SV_VL8), sumi1), d_broad);
            }
            *s = svaddv_f32(svptrue_pat_b32(SV_VL8), acc_sum);
            break;

        default:
            assert(false && "Unsupported vector length");
            break;
    }

#elif __ARM_NEON
    const uint8x16_t m3 = vdupq_n_u8(0x3);
    const uint8x16_t m4 = vdupq_n_u8(0xF);

    const int32x4_t vzero = vdupq_n_s32(0);

    ggml_int8x16x2_t q2bytes;
    uint8_t aux[16];

    float sum = 0;

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * GGML_FP16_TO_FP32(x[i].dmin);

        const uint8_t * __restrict__ q2 = x[i].qs;
        const int8_t  * __restrict__ q8 = y[i].qs;
        const uint8_t * __restrict__ sc = x[i].scales;

        const uint8x16_t mins_and_scales = vld1q_u8(sc);
        const uint8x16_t scales = vandq_u8(mins_and_scales, m4);
        vst1q_u8(aux, scales);

        const uint8x16_t mins = vshrq_n_u8(mins_and_scales, 4);
        const ggml_int16x8x2_t q8sums = ggml_vld1q_s16_x2(y[i].bsums);
        const ggml_int16x8x2_t mins16 = {{vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(mins))), vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(mins)))}};
        const int32x4_t s0 = vaddq_s32(vmull_s16(vget_low_s16 (mins16.val[0]), vget_low_s16 (q8sums.val[0])),
                                       vmull_s16(vget_high_s16(mins16.val[0]), vget_high_s16(q8sums.val[0])));
        const int32x4_t s1 = vaddq_s32(vmull_s16(vget_low_s16 (mins16.val[1]), vget_low_s16 (q8sums.val[1])),
                                       vmull_s16(vget_high_s16(mins16.val[1]), vget_high_s16(q8sums.val[1])));
        sum += dmin * vaddvq_s32(vaddq_s32(s0, s1));

        int isum = 0;
        int is = 0;

// We use this macro instead of a function call because for some reason
// the code runs 2-3% slower, even if the function is declared inline
#define MULTIPLY_ACCUM_WITH_SCALE(index)\
        isum += vaddvq_s32(ggml_vdotq_s32(vzero, q2bytes.val[0], q8bytes.val[0])) * aux[is+(index)];\
        isum += vaddvq_s32(ggml_vdotq_s32(vzero, q2bytes.val[1], q8bytes.val[1])) * aux[is+1+(index)];

#define SHIFT_MULTIPLY_ACCUM_WITH_SCALE(shift, index)\
        q8bytes = ggml_vld1q_s8_x2(q8); q8 += 32;\
        q2bytes.val[0] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits.val[0], (shift)), m3));\
        q2bytes.val[1] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits.val[1], (shift)), m3));\
        MULTIPLY_ACCUM_WITH_SCALE((index));

        for (int j = 0; j < QK_K/128; ++j) {
            const ggml_uint8x16x2_t q2bits = ggml_vld1q_u8_x2(q2); q2 += 32;

            ggml_int8x16x2_t q8bytes = ggml_vld1q_s8_x2(q8); q8 += 32;
            q2bytes.val[0] = vreinterpretq_s8_u8(vandq_u8(q2bits.val[0], m3));
            q2bytes.val[1] = vreinterpretq_s8_u8(vandq_u8(q2bits.val[1], m3));

            MULTIPLY_ACCUM_WITH_SCALE(0);

            SHIFT_MULTIPLY_ACCUM_WITH_SCALE(2, 2);
            SHIFT_MULTIPLY_ACCUM_WITH_SCALE(4, 4);
            SHIFT_MULTIPLY_ACCUM_WITH_SCALE(6, 6);

            is += 8;
        }

        sum += d * isum;
    }

    *s = sum;


#else

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {

        const uint8_t * q2 = x[i].qs;
        const  int8_t * q8 = y[i].qs;
        const uint8_t * sc = x[i].scales;

        int summs = 0;
        for (int j = 0; j < 16; ++j) {
            summs += y[i].bsums[j] * (sc[j] >> 4);
        }

        const float dall = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        const float dmin = y[i].d * GGML_FP16_TO_FP32(x[i].dmin);

        int isum = 0;
        int is = 0;
        int d;
        for (int k = 0; k < QK_K/128; ++k) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                d = sc[is++] & 0xF;
                int isuml = 0;
                for (int l =  0; l < 16; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += d * isuml;
                d = sc[is++] & 0xF;
                isuml = 0;
                for (int l = 16; l < 32; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += d * isuml;
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
        sumf += dall * isum - dmin * summs;
    }
    *s = sumf;
#endif
}

