#pragma once
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#include <assert.h>
#include <stdint.h> 
#include <math.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FMA)

#define GGML_SIMD

// F16 NEON

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    #define GGML_F16_STEP 32
    #define GGML_F16_EPR  8

    #define GGML_F16x8              float16x8_t
    #define GGML_F16x8_ZERO         vdupq_n_f16(0.0f)
    #define GGML_F16x8_SET1(x)      vdupq_n_f16(x)
    #define GGML_F16x8_LOAD(x)      vld1q_f16((const f16 *)(x))
    #define GGML_F16x8_STORE        vst1q_f16
    #define GGML_F16x8_FMA(a, b, c) vfmaq_f16(a, b, c)
    #define GGML_F16x8_ADD          vaddq_f16
    #define GGML_F16x8_MUL          vmulq_f16
    #define GGML_F16x8_REDUCE(res, x)                               \
    do {                                                            \
        int offset = GGML_F16_ARR >> 1;                             \
        for (int i = 0; i < offset; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[offset+i]);              \
        }                                                           \
        offset >>= 1;                                               \
        for (int i = 0; i < offset; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[offset+i]);              \
        }                                                           \
        offset >>= 1;                                               \
        for (int i = 0; i < offset; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[offset+i]);              \
        }                                                           \
        const float32x4_t t0 = vcvt_f32_f16(vget_low_f16 ((x)[0])); \
        const float32x4_t t1 = vcvt_f32_f16(vget_high_f16((x)[0])); \
        (res) = (ggml_float) vaddvq_f32(vaddq_f32(t0, t1));         \
    } while (0)

    #define GGML_F16_VEC                GGML_F16x8
    #define GGML_F16_VEC_ZERO           GGML_F16x8_ZERO
    #define GGML_F16_VEC_SET1           GGML_F16x8_SET1
    #define GGML_F16_VEC_LOAD(p, i)     GGML_F16x8_LOAD(p)
    #define GGML_F16_VEC_STORE(p, r, i) GGML_F16x8_STORE((f16 *)(p), (r)[i])
    #define GGML_F16_VEC_FMA            GGML_F16x8_FMA
    #define GGML_F16_VEC_ADD            GGML_F16x8_ADD
    #define GGML_F16_VEC_MUL            GGML_F16x8_MUL
    #define GGML_F16_VEC_REDUCE         GGML_F16x8_REDUCE
#else
    // if FP16 vector arithmetic is not supported, we use FP32 instead
    // and take advantage of the vcvt_ functions to convert to/from FP16

    #define GGML_F16_STEP 16
    #define GGML_F16_EPR  4

    #define GGML_F32Cx4              float32x4_t
    #define GGML_F32Cx4_ZERO         vdupq_n_f32(0.0f)
    #define GGML_F32Cx4_SET1(x)      vdupq_n_f32(x)
    #define GGML_F32Cx4_LOAD(x)      vcvt_f32_f16(vld1_f16((const f16 *)(x)))
    #define GGML_F32Cx4_STORE(x, y)  vst1_f16(x, vcvt_f16_f32(y))
    #define GGML_F32Cx4_FMA(a, b, c) vfmaq_f32(a, b, c)
    #define GGML_F32Cx4_ADD          vaddq_f32
    #define GGML_F32Cx4_MUL          vmulq_f32
    #define GGML_F32Cx4_REDUCE       GGML_F32x4_REDUCE

    #define GGML_F16_VEC                GGML_F32Cx4
    #define GGML_F16_VEC_ZERO           GGML_F32Cx4_ZERO
    #define GGML_F16_VEC_SET1           GGML_F32Cx4_SET1
    #define GGML_F16_VEC_LOAD(p, i)     GGML_F32Cx4_LOAD(p)
    #define GGML_F16_VEC_STORE(p, r, i) GGML_F32Cx4_STORE((f16 *)(p), r[i])
    #define GGML_F16_VEC_FMA            GGML_F32Cx4_FMA
    #define GGML_F16_VEC_ADD            GGML_F32Cx4_ADD
    #define GGML_F16_VEC_MUL            GGML_F32Cx4_MUL
    #define GGML_F16_VEC_REDUCE         GGML_F32Cx4_REDUCE
#endif

    #define GGML_F16_ARR (GGML_F16_STEP/GGML_F16_EPR)
#endif

static inline int nearest_int(float fval) {
    assert(fabsf(fval) <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

static float make_qkx2_quants(int n, int nmax, const float *__restrict__ x, const float *__restrict__ weights,
        uint8_t *__restrict__ L, float *__restrict__ the_min, uint8_t *__restrict__ Laux,
        float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0];
    float max = x[0];
    float sum_w = weights[0];
    float sum_x = sum_w * x[0];
#ifdef HAVE_BUGGY_APPLE_LINKER
    // use 'volatile' to prevent unroll and work around a bug in Apple ld64 1015.7
    for (volatile int i = 1; i < n; ++i) {
#else
    for (int i = 1; i < n; ++i) {
#endif
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0) min = 0;
    if (max == min) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        *the_min = -min;
        return 0.f;
    }
    float iscale = nmax/(max - min);
    float scale = 1/iscale;
    float best_mad = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale*(x[i] - min));
        L[i] = MAX(0, MIN(nmax, l));
        float diff = scale * L[i] + min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        float w = weights[i];
        best_mad += w * diff;
    }
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta*is + nmax)/(max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale*(x[i] - min));
            l = MAX(0, MIN(nmax, l));
            Laux[i] = l;
            float w = weights[i];
            sum_l += w*l;
            sum_l2 += w*l*l;
            sum_xl += w*l*x[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l)/D;
            float this_min   = (sum_l2 * sum_x - sum_l * sum_xl)/D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            float mad = 0;
            for (int i = 0; i < n; ++i) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? fabsf(diff) : diff * diff;
                float w = weights[i];
                mad += w * diff;
            }
            if (mad < best_mad) {
                for (int i = 0; i < n; ++i) {
                    L[i] = Laux[i];
                }
                best_mad = mad;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}