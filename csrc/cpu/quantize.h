#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <arm_neon.h>
#include <float.h>
#include <cassert>
#include <stdio.h>

#define GGML_COMMON_AGGR_U
#define GGML_COMMON_AGGR_S

#ifdef _MSC_VER
#define GGML_EXTENSION
#else // _MSC_VER
#define GGML_EXTENSION __extension__
#endif // _MSC_VER

typedef float16_t ggml_half;
typedef float32_t ggml_half2;
typedef float16_t ggml_fp16_t;
typedef float16_t ggml_float;
typedef float16_t f16;

typedef struct ggml_uint8x16x2_t {
    uint8x16_t val[2];
} ggml_uint8x16x2_t;

typedef struct ggml_int8x16x2_t {
    int8x16_t val[2];
} ggml_int8x16x2_t;

typedef struct ggml_uint16x8x2_t {
    uint16x8_t val[2];
} ggml_uint16x8x2_t;

typedef struct ggml_int16x8x2_t {
    int16x8_t val[2];
} ggml_int16x8x2_t;

inline static ggml_uint8x16x2_t ggml_vld1q_u8_x2(const uint8_t * ptr) {
    ggml_uint8x16x2_t res;

    res.val[0] = vld1q_u8(ptr + 0);
    res.val[1] = vld1q_u8(ptr + 16);

    return res;
}

inline static ggml_int8x16x2_t ggml_vld1q_s8_x2(const int8_t * ptr) {
    ggml_int8x16x2_t res;

    res.val[0] = vld1q_s8(ptr + 0);
    res.val[1] = vld1q_s8(ptr + 16);

    return res;
}

inline static ggml_uint16x8x2_t ggml_vld1q_u16_x2(const uint16_t * ptr) {
    ggml_uint16x8x2_t res;

    res.val[0] = vld1q_u16(ptr + 0);
    res.val[1] = vld1q_u16(ptr + 8);

    return res;
}

inline static ggml_int16x8x2_t ggml_vld1q_s16_x2(const int16_t * ptr) {
    ggml_int16x8x2_t res;

    res.val[0] = vld1q_s16(ptr + 0);
    res.val[1] = vld1q_s16(ptr + 8);

    return res;
}

inline static int32x4_t ggml_vdotq_s32(int32x4_t acc, int8x16_t a, int8x16_t b) {
    const int16x8_t p0 = vmull_s8(vget_low_s8 (a), vget_low_s8 (b));
    const int16x8_t p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));

    return vaddq_s32(acc, vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)));
}


#define QK4_0 32
typedef struct {
    ggml_half d;          // delta
    uint8_t qs[QK4_0 / 2];  // nibbles / quants  ggml_half
} block_q4_0;

#define QK8_0 32
typedef struct {
    ggml_half d;       // delta
    int8_t  qs[QK8_0]; // quants
} block_q8_0;

#define QK_K 256
typedef struct {
    uint8_t scales[QK_K/16]; // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K/4];      // quants
    GGML_EXTENSION union {
        struct {
            ggml_half d;    // super-block scale for quantized scales
            ggml_half dmin; // super-block scale for quantized mins
        } GGML_COMMON_AGGR_S;
        ggml_half2 dm;
    } GGML_COMMON_AGGR_U;
} block_q2_K;

typedef struct {
    float   d;              // delta
    int8_t  qs[QK_K];       // quants
    int16_t bsums[QK_K/16]; // sum of quants in groups of 16
} block_q8_K;


typedef void (*ggml_to_float_t)(const void  *__restrict__ x, float *__restrict__ y, int64_t k);

typedef void (*ggml_from_float_t)(const float *__restrict__ x, void  *__restrict__ y, int64_t k);

typedef void (*ggml_vec_dot_t)(int n, float *__restrict__ s, size_t bs, const void *__restrict__ x, size_t bx,
                               const void *__restrict__ y, size_t by, int nrc);


void dequantize_row_q4_0(const block_q4_0 * __restrict__ src, float * __restrict__ dst, int64_t k);
void dequantize_row_q8_0(const block_q8_0 *__restrict__ x, float *__restrict__ y, int64_t k);
void dequantize_row_q2_K(const block_q2_K *__restrict__ x, float *__restrict__ y, int64_t k);
void dequantize_row_q8_K(const block_q8_K *__restrict__ x, float *__restrict__ y, int64_t k);

void quantize_row_q4_0(const float *__restrict__ x, block_q4_0 *__restrict__ y, int64_t k);
void quantize_row_q8_0(const float *__restrict__ x, block_q8_0 *__restrict__ y, int64_t k);
void quantize_row_q2_K(const float *__restrict__ x, block_q2_K *__restrict__ y, int64_t k);
void quantize_row_q8_K(const float *__restrict__ x, block_q8_K *__restrict__ y, int64_t k);

void ggml_vec_dot_q4_0_q8_0(int n, float *__restrict__ s, size_t bs, const void *__restrict__ vx,
                            size_t bx, const void *__restrict__ vy, size_t by, int nrc);
void ggml_vec_dot_q8_0_q8_0(int n, float *__restrict__ s, size_t bs, const void *__restrict__ vx,
                            size_t bx, const void *__restrict__ vy, size_t by, int nrc);
void ggml_vec_dot_q2_K_q8_K(int n, float *__restrict__ s, size_t bs, const void *__restrict__ vx,
                            size_t bx, const void *__restrict__ vy, size_t by, int nrc);
        

void ggml_fp32_to_fp16_row(const float * x, ggml_fp16_t * y, int64_t n);
void ggml_fp16_to_fp32_row(const ggml_fp16_t * x, float * y, int64_t n);

void ggml_vec_dot_f16(int n, float * __restrict__ s, size_t bs, ggml_fp16_t * __restrict__ x, size_t bx, ggml_fp16_t * __restrict__ y, size_t by, int nrc);

#define GGML_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)
static inline float ggml_compute_fp16_to_fp32(ggml_half h) {
    __fp16 tmp;
    memcpy(&tmp, &h, sizeof(ggml_half));
    float res = tmp;
    return res;
}

#define GGML_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)
static inline ggml_half ggml_compute_fp32_to_fp16(float f) {
    ggml_half res;
    __fp16 tmp = f;
    memcpy(&res, &tmp, sizeof(ggml_half));
    return res;
}
