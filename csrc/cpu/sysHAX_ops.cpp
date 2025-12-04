#include <sys/types.h>
#include <ctype.h>
#include <cmath>
#include <cassert>
#include <numa.h>
#include <iostream>
#include <sstream>
#include <unistd.h>  // Linux
#include <csignal>
#include <sched.h>

#include "cpu_types.hpp"
#include "quantize.h"
#include "cpu_utils.h"
#include "paged_attention.h"

typedef unsigned int UINT32;
typedef unsigned long long UINT64;
typedef float f32;

extern void transpose_v(f16 *vt, const f16 *v, int n_tokens, int v_dim, int qkv_dim);
extern void prefill_attention(f16 *out_ptr, const f16 *qkv_ptr, const f16 *vt_ptr, int N_tokens, int N_seqs, const int *seq_lens);

float expf_f16_table[65536];

template<class scaler_t>
struct KernelVecType{
  using q_load_vec_t = void;
  using k_load_vec_t = void;
  using v_load_vec_t = void;
  using q_k_v_vec_t = void;
  using accum_vec_t = void;
};

template<>
struct KernelVecType<f16>{
  using q_load_vec_t = vec_op::FP16Vec8;
  using k_load_vec_t = vec_op::FP16Vec16;
  using v_load_vec_t = vec_op::FP16Vec16;
  using q_k_v_vec_t = vec_op::FP16Vec16;
  using accum_vec_t = vec_op::FP16Vec16;
};

enum ggml_type {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_COUNT,
};

typedef struct {
    const char *pcTypeName;
    UINT32 uiblkSize;
    UINT32 uiTypeSize;
    ggml_from_float_t quantize;
    ggml_to_float_t dequantize;
    enum ggml_type VecDotType;  /* 矩阵点积计算类型 */
    ggml_vec_dot_t VecDotFunc;  /* 矩阵点积计算函数 */
} BLOCK_DATA_INFO;

BLOCK_DATA_INFO g_BlockDataInfo[] = {
    {"f32", 1, sizeof(float), NULL, NULL, GGML_TYPE_F32, NULL},
    {"f16", 1, sizeof(uint16_t), (ggml_from_float_t)ggml_fp32_to_fp16_row, (ggml_to_float_t)ggml_fp16_to_fp32_row, GGML_TYPE_F16, (ggml_vec_dot_t)ggml_vec_dot_f16},
    {"q4_0", QK4_0, sizeof(block_q4_0), (ggml_from_float_t)quantize_row_q4_0, (ggml_to_float_t)dequantize_row_q4_0, GGML_TYPE_Q8_0, (ggml_vec_dot_t)ggml_vec_dot_q4_0_q8_0},
    {"", 0, 0, NULL, NULL, GGML_TYPE_COUNT, NULL},
    {"", 0, 0, NULL, NULL, GGML_TYPE_COUNT, NULL},
    {"", 0, 0, NULL, NULL, GGML_TYPE_COUNT, NULL},
    {"", 0, 0, NULL, NULL, GGML_TYPE_COUNT, NULL},
    {"", 0, 0, NULL, NULL, GGML_TYPE_COUNT, NULL},
    {"q8_0", QK8_0, sizeof(block_q8_0), (ggml_from_float_t)quantize_row_q8_0, (ggml_to_float_t)dequantize_row_q8_0, GGML_TYPE_Q8_0, (ggml_vec_dot_t)ggml_vec_dot_q8_0_q8_0},
    {"", 0, 0, NULL, NULL, GGML_TYPE_COUNT, NULL},
    {"q2_k", QK_K, sizeof(block_q2_K), (ggml_from_float_t)quantize_row_q2_K, (ggml_to_float_t)dequantize_row_q2_K, GGML_TYPE_Q8_K, (ggml_vec_dot_t)ggml_vec_dot_q2_K_q8_K},
    {"", 0, 0, NULL, NULL, GGML_TYPE_COUNT, NULL},
    {"", 0, 0, NULL, NULL, GGML_TYPE_COUNT, NULL},
    {"", 0, 0, NULL, NULL, GGML_TYPE_COUNT, NULL},
    {"", 0, 0, NULL, NULL, GGML_TYPE_COUNT, NULL},
    {"q8_k", QK_K, sizeof(block_q8_K), (ggml_from_float_t)quantize_row_q8_K, (ggml_to_float_t)dequantize_row_q8_K, GGML_TYPE_COUNT, NULL},
    {"", 0, 0, NULL, NULL, GGML_TYPE_COUNT, NULL},
    {"", 0, 0, NULL, NULL, GGML_TYPE_COUNT, NULL},
    {"", 0, 0, NULL, NULL, GGML_TYPE_COUNT, NULL},
    {"", 0, 0, NULL, NULL, GGML_TYPE_COUNT, NULL},
    {"", 0, 0, NULL, NULL, GGML_TYPE_COUNT, NULL},
    {"", 0, 0, NULL, NULL, GGML_TYPE_COUNT, NULL},
    {"", 0, 0, NULL, NULL, GGML_TYPE_COUNT, NULL},
    {"", 0, 0, NULL, NULL, GGML_TYPE_COUNT, NULL},
    {"", 0, 0, NULL, NULL, GGML_TYPE_COUNT, NULL},
    {"", 0, 0, NULL, NULL, GGML_TYPE_COUNT, NULL},
    {"", 0, 0, NULL, NULL, GGML_TYPE_COUNT, NULL},
    {"", 0, 0, NULL, NULL, GGML_TYPE_COUNT, NULL}
};

typedef struct {
    enum ggml_type DataType;  /* 张量的数据类型 */
    union {
        void *tensor1;
        void **tensor2;
        void ***tensor3;
    } Data;
} TENSOR_INFO;

/* 挂载模型位置 */
typedef struct weight {
    TENSOR_INFO embed_tokens_weight;                // [vocab_size, hidden_size]
    TENSOR_INFO input_layernorm_weight;             // [layer][hidden_size]
    TENSOR_INFO post_attention_layernorm_weight;    // [layer][hidden_size]
    TENSOR_INFO qkv_proj_weight;                    // [layer][numa][qkv_dim/numa, hidden_size]
    TENSOR_INFO o_proj_weight;                      // [layer][numa][hidden_size/numa, q_dim]
    TENSOR_INFO qkv_proj_bias;                      // [layer][qkv_dim]
    TENSOR_INFO gate_up_proj_weight;                // [layer][numa][2*intermediate_size/numa, hidden_size]
    TENSOR_INFO down_proj_weight;                   // [layer][numa][hidden_size/numa, intermediate_size]
    TENSOR_INFO lm_head_weight;                     // [numa][vocab_size/numa, hidden_size]
    TENSOR_INFO norm_weight;                        // [hidden_size]
    TENSOR_INFO q_norm_weight;                      // [layer][head_dim]
    TENSOR_INFO k_norm_weight;                      // [layer][head_dim]

    TENSOR_INFO gate_proj_weight;
    TENSOR_INFO up_proj_weight;
} WEIGHT;

/* 记录tensor的数据类型 */
typedef struct {
    int token_embd_weight;
    int attn_k_weight;
    int attn_k_bias;
    int attn_norm_weight;
    int attn_q_weight;
    int attn_q_bias;
    int attn_v_weight;
    int attn_v_bias;
    int ffn_down_weight;
    int ffn_gate_weight;
    int ffn_norm_weight;
    int ffn_up_weight;
    int attn_output_weight;
    int output_weight;
    int output_norm_weight;
    int q_norm_weight;
    int k_norm_weight;
} WeightTypes;

typedef struct {
    float *hidden_states;
    float *residual;
    void **quant_hidden_states;
    float *qkv;
    f16 *qkv_f16;

    float *attn_output;
    f16 *attn_output_f16;
    void **quant_attn_output;

    float *gate_up;
    float *fused_gate_up;
    void **quant_fused_gate_up;

    float *logits;
} MODEL_RUN_STATE;

typedef struct {
    int head_dim;
    int hidden_size;            /* embedding 维度 */
    int num_attention_heads;        /* 注意力头个数 */
    int num_key_value_heads;     /* kv的对数 */
    int intermediate_size;     /* ffn隐藏层维度 */
    int num_hidden_layers;       /* 模型层数 */
    int max_position_embeddings; /* 上下文长度 */
    float rms_norm_eps; /* eps */
    int vocab_size;        /* 词汇数量 */
    float rope_theta; /* rope频率 */
    f16 *cos_sin_cache;  /* rope历史数据 */
    int n_rotary;        /* rotary维度 */
    bool is_neox_style;  /* rope风格 */
    double attn_scale;   /* rope系数 */
} MODEL_HYPE_PARA;

__thread f16 qk_tmp_storage[131072];

int g_quantization_bit_code = 1;
int g_numas = numa_num_configured_nodes();
WEIGHT g_pstWeight;
WeightTypes g_pstWeightTypes;
MODEL_HYPE_PARA g_pstModelHypePara;

float f16_to_f32(f16 h){return h;}
f16 f32_to_f16(float h){return h;}

__attribute__((noinline))
f16 DOTPRODUCT_vv_f16(int M, const f16 *src0_ptr, const f16 *src1_ptr)
{
    __builtin_prefetch(&src0_ptr[0], 0 , 0);
    __builtin_prefetch(&src1_ptr[0], 0 , 0);
    if (M >= 128) {
        __builtin_prefetch(&src0_ptr[32], 0 , 0);
        __builtin_prefetch(&src1_ptr[32], 0 , 0);
        __builtin_prefetch(&src0_ptr[64], 0 , 0);
        __builtin_prefetch(&src1_ptr[64], 0 , 0);
        __builtin_prefetch(&src0_ptr[96], 0 , 0);
        __builtin_prefetch(&src1_ptr[96], 0 , 0);
    }
    float sumf = 0.0f;
    int j = 0;
#ifdef __ARM_NEON
    const int M_UNROLL = 8;
    const int M_SIMD = 8;
    float16x8_t sum[M_UNROLL] = {vdupq_n_f16(0.0f)};
    for (; j <= M - M_UNROLL * M_SIMD; j += M_UNROLL * M_SIMD) {
        __builtin_prefetch(&src0_ptr[j + 192], 0 , 0);
        __builtin_prefetch(&src1_ptr[j + 192], 0 , 0);
        __builtin_prefetch(&src0_ptr[j + 224], 0 , 0);
        __builtin_prefetch(&src1_ptr[j + 224], 0 , 0);
        __builtin_prefetch(&src0_ptr[j + 256], 0 , 0);
        __builtin_prefetch(&src1_ptr[j + 256], 0 , 0);
        __builtin_prefetch(&src0_ptr[j + 288], 0 , 0);
        __builtin_prefetch(&src1_ptr[j + 288], 0 , 0);
        __builtin_prefetch(&src0_ptr[j + 320], 0 , 0);
        __builtin_prefetch(&src1_ptr[j + 320], 0 , 0);
        for (int ss = 0; ss < M_UNROLL; ss++) {
            sum[ss] = vfmaq_f16(sum[ss], vld1q_f16(&src0_ptr[j + ss * M_SIMD]), vld1q_f16(&src1_ptr[j + ss * M_SIMD]));
        }
    }

    for (; j <= M - 8; j += 8) {
        sum[0] = vfmaq_f16(sum[0], vld1q_f16(&src0_ptr[j]), vld1q_f16(&src1_ptr[j]));
    }
    sum[0] = vaddq_f16(vaddq_f16(sum[0], sum[2]), vaddq_f16(sum[1], sum[3]));
    if (M_UNROLL > 4) {
        sum[4] = vaddq_f16(vaddq_f16(sum[4], sum[6]), vaddq_f16(sum[5], sum[7]));
        sum[0] = vaddq_f16(sum[0], sum[4]);
    }

    float32x4_t t0 = vcvt_f32_f16(vget_low_f16(sum[0]));
    float32x4_t t1 = vcvt_f32_f16(vget_high_f16(sum[0]));
    sumf = vaddvq_f32(vaddq_f32(t0, t1));
#endif
    for (; j < M; j++) {
        sumf += (f16_to_f32(src0_ptr[j]) * f16_to_f32(src1_ptr[j]));
    }

    return sumf;
}

void transpose_v(f16 *vt, const f16 *v, int n_tokens, int v_dim, int qkv_dim)
{
    for (int i = 0; i < n_tokens; i++) {
        int j = 0;
        for (int j = 0; j < v_dim; j++) {
            vt[j * n_tokens + i] = v[i * qkv_dim + j];
        }
    }
}

void prefill_attention(f16 *out_ptr, const f16 *qkv_ptr, const f16 *vt_ptr, int N_tokens, int N_seqs, const int *seq_lens)
{
    int head_dim = g_pstModelHypePara.head_dim;
    int num_attention_heads = g_pstModelHypePara.num_attention_heads;
    int num_key_value_heads = g_pstModelHypePara.num_key_value_heads;
    int q_dim = head_dim * num_attention_heads;
    int k_dim = head_dim * num_key_value_heads;
    int v_dim = head_dim * num_key_value_heads;
    int N_gqa = num_attention_heads / num_key_value_heads;
    int qkv_stride = q_dim + k_dim + v_dim;
    int head_size = head_dim;
    const f16 *q_ptr = qkv_ptr, *k_ptr = q_ptr + q_dim;

    std::vector<int> seqlen_prefix_sum;
    seqlen_prefix_sum.push_back(0);
    for (int i = 0, sum_seq_lens = 0; i < N_seqs; i++) {
        sum_seq_lens += seq_lens[i];
        seqlen_prefix_sum.push_back(sum_seq_lens);
    }

    get_total_thread_num();

    #pragma omp parallel for collapse(2)
    for (int seq = 0; seq < N_seqs; seq++) {
        for (int h_q = 0; h_q < num_attention_heads; h_q++) {
            f16 *qk_tmp = (f16 *)qk_tmp_storage;
            int seq_t_begin = seqlen_prefix_sum[seq], seq_t_end = seqlen_prefix_sum[seq + 1];
            int h_kv = h_q / N_gqa;

            for (int t = seq_t_begin; t < seq_t_end; t++) {
                const f16 *q_head_ptr = q_ptr + t * qkv_stride + h_q * head_size;
                int token_idx_in_seq = t - seq_t_begin;
                f16 row_max = -INFINITY;
                for (int i = 0; i <= token_idx_in_seq; i++) {
                    const f16 *k_head_ptr = k_ptr + (seq_t_begin + i) *qkv_stride + h_kv *head_size;
                    qk_tmp[i] = DOTPRODUCT_vv_f16(head_size, q_head_ptr, k_head_ptr);
                    row_max = qk_tmp[i] > row_max ? qk_tmp[i] : row_max;
                }
                f32 sumexp = 0.0f;
                for (int i = 0; i <= token_idx_in_seq; i++) {
                    f16 diff = qk_tmp[i] - row_max;
                    f32 exp_result = expf_f16_table[*(uint16_t *)&diff];
                    qk_tmp[i] = exp_result;
                    sumexp += exp_result;
                }

                for (int i = 0; i <= token_idx_in_seq; i++) {
                    qk_tmp[i] /= sumexp;
                }

                for (int iv = 0; iv < head_size; iv++) {
                    const f16 *vt_seq_ptr = vt_ptr + h_kv * head_size * N_tokens + iv *N_tokens + seq_t_begin;
                    out_ptr[t * q_dim + h_q * head_size + iv] = DOTPRODUCT_vv_f16(token_idx_in_seq + 1, vt_seq_ptr, qk_tmp);
                }
            }
        }
    }
}

void Quantize(void *Dst, float *src, enum ggml_type DataType, int size)
{
    g_BlockDataInfo[DataType].quantize(src, Dst, size);
}

void RmsNorm(float *DstData, float *SrcData, float *SrcWeight, float eps, int dataNum)
{
    float ss = 0.0f;
#ifdef __ARM_NEON
    int i = 0;
    float32x4_t ss_vec0 = vdupq_n_f32(0.0f);
    float32x4_t ss_vec1 = vdupq_n_f32(0.0f);
    float32x4_t ss_vec2 = vdupq_n_f32(0.0f);
    float32x4_t ss_vec3 = vdupq_n_f32(0.0f);

    for (; i <= dataNum - 16; i += 16) {
        __builtin_prefetch(SrcData + i + 64, 0, 0);

        float32x4_t data0 = vld1q_f32(SrcData + i);
        float32x4_t data1 = vld1q_f32(SrcData + i + 4);
        float32x4_t data2 = vld1q_f32(SrcData + i + 8);
        float32x4_t data3 = vld1q_f32(SrcData + i + 12);

        ss_vec0 = vmlaq_f32(ss_vec0, data0, data0);
        ss_vec1 = vmlaq_f32(ss_vec1, data1, data1);
        ss_vec2 = vmlaq_f32(ss_vec2, data2, data2);
        ss_vec3 = vmlaq_f32(ss_vec3, data3, data3);
    }

    ss_vec0 = vaddq_f32(ss_vec0, ss_vec1);
    ss_vec0 = vaddq_f32(ss_vec0, ss_vec2);
    ss_vec0 = vaddq_f32(ss_vec0, ss_vec3);
    ss = vaddvq_f32(ss_vec0);
    for (; i < dataNum; ++i) {
        ss += SrcData[i] * SrcData[i];
    }

    float scale = 1.0f / sqrtf(ss / dataNum + eps);
    i = 0;
    const float32x4_t scale_vec = vdupq_n_f32(scale);

    for (; i <= dataNum - 16; i += 16) {
        float32x4_t data0 = vld1q_f32(SrcData + i);
        float32x4_t weight0 = vld1q_f32(SrcWeight + i);
        float32x4_t data1 = vld1q_f32(SrcData + i + 4);
        float32x4_t weight1 = vld1q_f32(SrcWeight + i + 4);
        float32x4_t data2 = vld1q_f32(SrcData + i + 8);
        float32x4_t weight2 = vld1q_f32(SrcWeight + i + 8);
        float32x4_t data3 = vld1q_f32(SrcData + i + 12);
        float32x4_t weight3 = vld1q_f32(SrcWeight + i + 12);

        vst1q_f32(DstData + i, vfmaq_f32(vdupq_n_f32(0), weight0, vmulq_f32(data0, scale_vec)));
        vst1q_f32(DstData + i + 4, vfmaq_f32(vdupq_n_f32(0), weight1, vmulq_f32(data1, scale_vec)));
        vst1q_f32(DstData + i + 8, vfmaq_f32(vdupq_n_f32(0), weight2, vmulq_f32(data2, scale_vec)));
        vst1q_f32(DstData + i + 12, vfmaq_f32(vdupq_n_f32(0), weight3, vmulq_f32(data3, scale_vec)));
    }
    for (; i <= dataNum - 4; i += 4) {
        float32x4_t data = vld1q_f32(SrcData + i);
        float32x4_t weight = vld1q_f32(SrcWeight + i);
        float32x4_t result = vmulq_f32(weight, vmulq_f32(data, scale_vec));
        vst1q_f32(DstData + i, result);
    }
    for (; i < dataNum; ++i) {
        DstData[i] = SrcWeight[i] * (ss * SrcData[i]);
    }

#else
    for (int j = 0; j < dataNum; j++) {
        ss += SrcData[j] * SrcData[j];
    }
    ss /= dataNum;
    ss += eps;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < dataNum; j++) {
        DstData[j] = SrcWeight[j] * (ss * SrcData[j]);
    }
#endif

}

// 通用的残差连接加RmsNorm函数
void residual_add_rmsnorm_with_numa(
    const WorkDivider *work,
    f32 *output_data,               // 输出数据
    f32 *residual_target_data,      // 残差连接的目标数据
    f32 *input_data,                // 输入数据
    f32 *rmsnorm_weight,            // RmsNorm权重
    int token_count,                // token数量
    int hidden_size,                        // 维度
    float eps,                      // RmsNorm的eps参数
    const std::vector<int> *indices, // 可选的索引数组（用于处理last_token情况）
    const char* operation_name      // 用于调试的操作名称
) {
    SingleNumaWorkRange srange;
    int work_size = indices ? indices->size() : token_count;
    divide_work_first_numa(work, work_size, &srange);
    
    if (work->my_numa == 0) {
        for (int i = srange.begin_thread; i < srange.end_thread; i++) {
            int token_idx = indices ? (*indices)[i] : i;
            
            // 残差连接：target += input
            for (int j = 0; j < hidden_size; j++) {
                residual_target_data[token_idx * hidden_size + j] += input_data[token_idx * hidden_size + j];
            }
            
            // RmsNorm操作
            int output_idx = indices ? i : token_idx;
            RmsNorm(output_data + output_idx * hidden_size, residual_target_data + token_idx * hidden_size,
                    rmsnorm_weight, eps, hidden_size);
        }
    }
}

// 通用的bias添加函数
void add_bias_with_numa(
    const WorkDivider *work,
    f32 *data,                      // 要添加bias的数据
    f32 *bias,                      // bias数据
    int token_count,                // token数量
    int dim,                        // 每个token的维度
    const char* operation_name      // 用于调试的操作名称
) {
    SingleNumaWorkRange srange;
    divide_all_work(work, token_count * dim, &srange);
    
    for (int item = srange.begin_thread; item < srange.end_thread; item++) {
        int k = item / dim;  // token索引
        int i = item % dim;  // 维度索引
        data[k * dim + i] += bias[i];
    }
}

// Q和K的RMS归一化函数（按头归一化）
void qk_norm_with_numa(
    const WorkDivider *work,
    f32 *data,                      // QKV数据，布局：[q(q_dim), k(k_dim), v(v_dim)] per token
    f32 *q_norm_weight,             // Q归一化权重，大小为head_dim
    f32 *k_norm_weight,             // K归一化权重，大小为head_dim
    int token_count,                // token数量
    int num_attention_heads,        // Q头数
    int num_key_value_heads,        // K头数
    int head_dim,                   // 每个头的维度
    float eps                       // RMS norm的epsilon
) { 
    int q_dim = num_attention_heads * head_dim;
    int k_dim = num_key_value_heads * head_dim;
    int v_dim = num_key_value_heads * head_dim;
    int qkv_dim = q_dim + k_dim + v_dim;  // 总QKV维度
    
    f32 *q_ptr = data;                           // Q数据起始位置
    f32 *k_ptr = data + q_dim;                   // K数据起始位置（跳过Q）

    // q_norm - 对Q进行按头归一化
    SingleNumaWorkRange srange;
    divide_all_work(work, token_count * num_attention_heads, &srange);

    for (int i = srange.begin_thread; i < srange.end_thread; i++) {
        int t = i / num_attention_heads;    // token索引
        int h = i % num_attention_heads;    // head索引
        
        f32 *q_head_ptr = q_ptr + t * qkv_dim + h * head_dim;
        RmsNorm(q_head_ptr, q_head_ptr, q_norm_weight, eps, head_dim);
    }
    
    // k_norm - 对K进行按头归一化
    divide_all_work(work, token_count * num_key_value_heads, &srange);
    for (int i = srange.begin_thread; i < srange.end_thread; i++) {
        int t = i / num_key_value_heads;    // token索引
        int h = i % num_key_value_heads;    // head索引
        
        f32 *k_head_ptr = k_ptr + t * qkv_dim + h * head_dim;
        RmsNorm(k_head_ptr, k_head_ptr, k_norm_weight, eps, head_dim);
    }
}

// 通用的量化和NUMA分发函数
void quantize_and_distribute_to_numa(
    const WorkDivider *work,
    MODEL_RUN_STATE *pstRunState,
    void **target_numa_buffers,  // 目标NUMA缓冲区数组
    f32 *source_data,            // 源数据
    int srcType,                 // 源数据类型
    int total_elements,          // 总元素数
    const char* buffer_name      // 用于调试的缓冲区名称
) {
    enum ggml_type dstType = g_BlockDataInfo[srcType].VecDotType;
    UINT32 srcBlockNum = g_BlockDataInfo[srcType].uiblkSize;
    UINT32 srcBlocksize = g_BlockDataInfo[srcType].uiTypeSize;
    UINT32 dstBlockNum = g_BlockDataInfo[dstType].uiblkSize;
    UINT32 dstBlocksize = g_BlockDataInfo[dstType].uiTypeSize;

    SingleNumaWorkRange srange;
    divide_all_work(work, total_elements / dstBlockNum, &srange);

    Quantize((char *)target_numa_buffers[work->my_numa] + srange.begin_thread * dstBlocksize,
             source_data + srange.begin_thread * dstBlockNum,
             dstType, srange.work_per_thread * dstBlockNum);

    for (int target_numa = 0; target_numa < g_numas; target_numa++) {
        if (target_numa != work->my_numa) {
            memcpy((char *)target_numa_buffers[target_numa] + srange.begin_thread * dstBlocksize,
                    (char *)target_numa_buffers[work->my_numa] + srange.begin_thread * dstBlocksize,
                    srange.work_per_thread * dstBlocksize);
        }
    }
}

void gate_up_proj_matrix_multiply_with_numa(
    const WorkDivider *work,
    f32 *output_data,
    void *gate_proj_weight,
    void *up_proj_weight,
    void **input_numa_buffers,
    int srcType,
    int m,              // 等价于n_tokens
    int k,              // 等价于hidden_size和VecDotFunc的第一个参数
    int n,              // 等价于work_dim
    int layer_index
) {
    enum ggml_type dstType = g_BlockDataInfo[srcType].VecDotType;
    UINT32 srcBlockNum = g_BlockDataInfo[srcType].uiblkSize;
    UINT32 srcBlocksize = g_BlockDataInfo[srcType].uiTypeSize;
    UINT32 dstBlockNum = g_BlockDataInfo[dstType].uiblkSize;
    UINT32 dstBlocksize = g_BlockDataInfo[dstType].uiTypeSize;

    MultiNumaWorkRange mrange;
    divide_work_all_numas(work, n, &mrange);
    
    char *gate_proj_weight_ptr = (char *)((void***)gate_proj_weight)[work->my_numa][layer_index];
    char *up_proj_weight_ptr = (char *)((void***)up_proj_weight)[work->my_numa][layer_index];

    // 计算stride
    const int weight_stride = k / srcBlockNum * srcBlocksize;
    const int input_stride = k / dstBlockNum * dstBlocksize;
    const int output_stride = 2 * n;
    
    // 获取NRC值
    const int nrc = get_nrc_value();
    
#if defined(__ARM_FEATURE_MATMUL_INT8)
    const bool use_i8mm_optimization = (srcType == GGML_TYPE_Q4_0 || srcType == GGML_TYPE_Q8_0);
#else
    const bool use_i8mm_optimization = false;
#endif

    if (!use_i8mm_optimization) {
        // F16模式：简单双重循环
        for (int i = mrange.begin_thread; i < mrange.end_thread; i++) {
            for (int j = 0; j < m; j++) {
                for (int proj = 0; proj < 2; proj++) { // 0=gate, 1=up
                    f32 *output_ptr = output_data + (j * output_stride) + proj * n + mrange.begin_numa + i;
                    char *weight_ptr = (proj == 0) ? gate_proj_weight_ptr : up_proj_weight_ptr;
                    __builtin_prefetch(output_ptr, 1, 2);
                    
                    g_BlockDataInfo[srcType].VecDotFunc(k,
                        output_ptr, 0,
                        weight_ptr + i * weight_stride, 0,
                        (char *)input_numa_buffers[work->my_numa] + j * input_stride, 0, 1);
                }
            }
        }
    }
    else {
        // i8mm优化模式：根据nrc值选择不同的块处理策略
        const int i_end = mrange.end_thread;
        int i = mrange.begin_thread;
        
        if (nrc == 4) {
            // nrc=4: 4x4块处理策略
            // 主循环：先对矩阵按4x4的大块处理
            for (; i + 3 < i_end; i += 4) {
                int j = 0;
                
                // 内层4x4块处理
                for (; j + 3 < m; j += 4) {
                    for (int proj = 0; proj < 2; proj++) { // 0=gate, 1=up
                        f32 *base_output_ptr = output_data + (j * output_stride) + proj * n + mrange.begin_numa + i;
                        char *weight_ptr = (proj == 0) ? gate_proj_weight_ptr : up_proj_weight_ptr;
                        char *base_weight_ptr = weight_ptr + i * weight_stride;
                        char *base_input_ptr = (char *)input_numa_buffers[work->my_numa] + j * input_stride;
                        
                        // 预取优化
                        __builtin_prefetch(base_output_ptr, 1, 2);
                        __builtin_prefetch(base_output_ptr + output_stride, 1, 2);
                        __builtin_prefetch(base_output_ptr + 2 * output_stride, 1, 2);
                        __builtin_prefetch(base_output_ptr + 3 * output_stride, 1, 2);
                        
                        // 处理4x4块
                        g_BlockDataInfo[srcType].VecDotFunc(k,
                            base_output_ptr, output_stride,
                            base_weight_ptr, weight_stride,
                            base_input_ptr, input_stride, 4);
                    }
                }
                
                // 处理剩余token的4行输出：先用nrc=2的中块，最后用nrc=1小块
                int jj = j;
                
                // 先用2x2块处理剩余的token（对于这4行）
                for (; jj + 1 < m; jj += 2) {
                    for (int proj = 0; proj < 2; proj++) {
                        int ii = 0;
                        // 处理4行中的前2行，2个token（2x2块）
                        for (; ii + 1 < 4; ii += 2) {
                            f32 *base_output_ptr = output_data + (jj * output_stride) + proj * n + mrange.begin_numa + i + ii;
                            char *weight_ptr = (proj == 0) ? gate_proj_weight_ptr : up_proj_weight_ptr;
                            char *base_weight_ptr = weight_ptr + (i + ii) * weight_stride;
                            char *base_input_ptr = (char *)input_numa_buffers[work->my_numa] + jj * input_stride;
                            
                            // 预取优化
                            __builtin_prefetch(base_output_ptr, 1, 2);
                            __builtin_prefetch(base_output_ptr + output_stride, 1, 2);
                            
                            // 处理2x2块
                            g_BlockDataInfo[srcType].VecDotFunc(k,
                                base_output_ptr, output_stride,
                                base_weight_ptr, weight_stride,
                                base_input_ptr, input_stride, 2);
                        }
                    }
                }
                
                // 处理最后剩余的单个token（如果有）
                if (jj < m) {
                    for (int proj = 0; proj < 2; proj++) {
                        int ii = 0;
                        // 先用2x1块处理这4行中的前2行
                        for (; ii + 1 < 4; ii += 2) {
                            f32 *base_output_ptr = output_data + (jj * output_stride) + proj * n + mrange.begin_numa + i + ii;
                            char *weight_ptr = (proj == 0) ? gate_proj_weight_ptr : up_proj_weight_ptr;
                            char *base_input_ptr = (char *)input_numa_buffers[work->my_numa] + jj * input_stride;
                            
                            __builtin_prefetch(base_output_ptr, 1, 2);
                            
                            // 处理2行输出，单个token
                            for (int kk = 0; kk < 2; ++kk) {
                                g_BlockDataInfo[srcType].VecDotFunc(k,
                                    base_output_ptr + kk, 0,
                                    weight_ptr + (i + ii + kk) * weight_stride, 0,
                                    base_input_ptr, 0, 1);
                            }
                        }
                    }
                }
            }
            
            // 处理剩余的行：先用nrc=2，最后用nrc=1
            // 处理剩余的2行
            if (i + 1 < i_end) {
                int j = 0;
                
                // 先用2x2块处理token
                for (; j + 1 < m; j += 2) {
                    for (int proj = 0; proj < 2; proj++) {
                        f32 *base_output_ptr = output_data + (j * output_stride) + proj * n + mrange.begin_numa + i;
                        char *weight_ptr = (proj == 0) ? gate_proj_weight_ptr : up_proj_weight_ptr;
                        char *base_weight_ptr = weight_ptr + i * weight_stride;
                        char *base_input_ptr = (char *)input_numa_buffers[work->my_numa] + j * input_stride;
                        
                        // 预取优化
                        __builtin_prefetch(base_output_ptr, 1, 2);
                        __builtin_prefetch(base_output_ptr + output_stride, 1, 2);
                        
                        // 处理2x2块
                        g_BlockDataInfo[srcType].VecDotFunc(k,
                            base_output_ptr, output_stride,
                            base_weight_ptr, weight_stride,
                            base_input_ptr, input_stride, 2);
                    }
                }
                
                // 处理剩余token的2行输出
                if (j < m) {
                    for (int proj = 0; proj < 2; proj++) {
                        f32 *base_output_ptr = output_data + (j * output_stride) + proj * n + mrange.begin_numa + i;
                        char *weight_ptr = (proj == 0) ? gate_proj_weight_ptr : up_proj_weight_ptr;
                        char *base_input_ptr = (char *)input_numa_buffers[work->my_numa] + j * input_stride;
                        
                        __builtin_prefetch(base_output_ptr, 1, 2);
                        
                        // 处理2行输出，单个token
                        for (int ii = 0; ii < 2; ++ii) {
                            g_BlockDataInfo[srcType].VecDotFunc(k,
                                base_output_ptr + ii, 0,
                                weight_ptr + (i + ii) * weight_stride, 0,
                                base_input_ptr, 0, 1);
                        }
                    }
                }
                i += 2;
            }
            
            // 处理最后剩余的1行（如果有）
            if (i < i_end) {
                for (int j = 0; j < m; j++) {
                    for (int proj = 0; proj < 2; proj++) {
                        f32 *output_ptr = output_data + (j * output_stride) + proj * n + mrange.begin_numa + i;
                        char *weight_ptr = (proj == 0) ? gate_proj_weight_ptr : up_proj_weight_ptr;
                        
                        __builtin_prefetch(output_ptr, 1, 2);
                        g_BlockDataInfo[srcType].VecDotFunc(k,
                            output_ptr, 0,
                            weight_ptr + i * weight_stride, 0,
                            (char *)input_numa_buffers[work->my_numa] + j * input_stride, 0, 1);
                    }
                }
            }
        } else {
            // nrc=2: 2x2块处理策略（原始逻辑）
            // 主循环：2x2块处理
            for (; i + 1 < i_end; i += 2) {
                int j = 0;
                
                // 内层2x2块处理
                for (; j + 1 < m; j += 2) {
                    for (int proj = 0; proj < 2; proj++) { // 0=gate, 1=up
                        f32 *base_output_ptr = output_data + (j * output_stride) + proj * n + mrange.begin_numa + i;
                        char *weight_ptr = (proj == 0) ? gate_proj_weight_ptr : up_proj_weight_ptr;
                        char *base_weight_ptr = weight_ptr + i * weight_stride;
                        char *base_input_ptr = (char *)input_numa_buffers[work->my_numa] + j * input_stride;
                        
                        // 预取优化
                        __builtin_prefetch(base_output_ptr, 1, 2);
                        __builtin_prefetch(base_output_ptr + output_stride, 1, 2);
                        
                        // 处理2x2块
                        g_BlockDataInfo[srcType].VecDotFunc(k,
                            base_output_ptr, output_stride,
                            base_weight_ptr, weight_stride,
                            base_input_ptr, input_stride, 2);
                    }
                }
                
                // 处理剩余的单个token (2x1块)
                if (j < m) {
                    for (int proj = 0; proj < 2; proj++) {
                        f32 *base_output_ptr = output_data + (j * output_stride) + proj * n + mrange.begin_numa + i;
                        char *weight_ptr = (proj == 0) ? gate_proj_weight_ptr : up_proj_weight_ptr;
                        char *base_weight_ptr = weight_ptr + i * weight_stride;
                        char *base_input_ptr = (char *)input_numa_buffers[work->my_numa] + j * input_stride;
                        
                        __builtin_prefetch(base_output_ptr, 1, 2);
                        
                        // 处理2行输出，单个token
                        for (int ii = 0; ii < 2; ++ii) {
                            g_BlockDataInfo[srcType].VecDotFunc(k,
                                base_output_ptr + ii, 0,
                                base_weight_ptr + ii * weight_stride, 0,
                                base_input_ptr, 0, 1);
                        }
                    }
                }
            }
            
            // 处理剩余的单个输出行 (1x块)
            if (i < i_end) {
                for (int j = 0; j < m; j++) {
                    for (int proj = 0; proj < 2; proj++) {
                        f32 *output_ptr = output_data + (j * output_stride) + proj * n + mrange.begin_numa + i;
                        char *weight_ptr = (proj == 0) ? gate_proj_weight_ptr : up_proj_weight_ptr;
                        char *weight_ptr_single = weight_ptr + i * weight_stride;
                        char *input_ptr_single = (char *)input_numa_buffers[work->my_numa] + j * input_stride;
                        
                        __builtin_prefetch(output_ptr, 1, 2);
                        g_BlockDataInfo[srcType].VecDotFunc(k,
                            output_ptr, 0,
                            weight_ptr_single, 0,
                            input_ptr_single, 0, 1);
                    }
                }
            }
        }
    }
}



// 通用的矩阵点乘计算函数
void matrix_multiply_with_numa(
    const WorkDivider *work,
    f32 *output_data,           // 输出数据
    void *weight_data,          // 权重数据 (tensor2或tensor3)
    void **input_numa_buffers,  // 输入NUMA缓冲区数组
    int srcType,                // 源数据类型
    int work_dim,               // 工作维度 (传给divide_work_all_numas)
    int vec_dot_first_param,    // VecDotFunc的第一个参数
    int output_stride,          // 输出数据的stride
    int token_count,            // token数量
    int layer_index,            // 层索引
    bool is_tensor2,            // 权重是tensor2(true)还是tensor3(false)
    const char* operation_name  // 用于调试的操作名称
) {
    enum ggml_type dstType = g_BlockDataInfo[srcType].VecDotType;
    UINT32 srcBlockNum = g_BlockDataInfo[srcType].uiblkSize;
    UINT32 srcBlocksize = g_BlockDataInfo[srcType].uiTypeSize;
    UINT32 dstBlockNum = g_BlockDataInfo[dstType].uiblkSize;
    UINT32 dstBlocksize = g_BlockDataInfo[dstType].uiTypeSize;

    MultiNumaWorkRange mrange;
    divide_work_all_numas(work, work_dim, &mrange);
    
    // 获取权重指针
    char *weight_ptr;
    if (is_tensor2) {
        weight_ptr = (char *)((void**)weight_data)[work->my_numa];
    } else {
        weight_ptr = (char *)((void***)weight_data)[work->my_numa][layer_index];
    }

    // 计算stride和offset
    const int weight_stride = vec_dot_first_param / srcBlockNum * srcBlocksize;
    const int input_stride = vec_dot_first_param / dstBlockNum * dstBlocksize;

    // 获取NRC值
    const int nrc = get_nrc_value();

#if defined(__ARM_FEATURE_MATMUL_INT8)
    const bool use_i8mm_optimization = (srcType == GGML_TYPE_Q4_0 || srcType == GGML_TYPE_Q8_0);
#else
    const bool use_i8mm_optimization = false;
#endif

    if (!use_i8mm_optimization) {
        // F16模式：单个元素处理
        for (int i = mrange.begin_thread; i < mrange.end_thread; i++) {
            for (int k = 0; k < token_count; k++) {
                __builtin_prefetch(output_data + mrange.begin_numa + k * output_stride + i, 1, 2);
                g_BlockDataInfo[srcType].VecDotFunc(vec_dot_first_param,
                    output_data + mrange.begin_numa + k * output_stride + i, 0,
                    weight_ptr + i * weight_stride, 0,
                    (char *)input_numa_buffers[work->my_numa] + k * input_stride, 0, 1);
            }
        }
    }
    else {
        // i8mm优化模式：根据nrc值选择不同的块处理策略
        const int i_end = mrange.end_thread;
        int i = mrange.begin_thread;
        
        if (nrc == 4) {
            // nrc=4: 4x4块处理策略
            // 主循环：4x4块处理
            for (; i + 3 < i_end; i += 4) {
                int k = 0;
                
                // 内层4x4块处理
                for (; k + 3 < token_count; k += 4) {
                    f32 *base_output_ptr = output_data + mrange.begin_numa + k * output_stride + i;
                    char *base_weight_ptr = weight_ptr + i * weight_stride;
                    char *base_input_ptr = (char *)input_numa_buffers[work->my_numa] + k * input_stride;
                    
                    // 预取优化
                    __builtin_prefetch(base_output_ptr, 1, 2);
                    __builtin_prefetch(base_output_ptr + output_stride, 1, 2);
                    __builtin_prefetch(base_output_ptr + 2 * output_stride, 1, 2);
                    __builtin_prefetch(base_output_ptr + 3 * output_stride, 1, 2);
                    
                    // 处理4x4块
                    g_BlockDataInfo[srcType].VecDotFunc(vec_dot_first_param,
                            base_output_ptr, output_stride,
                            base_weight_ptr, weight_stride,
                            base_input_ptr, input_stride, 4);
                }
                
                // 处理剩余token的4行输出：先用nrc=2，最后用nrc=1
                int kk = k;
                
                // 先用2x2块处理剩余的token（对于这4行）
                for (; kk + 1 < token_count; kk += 2) {
                    int ii = 0;
                    // 处理4行中的前2行，2个token（2x2块）
                    for (; ii + 1 < 4; ii += 2) {
                        f32 *base_output_ptr = output_data + mrange.begin_numa + kk * output_stride + i + ii;
                        char *base_weight_ptr = weight_ptr + (i + ii) * weight_stride;
                        char *base_input_ptr = (char *)input_numa_buffers[work->my_numa] + kk * input_stride;
                        
                        // 预取优化
                        __builtin_prefetch(base_output_ptr, 1, 2);
                        __builtin_prefetch(base_output_ptr + output_stride, 1, 2);
                        
                        // 处理2x2块
                        g_BlockDataInfo[srcType].VecDotFunc(vec_dot_first_param,
                                base_output_ptr, output_stride,
                                base_weight_ptr, weight_stride,
                                base_input_ptr, input_stride, 2);
                    }
                }
                
                // 处理最后剩余的单个token（如果有）
                if (kk < token_count) {
                    int ii = 0;
                    // 先用2x1块处理这4行中的前2行
                    for (; ii + 1 < 4; ii += 2) {
                        f32 *base_output_ptr = output_data + mrange.begin_numa + kk * output_stride + i + ii;
                        char *base_weight_ptr = weight_ptr + (i + ii) * weight_stride;
                        char *base_input_ptr = (char *)input_numa_buffers[work->my_numa] + kk * input_stride;
                        
                        __builtin_prefetch(base_output_ptr, 1, 2);
                        
                        // 处理2行输出，单个token
                        for (int jj = 0; jj < 2; ++jj) {
                            g_BlockDataInfo[srcType].VecDotFunc(vec_dot_first_param,
                                    base_output_ptr + jj, 0,
                                    base_weight_ptr + jj * weight_stride, 0,
                                    base_input_ptr, 0, 1);
                        }
                    }
                }
            }
            
            // 处理剩余的行：先用nrc=2，最后用nrc=1
            // 处理剩余的2行
            if (i + 1 < i_end) {
                int k = 0;
                
                // 先用2x2块处理token  
                for (; k + 1 < token_count; k += 2) {
                    f32 *base_output_ptr = output_data + mrange.begin_numa + k * output_stride + i;
                    char *base_weight_ptr = weight_ptr + i * weight_stride;
                    char *base_input_ptr = (char *)input_numa_buffers[work->my_numa] + k * input_stride;
                    
                    // 预取优化
                    __builtin_prefetch(base_output_ptr, 1, 2);
                    __builtin_prefetch(base_output_ptr + output_stride, 1, 2);
                    
                    // 处理2x2块
                    g_BlockDataInfo[srcType].VecDotFunc(vec_dot_first_param,
                            base_output_ptr, output_stride,
                            base_weight_ptr, weight_stride,
                            base_input_ptr, input_stride, 2);
                }
                
                // 处理剩余token的2行输出
                if (k < token_count) {
                    f32 *base_output_ptr = output_data + mrange.begin_numa + k * output_stride + i;
                    char *base_weight_ptr = weight_ptr + i * weight_stride;
                    char *base_input_ptr = (char *)input_numa_buffers[work->my_numa] + k * input_stride;
                    
                    __builtin_prefetch(base_output_ptr, 1, 2);
                    
                    // 处理2行输出，单个token
                    for (int ii = 0; ii < 2; ++ii) {
                        g_BlockDataInfo[srcType].VecDotFunc(vec_dot_first_param,
                                base_output_ptr + ii, 0,
                                base_weight_ptr + ii * weight_stride, 0,
                                base_input_ptr, 0, 1);
                    }
                }
                i += 2;
            }
            
            // 处理最后剩余的1行（如果有）
            if (i < i_end) {
                for (int k = 0; k < token_count; k++) {
                    f32 *output_ptr = output_data + mrange.begin_numa + k * output_stride + i;
                    char *weight_ptr_single = weight_ptr + i * weight_stride;
                    char *input_ptr_single = (char *)input_numa_buffers[work->my_numa] + k * input_stride;
                    
                    __builtin_prefetch(output_ptr, 1, 2);
                    g_BlockDataInfo[srcType].VecDotFunc(vec_dot_first_param,
                            output_ptr, 0,
                            weight_ptr_single, 0,
                            input_ptr_single, 0, 1);
                }
            }
        } else {
            // nrc=2: 2x2块处理策略（原始逻辑）
            // 主循环：2x2块处理
            for (; i + 1 < i_end; i += 2) {
                int k = 0;
                
                // 内层2x2块处理
                for (; k + 1 < token_count; k += 2) {
                    f32 *base_output_ptr = output_data + mrange.begin_numa + k * output_stride + i;
                    char *base_weight_ptr = weight_ptr + i * weight_stride;
                    char *base_input_ptr = (char *)input_numa_buffers[work->my_numa] + k * input_stride;
                    
                    // 预取优化
                    __builtin_prefetch(base_output_ptr, 1, 2);
                    __builtin_prefetch(base_output_ptr + output_stride, 1, 2);
                    
                    // 处理2x2块
                    g_BlockDataInfo[srcType].VecDotFunc(vec_dot_first_param,
                            base_output_ptr, output_stride,
                            base_weight_ptr, weight_stride,
                            base_input_ptr, input_stride, 2);
                }
                
                // 处理剩余的单个token (2x1块)
                if (k < token_count) {
                    f32 *base_output_ptr = output_data + mrange.begin_numa + k * output_stride + i;
                    char *base_weight_ptr = weight_ptr + i * weight_stride;
                    char *base_input_ptr = (char *)input_numa_buffers[work->my_numa] + k * input_stride;
                    
                    __builtin_prefetch(base_output_ptr, 1, 2);
                    
                    // 处理2行输出，单个token
                    for (int ii = 0; ii < 2; ++ii) {
                        g_BlockDataInfo[srcType].VecDotFunc(vec_dot_first_param,
                                base_output_ptr + ii, 0,
                                base_weight_ptr + ii * weight_stride, 0,
                                base_input_ptr, 0, 1);
                    }
                }
            }
            
            // 处理剩余的单个输出行 (1x块)
            if (i < i_end) {
                for (int k = 0; k < token_count; k++) {
                    f32 *output_ptr = output_data + mrange.begin_numa + k * output_stride + i;
                    char *weight_ptr_single = weight_ptr + i * weight_stride;
                    char *input_ptr_single = (char *)input_numa_buffers[work->my_numa] + k * input_stride;
                    
                    __builtin_prefetch(output_ptr, 1, 2);
                    g_BlockDataInfo[srcType].VecDotFunc(vec_dot_first_param,
                            output_ptr, 0,
                            weight_ptr_single, 0,
                            input_ptr_single, 0, 1);
                }
            }
        }
    }
}



/* 反量化 */
void Dequantize(void *DstData, void *SrcData, WEIGHT *pstWeight, int dataNum)
{
    enum ggml_type TokenType = pstWeight->embed_tokens_weight.DataType;

    /* 需要反量化情况 */
    if (TokenType != GGML_TYPE_F32 && pstWeight->input_layernorm_weight.DataType == GGML_TYPE_F32) {
        g_BlockDataInfo[TokenType].dequantize(SrcData, static_cast<float*>(DstData), dataNum);
    }
}

__attribute__((noinline))
void Rope_embedding_impl(bool rope_type, int n_rotary, f16 *head_ptr, const f16 *cos_sin_cache, int position)
{
    const f16 *cos_sin_ptr = cos_sin_cache + position * n_rotary;
    int embed_dim = n_rotary >> 1;

    /* rope_neox */
    if (rope_type == true) {
        int xx = 0, yy = embed_dim;
        for (; xx <= embed_dim - 8; xx += 8, yy += 8) {
            __builtin_prefetch(&head_ptr[xx + 32], 1, 2);
            __builtin_prefetch(&head_ptr[yy + 32], 1, 2);
            const float16x8_t qx = vld1q_f16(&head_ptr[xx]), qy = vld1q_f16(&head_ptr[yy]);
            const float16x8_t csx = vld1q_f16(&cos_sin_ptr[xx]), csy = vld1q_f16(&cos_sin_ptr[yy]);
            vst1q_f16(&head_ptr[xx], vfmaq_f16(vmulq_f16(qx, csx), vnegq_f16(qy), csy));
            vst1q_f16(&head_ptr[yy], vfmaq_f16(vmulq_f16(qy, csx), qx, csy));
        }
        for (; xx < embed_dim; xx++, yy++) {
            const f16 qx = head_ptr[xx], qy = head_ptr[yy];
            head_ptr[xx] = qx * cos_sin_ptr[xx] - qy * cos_sin_ptr[yy];
            head_ptr[yy] = qy * cos_sin_ptr[xx] + qx * cos_sin_ptr[yy];
        }
     } else { /* rope_gptj */
        for (int j = 0; j < embed_dim; j++) {
            const f16 qx = head_ptr[2 * j], qy = head_ptr[2 * j + 1];
            const f16 cos = cos_sin_ptr[j], sin = cos_sin_ptr[embed_dim + j];
            head_ptr[2 * j] = qx * cos - qy * sin;
            head_ptr[2 * j + 1] = qy * cos + qx * sin;
        }
     }
}

void qkv_rope_and_cache_numa(
    const WorkDivider *work,
    f16 *q_ptr,                     // Q数据指针
    f16 *k_ptr,                     // K数据指针
    f16 *v_ptr,                     // V数据指针  
    f16 *kcache_ptr,                // K cache指针
    f16 *vcache_ptr,                // V cache指针
    int64_t *slot_mapping_ptr,      // slot mapping
    int64_t *pos,                   // position数组
    int n_tokens,                   // token数量
    int num_attention_heads,        // head数量
    int num_key_value_heads,        // KV head数量
    int head_size,                  // head size
    int qkv_dim,                    // qkv维度
    int block_size,                 // block size
    int kv_cache_block_elem_num,    // kv cache block元素数量
    double attn_scale,              // attention scale
    const char* operation_name      // 用于调试的操作名称
) {
    SingleNumaWorkRange srange;
    
    // 首先处理KV cache保存和K的Rope计算
    divide_all_work(work, n_tokens * num_key_value_heads, &srange);
    for (int i = srange.begin_thread; i < srange.end_thread; i++) {
        int t = i / num_key_value_heads;
        int h = i % num_key_value_heads;
        const int64_t slot = slot_mapping_ptr[t];
        if (slot < 0) {
            continue;
        }
        int64_t block_idx = slot / block_size, block_offset = slot % block_size;
        f16 *k_head_ptr = k_ptr + t * qkv_dim + h * head_size;
        f16 *kcache_head_ptr = kcache_ptr + kv_cache_block_elem_num * block_idx + h * block_size * head_size;
        const f16 *v_head_ptr = v_ptr + t * qkv_dim + h * head_size;
        f16 *vcache_head_ptr = vcache_ptr + kv_cache_block_elem_num * block_idx + h * block_size * head_size;
        
        // K的Rope计算
        Rope_embedding_impl(g_pstModelHypePara.is_neox_style, g_pstModelHypePara.n_rotary, k_head_ptr, g_pstModelHypePara.cos_sin_cache, pos[t]);
        
        // 保存K和V到cache
        for (int idx = 0; idx < head_size; idx += 8) {   //8 = 16 / sizeof(f16)
            for (int vidx = idx; vidx < idx + 8; vidx++) {
                vcache_head_ptr[vidx * block_size + block_offset] = v_head_ptr[vidx];
            }
            std::copy_n(k_head_ptr + idx, 8, kcache_head_ptr + idx * block_size + block_offset * 8);
        }
    }
    
    // 然后处理Q的Rope计算和缩放
    divide_all_work(work, n_tokens * num_attention_heads, &srange);
    for (int i = srange.begin_thread; i < srange.end_thread; i++) {
        int t = i / num_attention_heads;
        int h = i % num_attention_heads;
        
        // Q的Rope计算
        Rope_embedding_impl(g_pstModelHypePara.is_neox_style, g_pstModelHypePara.n_rotary, q_ptr + t * qkv_dim + h * head_size,
                            g_pstModelHypePara.cos_sin_cache, pos[t]);
        
    }
}

float silu_table(float x) {
    float result;
                
    if (x >= 4.0f) {
        result = x;
    } else if (x < -10.0f) {
        result = 0.0f;
    } else if (x >= 0.0f) {
        f16 neg_x = -x;
        f32 exp_neg_x = expf_f16_table[*(uint16_t *)&neg_x];
        result = x / (1.0f + exp_neg_x);
    } else {
        f16 x_f16 = x;
        f32 exp_x = expf_f16_table[*(uint16_t *)&x_f16];
        result = x * exp_x / (1.0f + exp_x);
    }

    return result;
}

void neon_silu_computation_with_table(
    float* input_data,
    float* output_data,
    int token_count,
    int intermediate_size,
    const SingleNumaWorkRange& srange)
{
    for (int i = 0; i < token_count; i++) {
        float* gate_data = input_data + i * intermediate_size * 2;           // w1部分
        float* up_data = input_data + i * intermediate_size * 2 + intermediate_size; // w3部分
        float* result = output_data + i * intermediate_size;          // 输出位置

        int j = srange.begin_thread;
        for (; j + 4 <= srange.end_thread; j += 4) {
            // 加载数据
            float32x4_t up_data_vec = vld1q_f32(up_data + j);
            
            float res0 = silu_table(gate_data[j]);
            float res1 = silu_table(gate_data[j + 1]);
            float res2 = silu_table(gate_data[j + 2]);
            float res3 = silu_table(gate_data[j + 3]);

            float32x4_t result_vec = {res0, res1, res2, res3};

            float32x4_t output = vmulq_f32(result_vec, up_data_vec);
            vst1q_f32(result + j, output);
        }
        
        // 处理剩余元素
        for (; j < srange.end_thread; j++) {
            float x = gate_data[j];
            float res = silu_table(x);
            result[j] = res * up_data[j];
        }
    }
}

// 通用的SiLU激活函数
void silu_activation_with_numa(
    const WorkDivider *work,
    f32 *output_data,           // 输出数据 (ffn_Gate)
    f32 *input_data,            // 输入数据 (w1w3结果)
    int token_count,            // token数量
    int intermediate_size,             // 隐藏层维度
    const char* operation_name  // 用于调试的操作名称
) {
    SingleNumaWorkRange srange;
    divide_all_work(work, intermediate_size, &srange);
    
#ifdef __ARM_NEON
        neon_silu_computation_with_table(input_data, output_data, token_count, intermediate_size, srange);
#else
    for (int i = 0; i < token_count; i++) {
        f32 *gate_data = input_data + i * 2 * intermediate_size;           // gate部分
        f32 *up_data = input_data + i * 2 * intermediate_size + intermediate_size; // up部分
        f32 *result = output_data + i * intermediate_size;          // 输出位置
        for (int j = srange.begin_thread; j < srange.end_thread; j++) {
            // SiLU激活函数: silu(x) = x / (1 + exp(-x))
            f16 neg_gate = -gate_data[j];
            f32 silu_f32 = gate_data[j] / (1.0 + expf_f16_table[*(uint16_t *)&neg_gate]);
            
            // SiLU(gate) * up
            result[j] = silu_f32 * up_data[j];
        }
    }
#endif
}

void quantization_weight_strategy(void *dst, void *src, int64_t quantization_bit_code, size_t Size)
{
    int quant_blcok = 256;
    float Buffer[quant_blcok];
    assert(Size % quant_blcok == 0);
    int block_num = Size / quant_blcok;

    /* 反量化 */
    if (quantization_bit_code == GGML_TYPE_F32) {
        g_BlockDataInfo[GGML_TYPE_F16].dequantize(src, static_cast<float*>(dst), Size);
    } else {
        if (quantization_bit_code != GGML_TYPE_F16) {
            int offset = quant_blcok / g_BlockDataInfo[quantization_bit_code].uiblkSize * g_BlockDataInfo[quantization_bit_code].uiTypeSize;
            for (int i = 0; i < block_num; i++) {
                g_BlockDataInfo[GGML_TYPE_F16].dequantize((char *)src + i * quant_blcok * sizeof(f16), Buffer, quant_blcok);
                g_BlockDataInfo[quantization_bit_code].quantize(Buffer, (char *)dst + i * offset, quant_blcok);
            }
        } else { /* 直接复制权重 */
            memcpy(dst, src, Size * sizeof(f16));
        }
    }
}

void load_weight_and_malloc_active_tensor(
    torch::Tensor embed_tokens_weight,                  // [vocab_size, hidden_size]
    torch::Tensor input_layernorm_weight,               // [num_hidden_layers][hidden_size]
    torch::Tensor post_attention_layernorm_weight,      // [num_hidden_layers][hidden_size]
    torch::Tensor qkv_proj_weight,                      // [num_hidden_layers][q_dim + k_dim + v_dim, hidden_size]
    torch::Tensor o_proj_weight,                        // [num_hidden_layers][hidden_size, q_dim]
    torch::Tensor qkv_proj_bias,                        // [num_hidden_layers][q_dim + k_dim + v_dim]
    torch::Tensor gate_up_proj_weight,                  // [num_hidden_layers][2 * intermediate_size, hidden_size]
    torch::Tensor down_proj_weight,                     // [num_hidden_layers][hidden_size, intermediate_size]
    torch::Tensor norm_weight,                          // [hidden_size]
    torch::Tensor lm_head_weight                        // [vocab_size, hidden_size]
){

    assert(qkv_proj_weight.dtype() == torch::kFloat16);
    int64_t head_dim = g_pstModelHypePara.head_dim;
    int64_t hidden_size = g_pstModelHypePara.hidden_size;
    int64_t intermediate_size = g_pstModelHypePara.intermediate_size;
    int64_t num_attention_heads = g_pstModelHypePara.num_attention_heads;
    int64_t num_hidden_layers = g_pstModelHypePara.num_hidden_layers;
    int64_t vocab_size = g_pstModelHypePara.vocab_size;
    int64_t num_key_value_heads = g_pstModelHypePara.num_key_value_heads;
    int64_t max_position_embeddings = g_pstModelHypePara.max_position_embeddings;
    int64_t N_gqa = num_attention_heads / num_key_value_heads;

    int64_t q_dim = head_dim * num_attention_heads;
    int64_t k_dim = head_dim * num_key_value_heads;
    int64_t v_dim = head_dim * num_key_value_heads;

    size_t tokens_embedding_weight_size = (size_t)hidden_size * vocab_size / g_BlockDataInfo[g_pstWeightTypes.token_embd_weight].uiblkSize * g_BlockDataInfo[g_pstWeightTypes.token_embd_weight].uiTypeSize;

    size_t attention_q_size_per_layer = (size_t)hidden_size * q_dim / g_BlockDataInfo[g_pstWeightTypes.attn_q_weight].uiblkSize * g_BlockDataInfo[g_pstWeightTypes.attn_q_weight].uiTypeSize;
    size_t attention_k_size_per_layer = (size_t)hidden_size * k_dim / g_BlockDataInfo[g_pstWeightTypes.attn_k_weight].uiblkSize * g_BlockDataInfo[g_pstWeightTypes.attn_k_weight].uiTypeSize;
    size_t attention_v_size_per_layer = (size_t)hidden_size * v_dim / g_BlockDataInfo[g_pstWeightTypes.attn_v_weight].uiblkSize * g_BlockDataInfo[g_pstWeightTypes.attn_v_weight].uiTypeSize;
    size_t attention_size_per_layer = attention_q_size_per_layer + attention_k_size_per_layer + attention_v_size_per_layer;

    size_t bias_q_size_per_layer = (size_t)q_dim / g_BlockDataInfo[g_pstWeightTypes.attn_q_bias].uiblkSize * g_BlockDataInfo[g_pstWeightTypes.attn_q_bias].uiTypeSize;
    size_t bias_k_size_per_layer = (size_t)k_dim / g_BlockDataInfo[g_pstWeightTypes.attn_k_bias].uiblkSize * g_BlockDataInfo[g_pstWeightTypes.attn_k_bias].uiTypeSize;
    size_t bias_v_size_per_layer = (size_t)v_dim / g_BlockDataInfo[g_pstWeightTypes.attn_v_bias].uiblkSize * g_BlockDataInfo[g_pstWeightTypes.attn_v_bias].uiTypeSize;
    size_t bias_qkv_size_per_layer = bias_q_size_per_layer + bias_k_size_per_layer + bias_v_size_per_layer;

    size_t attention_norm_size_per_layer = (size_t)hidden_size / g_BlockDataInfo[g_pstWeightTypes.attn_norm_weight].uiblkSize * g_BlockDataInfo[g_pstWeightTypes.attn_norm_weight].uiTypeSize;

    size_t ffn_down_size_per_layer = (size_t)intermediate_size * hidden_size / g_BlockDataInfo[g_pstWeightTypes.ffn_down_weight].uiblkSize * g_BlockDataInfo[g_pstWeightTypes.ffn_down_weight].uiTypeSize;
    size_t ffn_gate_size_per_layer = (size_t)hidden_size * intermediate_size / g_BlockDataInfo[g_pstWeightTypes.ffn_gate_weight].uiblkSize * g_BlockDataInfo[g_pstWeightTypes.ffn_gate_weight].uiTypeSize;
    size_t ffn_norm_size_per_layer = (size_t)hidden_size / g_BlockDataInfo[g_pstWeightTypes.ffn_norm_weight].uiblkSize * g_BlockDataInfo[g_pstWeightTypes.ffn_norm_weight].uiTypeSize;
    size_t ffn_up_size_per_layer = (size_t)hidden_size * intermediate_size / g_BlockDataInfo[g_pstWeightTypes.ffn_up_weight].uiblkSize * g_BlockDataInfo[g_pstWeightTypes.ffn_up_weight].uiTypeSize;
    size_t w1w3_size_per_layer = ffn_gate_size_per_layer + ffn_up_size_per_layer;

    size_t attention_output_size_per_layer = (size_t)hidden_size * q_dim / g_BlockDataInfo[g_pstWeightTypes.attn_output_weight].uiblkSize * g_BlockDataInfo[g_pstWeightTypes.attn_output_weight].uiTypeSize;
    size_t output_size = (size_t)hidden_size * vocab_size / g_BlockDataInfo[g_pstWeightTypes.output_weight].uiblkSize * g_BlockDataInfo[g_pstWeightTypes.output_weight].uiTypeSize;
    size_t output_norm_size = (size_t)hidden_size / g_BlockDataInfo[g_pstWeightTypes.output_norm_weight].uiblkSize * g_BlockDataInfo[g_pstWeightTypes.output_norm_weight].uiTypeSize;

    g_pstWeight.input_layernorm_weight.Data.tensor2 = static_cast<void**>(numa_alloc_onnode(num_hidden_layers * sizeof(void *), 0));
    g_pstWeight.post_attention_layernorm_weight.Data.tensor2 = static_cast<void**>(numa_alloc_onnode(num_hidden_layers * sizeof(void *), 0));
    g_pstWeight.qkv_proj_bias.Data.tensor2 = static_cast<void**>(numa_alloc_onnode(num_hidden_layers * sizeof(void *), 0));

    for (int i = 0; i < num_hidden_layers; i++) {
        g_pstWeight.input_layernorm_weight.Data.tensor2[i] = numa_alloc_onnode(attention_norm_size_per_layer, 0);
        g_pstWeight.post_attention_layernorm_weight.Data.tensor2[i] = numa_alloc_onnode(ffn_norm_size_per_layer, 0);
        g_pstWeight.qkv_proj_bias.Data.tensor2[i] = numa_alloc_onnode(bias_qkv_size_per_layer, 0);
    }
    g_pstWeight.norm_weight.Data.tensor1 = numa_alloc_onnode(output_norm_size, 0);

    g_pstWeight.qkv_proj_weight.Data.tensor3 = static_cast<void***>(numa_alloc_onnode(g_numas * sizeof(void **), 0));
    g_pstWeight.o_proj_weight.Data.tensor3 = static_cast<void***>(numa_alloc_onnode(g_numas * sizeof(void **), 0));
    g_pstWeight.down_proj_weight.Data.tensor3 = static_cast<void***>(numa_alloc_onnode(g_numas * sizeof(void **), 0));
    g_pstWeight.lm_head_weight.Data.tensor2 = static_cast<void**>(numa_alloc_onnode(g_numas * sizeof(void *), 0));
    g_pstWeight.embed_tokens_weight.Data.tensor1 = numa_alloc_onnode(tokens_embedding_weight_size, 0);

    g_pstWeight.gate_proj_weight.Data.tensor3 = static_cast<void***>(numa_alloc_onnode(g_numas * sizeof(void **), 0));
    g_pstWeight.up_proj_weight.Data.tensor3 = static_cast<void***>(numa_alloc_onnode(g_numas * sizeof(void **), 0));

    for (int i = 0; i < g_numas; i++) {
        g_pstWeight.qkv_proj_weight.Data.tensor3[i] = static_cast<void**>(numa_alloc_onnode(num_hidden_layers * sizeof(void *), i));
        g_pstWeight.o_proj_weight.Data.tensor3[i] = static_cast<void**>(numa_alloc_onnode(num_hidden_layers * sizeof(void *), i));
        g_pstWeight.down_proj_weight.Data.tensor3[i] = static_cast<void**>(numa_alloc_onnode(num_hidden_layers * sizeof(void *), i));
        g_pstWeight.lm_head_weight.Data.tensor2[i] = (void *)numa_alloc_onnode(output_size / g_numas, i);

        g_pstWeight.gate_proj_weight.Data.tensor3[i] = static_cast<void**>(numa_alloc_onnode(num_hidden_layers * sizeof(void *), i));
        g_pstWeight.up_proj_weight.Data.tensor3[i] = static_cast<void**>(numa_alloc_onnode(num_hidden_layers * sizeof(void *), i));

        for (int j = 0; j < num_hidden_layers; j++) {
            g_pstWeight.qkv_proj_weight.Data.tensor3[i][j] = numa_alloc_onnode(attention_size_per_layer / g_numas, i);
            g_pstWeight.o_proj_weight.Data.tensor3[i][j] = numa_alloc_onnode(attention_output_size_per_layer / g_numas, i);
            g_pstWeight.down_proj_weight.Data.tensor3[i][j] = numa_alloc_onnode(ffn_down_size_per_layer / g_numas, i);

            g_pstWeight.gate_proj_weight.Data.tensor3[i][j] = numa_alloc_onnode(ffn_gate_size_per_layer / g_numas, i);
            g_pstWeight.up_proj_weight.Data.tensor3[i][j] = numa_alloc_onnode(ffn_up_size_per_layer / g_numas, i);

        }
    }

    std::cout << "load_weight start ..." << std::endl;

    /* 量化权重 */
    for (int layerNum = 0; layerNum < num_hidden_layers; layerNum++) {
        quantization_weight_strategy(g_pstWeight.input_layernorm_weight.Data.tensor2[layerNum], input_layernorm_weight.index(torch::indexing::TensorIndex(layerNum)).data_ptr(),
                                     g_pstWeightTypes.attn_norm_weight, hidden_size);
        quantization_weight_strategy(g_pstWeight.post_attention_layernorm_weight.Data.tensor2[layerNum], post_attention_layernorm_weight.index(torch::indexing::TensorIndex(layerNum)).data_ptr(),
                                     g_pstWeightTypes.ffn_norm_weight, hidden_size);
        int qkv_dim = q_dim + k_dim + v_dim;
        for (int j = 0; j < g_numas; ++j) {
            f16 *qkv_pointer = (f16 *)qkv_proj_weight.index(torch::indexing::TensorIndex(layerNum)).data_ptr() + qkv_dim / g_numas * hidden_size * j;
            quantization_weight_strategy(g_pstWeight.qkv_proj_weight.Data.tensor3[j][layerNum], (char *)qkv_pointer, g_pstWeightTypes.attn_k_weight,
                                         qkv_dim / g_numas * hidden_size);

            f16 *wo_pointer = (f16 *)o_proj_weight.index(torch::indexing::TensorIndex(layerNum)).data_ptr() + hidden_size / g_numas * q_dim * j;
            quantization_weight_strategy(g_pstWeight.o_proj_weight.Data.tensor3[j][layerNum], (char *)wo_pointer,
                                         g_pstWeightTypes.attn_output_weight, hidden_size * q_dim / g_numas);

            f16 *ffn_down_pointer = (f16 *)down_proj_weight.index(torch::indexing::TensorIndex(layerNum)).data_ptr() + intermediate_size / g_numas * hidden_size * j;
            quantization_weight_strategy(g_pstWeight.down_proj_weight.Data.tensor3[j][layerNum], (char *)ffn_down_pointer,
                                         g_pstWeightTypes.ffn_down_weight, hidden_size * intermediate_size / g_numas);

            f16 *gate_proj_pointer = (f16 *)gate_up_proj_weight.index(torch::indexing::TensorIndex(layerNum)).data_ptr() + intermediate_size / g_numas * hidden_size * j;
            quantization_weight_strategy(g_pstWeight.gate_proj_weight.Data.tensor3[j][layerNum], (char *)gate_proj_pointer,
                                         g_pstWeightTypes.ffn_gate_weight, hidden_size * intermediate_size / g_numas);

            f16 *up_proj_pointer = (f16 *)gate_up_proj_weight.index(torch::indexing::TensorIndex(layerNum)).data_ptr() + intermediate_size * hidden_size + intermediate_size / g_numas * hidden_size * j;
            quantization_weight_strategy(g_pstWeight.up_proj_weight.Data.tensor3[j][layerNum], (char *)up_proj_pointer,
                                         g_pstWeightTypes.ffn_up_weight, hidden_size * intermediate_size / g_numas);
        }

        quantization_weight_strategy(g_pstWeight.qkv_proj_bias.Data.tensor2[layerNum], qkv_proj_bias.index(torch::indexing::TensorIndex(layerNum)).data_ptr(),
                                     g_pstWeightTypes.attn_q_bias, qkv_dim);
    }

    for (int i = 0; i < g_numas; i++) {
        f16 *output_pointer = (f16 *)lm_head_weight.data_ptr() + vocab_size / g_numas * hidden_size * i;
        quantization_weight_strategy(g_pstWeight.lm_head_weight.Data.tensor2[i], output_pointer,
                                     g_pstWeightTypes.output_weight, hidden_size * vocab_size /  g_numas);
    }

    quantization_weight_strategy(g_pstWeight.embed_tokens_weight.Data.tensor1, embed_tokens_weight.data_ptr(), g_pstWeightTypes.token_embd_weight, hidden_size * vocab_size);
    quantization_weight_strategy(g_pstWeight.norm_weight.Data.tensor1, norm_weight.data_ptr(), g_pstWeightTypes.output_norm_weight, hidden_size);

    std::cout << "load_weight end." << std::endl;
}

// #define DEBUG_TIME 1
static inline uint64_t get_time_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec * 1000 * 1000 * 1000 + ts.tv_nsec;
}

void get_next_token(void* output, MODEL_HYPE_PARA *pstModelHypePara, WEIGHT *pstLlama, MODEL_RUN_STATE *pstRunState,
                      bool is_prompt,
                      torch::Tensor& block_tables,
                      torch::Tensor& seq_lens,
                      torch::Tensor& slot_mapping,
                      void *hidden_state, int64_t *pos,
                      std::vector<torch::Tensor>& kv_caches,
                      int64_t block_size,
                      int n_tokens,
                      bool is_qkv_bias,
                      bool is_qk_norm)
{
    int hidden_size = pstModelHypePara->hidden_size;
    int num_key_value_heads = pstModelHypePara->num_key_value_heads;
    int num_attention_heads = pstModelHypePara->num_attention_heads;
    int head_dim = pstModelHypePara->head_dim;
    int q_dim = head_dim * num_attention_heads;
    int k_dim = head_dim * num_key_value_heads;
    int v_dim = head_dim * num_key_value_heads;
    int intermediate_size =  pstModelHypePara->intermediate_size;
    int layers = pstModelHypePara->num_hidden_layers;
    int vocab_size = pstModelHypePara->vocab_size;
    float eps = pstModelHypePara->rms_norm_eps;
    UINT32 srcBlockNum, srcBlocksize;
    UINT32 dstBlockNum, dstBlocksize;
    int srcType;
    enum ggml_type dstType;
    
    int qkv_dim = q_dim + k_dim + v_dim;
    int head_size = head_dim;
    int kv_head_dim = block_size * head_size;
    f16 *VT = (f16 *)numa_alloc_onnode(n_tokens * q_dim * sizeof(f16), 0);

    quantization_weight_strategy(pstRunState->hidden_states, (char *)hidden_state, g_pstWeightTypes.token_embd_weight, n_tokens * hidden_size);

#ifdef DEBUG_TIME
    uint64_t t0 = get_time_ns();
    uint64_t time[25] = {0};
    uint64_t tt1 = get_time_ns();
#endif

    int total_thread_num = get_total_thread_num();
    std::vector<bool> is_init_process_affinity(total_thread_num, false);

    for(int L = 0; L < layers; L++) {
#pragma omp parallel
{
        int current_thread_num = omp_get_thread_num();
        if(!is_init_process_affinity[current_thread_num]) {
            init_process_affinity();
            is_init_process_affinity[current_thread_num] = true;
        }
        WorkDivider work;
        init_work_divider(&work, g_numas);
        SingleNumaWorkRange srange;
        MultiNumaWorkRange mrange;

        residual_add_rmsnorm_with_numa(&work,
                                       (f32 *)pstRunState->hidden_states,
                                       (f32 *)pstRunState->residual,
                                       (f32 *)pstRunState->hidden_states,
                                       (f32 *)pstLlama->input_layernorm_weight.Data.tensor2[L],
                                       n_tokens,
                                       hidden_size,
                                       eps,
                                       nullptr,
                                       "attention_residual_norm");

#ifdef DEBUG_TIME
    if (work.tid == 0) {
        time[0] += get_time_ns() - tt1;
        tt1 = get_time_ns();
    }
#endif

#pragma omp barrier
        quantize_and_distribute_to_numa(&work, pstRunState, pstRunState->quant_hidden_states,
                                                (f32 *)pstRunState->hidden_states, g_pstWeightTypes.attn_k_weight,
                                                n_tokens * hidden_size, "qkv_input");

#ifdef DEBUG_TIME
if (work.tid == 0) {
    time[1] += get_time_ns() - tt1;
    tt1 = get_time_ns();
}
#endif

#pragma omp barrier
        /* 计算qkv */
        matrix_multiply_with_numa(&work, 
                                  (f32 *)pstRunState->qkv,
                                  g_pstWeight.qkv_proj_weight.Data.tensor3,
                                  pstRunState->quant_hidden_states,
                                  g_pstWeightTypes.attn_k_weight,
                                  qkv_dim,
                                  hidden_size,
                                  qkv_dim,
                                  n_tokens,
                                  L,
                                  false,
                                  "qkv_multiply");

#ifdef DEBUG_TIME
    if (work.tid == 0) {
        time[2] += get_time_ns() - tt1;
        tt1 = get_time_ns();
    }
#endif

#pragma omp barrier

        if (is_qkv_bias) {
            add_bias_with_numa(&work,
                            pstRunState->qkv,
                            (f32 *)g_pstWeight.qkv_proj_bias.Data.tensor2[L],
                            n_tokens,
                            qkv_dim,
                            "qkv_bias_add");
        }
        if (is_qk_norm) {
            qk_norm_with_numa(&work,
                              pstRunState->qkv,
                              (f32 *)g_pstWeight.q_norm_weight.Data.tensor2[L],
                              (f32 *)g_pstWeight.k_norm_weight.Data.tensor2[L],
                              n_tokens,
                              num_attention_heads,
                              num_key_value_heads,
                              head_size,
                              eps);
        }

#pragma omp barrier
        // 单线程执行量化操作
        if (work.tid == 0) {
            g_BlockDataInfo[GGML_TYPE_F16].quantize(pstRunState->qkv, pstRunState->qkv_f16, n_tokens * qkv_dim);
        }

#ifdef DEBUG_TIME
    if (work.tid == 0) {
        time[3] += get_time_ns() - tt1;
        tt1 = get_time_ns();
    }
#endif

// 中间数据f32 -> f16, 减少attention修改
#pragma omp barrier
        f16 *q_ptr = pstRunState->qkv_f16, *k_ptr = pstRunState->qkv_f16 + q_dim, *v_ptr = pstRunState->qkv_f16 + q_dim + k_dim;
        f16 *kcache_ptr = (f16 *)kv_caches[L][0].data_ptr(), *vcache_ptr = (f16 *)kv_caches[L][1].data_ptr();
        int64_t *slot_mapping_ptr = (int64_t *)slot_mapping.data_ptr();
        int kv_cache_block_elem_num = num_key_value_heads * head_size * block_size;

        qkv_rope_and_cache_numa(&work,
                                q_ptr, k_ptr, v_ptr, kcache_ptr, vcache_ptr,
                                slot_mapping_ptr, pos,
                                n_tokens, num_attention_heads, num_key_value_heads, head_size, qkv_dim,
                                block_size, kv_cache_block_elem_num,
                                g_pstModelHypePara.attn_scale,
                                "qkv_rope_and_cache");

#ifdef DEBUG_TIME
    if (work.tid == 0) {
        time[4] += get_time_ns() - tt1;
        tt1 = get_time_ns();
    }
#endif

}
        //divide_kv_cache_numa(&work, n_head, &srange);
        if (is_prompt == true) {
            f16 *v_ptr = (f16 *)pstRunState->qkv_f16 + q_dim + k_dim;
            transpose_v(VT, v_ptr, n_tokens, v_dim, qkv_dim);
            prefill_attention(pstRunState->attn_output_f16, pstRunState->qkv_f16, VT, n_tokens, seq_lens.size(0),
                              (int *)seq_lens.data_ptr());
        } else {
            int kv_block_row = kv_caches[L][0].stride(0);
            paged_attention_v1_impl(pstRunState->attn_output_f16, pstRunState->qkv_f16, (f16 *)kv_caches[L][0].data_ptr(),
                                    (f16 *)kv_caches[L][1].data_ptr(), num_key_value_heads, (int *)block_tables.data_ptr(),
                                    (int *)seq_lens.data_ptr(), block_tables.size(1), qkv_dim, kv_block_row,
                                    kv_head_dim, n_tokens, num_attention_heads, head_size);
        }

#ifdef DEBUG_TIME
    time[5] += get_time_ns() - tt1;
    tt1 = get_time_ns();
#endif

// 数据转换f16 —> f32
#pragma omp parallel 
{
        WorkDivider work;
        init_work_divider(&work, g_numas);
        SingleNumaWorkRange srange;
        MultiNumaWorkRange mrange;

        if (work.tid == 1) {
            g_BlockDataInfo[GGML_TYPE_F16].dequantize(pstRunState->attn_output_f16, pstRunState->attn_output, n_tokens * q_dim);
        }

#ifdef DEBUG_TIME
    if (work.tid == 0) {
        time[6] += get_time_ns() - tt1;
        tt1 = get_time_ns();
    }
#endif

#pragma omp barrier
        quantize_and_distribute_to_numa(&work, pstRunState, pstRunState->quant_attn_output,
                                                (f32 *)pstRunState->attn_output, g_pstWeightTypes.attn_output_weight,
                                                n_tokens * q_dim, "attn_output");

#pragma omp barrier
        // 矩阵点乘计算
        matrix_multiply_with_numa(&work,
                                  (f32 *)pstRunState->hidden_states,
                                  g_pstWeight.o_proj_weight.Data.tensor3,
                                  pstRunState->quant_attn_output,
                                  g_pstWeightTypes.attn_output_weight,
                                  hidden_size,
                                  q_dim,
                                  hidden_size,
                                  n_tokens,
                                  L,
                                  false,
                                  "attn_output_multiply");

#ifdef DEBUG_TIME
    if (work.tid == 0) {
        time[7] += get_time_ns() - tt1;
        tt1 = get_time_ns();
    }
#endif

#pragma omp barrier
        residual_add_rmsnorm_with_numa(&work,
                                       (f32 *)pstRunState->hidden_states,
                                       (f32 *)pstRunState->residual,
                                       (f32 *)pstRunState->hidden_states,
                                       (f32 *)pstLlama->post_attention_layernorm_weight.Data.tensor2[L],
                                       n_tokens,
                                       hidden_size,
                                       eps,
                                       nullptr,
                                       "ffn_residual_norm");

#ifdef DEBUG_TIME
    if (work.tid == 0) {
        time[8] += get_time_ns() - tt1;
        tt1 = get_time_ns();
    }
#endif

#pragma omp barrier
        quantize_and_distribute_to_numa(&work, pstRunState, pstRunState->quant_hidden_states,
                                                (f32 *)pstRunState->hidden_states, g_pstWeightTypes.ffn_up_weight,
                                                n_tokens * hidden_size, "ffn_input");
#pragma omp barrier
        gate_up_proj_matrix_multiply_with_numa(&work,
                                               (f32 *)pstRunState->gate_up,
                                               g_pstWeight.gate_proj_weight.Data.tensor3,
                                               g_pstWeight.up_proj_weight.Data.tensor3,
                                               pstRunState->quant_hidden_states,
                                               g_pstWeightTypes.ffn_up_weight,
                                               n_tokens,
                                               hidden_size,
                                               intermediate_size,
                                               L);

#ifdef DEBUG_TIME
    if (work.tid == 0) {
        time[9] += get_time_ns() - tt1;
        tt1 = get_time_ns();
    }
#endif

        /* silu激活函数 */
        silu_activation_with_numa(&work,
                                  pstRunState->fused_gate_up,
                                  pstRunState->gate_up,
                                  n_tokens,
                                  intermediate_size,
                                  "silu_activation");

#ifdef DEBUG_TIME
    if (work.tid == 0) {
        time[10] += get_time_ns() - tt1;
        tt1 = get_time_ns();
    }
#endif

#pragma omp barrier
        quantize_and_distribute_to_numa(&work, pstRunState, pstRunState->quant_fused_gate_up,
                                                (f32 *)pstRunState->fused_gate_up, g_pstWeightTypes.ffn_down_weight,
                                                n_tokens * intermediate_size, "ffn_down");
#pragma omp barrier
        matrix_multiply_with_numa(&work,
                                  (f32 *)pstRunState->hidden_states,
                                  g_pstWeight.down_proj_weight.Data.tensor3,
                                  pstRunState->quant_fused_gate_up,
                                  g_pstWeightTypes.ffn_down_weight,
                                  hidden_size,
                                  intermediate_size,
                                  hidden_size,
                                  n_tokens,
                                  L,
                                  false,
                                  "ffn_down_multiply");

#ifdef DEBUG_TIME
    if (work.tid == 0) {
        time[11] += get_time_ns() - tt1;
        tt1 = get_time_ns();
    }
#endif

} //end omp
    }

        std::vector<int> last_token_indices;
        if (is_prompt == true) {
            for (int i = 0, sum_seq_lens = 0; i < seq_lens.size(0); i++) {
                sum_seq_lens += ((int *)seq_lens.data_ptr())[i];
                last_token_indices.push_back(sum_seq_lens - 1);
            }
        } else {
            for (int i = 0; i < n_tokens; i++) {
                last_token_indices.push_back(i);
            }
        }

#pragma omp parallel
{
        WorkDivider work;
        init_work_divider(&work, g_numas);
        SingleNumaWorkRange srange;
        MultiNumaWorkRange mrange;

        residual_add_rmsnorm_with_numa(&work,
                                       (f32 *)pstRunState->hidden_states,
                                       (f32 *)pstRunState->residual,
                                       (f32 *)pstRunState->hidden_states,
                                       (f32 *)pstLlama->norm_weight.Data.tensor1,
                                       n_tokens,
                                       hidden_size,
                                       eps,
                                       &last_token_indices,
                                       "output_residual_norm");

#ifdef DEBUG_TIME
    if (work.tid == 0) {
        time[12] += get_time_ns() - tt1;
        tt1 = get_time_ns();
    }
#endif

#pragma omp barrier
        quantize_and_distribute_to_numa(&work, pstRunState, pstRunState->quant_hidden_states,
                                                (f32 *)pstRunState->hidden_states, g_pstWeightTypes.output_weight,
                                                last_token_indices.size() * hidden_size, "output");
#pragma omp barrier
        // 矩阵点乘计算
        matrix_multiply_with_numa(&work,
                                  (f32 *)pstRunState->logits,
                                  g_pstWeight.lm_head_weight.Data.tensor2,
                                  pstRunState->quant_hidden_states,
                                  g_pstWeightTypes.output_weight,
                                  vocab_size,
                                  hidden_size,
                                  vocab_size,
                                  last_token_indices.size(),
                                  0,
                                  true,
                                  "output_multiply");

#ifdef DEBUG_TIME
    if (work.tid == 0) {
        time[13] += get_time_ns() - tt1;
        tt1 = get_time_ns();
    }
#endif

}
    g_BlockDataInfo[GGML_TYPE_F16].quantize(pstRunState->logits, output, last_token_indices.size() * vocab_size);

#ifdef DEBUG_TIME
    time[14] += get_time_ns() - tt1;
    tt1 = get_time_ns();
#endif
    numa_free(VT, n_tokens * q_dim * sizeof(f16));

#ifdef DEBUG_TIME
    uint64_t t1 = get_time_ns();
    if (is_prompt == true) {
        fprintf(stderr, " bs=%d prefill=%.3f ms, %.3f token/s\n", n_tokens, (t1 - t0) / 1000000.0, 1.0 * n_tokens / ((t1 - t0) / 1000000000.0));
    } else {
        fprintf(stderr, " bs=%d decode=%.3f ms, %.3f token/s\n", n_tokens, (t1 - t0) / 1000000.0, 1.0 * n_tokens / ((t1 - t0) / 1000000000.0));
    }

    fprintf(stderr, "[0] first rms_norm ——> %8.3lf ms\n", time[0] / 1000000.0);
    fprintf(stderr, "[1] qkv quantize and memcpy ——> %8.3lf ms\n", time[1] / 1000000.0);
    fprintf(stderr, "[2] qkv matmul ——> %8.3lf ms\n", time[2] / 1000000.0);
    fprintf(stderr, "[3] qkv add and quantize f16 ——> %8.3lf ms\n", time[3] / 1000000.0);
    fprintf(stderr, "[4] rope operator ——> %8.3lf ms\n", time[4] / 1000000.0);
    fprintf(stderr, "[5] page attention operator ——> %8.3lf ms\n", time[5] / 1000000.0);
    fprintf(stderr, "[6] dequantize f32 ——> %8.3lf ms\n", time[6] / 1000000.0);
    fprintf(stderr, "[7] (wo)quantize-memcpy-matmul ——> %8.3lf ms\n", time[7] / 1000000.0);
    fprintf(stderr, "[8] ffn add and rmsnorm ——> %8.3lf ms\n", time[8] / 1000000.0);
    fprintf(stderr, "[9] (w1w3)quantize-memcpy-matmul ——> %8.3lf ms\n", time[9] / 1000000.0);
    fprintf(stderr, "[10] silu activation function ——> %8.3lf ms\n", time[10] / 1000000.0);
    fprintf(stderr, "[11] (w2)quantize-memcpy-matmul ——> %8.3lf ms\n", time[11] / 1000000.0);
    fprintf(stderr, "[12] output_norm add and rmsnorm ——> %8.3lf ms\n", time[12] / 1000000.0);
    fprintf(stderr, "[13] (output)quantize-memcpy-matmul ——> %8.3lf ms\n", time[13] / 1000000.0);
    fprintf(stderr, "[14] output quantize f16 ——> %8.3lf ms\n\n", time[14] / 1000000.0);
    fprintf(stderr, "[sum] sum matmul ——> %8.3lf ms\n\n", (time[2] + time[7] + time[9] + time[11]  + time[13])/ 1000000.0);
#endif
}

void destroy_run_state(MODEL_RUN_STATE* state, const MODEL_HYPE_PARA* para, bool is_prompt, int64_t N_tokens, int64_t seq_num) {
    if (!state) return;

    // ========== 模型参数解析 ==========
    int64_t head_dim       = para->head_dim;
    int64_t hidden_size            = para->hidden_size;
    int64_t num_attention_heads         = para->num_attention_heads;
    int64_t num_key_value_heads     = para->num_key_value_heads;
    int64_t q_dim          = head_dim * num_attention_heads;
    int64_t k_dim          = head_dim * num_key_value_heads;
    int64_t v_dim          = head_dim * num_key_value_heads;
    int64_t intermediate_size     = para->intermediate_size;
    int64_t vocab_size        = para->vocab_size;
    int64_t qkv_dim        = q_dim + k_dim + v_dim;

    // ========== 释放Token处理缓冲区 ==========
    if (state->hidden_states) numa_free(state->hidden_states, hidden_size * N_tokens * sizeof(f32));
    if (state->residual) numa_free(state->residual, hidden_size * N_tokens * sizeof(f32));
    if (state->qkv) numa_free(state->qkv, qkv_dim * N_tokens * sizeof(f32));
    if (state->qkv_f16) numa_free(state->qkv_f16, qkv_dim * N_tokens * sizeof(f16));
    if (state->attn_output) numa_free(state->attn_output, q_dim * N_tokens * sizeof(f32));
    if (state->attn_output_f16) numa_free(state->attn_output_f16, q_dim * N_tokens * sizeof(f16));
    if (state->gate_up) numa_free(state->gate_up, 2 * intermediate_size * N_tokens * sizeof(f32));
    if (state->fused_gate_up) numa_free(state->fused_gate_up, intermediate_size * N_tokens * sizeof(f32));
    if (state->logits) numa_free(state->logits, vocab_size * N_tokens * sizeof(f32));

    // ========== 释放NUMA多节点缓冲区 ==========
    if (state->quant_hidden_states) {
        for (int i = 0; i < g_numas; i++) {
            if (state->quant_hidden_states[i]) {
                numa_free(state->quant_hidden_states[i], hidden_size * N_tokens * sizeof(f32));
            }
        }
        numa_free(state->quant_hidden_states, g_numas * sizeof(void*));
    }
    
    if (state->quant_attn_output) {
        for (int i = 0; i < g_numas; i++) {
            if (state->quant_attn_output[i]) {
                numa_free(state->quant_attn_output[i], q_dim * N_tokens * sizeof(f32));
            }
        }
        numa_free(state->quant_attn_output, g_numas * sizeof(void*));
    }

    if (state->quant_fused_gate_up) {
        for (int i = 0; i < g_numas; i++) {
            if (state->quant_fused_gate_up[i]) {
                numa_free(state->quant_fused_gate_up[i], intermediate_size * N_tokens * sizeof(f32));
            }
        }
        numa_free(state->quant_fused_gate_up, g_numas * sizeof(void*));
    }

    // ========== 释放运行状态对象 ==========
    delete state;
}

MODEL_RUN_STATE* create_run_state(const MODEL_HYPE_PARA* para, bool is_prompt, int64_t N_tokens, int64_t seq_num) {
    // ========== 模型参数解析 ==========
    int64_t head_dim       = para->head_dim;
    int64_t hidden_size            = para->hidden_size;
    int64_t num_attention_heads         = para->num_attention_heads;
    int64_t num_key_value_heads     = para->num_key_value_heads;
    int64_t q_dim          = head_dim * num_attention_heads;
    int64_t k_dim          = head_dim * num_key_value_heads;
    int64_t v_dim          = head_dim * num_key_value_heads;
    int64_t intermediate_size     = para->intermediate_size;
    int64_t num_hidden_layers       = para->num_hidden_layers;
    int64_t max_position_embeddings = para->max_position_embeddings;
    int64_t vocab_size        = para->vocab_size;
    int64_t qkv_dim        = q_dim + k_dim + v_dim;

    // ========== 创建运行状态对象 ==========
    MODEL_RUN_STATE* state = new MODEL_RUN_STATE();

    // ========== Token处理缓冲区 ==========
    state->hidden_states = (f32*)numa_alloc_onnode(hidden_size * N_tokens * sizeof(f32), 0);
    state->residual = (f32*)numa_alloc_onnode(hidden_size * N_tokens * sizeof(f32), 0);
    state->qkv = (f32*)numa_alloc_onnode(qkv_dim * N_tokens * sizeof(f32), 0);
    state->qkv_f16 = (f16*)numa_alloc_onnode(qkv_dim * N_tokens * sizeof(f16), 0);
    state->attn_output = (f32*)numa_alloc_onnode(q_dim * N_tokens * sizeof(f32), 0);
    state->attn_output_f16 = (f16*)numa_alloc_onnode(q_dim * N_tokens * sizeof(f16), 0);
    state->gate_up = (f32*)numa_alloc_onnode(2 * intermediate_size * N_tokens * sizeof(f32), 0);
    state->fused_gate_up = (f32*)numa_alloc_onnode(intermediate_size * N_tokens * sizeof(f32), 0);
    state->logits = (f32*)numa_alloc_onnode(vocab_size * N_tokens * sizeof(f32), 0);

    // ========== NUMA多节点缓冲区数组分配 ==========
    state->quant_hidden_states = (void**)numa_alloc_onnode(g_numas * sizeof(void*), 0);
    state->quant_attn_output = (void**)numa_alloc_onnode(g_numas * sizeof(void*), 0);
    state->quant_fused_gate_up = (void**)numa_alloc_onnode(g_numas * sizeof(void*), 0);

    // ========== 内存分配检查 ==========
    if (!state->hidden_states || !state->residual || !state->qkv || !state->qkv_f16 || !state->attn_output ||
        !state->attn_output_f16 || !state->gate_up || !state->fused_gate_up || !state->logits ||
        !state->quant_hidden_states || !state->quant_attn_output || !state->quant_fused_gate_up) {
        fprintf(stderr, "Error: Primary memory allocation failed! (File: %s, Line: %d)\n", __FILE__, __LINE__);
        destroy_run_state(state, para, is_prompt, N_tokens, seq_num);
        return nullptr;
    }

    // ========== 为每个NUMA节点分配缓冲区 ==========
    bool allocation_success = true;
    for (int i = 0; i < g_numas; i++) {
        state->quant_hidden_states[i] = (void*)numa_alloc_onnode(hidden_size * N_tokens * sizeof(f32), i);
        state->quant_attn_output[i] = (void*)numa_alloc_onnode(q_dim * N_tokens * sizeof(f32), i);
        state->quant_fused_gate_up[i] = (void*)numa_alloc_onnode(intermediate_size * N_tokens * sizeof(f32), i);
        
        if (!state->quant_hidden_states[i] || !state->quant_attn_output[i] || !state->quant_fused_gate_up[i]) {
            fprintf(stderr, "Error: NUMA node %d memory allocation failed! (File: %s, Line: %d)\n", i, __FILE__, __LINE__);
            allocation_success = false;
            break;
        }
    }

    if (!allocation_success) {
        destroy_run_state(state, para, is_prompt, N_tokens, seq_num);
        return nullptr;
    }

    return state;
}

void get_next_token_for_torch(
    torch::Tensor model_output,   // WEIGHT.embed_tokens_weight
    torch::Tensor hidden_stats,

    bool is_prompt,
    torch::Tensor block_tables,
    torch::Tensor seq_lens,
    torch::Tensor& slot_mapping,
    torch::Tensor positions,
    std::vector<torch::Tensor> kv_caches,
    int64_t block_size,
    int64_t N_tokens,
    bool is_qkv_bias,
    bool is_qk_norm)
{
    get_affinity_cpus(cpu_ids);
    void* hd = static_cast<void*>(hidden_stats.data_ptr());
    int64_t *pos = static_cast<int64_t *>(positions.data_ptr());
    void* output = static_cast<void*>(model_output.data_ptr());

    int64_t seq_num = seq_lens.size(0);
    MODEL_RUN_STATE* state = create_run_state(&g_pstModelHypePara, is_prompt, N_tokens, seq_num);

    get_next_token(output,
        &g_pstModelHypePara, &g_pstWeight, state, 
        is_prompt, 
        block_tables,
        seq_lens,
        slot_mapping,
        hd, 
        pos, 
        kv_caches,
        block_size,
        N_tokens,
        is_qkv_bias,
        is_qk_norm
    );

    destroy_run_state(state, &g_pstModelHypePara, is_prompt, N_tokens, seq_num);
}

void load_weight_and_malloc_active_tensor_qwen3(
    torch::Tensor embed_tokens_weight,                  // [vocab_size, hidden_size]
    torch::Tensor input_layernorm_weight,               // [num_hidden_layers][hidden_size]
    torch::Tensor post_attention_layernorm_weight,      // [num_hidden_layers][hidden_size]
    torch::Tensor qkv_proj_weight,                      // [num_hidden_layers][q_dim + k_dim + v_dim, hidden_size]
    torch::Tensor o_proj_weight,                        // [num_hidden_layers][hidden_size, q_dim]
    torch::Tensor qkv_proj_bias,                        // [num_hidden_layers][q_dim + k_dim + v_dim]
    torch::Tensor gate_up_proj_weight,                  // [num_hidden_layers][2 * intermediate_size, hidden_size]
    torch::Tensor down_proj_weight,                     // [num_hidden_layers][hidden_size, intermediate_size]
    torch::Tensor norm_weight,                          // [hidden_size]
    torch::Tensor lm_head_weight,                       // [vocab_size, hidden_size]
    torch::Tensor q_norm_weight,                        // [num_hidden_layers][head_dim]
    torch::Tensor k_norm_weight                         // [num_hidden_layers][head_dim]
){

    assert(qkv_proj_weight.dtype() == torch::kFloat16);
    int64_t head_dim = g_pstModelHypePara.head_dim;
    int64_t hidden_size = g_pstModelHypePara.hidden_size;
    int64_t intermediate_size = g_pstModelHypePara.intermediate_size;
    int64_t num_attention_heads = g_pstModelHypePara.num_attention_heads;
    int64_t num_hidden_layers = g_pstModelHypePara.num_hidden_layers;
    int64_t vocab_size = g_pstModelHypePara.vocab_size;
    int64_t num_key_value_heads = g_pstModelHypePara.num_key_value_heads;
    int64_t max_position_embeddings = g_pstModelHypePara.max_position_embeddings;
    int64_t N_gqa = num_attention_heads / num_key_value_heads;

    int64_t q_dim = head_dim * num_attention_heads;
    int64_t k_dim = head_dim * num_key_value_heads;
    int64_t v_dim = head_dim * num_key_value_heads;

    size_t tokens_embedding_weight_size = (size_t)hidden_size * vocab_size / g_BlockDataInfo[g_pstWeightTypes.token_embd_weight].uiblkSize * g_BlockDataInfo[g_pstWeightTypes.token_embd_weight].uiTypeSize;

    size_t attention_q_size_per_layer = (size_t)hidden_size* q_dim / g_BlockDataInfo[g_pstWeightTypes.attn_q_weight].uiblkSize * g_BlockDataInfo[g_pstWeightTypes.attn_q_weight].uiTypeSize;
    size_t attention_k_size_per_layer = (size_t)hidden_size * k_dim / g_BlockDataInfo[g_pstWeightTypes.attn_k_weight].uiblkSize * g_BlockDataInfo[g_pstWeightTypes.attn_k_weight].uiTypeSize;
    size_t attention_v_size_per_layer = (size_t)hidden_size * v_dim / g_BlockDataInfo[g_pstWeightTypes.attn_v_weight].uiblkSize * g_BlockDataInfo[g_pstWeightTypes.attn_v_weight].uiTypeSize;
    size_t attention_size_per_layer = attention_q_size_per_layer + attention_k_size_per_layer + attention_v_size_per_layer;

    size_t bias_q_size_per_layer = (size_t)q_dim / g_BlockDataInfo[g_pstWeightTypes.attn_q_bias].uiblkSize * g_BlockDataInfo[g_pstWeightTypes.attn_q_bias].uiTypeSize;
    size_t bias_k_size_per_layer = (size_t)k_dim / g_BlockDataInfo[g_pstWeightTypes.attn_k_bias].uiblkSize * g_BlockDataInfo[g_pstWeightTypes.attn_k_bias].uiTypeSize;
    size_t bias_v_size_per_layer = (size_t)v_dim / g_BlockDataInfo[g_pstWeightTypes.attn_v_bias].uiblkSize * g_BlockDataInfo[g_pstWeightTypes.attn_v_bias].uiTypeSize;
    size_t bias_qkv_size_per_layer = bias_q_size_per_layer + bias_k_size_per_layer + bias_v_size_per_layer;

    size_t attention_norm_size_per_layer = (size_t)hidden_size / g_BlockDataInfo[g_pstWeightTypes.attn_norm_weight].uiblkSize * g_BlockDataInfo[g_pstWeightTypes.attn_norm_weight].uiTypeSize;

    size_t ffn_down_size_per_layer = (size_t)intermediate_size * hidden_size / g_BlockDataInfo[g_pstWeightTypes.ffn_down_weight].uiblkSize * g_BlockDataInfo[g_pstWeightTypes.ffn_down_weight].uiTypeSize;
    size_t ffn_gate_size_per_layer = (size_t)hidden_size * intermediate_size / g_BlockDataInfo[g_pstWeightTypes.ffn_gate_weight].uiblkSize * g_BlockDataInfo[g_pstWeightTypes.ffn_gate_weight].uiTypeSize;
    size_t ffn_norm_size_per_layer = (size_t)hidden_size / g_BlockDataInfo[g_pstWeightTypes.ffn_norm_weight].uiblkSize * g_BlockDataInfo[g_pstWeightTypes.ffn_norm_weight].uiTypeSize;
    size_t ffn_up_size_per_layer = (size_t)hidden_size * intermediate_size / g_BlockDataInfo[g_pstWeightTypes.ffn_up_weight].uiblkSize * g_BlockDataInfo[g_pstWeightTypes.ffn_up_weight].uiTypeSize;
    size_t w1w3_size_per_layer = ffn_gate_size_per_layer + ffn_up_size_per_layer;

    size_t attention_output_size_per_layer = (size_t)hidden_size * q_dim / g_BlockDataInfo[g_pstWeightTypes.attn_output_weight].uiblkSize * g_BlockDataInfo[g_pstWeightTypes.attn_output_weight].uiTypeSize;
    size_t output_size = (size_t)hidden_size * vocab_size / g_BlockDataInfo[g_pstWeightTypes.output_weight].uiblkSize * g_BlockDataInfo[g_pstWeightTypes.output_weight].uiTypeSize;
    size_t output_norm_size = (size_t)hidden_size / g_BlockDataInfo[g_pstWeightTypes.output_norm_weight].uiblkSize * g_BlockDataInfo[g_pstWeightTypes.output_norm_weight].uiTypeSize;

    // qwen3特有的q_norm和k_norm权重大小计算
    size_t q_norm_size_per_layer = (size_t)head_dim / g_BlockDataInfo[g_pstWeightTypes.q_norm_weight].uiblkSize * g_BlockDataInfo[g_pstWeightTypes.q_norm_weight].uiTypeSize;
    size_t k_norm_size_per_layer = (size_t)head_dim / g_BlockDataInfo[g_pstWeightTypes.k_norm_weight].uiblkSize * g_BlockDataInfo[g_pstWeightTypes.k_norm_weight].uiTypeSize;

    g_pstWeight.input_layernorm_weight.Data.tensor2 = static_cast<void**>(numa_alloc_onnode(num_hidden_layers * sizeof(void *), 0));
    g_pstWeight.post_attention_layernorm_weight.Data.tensor2 = static_cast<void**>(numa_alloc_onnode(num_hidden_layers * sizeof(void *), 0));
    g_pstWeight.qkv_proj_bias.Data.tensor2 = static_cast<void**>(numa_alloc_onnode(num_hidden_layers * sizeof(void *), 0));
    g_pstWeight.q_norm_weight.Data.tensor2 = static_cast<void**>(numa_alloc_onnode(num_hidden_layers * sizeof(void *), 0));
    g_pstWeight.k_norm_weight.Data.tensor2 = static_cast<void**>(numa_alloc_onnode(num_hidden_layers * sizeof(void *), 0));

    for (int i = 0; i < num_hidden_layers; i++) {
        g_pstWeight.input_layernorm_weight.Data.tensor2[i] = numa_alloc_onnode(attention_norm_size_per_layer, 0);
        g_pstWeight.post_attention_layernorm_weight.Data.tensor2[i] = numa_alloc_onnode(ffn_norm_size_per_layer, 0);
        g_pstWeight.qkv_proj_bias.Data.tensor2[i] = numa_alloc_onnode(bias_qkv_size_per_layer, 0);
        
        g_pstWeight.q_norm_weight.Data.tensor2[i] = numa_alloc_onnode(q_norm_size_per_layer, 0);
        g_pstWeight.k_norm_weight.Data.tensor2[i] = numa_alloc_onnode(k_norm_size_per_layer, 0);
    }
    g_pstWeight.norm_weight.Data.tensor1 = numa_alloc_onnode(output_norm_size, 0);

    g_pstWeight.qkv_proj_weight.Data.tensor3 = static_cast<void***>(numa_alloc_onnode(g_numas * sizeof(void **), 0));
    g_pstWeight.o_proj_weight.Data.tensor3 = static_cast<void***>(numa_alloc_onnode(g_numas * sizeof(void **), 0));
    g_pstWeight.gate_up_proj_weight.Data.tensor3 = static_cast<void***>(numa_alloc_onnode(g_numas * sizeof(void **), 0));
    g_pstWeight.down_proj_weight.Data.tensor3 = static_cast<void***>(numa_alloc_onnode(g_numas * sizeof(void **), 0));
    g_pstWeight.lm_head_weight.Data.tensor2 = static_cast<void**>(numa_alloc_onnode(g_numas * sizeof(void *), 0));
    g_pstWeight.embed_tokens_weight.Data.tensor1 = numa_alloc_onnode(tokens_embedding_weight_size, 0);

    for (int i = 0; i < g_numas; i++) {
        g_pstWeight.qkv_proj_weight.Data.tensor3[i] = static_cast<void**>(numa_alloc_onnode(num_hidden_layers * sizeof(void *), i));
        g_pstWeight.o_proj_weight.Data.tensor3[i] = static_cast<void**>(numa_alloc_onnode(num_hidden_layers * sizeof(void *), i));
        g_pstWeight.gate_up_proj_weight.Data.tensor3[i] = static_cast<void**>(numa_alloc_onnode(num_hidden_layers * sizeof(void *), i));
        g_pstWeight.down_proj_weight.Data.tensor3[i] = static_cast<void**>(numa_alloc_onnode(num_hidden_layers * sizeof(void *), i));
        g_pstWeight.lm_head_weight.Data.tensor2[i] = (void *)numa_alloc_onnode(output_size / g_numas, i);

        for (int j = 0; j < num_hidden_layers; j++) {
            g_pstWeight.qkv_proj_weight.Data.tensor3[i][j] = numa_alloc_onnode(attention_size_per_layer / g_numas, i);
            g_pstWeight.o_proj_weight.Data.tensor3[i][j] = numa_alloc_onnode(attention_output_size_per_layer / g_numas, i);
            g_pstWeight.gate_up_proj_weight.Data.tensor3[i][j] = numa_alloc_onnode(w1w3_size_per_layer / g_numas, i);
            g_pstWeight.down_proj_weight.Data.tensor3[i][j] = numa_alloc_onnode(ffn_down_size_per_layer / g_numas, i);
        }
    }

    std::cout << "load_weight qwen3 start ..." << std::endl;

    /* 量化权重 */
    for (int layerNum = 0; layerNum < num_hidden_layers; layerNum++) {
        quantization_weight_strategy(g_pstWeight.input_layernorm_weight.Data.tensor2[layerNum], input_layernorm_weight.index(torch::indexing::TensorIndex(layerNum)).data_ptr(),
                                     g_pstWeightTypes.attn_norm_weight, hidden_size);
        quantization_weight_strategy(g_pstWeight.post_attention_layernorm_weight.Data.tensor2[layerNum], post_attention_layernorm_weight.index(torch::indexing::TensorIndex(layerNum)).data_ptr(),
                                     g_pstWeightTypes.ffn_norm_weight, hidden_size);
        int qkv_dim = q_dim + k_dim + v_dim;
        for (int j = 0; j < g_numas; ++j) {
            f16 *qkv_pointer = (f16 *)qkv_proj_weight.index(torch::indexing::TensorIndex(layerNum)).data_ptr() + qkv_dim / g_numas * hidden_size * j;
            quantization_weight_strategy(g_pstWeight.qkv_proj_weight.Data.tensor3[j][layerNum], (char *)qkv_pointer, g_pstWeightTypes.attn_k_weight,
                                         qkv_dim / g_numas * hidden_size);

            f16 *wo_pointer = (f16 *)o_proj_weight.index(torch::indexing::TensorIndex(layerNum)).data_ptr() + hidden_size / g_numas * q_dim * j;
            quantization_weight_strategy(g_pstWeight.o_proj_weight.Data.tensor3[j][layerNum], (char *)wo_pointer,
                                         g_pstWeightTypes.attn_output_weight, hidden_size * q_dim / g_numas);

            f16 *w1w3_pointer = (f16 *)gate_up_proj_weight.index(torch::indexing::TensorIndex(layerNum)).data_ptr() + 2 * intermediate_size / g_numas * hidden_size * j;
            quantization_weight_strategy(g_pstWeight.gate_up_proj_weight.Data.tensor3[j][layerNum], (char *)w1w3_pointer,
                                         g_pstWeightTypes.ffn_up_weight, 2 * intermediate_size / g_numas * hidden_size);

            f16 *ffn_down_pointer = (f16 *)down_proj_weight.index(torch::indexing::TensorIndex(layerNum)).data_ptr() + intermediate_size / g_numas * hidden_size * j;
            quantization_weight_strategy(g_pstWeight.down_proj_weight.Data.tensor3[j][layerNum], (char *)ffn_down_pointer,
                                         g_pstWeightTypes.ffn_down_weight, hidden_size * intermediate_size / g_numas);
        }

        quantization_weight_strategy(g_pstWeight.qkv_proj_bias.Data.tensor2[layerNum], qkv_proj_bias.index(torch::indexing::TensorIndex(layerNum)).data_ptr(),
                                     g_pstWeightTypes.attn_q_bias, qkv_dim);

        quantization_weight_strategy(g_pstWeight.q_norm_weight.Data.tensor2[layerNum], q_norm_weight.index(torch::indexing::TensorIndex(layerNum)).data_ptr(),
                                     g_pstWeightTypes.q_norm_weight, head_dim);
        quantization_weight_strategy(g_pstWeight.k_norm_weight.Data.tensor2[layerNum], k_norm_weight.index(torch::indexing::TensorIndex(layerNum)).data_ptr(),
                                     g_pstWeightTypes.k_norm_weight, head_dim);
    }

    for (int i = 0; i < g_numas; i++) {
        f16 *output_pointer = (f16 *)lm_head_weight.data_ptr() + vocab_size / g_numas * hidden_size * i;
        quantization_weight_strategy(g_pstWeight.lm_head_weight.Data.tensor2[i], output_pointer,
                                     g_pstWeightTypes.output_weight, hidden_size * vocab_size /  g_numas);
    }

    quantization_weight_strategy(g_pstWeight.embed_tokens_weight.Data.tensor1, embed_tokens_weight.data_ptr(), g_pstWeightTypes.token_embd_weight, hidden_size * vocab_size);
    quantization_weight_strategy(g_pstWeight.norm_weight.Data.tensor1, norm_weight.data_ptr(), g_pstWeightTypes.output_norm_weight, hidden_size);

    std::cout << "load_weight qwen3 end." << std::endl;
}

// 统一的权重加载接口
void load_weight_unified(
    const std::string& model_type,
    torch::Tensor embed_tokens_weight,                  // [vocab_size, hidden_size]
    torch::Tensor input_layernorm_weight,               // [num_hidden_layers][hidden_size]
    torch::Tensor post_attention_layernorm_weight,      // [num_hidden_layers][hidden_size]
    torch::Tensor qkv_proj_weight,                      // [num_hidden_layers][q_dim + k_dim + v_dim, hidden_size]
    torch::Tensor o_proj_weight,                        // [num_hidden_layers][hidden_size, q_dim]
    torch::Tensor qkv_proj_bias,                        // [num_hidden_layers][q_dim + k_dim + v_dim]
    torch::Tensor gate_up_proj_weight,                  // [num_hidden_layers][2 * intermediate_size, hidden_size]
    torch::Tensor down_proj_weight,                     // [num_hidden_layers][hidden_size, intermediate_size]
    torch::Tensor norm_weight,                          // [hidden_size]
    torch::Tensor lm_head_weight,                       // [vocab_size, hidden_size]
    torch::Tensor q_norm_weight,                        // [num_hidden_layers][head_dim]
    torch::Tensor k_norm_weight                         // [num_hidden_layers][head_dim]
) {
    if (model_type == "qwen2") {
        // 调用原来的qwen2函数，qwen2不需要q_norm和k_norm
        load_weight_and_malloc_active_tensor(
            embed_tokens_weight, input_layernorm_weight, post_attention_layernorm_weight, qkv_proj_weight, o_proj_weight,
            qkv_proj_bias, gate_up_proj_weight, down_proj_weight, norm_weight, lm_head_weight
        );
    } else if (model_type == "qwen3") {
        // 检查qwen3必需的参数
        if (!q_norm_weight.defined() || !k_norm_weight.defined()) {
            throw std::runtime_error("qwen3 model requires q_norm and k_norm parameters");
        }
        // 调用qwen3专用函数
        load_weight_and_malloc_active_tensor_qwen3(
            embed_tokens_weight, input_layernorm_weight, post_attention_layernorm_weight, qkv_proj_weight, o_proj_weight,
            qkv_proj_bias, gate_up_proj_weight, down_proj_weight, norm_weight, lm_head_weight, q_norm_weight, k_norm_weight
        );
    } else {
        throw std::runtime_error("Unsupported model type: " + model_type);
    }
}

void load_model_config(
    const std::string& model_type,
    int64_t head_dim, int64_t hidden_size, int64_t intermediate_size, 
    int64_t num_attention_heads, int64_t num_hidden_layers, int64_t vocab_size, int64_t num_key_value_heads, int64_t context_length,
    double rms_norm_eps, double rope_freq_base, double attn_scale, int64_t is_neox_style,
    torch::Tensor const &cos_sin_cache, 
    int64_t quantization_bit_code
) {
    // 初始化 i8mm 指令集检测
    init_i8mm_flag();

    // 设置全局量化编码
    g_quantization_bit_code = quantization_bit_code;

    // 设置模型超参数
    g_pstModelHypePara.head_dim = head_dim;
    g_pstModelHypePara.hidden_size = hidden_size;            /* embedding 维度 */
    g_pstModelHypePara.num_attention_heads = num_attention_heads;        /* 注意力头个数 */
    g_pstModelHypePara.num_key_value_heads = num_key_value_heads;     /* kv的对数 */
    g_pstModelHypePara.intermediate_size = intermediate_size;     /* ffn隐藏层维度 */
    g_pstModelHypePara.num_hidden_layers = num_hidden_layers;       /* 模型层数 */
    g_pstModelHypePara.max_position_embeddings = context_length; /* 上下文长度 */
    g_pstModelHypePara.rms_norm_eps = rms_norm_eps; /* eps */
    g_pstModelHypePara.vocab_size = vocab_size;        /* 词汇数量 */
    g_pstModelHypePara.rope_theta = rope_freq_base; /* rope频率 */
    g_pstModelHypePara.cos_sin_cache = (f16 *)cos_sin_cache.data_ptr();
    g_pstModelHypePara.n_rotary = cos_sin_cache.size(1);
    g_pstModelHypePara.is_neox_style = is_neox_style;
    g_pstModelHypePara.attn_scale = attn_scale;

    g_pstWeightTypes.token_embd_weight = GGML_TYPE_F32;
    g_pstWeightTypes.attn_k_weight = quantization_bit_code;
    g_pstWeightTypes.attn_norm_weight = GGML_TYPE_F32;
    g_pstWeightTypes.attn_q_weight = quantization_bit_code;
    g_pstWeightTypes.attn_v_weight = quantization_bit_code;
    g_pstWeightTypes.ffn_down_weight = quantization_bit_code;
    g_pstWeightTypes.ffn_gate_weight = quantization_bit_code;
    g_pstWeightTypes.ffn_norm_weight = GGML_TYPE_F32;
    g_pstWeightTypes.ffn_up_weight = quantization_bit_code;
    g_pstWeightTypes.attn_output_weight = quantization_bit_code;
    g_pstWeightTypes.output_weight = quantization_bit_code;
    g_pstWeightTypes.output_norm_weight = GGML_TYPE_F32;
    
    if (model_type == "qwen2") {
        g_pstWeightTypes.attn_k_bias = GGML_TYPE_F32;
        g_pstWeightTypes.attn_q_bias = GGML_TYPE_F32;
        g_pstWeightTypes.attn_v_bias = GGML_TYPE_F32;
    }

    if (model_type == "qwen3") {
        g_pstWeightTypes.q_norm_weight = GGML_TYPE_F32;
        g_pstWeightTypes.k_norm_weight = GGML_TYPE_F32;
    }

    for(int i = 0; i < (1 << 16); ++i) {
        float f = f16_to_f32(*(f16*)(&i));
        expf_f16_table[i] = f32_to_f16(expf(f));
    }
}


