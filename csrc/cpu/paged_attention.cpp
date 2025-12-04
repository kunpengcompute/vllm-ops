/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: Paged Attention
 * Create: 2025-10-22
 * Notes: NA
 */

#include <arm_sve.h>
#include <arm_neon.h>
#include <ATen/ops/empty.h>
#include "paged_attention_utils.h"
#include "paged_attention.h"

#define BLOCK_SIZE                                       16

#define KERNEL_LOOP_UNFOLD                               8


static void SoftMaxFusionKernel(float *in, float16_t *out, int64_t size)
{
    const int64_t vlf = svcntw();
    const int64_t vlh = svcnth();
    svfloat32_t vecMax = svdup_f32(-1e9);
    svfloat32_t reduce = svdup_f32(0.f);
    for (int64_t i = 0; i < size; i += vlf) {
        svbool_t pg = svwhilelt_b32(i, size);
        svfloat32_t values = svld1(pg, in + i);
        vecMax = svmax_m(pg, vecMax, values);
    }
    float32_t scaMax = svmaxv(svptrue_b32(), vecMax);
    for (int64_t i = 0; i < size; i += vlf) {
        svbool_t pg = svwhilelt_b32(i, size);
        svfloat32_t values = svld1(pg, in + i);
        values = FastExp(pg, svsub_x(pg, values, scaMax));
        reduce = svadd_m(pg, reduce, values);
        svst1_f32(pg, in + i, values);
    }
    float expSum = svaddv(svptrue_b32(), reduce);
    svfloat32_t sumInv = svdup_f32(1 / expSum);
    for (int64_t i = 0; i < size; i += vlh) {
        svbool_t pg0 = svwhilelt_b32(i, size);
        svbool_t pg1 = svwhilelt_b32(i + vlf, size);
        svbool_t pg2 = svwhilelt_b16(i, size);
        svfloat32_t values0 = svld1(pg0, in + i);
        svfloat32_t values1 = svld1(pg1, in + vlf + i);
        values0 = svmul_x(pg0, values0, sumInv);
        values1 = svmul_x(pg1, values1, sumInv);
        svfloat16_t res0 = svcvt_f16_x(pg0, values0);
        svfloat16_t res1 = svcvt_f16_x(pg1, values1);
        svfloat16_t res = svuzp1_f16(res0, res1);
        svst1_f16(pg2, (out + i), res);
    }
}

static float16_t ReduceSum(float16x8_t vecs)
{
    float16x4_t low = vget_low_f16(vecs);
    float16x4_t high = vget_high_f16(vecs);
    float16x4_t sum2 = vadd_f16(low, high);
    float16x4_t sum4 = vpadd_f16(sum2, sum2);
    float16x4_t sum8 = vpadd_f16(sum4, sum4);
    return vget_lane_f16(sum8, 0);
}

static void DotProductAttn(const float16_t *input1, const float16_t *input2, float *dst,
    int nGroups, int nElems, int len, float scale)
{
    int i = 0;
    for (i = 0; i < len / KERNEL_LOOP_UNFOLD; i++) {
        float16x8_t sum[KERNEL_LOOP_UNFOLD];
        LoopUnFoldFunc<int, KERNEL_LOOP_UNFOLD>([&](int idx) {
            sum[idx] = vmovq_n_f16(0.0);
        });
        for (int j = 0; j < nGroups; j++) {
            float16x8_t in1 = vld1q_f16(input1 + j * nElems);
            LoopUnFoldFunc<int, KERNEL_LOOP_UNFOLD>([&](int idx) {
                int groupOffset = i * KERNEL_LOOP_UNFOLD + idx;
                float16x8_t in2 = vld1q_f16(input2 + j * BLOCK_SIZE * nElems + groupOffset * nElems);
                sum[idx] = vfmaq_f16(sum[idx], in1, in2);
            });
        }
        LoopUnFoldFunc<int, KERNEL_LOOP_UNFOLD>([&](int idx) {
            int groupOffset = i * KERNEL_LOOP_UNFOLD + idx;
            dst[groupOffset] = static_cast<float>(ReduceSum(sum[idx])) * scale;
        });
    }
    for (i = i * KERNEL_LOOP_UNFOLD; i < len; i++) {
        float16x8_t sum = vmovq_n_f16(0.0);
        for (int j = 0; j < nGroups; j++) {
            float16x8_t in1 = vld1q_f16(input1 + j * nElems);
            float16x8_t in2 = vld1q_f16(input2 + j * BLOCK_SIZE * nElems + i * nElems);
            sum = vfmaq_f16(sum, in1, in2);
        }
        dst[i] = static_cast<float>(ReduceSum(sum)) * scale;
    }
}


static void DotProductOutput(const float16_t *input1, const float16_t *input2, float16_t *dst, int k, int len, bool acc)
{
    int i = 0;
    svbool_t pg = svwhilelt_b16(0, len);
    svfloat16_t in1 = svld1_f16(pg, input1);
    for (i = 0; i < k / KERNEL_LOOP_UNFOLD; i++) {
        LoopUnFoldFunc<int, KERNEL_LOOP_UNFOLD>([&](int idx) {
            int offset = i * KERNEL_LOOP_UNFOLD + idx;
            svfloat16_t in2 = svld1_f16(pg, input2 + offset * BLOCK_SIZE);
            svfloat16_t res = svmul_f16_x(pg, in1, in2);
            dst[offset] = (svaddv_f16(pg, res)) + (acc ? dst[offset] : 0.0);
        });
    }
    for (i = i * KERNEL_LOOP_UNFOLD; i < k; i++) {
        svfloat16_t in2 = svld1_f16(pg, input2 + i * BLOCK_SIZE);
        svfloat16_t res = svmul_f16_x(pg, in1, in2);
        dst[i] = (svaddv_f16(pg, res)) + (acc ? dst[i] : 0.0);
    }
}

template<class T>
void paged_attention_v1_impl(T* __restrict__ out, const T* __restrict__ query,
    const T* __restrict__ kCaches, const T* __restrict__ vCaches, const int nHeadsKV,
    const int* __restrict__ blockTables, const int* __restrict__ seqLens, const int maxBlocksNumPerSeq,
    const int qStride, const int kvBlockStride, const int kvHeadStride,
    const int numSeqs, const int nHeads, const int headSize)
{
    constexpr bool isReducedType = std::is_same_v<T, float16_t>;
    constexpr int groupSize = 16;
    int maxSeqLen = maxBlocksNumPerSeq * BLOCK_SIZE;
    float scale = static_cast<float>(1.0 / sqrt(headSize));
    constexpr int numElem = groupSize / sizeof(T);
    int nGroupsElem = headSize / numElem;
    int numQueriesPerKV = nHeads / nHeadsKV;
    int nThreads = omp_get_max_threads();

    at::Tensor buf = at::empty({nThreads, maxSeqLen}, at::dtype(at::kFloat));
    at::Tensor bufReduced = at::empty({nThreads, isReducedType ? maxSeqLen : 0}, at::dtype(at::kHalf));
    float *bufData = TensorDataPtr<float>(buf);
    T *bufRedecedData = isReducedType ? TensorDataPtr<T>(bufReduced) : nullptr;

// #pragma omp parallel for collapse(2) schedule(dynamic, 1)
#pragma omp parallel for collapse(2)
    for (int i = 0; i < numSeqs; i++) {
        for (int j = 0; j < nHeads; j++) {
            const int *blockTable = blockTables + maxBlocksNumPerSeq * i;
            int seqLen = seqLens[i];
            const T *qdata = query + i * qStride + j * headSize;
            int ompIdx = omp_get_thread_num();
            float *bufPtr = bufData + ompIdx * maxSeqLen;
            float *qkData = bufPtr;
            T *outData = out + i * nHeads * headSize + j * headSize;
            T *qkReducedData = isReducedType ? bufRedecedData + ompIdx * maxSeqLen : nullptr;
            for (int k = 0; k < seqLen; k += BLOCK_SIZE) {
                int64_t kSize = static_cast<int64_t>(std::min(BLOCK_SIZE, seqLen - k));
                int blockIdx = blockTable[k / BLOCK_SIZE];
                const T *kdata = kCaches + blockIdx * kvBlockStride + (j / numQueriesPerKV) * kvHeadStride;
                DotProductAttn(qdata, kdata, qkData + k, nGroupsElem, numElem, kSize, scale);
            }

            SoftMaxFusionKernel(qkData, qkReducedData, seqLen);

            for (int k = 0; k < seqLen; k += BLOCK_SIZE) {
                int64_t vSize = static_cast<int64_t>(std::min(BLOCK_SIZE, seqLen - k));
                int blockIdx = blockTable[k / BLOCK_SIZE];
                const T *vdata = vCaches + blockIdx * kvBlockStride + (j / numQueriesPerKV) * kvHeadStride;
                DotProductOutput(qkReducedData + k, vdata, outData, headSize, vSize, k > 0);
            }
        }
    }
}


#define PAGED_ATTENTION_KERNEL(TYPE, ...)    \
    paged_attention_v1_impl<TYPE>(__VA_ARGS__)

at::Tensor ScaledDotProductPagedAttention(
    const at::Tensor& query,
    const at::Tensor& kCaches,
    const at::Tensor& vCaches,
    const at::Tensor& blockTables,
    const at::Tensor& seqLens)
{
    at::Tensor output = at::empty_like(query, query.options());
    int nHeadsKV = static_cast<int>(kCaches.size(1));
    int maxBlocksNumPerSeq = static_cast<int>(blockTables.size(1));
    int qStride = static_cast<int>(query.stride(0));
    int kvBlockStride = static_cast<int>(kCaches.stride(0));
    int kvHeadStride = static_cast<int>(kCaches.stride(1));
    int numSeqs = static_cast<int>(query.size(0));
    int nHeads = static_cast<int>(query.size(1));
    int headSize = static_cast<int>(query.size(2));
    KPEX_DISPATCH_HALF(query.scalar_type(), T, [&] {
        PAGED_ATTENTION_KERNEL(T, TensorDataPtr<T>(output), TensorConstDataPtr<T>(query),
            TensorConstDataPtr<T>(kCaches), TensorConstDataPtr<T>(vCaches), nHeadsKV,
            TensorConstDataPtr<int>(blockTables), TensorConstDataPtr<int>(seqLens),
            maxBlocksNumPerSeq, qStride, kvBlockStride, kvHeadStride,
            numSeqs, nHeads, headSize);
    });

    return output;
}
