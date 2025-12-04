/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: Paged_attention optimize utils
 * Create: 2025-10-24
 */

#pragma once
#include <arm_sve.h>
#include <cmath>
#include <arm_fp16.h>
#include <torch/torch.h>
#include <type_traits>
#include <utility>

// ===== 调度宏 =====
#define KPEX_DISPATCH_HALF_CASE(HINT, ...)                         \
    if (tensorType == at::kHalf) {                                 \
        using HINT [[maybe_unused]] = float16_t;                   \
        __VA_ARGS__();                                             \
        dispatched = true;                                         \
    }

#define KPEX_DISPATCH_CONDITIONS(TYPE, ...)                        \
    [&] {                                                          \
        const auto &tensorType = TYPE;                             \
        bool dispatched = false;                                   \
        __VA_ARGS__                                                \
        if (!dispatched) {                                         \
            throw std::runtime_error("unsupported data type");     \
        }                                                          \
    } ()

#define KPEX_DISPATCH_HALF(TYPE, HINT, ...)                        \
    KPEX_DISPATCH_CONDITIONS(                                      \
        TYPE,                                                      \
        KPEX_DISPATCH_HALF_CASE(HINT, __VA_ARGS__))

// ===== LoopUnFold 模板 =====
template <typename T, T... indexes, typename F>
constexpr void LoopUnFoldHelper(std::integer_sequence<T, indexes...>, F &&f)
{
    (f(std::integral_constant<T, indexes>{}), ...);
}

template <typename T, T count, typename F,
          typename = std::enable_if_t<std::is_invocable_v<F, T>>>
constexpr void LoopUnFoldFunc(F &&f)
{
    LoopUnFoldHelper(std::make_integer_sequence<T, count>{}, std::forward<F>(f));
}

// ===== Tensor DataPtr 辅助函数 =====
template <typename T>
inline T* TensorDataPtr(const at::Tensor& t)
{
    using scalarType = std::conditional_t<std::is_same_v<T, float16_t>, c10::Half, T>;
    return reinterpret_cast<T*>(t.data_ptr<scalarType>());
}

template <typename T>
inline const T* TensorConstDataPtr(const at::Tensor& t)
{
    using scalarType = std::conditional_t<std::is_same_v<T, float16_t>, c10::Half, T>;
    return reinterpret_cast<const T*>(t.const_data_ptr<scalarType>());
}

// ===== FastExp 声明 =====
svfloat32_t FastExp(svbool_t pg, svfloat32_t values);
