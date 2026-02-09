#pragma once
#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include "core/dtype.h"
#include "utils/utils.h"

namespace infini {
// Generate random data for testing
// Supports generating random vectors of floating-point numbers or integers
// Usage: auto data = generateRandomData<float>(1000, -10.0f, 10.0f);
// Parameters:
//   size: Data size
//   min: Minimum value (default 0)
//   max: Maximum value (default 100)
template <typename T>
std::vector<T> generateRandomData(size_t size, T min = static_cast<T>(0),
                                  T max = static_cast<T>(100)) {
    std::vector<T> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());

    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
        // Floating-point type: use uniform distribution
        std::uniform_real_distribution<double> dis(static_cast<double>(min),
                                                   static_cast<double>(max));
        for (size_t i = 0; i < size; ++i) {
            data[i] = static_cast<T>(dis(gen));
        }
    } else {
        // Integer type: use uniform distribution
        std::uniform_int_distribution<int64_t> dis(static_cast<int64_t>(min),
                                                   static_cast<int64_t>(max));
        for (size_t i = 0; i < size; ++i) {
            data[i] = static_cast<T>(dis(gen));
        }
    }
    return data;
}

// Generate sequential data (ascending or descending) for deterministic testing
// Usage: auto data = generateSequentialData<float>(100, 1.0f, 0.5f);
// // 1.0, 1.5, 2.0, ... Parameters:
//   size: Data size
//   start: Starting value (default 1)
//   step: Step size, supports negative values for descending (default 1)
template <typename T>
std::vector<T> generateSequentialData(size_t size, T start = static_cast<T>(1),
                                      T step = static_cast<T>(1)) {
    std::vector<T> data(size);
    T current = start;
    for (size_t i = 0; i < size; ++i) {
        data[i] = current;
        current += step;
    }
    return data;
}
// Generic data copy and conversion function
// Copy data from device memory (hostPtr) to target vector (output), with type
// conversion as needed Parameters:
//   output: Target output vector
//   hostPtr: Source data pointer (pointer after device memory is copied to
//   host) numElements: Number of elements dataType: Actual data type of source
//   data
template <typename T>
void copyAndConvertData(std::vector<T> &output, const void *hostPtr,
                        size_t numElements, const DataType &dataType) {
    constexpr size_t typeSize = sizeof(T);
    const size_t dtypeSize = dataType.getSize();

    if (typeSize == dtypeSize) {
        // Type size matches, direct copy
        std::memcpy(output.data(), hostPtr, numElements * typeSize);
    } else {
        // Type size mismatch, conversion needed
        if constexpr (std::is_same_v<T, float>) {
            if (dataType.getType() == INFINI_DTYPE_F16) {
                // FP16 (16-bit) -> FP32 (32-bit)
                const uint16_t *srcData =
                    static_cast<const uint16_t *>(hostPtr);
                for (size_t i = 0; i < numElements; ++i) {
                    output[i] = fp16_to_fp32(srcData[i]);
                }
            } else if (dataType.getType() == INFINI_DTYPE_F64) {
                // F64 (64-bit) -> F32 (32-bit)
                const double *srcData = static_cast<const double *>(hostPtr);
                for (size_t i = 0; i < numElements; ++i) {
                    output[i] = static_cast<float>(srcData[i]);
                }
            } else {
                // Other types -> float (integer to float)
                switch (dataType.getType()) {
                case INFINI_DTYPE_I8: {
                    const int8_t *src = static_cast<const int8_t *>(hostPtr);
                    for (size_t i = 0; i < numElements; ++i)
                        output[i] = static_cast<float>(src[i]);
                    break;
                }
                case INFINI_DTYPE_I16: {
                    const int16_t *src = static_cast<const int16_t *>(hostPtr);
                    for (size_t i = 0; i < numElements; ++i)
                        output[i] = static_cast<float>(src[i]);
                    break;
                }
                case INFINI_DTYPE_I32: {
                    const int32_t *src = static_cast<const int32_t *>(hostPtr);
                    for (size_t i = 0; i < numElements; ++i)
                        output[i] = static_cast<float>(src[i]);
                    break;
                }
                case INFINI_DTYPE_I64: {
                    const int64_t *src = static_cast<const int64_t *>(hostPtr);
                    for (size_t i = 0; i < numElements; ++i)
                        output[i] = static_cast<float>(src[i]);
                    break;
                }
                case INFINI_DTYPE_U8: {
                    const uint8_t *src = static_cast<const uint8_t *>(hostPtr);
                    for (size_t i = 0; i < numElements; ++i)
                        output[i] = static_cast<float>(src[i]);
                    break;
                }
                case INFINI_DTYPE_U16: {
                    const uint16_t *src =
                        static_cast<const uint16_t *>(hostPtr);
                    for (size_t i = 0; i < numElements; ++i)
                        output[i] = static_cast<float>(src[i]);
                    break;
                }
                case INFINI_DTYPE_U32: {
                    const uint32_t *src =
                        static_cast<const uint32_t *>(hostPtr);
                    for (size_t i = 0; i < numElements; ++i)
                        output[i] = static_cast<float>(src[i]);
                    break;
                }
                case INFINI_DTYPE_U64: {
                    const uint64_t *src =
                        static_cast<const uint64_t *>(hostPtr);
                    for (size_t i = 0; i < numElements; ++i)
                        output[i] = static_cast<float>(src[i]);
                    break;
                }
                default:
                    throw std::runtime_error(
                        "Unsupported type conversion to float: " +
                        dataType.toString());
                }
            }
        } else if constexpr (std::is_same_v<T, uint16_t>) {
            if (dataType.getType() == INFINI_DTYPE_F32) {
                // FP32 (32-bit) -> FP16 (16-bit)
                const float *srcData = static_cast<const float *>(hostPtr);
                for (size_t i = 0; i < numElements; ++i) {
                    output[i] = fp32_to_fp16(srcData[i]);
                }
            } else if (dataType.getType() == INFINI_DTYPE_F64) {
                // F64 (64-bit) -> FP16 (16-bit)
                const double *srcData = static_cast<const double *>(hostPtr);
                for (size_t i = 0; i < numElements; ++i) {
                    output[i] = fp32_to_fp16(static_cast<float>(srcData[i]));
                }
            } else {
                throw std::runtime_error(
                    "Unsupported type conversion to uint16_t: " +
                    dataType.toString());
            }
        } else if constexpr (std::is_same_v<T, double>) {
            if (dataType.getType() == INFINI_DTYPE_F32) {
                // F32 (32-bit) -> F64 (64-bit)
                const float *srcData = static_cast<const float *>(hostPtr);
                for (size_t i = 0; i < numElements; ++i) {
                    output[i] = static_cast<double>(srcData[i]);
                }
            } else if (dataType.getType() == INFINI_DTYPE_F16) {
                // FP16 (16-bit) -> F64 (64-bit)
                const uint16_t *srcData =
                    static_cast<const uint16_t *>(hostPtr);
                for (size_t i = 0; i < numElements; ++i) {
                    output[i] = static_cast<double>(fp16_to_fp32(srcData[i]));
                }
            } else {
                // Other types -> double
                switch (dataType.getType()) {
                case INFINI_DTYPE_I8: {
                    const int8_t *src = static_cast<const int8_t *>(hostPtr);
                    for (size_t i = 0; i < numElements; ++i)
                        output[i] = static_cast<double>(src[i]);
                    break;
                }
                case INFINI_DTYPE_I16: {
                    const int16_t *src = static_cast<const int16_t *>(hostPtr);
                    for (size_t i = 0; i < numElements; ++i)
                        output[i] = static_cast<double>(src[i]);
                    break;
                }
                case INFINI_DTYPE_I32: {
                    const int32_t *src = static_cast<const int32_t *>(hostPtr);
                    for (size_t i = 0; i < numElements; ++i)
                        output[i] = static_cast<double>(src[i]);
                    break;
                }
                case INFINI_DTYPE_I64: {
                    const int64_t *src = static_cast<const int64_t *>(hostPtr);
                    for (size_t i = 0; i < numElements; ++i)
                        output[i] = static_cast<double>(src[i]);
                    break;
                }
                case INFINI_DTYPE_U8: {
                    const uint8_t *src = static_cast<const uint8_t *>(hostPtr);
                    for (size_t i = 0; i < numElements; ++i)
                        output[i] = static_cast<double>(src[i]);
                    break;
                }
                case INFINI_DTYPE_U16: {
                    const uint16_t *src =
                        static_cast<const uint16_t *>(hostPtr);
                    for (size_t i = 0; i < numElements; ++i)
                        output[i] = static_cast<double>(src[i]);
                    break;
                }
                case INFINI_DTYPE_U32: {
                    const uint32_t *src =
                        static_cast<const uint32_t *>(hostPtr);
                    for (size_t i = 0; i < numElements; ++i)
                        output[i] = static_cast<double>(src[i]);
                    break;
                }
                case INFINI_DTYPE_U64: {
                    const uint64_t *src =
                        static_cast<const uint64_t *>(hostPtr);
                    for (size_t i = 0; i < numElements; ++i)
                        output[i] = static_cast<double>(src[i]);
                    break;
                }
                default:
                    throw std::runtime_error(
                        "Unsupported type conversion to double: " +
                        dataType.toString());
                }
            }
        } else if constexpr (std::is_integral_v<T>) {
            // Generic integer type handling
            switch (dataType.getType()) {
            case INFINI_DTYPE_I8: {
                const int8_t *src = static_cast<const int8_t *>(hostPtr);
                for (size_t i = 0; i < numElements; ++i)
                    output[i] = static_cast<T>(src[i]);
                break;
            }
            case INFINI_DTYPE_I16: {
                const int16_t *src = static_cast<const int16_t *>(hostPtr);
                for (size_t i = 0; i < numElements; ++i)
                    output[i] = static_cast<T>(src[i]);
                break;
            }
            case INFINI_DTYPE_I32: {
                const int32_t *src = static_cast<const int32_t *>(hostPtr);
                for (size_t i = 0; i < numElements; ++i)
                    output[i] = static_cast<T>(src[i]);
                break;
            }
            case INFINI_DTYPE_I64: {
                const int64_t *src = static_cast<const int64_t *>(hostPtr);
                for (size_t i = 0; i < numElements; ++i)
                    output[i] = static_cast<T>(src[i]);
                break;
            }
            case INFINI_DTYPE_U8: {
                const uint8_t *src = static_cast<const uint8_t *>(hostPtr);
                for (size_t i = 0; i < numElements; ++i)
                    output[i] = static_cast<T>(src[i]);
                break;
            }
            case INFINI_DTYPE_U16: {
                const uint16_t *src = static_cast<const uint16_t *>(hostPtr);
                for (size_t i = 0; i < numElements; ++i)
                    output[i] = static_cast<T>(src[i]);
                break;
            }
            case INFINI_DTYPE_U32: {
                const uint32_t *src = static_cast<const uint32_t *>(hostPtr);
                for (size_t i = 0; i < numElements; ++i)
                    output[i] = static_cast<T>(src[i]);
                break;
            }
            case INFINI_DTYPE_U64: {
                const uint64_t *src = static_cast<const uint64_t *>(hostPtr);
                for (size_t i = 0; i < numElements; ++i)
                    output[i] = static_cast<T>(src[i]);
                break;
            }
            case INFINI_DTYPE_F16: {
                // FP16 -> integer (needs conversion to FP32 first)
                const uint16_t *src = static_cast<const uint16_t *>(hostPtr);
                for (size_t i = 0; i < numElements; ++i)
                    output[i] = static_cast<T>(fp16_to_fp32(src[i]));
                break;
            }
            case INFINI_DTYPE_F32: {
                const float *src = static_cast<const float *>(hostPtr);
                for (size_t i = 0; i < numElements; ++i)
                    output[i] = static_cast<T>(src[i]);
                break;
            }
            case INFINI_DTYPE_F64: {
                const double *src = static_cast<const double *>(hostPtr);
                for (size_t i = 0; i < numElements; ++i)
                    output[i] = static_cast<T>(src[i]);
                break;
            }
            default:
                throw std::runtime_error(
                    "Unsupported type conversion to integer: " +
                    dataType.toString());
            }
        } else {
            throw std::runtime_error("Unsupported type conversion: T size=" +
                                     std::to_string(typeSize) +
                                     ", dataType=" + dataType.toString());
        }
    }
}

} // namespace infini
#endif // TEST_UTILS_H