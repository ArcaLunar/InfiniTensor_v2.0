#pragma once
#ifndef UTIL_H
#define UTIL_H

#include "core/common.h"
#include "core/expr.h"
#include <numeric>
#include <random>

namespace infini {

// Calculate stride after broadcasting
// inputShape: Shape of the input tensor
// inputStride: Original stride of the input tensor
// outputShape: Shape of the output tensor (after broadcasting)
// Returns: Stride after broadcasting, stride is set to 0 for broadcasted
// dimensions
inline StrideExpr broadcastStride(const ShapeExpr &inputShape,
                                  const StrideExpr &inputStride,
                                  const ShapeExpr &outputShape) {
    IT_ASSERT(inputShape->size() == inputStride->size(),
              "Input shape and stride must have the same rank");

    size_t inputRank = inputShape->size();
    size_t outputRank = outputShape->size();

    // Align dimensions from right to left (NumPy broadcasting rules)
    std::vector<Expr> broadcastedStride(outputRank);

    for (size_t outIdx = 0; outIdx < outputRank; ++outIdx) {
        // Calculate the corresponding input dimension index (aligned from right
        // to left)
        int inIdx = static_cast<int>(outIdx) - static_cast<int>(outputRank) +
                    static_cast<int>(inputRank);

        if (inIdx < 0) {
            // Input dimension does not exist, equivalent to dimension being 1,
            // stride is 0
            broadcastedStride[outIdx] = ExprObj::constant(0);
        } else {
            // Check if input dimension is 1 (broadcast dimension) or equal to
            // output dimension
            const Expr &inputDim = (*inputShape)[inIdx];
            const Expr &outputDim = (*outputShape)[outIdx];

            // If input dimension is constant 1, it is a broadcast dimension,
            // stride is 0
            auto inputDimConst = inputDim->asConstant();
            if (inputDimConst.has_value() && *inputDimConst == 1) {
                broadcastedStride[outIdx] = ExprObj::constant(0);
            } else {
                // Dimensions are equal or unequal but valid (guaranteed by
                // infer_broadcast), use original stride
                broadcastedStride[outIdx] = (*inputStride)[inIdx];
            }
        }
    }

    return make_ref<StrideExprObj>(broadcastedStride);
}

inline ShapeExpr infer_broadcast(const ShapeExpr &A, const ShapeExpr &B) {
    size_t rankA = A->size();
    size_t rankB = B->size();
    size_t rank = std::max(rankA, rankB);

    std::vector<Expr> resultDims;
    for (size_t i = 0; i < rank; ++i) {
        // Align from right to left (NumPy broadcasting rules)
        int idxA = static_cast<int>(i) - static_cast<int>(rank) +
                   static_cast<int>(rankA);
        int idxB = static_cast<int>(i) - static_cast<int>(rank) +
                   static_cast<int>(rankB);

        // If index is negative, the dimension does not exist, equivalent to
        // dimension being 1
        Expr aDim = (idxA < 0) ? ExprObj::constant(1) : (*A)[idxA];
        Expr bDim = (idxB < 0) ? ExprObj::constant(1) : (*B)[idxB];

        // Validate broadcasting rules: dimensions must be equal, or one of them
        // is 1
        IT_ASSERT(aDim == bDim || aDim == ExprObj::constant(1) ||
                  bDim == ExprObj::constant(1));

        // Broadcasting result: if a dimension is 1, take b dimension, otherwise
        // take a dimension
        auto shapeEle = aDim == ExprObj::constant(1) ? bDim : aDim;
        resultDims.emplace_back(shapeEle);
    }

    return make_ref<ShapeExprObj>(resultDims);
}

inline size_t calculateLinearOffset(size_t index, Shape shape, Stride stride) {
    size_t rank = shape.size();
    std::vector<size_t> indices(rank);
    size_t remaining = index;
    for (size_t i = 0; i < rank; ++i) {
        size_t dim = rank - 1 - i;
        indices[dim] = remaining % shape.at(dim);
        remaining /= shape.at(dim);
    }
    size_t offset = 0;
    for (size_t i = 0; i < rank; ++i) {
        offset += indices[i] * stride.at(i);
    }
    return offset;
}

// FP16 to FP32 conversion utility
inline float fp16_to_fp32(uint16_t fp16) {
    // Union for safe type punning
    union {
        uint32_t u;
        float f;
    } converter;

    // Extract components from FP16
    uint32_t sign = (fp16 >> 15) & 0x1;
    uint32_t exponent = (fp16 >> 10) & 0x1F;
    uint32_t mantissa = fp16 & 0x3FF;

    // Handle special cases
    if (exponent == 0) {
        if (mantissa == 0) {
            // Zero
            converter.u = sign << 31;
            return converter.f;
        } else {
            // Subnormal number: normalize it
            while (!(mantissa & 0x400)) {
                mantissa <<= 1;
                exponent--;
            }
            exponent++;
            mantissa &= 0x3FF;
        }
    } else if (exponent == 31) {
        // Infinity or NaN
        converter.u = (sign << 31) | 0x7F800000;
        if (mantissa) {
            converter.u |= mantissa; // NaN
        }
        return converter.f;
    }

    // Convert to FP32
    // FP32: 1 sign bit, 8 exponent bits (bias 127), 23 mantissa bits
    // FP16: 1 sign bit, 5 exponent bits (bias 15), 10 mantissa bits
    converter.u = (sign << 31) | ((exponent + 112) << 23) | (mantissa << 13);
    return converter.f;
}

// FP32 to FP16 conversion utility
inline uint16_t fp32_to_fp16(float fp32) {
    // Union for safe type punning
    union {
        uint32_t u;
        float f;
    } converter;
    converter.f = fp32;

    // Extract components from FP32
    uint32_t sign = (converter.u >> 31) & 0x1;
    uint32_t exponent = (converter.u >> 23) & 0xFF;
    uint32_t mantissa = converter.u & 0x7FFFFF;

    // Handle special cases
    if (exponent == 0) {
        // Zero or subnormal FP32 (very small, treat as zero in FP16)
        return static_cast<uint16_t>(sign << 15);
    } else if (exponent == 255) {
        // Infinity or NaN
        uint16_t result = static_cast<uint16_t>((sign << 15) | 0x7C00);
        if (mantissa) {
            // NaN - preserve some mantissa bits
            result |= static_cast<uint16_t>(mantissa >> 13);
        }
        return result;
    }

    // Convert to FP16
    // FP32 exponent bias: 127, FP16 exponent bias: 15
    // New exponent = exponent - 127 + 15 = exponent - 112
    int32_t newExponent = static_cast<int32_t>(exponent) - 112;

    // Handle overflow/underflow
    if (newExponent >= 31) {
        // Overflow: return infinity
        return static_cast<uint16_t>((sign << 15) | 0x7C00);
    } else if (newExponent <= 0) {
        // Underflow: return zero (could also handle subnormals)
        return static_cast<uint16_t>(sign << 15);
    }

    // Normal number: round to nearest even
    uint32_t roundedMantissa = mantissa + 0x1000; // Add rounding bias
    if (roundedMantissa & 0x800000) {
        // Rounding caused overflow
        roundedMantissa = 0;
        newExponent++;
        if (newExponent >= 31) {
            // Overflow after rounding
            return static_cast<uint16_t>((sign << 15) | 0x7C00);
        }
    }

    // Assemble FP16
    return static_cast<uint16_t>((sign << 15) | (newExponent << 10) |
                                 (roundedMantissa >> 13));
}

} // namespace infini

#endif
