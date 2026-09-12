// CPU FP32 reference implementations and host-side precision helpers.
//
// These are deliberately simple: they define the ground truth the GPU kernels are
// compared against, and they pin down the matrix/sign convention. Performance is
// irrelevant here (the O(d^2) matmul only runs on small shapes).
#include "hadamard/hadamard.h"

#include <cmath>
#include <cstdint>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace hadamard {
namespace {

// Sylvester construction: H_1 = [1], H_2n = [[H_n, H_n], [H_n, -H_n]].
std::vector<float> build_sylvester(int d) {
    std::vector<float> h(1, 1.0f);
    while (static_cast<int>(h.size()) < d) {
        const int n = static_cast<int>(h.size());
        std::vector<float> next(static_cast<size_t>(4) * n * n, 0.0f);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                const float v = h[static_cast<size_t>(i) * n + j];
                const size_t top = static_cast<size_t>(i) * 2 * n + j;
                const size_t bottom = static_cast<size_t>(i + n) * 2 * n + j;
                next[top] = v;
                next[top + n] = v;
                next[bottom] = v;
                next[bottom + n] = -v;
            }
        }
        h.swap(next);
    }
    return h;
}

}  // namespace

void ref_matmul_fp32(const float* x, float* y, long rows, int d) {
    const std::vector<float> h = build_sylvester(d);
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));
    for (long r = 0; r < rows; ++r) {
        const float* row_x = x + r * d;
        float* row_y = y + r * d;
        for (int i = 0; i < d; ++i) {
            const float* h_row = h.data() + static_cast<size_t>(i) * d;
            float acc = 0.0f;
            for (int j = 0; j < d; ++j) {
                acc += h_row[j] * row_x[j];
            }
            row_y[i] = acc * scale;
        }
    }
}

void ref_fwht_fp32(const float* x, float* y, long rows, int d) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));
    for (long r = 0; r < rows; ++r) {
        const float* row_x = x + r * d;
        float* row_y = y + r * d;
        for (int i = 0; i < d; ++i) {
            row_y[i] = row_x[i];
        }
        // log2(d) butterfly stages, each halving the number of independent pairs.
        for (int len = 1; len < d; len <<= 1) {
            for (int base = 0; base < d; base += (len << 1)) {
                for (int j = 0; j < len; ++j) {
                    const float a = row_y[base + j];
                    const float b = row_y[base + j + len];
                    row_y[base + j] = a + b;
                    row_y[base + j + len] = a - b;
                }
            }
        }
        for (int i = 0; i < d; ++i) {
            row_y[i] *= scale;
        }
    }
}

void pack_fp32(const float* src, long n, uint16_t* dst, DType dtype) {
    if (dtype == DType::kFp16) {
        for (long i = 0; i < n; ++i) {
            dst[i] = __half_as_ushort(__float2half_rn(src[i]));
        }
        return;
    }
    for (long i = 0; i < n; ++i) {
        dst[i] = __bfloat16_as_ushort(__float2bfloat16_rn(src[i]));
    }
}

void unpack_fp32(const uint16_t* src, long n, float* dst, DType dtype) {
    if (dtype == DType::kFp16) {
        for (long i = 0; i < n; ++i) {
            dst[i] = __half2float(__ushort_as_half(src[i]));
        }
        return;
    }
    for (long i = 0; i < n; ++i) {
        dst[i] = __bfloat162float(__ushort_as_bfloat16(src[i]));
    }
}

}  // namespace hadamard
