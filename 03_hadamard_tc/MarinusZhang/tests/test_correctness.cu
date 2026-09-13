// Correctness harness.
//
// M1 covers the CPU references; the GPU comparisons are already wired so that M2
// and M3 only have to fill in the launchers. Comparison methodology: the input is
// rounded to the target precision first, and the reference is computed on that
// rounded input, so what the tolerance measures is kernel error, not input
// quantization error.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "hadamard/common.h"
#include "hadamard/cuda_utils.h"
#include "hadamard/hadamard.h"

// The helpers below are declared at file scope, so pull the project symbols in.
using namespace hadamard;

namespace {

int g_failed = 0;
int g_skipped = 0;

struct Case {
    int batch;
    int seq_len;
    int num_heads;
    int head_dim;
};

const Case kCases[] = {
    {1, 1, 1, 2}, {1, 1, 1, 64}, {2, 3, 4, 128}, {1, 7, 8, 256}, {3, 5, 6, 512},
};

void report(bool ok, const std::string& name, const std::string& detail) {
    std::printf("[%s] %-46s %s\n", ok ? "PASS" : "FAIL", name.c_str(), detail.c_str());
    if (!ok) {
        ++g_failed;
    }
}

void skip(const std::string& name, const std::string& detail) {
    std::printf("[SKIP] %-46s %s\n", name.c_str(), detail.c_str());
    ++g_skipped;
}

std::string fmt(const Case& c) {
    return "d=" + std::to_string(c.head_dim) + " rows=" +
           std::to_string(static_cast<long>(c.batch) * c.seq_len * c.num_heads);
}

float max_abs_diff(const float* a, const float* b, long n) {
    float worst = 0.0f;
    for (long i = 0; i < n; ++i) {
        worst = std::max(worst, std::fabs(a[i] - b[i]));
    }
    return worst;
}

// The O(d^2) matmul reference and the butterfly must agree: they are two
// independent constructions of the same matrix.
void test_reference_agreement() {
    const float kTol = 1e-4f;
    for (const Case& c : kCases) {
        const Shape shape{c.batch, c.seq_len, c.num_heads, c.head_dim};
        const long n = shape.elems();
        const std::vector<float> x = random_host_data(n, 0xABCDu + c.head_dim);
        std::vector<float> y_matmul(n);
        std::vector<float> y_fwht(n);
        ref_matmul_fp32(x.data(), y_matmul.data(), shape.rows(), shape.head_dim);
        ref_fwht_fp32(x.data(), y_fwht.data(), shape.rows(), shape.head_dim);
        const float diff = max_abs_diff(y_matmul.data(), y_fwht.data(), n);
        report(diff <= kTol, "reference agreement (matmul vs butterfly) " + fmt(c),
               "max|diff|=" + std::to_string(diff));
    }
}

// H is orthogonal up to the scale factor, so applying the scaled transform twice
// must return the input. This catches sign, ordering, and normalization mistakes.
void test_orthogonality() {
    const float kTol = 1e-4f;
    for (const Case& c : kCases) {
        const Shape shape{c.batch, c.seq_len, c.num_heads, c.head_dim};
        const long n = shape.elems();
        const std::vector<float> x = random_host_data(n, 0x1234u + c.head_dim);
        std::vector<float> tmp(n);
        std::vector<float> back(n);
        ref_fwht_fp32(x.data(), tmp.data(), shape.rows(), shape.head_dim);
        ref_fwht_fp32(tmp.data(), back.data(), shape.rows(), shape.head_dim);
        const float diff = max_abs_diff(x.data(), back.data(), n);
        report(diff <= kTol, "orthogonality (transform applied twice) " + fmt(c),
               "max|diff|=" + std::to_string(diff));
    }
}

// Sanity check on the tolerances: rounding an input in [-1, 1) to the target
// precision must cost far less than the allowed absolute error.
void test_precision_roundtrip() {
    for (DType dtype : {DType::kFp16, DType::kBFloat16}) {
        const long n = 4096;
        const std::vector<float> x = random_host_data(n, 0x5EEDu);
        std::vector<uint16_t> packed(n);
        std::vector<float> back(n);
        pack_fp32(x.data(), n, packed.data(), dtype);
        unpack_fp32(packed.data(), n, back.data(), dtype);
        const float diff = max_abs_diff(x.data(), back.data(), n);
        const float limit = 0.1f * tolerance_for(dtype);
        report(diff <= limit,
               std::string("precision round-trip ") + dtype_name(dtype),
               "max|diff|=" + std::to_string(diff) + " limit=" + std::to_string(limit));
    }
}

// Full GPU comparison, ready for M2/M3. Reports SKIP while the launcher is a stub.
void test_gpu_implementation(const char* impl_name,
                             bool (*launch)(const void*, void*, const Shape&, DType,
                                            cudaStream_t)) {
    for (const Case& c : kCases) {
        const Shape shape{c.batch, c.seq_len, c.num_heads, c.head_dim};
        const long rows = shape.rows();
        const long n = shape.elems();
        for (DType dtype : {DType::kFp16, DType::kBFloat16}) {
            const std::string name =
                std::string(impl_name) + " " + dtype_name(dtype) + " " + fmt(c);

            const std::vector<float> host_x = random_host_data(n, 0xBEEFu + c.head_dim);
            std::vector<uint16_t> packed_x(n);
            pack_fp32(host_x.data(), n, packed_x.data(), dtype);

            // Reference computed on the rounded input, i.e. the exact input the
            // kernel sees.
            std::vector<float> rounded_x(n);
            unpack_fp32(packed_x.data(), n, rounded_x.data(), dtype);
            std::vector<float> expected(n);
            ref_matmul_fp32(rounded_x.data(), expected.data(), rows, shape.head_dim);

            void* d_x = nullptr;
            void* d_y = nullptr;
            HW_CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(uint16_t)));
            HW_CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(uint16_t)));
            HW_CUDA_CHECK(cudaMemcpy(d_x, packed_x.data(), n * sizeof(uint16_t),
                                     cudaMemcpyHostToDevice));

            const bool available = launch(d_x, d_y, shape, dtype, nullptr);
            if (!available) {
                skip(name, "launcher is still a stub");
                HW_CUDA_CHECK(cudaFree(d_x));
                HW_CUDA_CHECK(cudaFree(d_y));
                continue;
            }

            std::vector<uint16_t> packed_y(n);
            HW_CUDA_CHECK(cudaMemcpy(packed_y.data(), d_y, n * sizeof(uint16_t),
                                     cudaMemcpyDeviceToHost));
            HW_CUDA_CHECK(cudaFree(d_x));
            HW_CUDA_CHECK(cudaFree(d_y));

            std::vector<float> got(n);
            unpack_fp32(packed_y.data(), n, got.data(), dtype);
            const float diff = max_abs_diff(expected.data(), got.data(), n);
            const float limit = tolerance_for(dtype);
            report(diff <= limit, name,
                   "max|diff|=" + std::to_string(diff) + " limit=" + std::to_string(limit));
        }
    }
}

}  // namespace

int main() {
    std::printf("== Hadamard correctness harness ==\n\n");
    test_reference_agreement();
    test_orthogonality();
    test_precision_roundtrip();
    std::printf("\n-- GPU implementations (filled in during M2/M3) --\n");
    test_gpu_implementation("fwht baseline", &hadamard::launch_fwht_baseline);
    test_gpu_implementation("tensor core", &hadamard::launch_hadamard_tc);

    std::printf("\n%d failed, %d skipped\n", g_failed, g_skipped);
    return g_failed == 0 ? 0 : 1;
}
