// GPU entry points: the M0 toolchain probe plus stubs that later milestones fill
// in. Nothing here implements the transform yet on purpose.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>

#include <cstdio>
#include <vector>

#include "hadamard/cuda_utils.h"
#include "hadamard/hadamard.h"

namespace hadamard {
namespace {

__global__ void hello_kernel(int* out, int value) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *out = value;
    }
}

// Half/bfloat16 helpers so the probe can be written once for both element types.
template <typename T>
struct TypeTraits;

template <>
struct TypeTraits<__half> {
    static __half from_float(float v) { return __float2half_rn(v); }
    static float to_float(__half v) { return __half2float(v); }
};

template <>
struct TypeTraits<__nv_bfloat16> {
    static __nv_bfloat16 from_float(float v) { return __float2bfloat16_rn(v); }
    static float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
};

// C = A * B for one 16x16x16 WMMA tile. Only used to prove the Tensor Core path
// works for a given element type; the real kernel comes in M3.
template <typename T>
__global__ void wmma_identity_probe_kernel(const T* a, const T* b, float* c) {
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, T, nvcuda::wmma::row_major> fa;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, T, nvcuda::wmma::col_major> fb;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> fc;
    nvcuda::wmma::fill_fragment(fc, 0.0f);
    nvcuda::wmma::load_matrix_sync(fa, a, 16);
    nvcuda::wmma::load_matrix_sync(fb, b, 16);
    nvcuda::wmma::mma_sync(fc, fa, fb, fc);
    nvcuda::wmma::store_matrix_sync(c, fc, 16, nvcuda::wmma::mem_row_major);
}

// A is filled with 0..255 (exact in fp16 and bf16) and B with the identity, so the
// expected result is exactly A.
template <typename T>
bool run_identity_probe(bool* correct, cudaStream_t stream) {
    constexpr int kDim = 16;
    constexpr int kElems = kDim * kDim;
    std::vector<T> h_a(kElems);
    std::vector<T> h_b(kElems);
    std::vector<float> h_expected(kElems);
    std::vector<float> h_got(kElems, 0.0f);
    for (int i = 0; i < kDim; ++i) {
        for (int j = 0; j < kDim; ++j) {
            const int idx = i * kDim + j;
            h_a[idx] = TypeTraits<T>::from_float(static_cast<float>(idx));
            h_b[idx] = TypeTraits<T>::from_float(i == j ? 1.0f : 0.0f);
            h_expected[idx] = static_cast<float>(idx);
        }
    }

    T* d_a = nullptr;
    T* d_b = nullptr;
    float* d_c = nullptr;
    HW_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_a), sizeof(T) * kElems));
    HW_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_b), sizeof(T) * kElems));
    HW_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_c), sizeof(float) * kElems));
    HW_CUDA_CHECK(cudaMemcpyAsync(d_a, h_a.data(), sizeof(T) * kElems,
                                  cudaMemcpyHostToDevice, stream));
    HW_CUDA_CHECK(cudaMemcpyAsync(d_b, h_b.data(), sizeof(T) * kElems,
                                  cudaMemcpyHostToDevice, stream));
    wmma_identity_probe_kernel<T><<<1, 32, 0, stream>>>(d_a, d_b, d_c);
    HW_CUDA_CHECK(cudaGetLastError());
    HW_CUDA_CHECK(cudaMemcpyAsync(h_got.data(), d_c, sizeof(float) * kElems,
                                  cudaMemcpyDeviceToHost, stream));
    HW_CUDA_CHECK(cudaStreamSynchronize(stream));
    HW_CUDA_CHECK(cudaFree(d_a));
    HW_CUDA_CHECK(cudaFree(d_b));
    HW_CUDA_CHECK(cudaFree(d_c));

    bool ok = true;
    for (int i = 0; i < kElems; ++i) {
        if (h_got[i] != h_expected[i]) {
            std::fprintf(stderr, "wmma probe mismatch at %d: got %f expected %f\n", i, h_got[i],
                         h_expected[i]);
            ok = false;
            break;
        }
    }
    *correct = ok;
    return true;
}

bool required_arch_ok(DType dtype) {
    int major = 0;
    int minor = 0;
    HW_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0));
    HW_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0));
    const bool supports_fp16_wmma = (major * 10 + minor) >= 70;
    if (!supports_fp16_wmma) {
        return false;
    }
    // bf16 WMMA landed with Ampere (sm_80).
    return dtype == DType::kFp16 || (major * 10 + minor) >= 80;
}

}  // namespace

void launch_hello(int* out, int value, cudaStream_t stream) {
    hello_kernel<<<1, 32, 0, stream>>>(out, value);
    HW_CUDA_CHECK(cudaGetLastError());
}

bool launch_wmma_identity_probe(DType dtype, bool* correct, cudaStream_t stream) {
    if (!required_arch_ok(dtype)) {
        return false;
    }
    if (dtype == DType::kFp16) {
        return run_identity_probe<__half>(correct, stream);
    }
    return run_identity_probe<__nv_bfloat16>(correct, stream);
}

bool launch_fwht_baseline(const void* x, void* y, const Shape& shape, DType dtype,
                          cudaStream_t stream) {
    (void)x;
    (void)y;
    (void)shape;
    (void)dtype;
    (void)stream;
    // TODO(M2): shared-memory butterfly FWHT, vectorized IO, covers 64/128/256 and
    // larger powers of two.
    return false;
}

bool launch_hadamard_tc(const void* x, void* y, const Shape& shape, DType dtype,
                        cudaStream_t stream) {
    (void)x;
    (void)y;
    (void)shape;
    (void)dtype;
    (void)stream;
    // TODO(M3): X_tile (Mtile x d) times the constant matrix H_d (d x d, fp16) via
    // WMMA, fp32 accumulate, epilogue scaling by 1/sqrt(d).
    return false;
}

}  // namespace hadamard
