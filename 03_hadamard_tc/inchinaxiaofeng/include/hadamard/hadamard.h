// Public API of the Hadamard transform implementations.
//
// Convention used everywhere in this project (see docs/learning_notes.md):
//     y = H_d * x / sqrt(d)
// with H_d the Sylvester matrix, H_1 = [1] and H_2n = [[H_n, H_n], [H_n, -H_n]].
// The 1/sqrt(d) factor makes the transform orthonormal: scaled twice it is the
// identity. TODO(M1): confirm the factor against the reference library before
// benchmarking.
#ifndef HADAMARD_HADAMARD_H_
#define HADAMARD_HADAMARD_H_

#include <cuda_runtime.h>

#include <cstdint>

#include "hadamard/common.h"

namespace hadamard {

// ---------------------------------------------------------------------------
// CPU FP32 reference implementations (ground truth for the correctness harness)
// ---------------------------------------------------------------------------

// Explicit H_d matrix multiply, O(d^2) per row. Slow on purpose: it pins down the
// sign/matrix convention independently of the butterfly implementation.
void ref_matmul_fp32(const float* x, float* y, long rows, int d);

// Butterfly fast Walsh-Hadamard transform, O(d log2 d) per row.
void ref_fwht_fp32(const float* x, float* y, long rows, int d);

// ---------------------------------------------------------------------------
// Host-side precision helpers: build the inputs the GPU kernels actually consume
// and dequantize their outputs before comparing.
// ---------------------------------------------------------------------------
void pack_fp32(const float* src, long n, uint16_t* dst, DType dtype);
void unpack_fp32(const uint16_t* src, long n, float* dst, DType dtype);

// ---------------------------------------------------------------------------
// GPU entry points
//
// Each launcher returns false while the implementation is still a stub so the
// harness reports SKIP instead of FAIL. Milestones fill them in:
//   launch_fwht_baseline -> M2 (shared-memory butterfly, non Tensor Core)
//   launch_hadamard_tc   -> M3 (WMMA GEMM against the constant H_d matrix)
// ---------------------------------------------------------------------------
bool launch_fwht_baseline(const void* x, void* y, const Shape& shape, DType dtype,
                          cudaStream_t stream = nullptr);

bool launch_hadamard_tc(const void* x, void* y, const Shape& shape, DType dtype,
                        cudaStream_t stream = nullptr);

// Toy kernel proving the toolchain works end to end (M0 deliverable).
void launch_hello(int* out, int value, cudaStream_t stream = nullptr);

// Tensor Core probe: computes C = A * I on 16x16 fragments and checks the result
// equals A. Verifies that the WMMA path for `dtype` is usable on this GPU.
// Returns false when the element type is unsupported on the current device.
bool launch_wmma_identity_probe(DType dtype, bool* correct, cudaStream_t stream = nullptr);

}  // namespace hadamard

#endif  // HADAMARD_HADAMARD_H_
