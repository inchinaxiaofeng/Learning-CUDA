// Benchmark driver: kernel time in ms plus effective bandwidth in GB/s.
//
// Shapes cover the head_dim values the spec calls out (64/128/256, plus 512) with
// different batch/sequence/head combinations. GPU rows appear as "n/a" until the
// corresponding milestone lands.
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "hadamard/common.h"
#include "hadamard/cuda_utils.h"
#include "hadamard/hadamard.h"

namespace {

struct BenchCase {
    int batch;
    int seq_len;
    int num_heads;
    int head_dim;
};

const BenchCase kCases[] = {
    {1, 128, 12, 64},
    {2, 256, 16, 128},
    {1, 512, 8, 256},
    {4, 512, 16, 128},
    {1, 4096, 12, 64},
};

constexpr int kGpuIters = 50;
constexpr int kCpuIters = 5;

std::string case_name(const BenchCase& c) {
    return "[" + std::to_string(c.batch) + "," + std::to_string(c.seq_len) + "," +
           std::to_string(c.num_heads) + "," + std::to_string(c.head_dim) + "]";
}

double bench_cpu_fwht(const std::vector<float>& x, std::vector<float>& y, long rows, int d) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kCpuIters; ++i) {
        hadamard::ref_fwht_fp32(x.data(), y.data(), rows, d);
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return total_ms / kCpuIters;
}

void bench_gpu_implementation(const char* impl_name,
                              bool (*launch)(const void*, void*, const hadamard::Shape&,
                                             hadamard::DType, cudaStream_t),
                              const BenchCase& c, hadamard::DType dtype) {
    using namespace hadamard;
    const Shape shape{c.batch, c.seq_len, c.num_heads, c.head_dim};
    const long n = shape.elems();
    const size_t bytes = static_cast<size_t>(n) * sizeof(uint16_t) * 2;  // read + write

    void* d_x = nullptr;
    void* d_y = nullptr;
    HW_CUDA_CHECK(cudaMalloc(&d_x, static_cast<size_t>(n) * sizeof(uint16_t)));
    HW_CUDA_CHECK(cudaMalloc(&d_y, static_cast<size_t>(n) * sizeof(uint16_t)));
    HW_CUDA_CHECK(cudaMemset(d_x, 0, static_cast<size_t>(n) * sizeof(uint16_t)));

    const bool available = launch(d_x, d_y, shape, dtype, nullptr);
    if (!available) {
        std::printf("%-24s %-5s %-16s %10s %10s\n", case_name(c).c_str(), dtype_name(dtype),
                    impl_name, "n/a", "n/a");
        HW_CUDA_CHECK(cudaFree(d_x));
        HW_CUDA_CHECK(cudaFree(d_y));
        return;
    }

    CudaEventTimer timer;
    launch(d_x, d_y, shape, dtype, nullptr);  // warm-up
    timer.start();
    for (int i = 0; i < kGpuIters; ++i) {
        launch(d_x, d_y, shape, dtype, nullptr);
    }
    timer.stop();
    const float ms = timer.elapsed_ms() / kGpuIters;

    std::printf("%-24s %-5s %-16s %10.4f %10.1f\n", case_name(c).c_str(), dtype_name(dtype),
                impl_name, static_cast<double>(ms), effective_bandwidth_gbps(bytes, ms));
    HW_CUDA_CHECK(cudaFree(d_x));
    HW_CUDA_CHECK(cudaFree(d_y));
}

}  // namespace

int main() {
    using namespace hadamard;

    const DeviceInfo info = query_device(0);
    std::printf("device: %s (sm_%d%d, %d SMs)\n", info.name.c_str(), info.major, info.minor,
                info.sm_count);
    std::printf("iters : gpu=%d, cpu=%d\n\n", kGpuIters, kCpuIters);

    std::printf("%-24s %-5s %-16s %10s %10s\n", "shape (B,S,H,D)", "dtype", "impl", "ms", "GB/s");
    std::printf("--------------------------------------------------------------------------\n");

    for (const BenchCase& c : kCases) {
        const Shape shape{c.batch, c.seq_len, c.num_heads, c.head_dim};
        const long n = shape.elems();
        std::vector<float> x = random_host_data(n, 0xF00Du + c.head_dim);
        std::vector<float> y(n);
        const double cpu_ms = bench_cpu_fwht(x, y, shape.rows(), shape.head_dim);
        const double cpu_bytes_gbps =
            effective_bandwidth_gbps(static_cast<size_t>(n) * sizeof(float) * 2,
                                     static_cast<float>(cpu_ms));
        std::printf("%-24s %-5s %-16s %10.4f %10.1f\n", case_name(c).c_str(), "fp32",
                    "cpu reference", cpu_ms, cpu_bytes_gbps);

        for (DType dtype : {DType::kFp16, DType::kBFloat16}) {
            bench_gpu_implementation("fwht baseline", &launch_fwht_baseline, c, dtype);
            bench_gpu_implementation("tensor core", &launch_hadamard_tc, c, dtype);
        }
    }

    std::printf("\nnotes: GB/s counts one read plus one write of the tensor.\n");
    return 0;
}
