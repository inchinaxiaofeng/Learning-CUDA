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
    // Traffic (one read plus one write) stays well below the 72 MiB L2 of the target
    // GPU, so across the timed iterations these rows measure L2-bound bandwidth.
    {1, 128, 12, 64},   // 0.4 MiB
    {2, 256, 16, 128},  // 4 MiB
    {1, 512, 8, 256},   // 4 MiB
    {4, 512, 16, 128},  // 16 MiB
    {1, 4096, 12, 64},  // 12 MiB
    // 8 * 4096 * 16 * 128 elements = 256 MiB of traffic, i.e. larger than L2: this is
    // the row that shows what the kernel achieves against HBM (see the report).
    {8, 4096, 16, 128},
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

template <typename Fn>
double time_gpu(Fn&& fn) {
    hadamard::CudaEventTimer timer;
    fn();  // warm-up
    timer.start();
    for (int i = 0; i < kGpuIters; ++i) {
        fn();
    }
    timer.stop();
    return timer.elapsed_ms() / kGpuIters;
}

// FP8 rows. The byte model is the point of these rows: a fused kernel reads the
// activation once and writes the codes, a two-stage pipeline pays for the transform's
// own read/write as well. Per-row scales add 4 B per row, i.e. 4/d B per element.
size_t fp8_code_bytes(long n, long rows) {
    return static_cast<size_t>(n) + static_cast<size_t>(rows) * sizeof(float);
}

void bench_gpu_fp8(const char* impl_name,
                   bool (*launch)(const void*, uint8_t*, float*, const hadamard::Shape&,
                                  hadamard::DType, cudaStream_t),
                   const BenchCase& c, hadamard::DType dtype) {
    using namespace hadamard;
    const Shape shape{c.batch, c.seq_len, c.num_heads, c.head_dim};
    const long n = shape.elems();
    const long rows = shape.rows();
    const size_t bytes = static_cast<size_t>(n) * sizeof(uint16_t) + fp8_code_bytes(n, rows);

    void* d_x = nullptr;
    uint8_t* d_q = nullptr;
    float* d_s = nullptr;
    HW_CUDA_CHECK(cudaMalloc(&d_x, static_cast<size_t>(n) * sizeof(uint16_t)));
    HW_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_q), static_cast<size_t>(n)));
    HW_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_s), rows * sizeof(float)));
    HW_CUDA_CHECK(cudaMemset(d_x, 0, static_cast<size_t>(n) * sizeof(uint16_t)));

    if (!launch(d_x, d_q, d_s, shape, dtype, nullptr)) {
        std::printf("%-24s %-5s %-16s %10s %10s\n", case_name(c).c_str(), dtype_name(dtype),
                    impl_name, "n/a", "n/a");
    } else {
        const double ms = time_gpu([&] { launch(d_x, d_q, d_s, shape, dtype, nullptr); });
        std::printf("%-24s %-5s %-16s %10.4f %10.1f\n", case_name(c).c_str(), dtype_name(dtype),
                    impl_name, ms, effective_bandwidth_gbps(bytes, static_cast<float>(ms)));
    }
    HW_CUDA_CHECK(cudaFree(d_x));
    HW_CUDA_CHECK(cudaFree(d_q));
    HW_CUDA_CHECK(cudaFree(d_s));
}

void bench_gpu_fp8_two_stage(const char* impl_name,
                             bool (*xform)(const void*, void*, const hadamard::Shape&,
                                           hadamard::DType, cudaStream_t),
                             bool (*quant)(const void*, uint8_t*, float*, const hadamard::Shape&,
                                           hadamard::DType, cudaStream_t),
                             const BenchCase& c, hadamard::DType dtype) {
    using namespace hadamard;
    const Shape shape{c.batch, c.seq_len, c.num_heads, c.head_dim};
    const long n = shape.elems();
    const long rows = shape.rows();
    // Transform: read + write an activation tensor. Quantizer: read it back and write
    // the codes. Both stages are timed together, and both are charged to the byte model.
    const size_t bytes = 3 * static_cast<size_t>(n) * sizeof(uint16_t) + fp8_code_bytes(n, rows);

    void* d_x = nullptr;
    void* d_y = nullptr;
    uint8_t* d_q = nullptr;
    float* d_s = nullptr;
    HW_CUDA_CHECK(cudaMalloc(&d_x, static_cast<size_t>(n) * sizeof(uint16_t)));
    HW_CUDA_CHECK(cudaMalloc(&d_y, static_cast<size_t>(n) * sizeof(uint16_t)));
    HW_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_q), static_cast<size_t>(n)));
    HW_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_s), rows * sizeof(float)));
    HW_CUDA_CHECK(cudaMemset(d_x, 0, static_cast<size_t>(n) * sizeof(uint16_t)));

    const bool available =
        xform(d_x, d_y, shape, dtype, nullptr) && quant(d_y, d_q, d_s, shape, dtype, nullptr);
    if (!available) {
        std::printf("%-24s %-5s %-16s %10s %10s\n", case_name(c).c_str(), dtype_name(dtype),
                    impl_name, "n/a", "n/a");
    } else {
        const double ms = time_gpu([&] {
            xform(d_x, d_y, shape, dtype, nullptr);
            quant(d_y, d_q, d_s, shape, dtype, nullptr);
        });
        std::printf("%-24s %-5s %-16s %10.4f %10.1f\n", case_name(c).c_str(), dtype_name(dtype),
                    impl_name, ms, effective_bandwidth_gbps(bytes, static_cast<float>(ms)));
    }
    HW_CUDA_CHECK(cudaFree(d_x));
    HW_CUDA_CHECK(cudaFree(d_y));
    HW_CUDA_CHECK(cudaFree(d_q));
    HW_CUDA_CHECK(cudaFree(d_s));
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
        const double cpu_bytes_gbps = effective_bandwidth_gbps(
            static_cast<size_t>(n) * sizeof(float) * 2, static_cast<float>(cpu_ms));
        std::printf("%-24s %-5s %-16s %10.4f %10.1f\n", case_name(c).c_str(), "fp32",
                    "cpu reference", cpu_ms, cpu_bytes_gbps);

        for (DType dtype : {DType::kFp16, DType::kBFloat16}) {
            bench_gpu_implementation("fwht baseline", &launch_fwht_baseline, c, dtype);
            bench_gpu_implementation("tensor core", &launch_hadamard_tc, c, dtype);
        }

        // FP8 rows run on fp16 activations only: bf16 traffic is identical (2 B/elem),
        // and the interesting question is whether fusing beats doing it in two passes.
        bench_gpu_fp8("fp8 quantize only", &launch_quantize_fp8, c, DType::kFp16);
        bench_gpu_fp8("fwht+fp8 fused", &launch_fwht_baseline_fp8, c, DType::kFp16);
        bench_gpu_fp8("tc+fp8 fused", &launch_hadamard_tc_fp8, c, DType::kFp16);
        bench_gpu_fp8_two_stage("tc->fp8 2-stage", &launch_hadamard_tc, &launch_quantize_fp8, c,
                                DType::kFp16);
        std::printf("%-24s %-5s %-16s %10s %10s\n", "", "", "", "", "");
    }

    std::printf("\nnotes: GB/s counts the traffic each row is expected to move. Rows whose\n");
    std::printf("       working set fits in L2 are L2-bound; the last case is DRAM-bound.\n");
    std::printf("       Transform rows move 2+2 B/elem. FP8 rows move 2 B/elem in and 1 B/elem\n");
    std::printf("       out plus 4 B per row of scales, so fusing in 3 B/elem against 7 B/elem\n");
    std::printf("       for the two-pass pipeline should be worth about 7/3 if both are purely\n");
    std::printf("       bandwidth-bound, and less when the transform is math-bound in L2.\n");
    return 0;
}
