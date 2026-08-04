#include <cstddef>
#include <cuda_fp16.h>
#include <vector>

#include "../tester/utils.h"

// =====================================================================
// CUDA 核函数辅助类型转换工具 (确保同时完美兼容 float 和 half)
// =====================================================================
template <typename T> __device__ __forceinline__ float to_float(T val) {
  return static_cast<float>(val);
}

template <> __device__ __forceinline__ float to_float<half>(half val) {
  return __half2float(val);
}

template <typename T> __device__ __forceinline__ T from_float(float val) {
  return static_cast<T>(val);
}

template <> __device__ __forceinline__ half from_float<half>(float val) {
  return __float2half(val);
}

// =====================================================================
// RMSNorm CUDA Kernel 实现
// =====================================================================
template <typename T>
__global__ void rmsNormKernel(const T *input, const T *weight, T *output,
                              size_t rows, size_t hidden_dim, float eps) {
  // 每个 Block 负责处理矩阵中的一个 Token (一行)
  size_t i = blockIdx.x;
  if (i >= rows)
    return;

  // 定位当前行的起始指针
  const T *row_input = input + i * hidden_dim;
  T *row_output = output + i * hidden_dim;

  // 动态共享内存，用于 Block 内部线程协同求和 (大小由启动时的第三个参数决定)
  extern __shared__ float sdata[];
  size_t tid = threadIdx.x;

  // 1. 每个线程并行计算自己分到的那一批元素的平方和
  float thread_sum = 0.0f;
  for (size_t j = tid; j < hidden_dim; j += blockDim.x) {
    float val = to_float(row_input[j]);
    thread_sum += val * val;
  }
  sdata[tid] = thread_sum;
  __syncthreads(); // 等待全块线程完成局部平方和写入

  // 2. 块内折半规约 (Block Reduction)：将所有线程的和累加到 sdata[0]
  // 保证 blockDim.x 是 2 的幂次（这里固定为 256），此逻辑绝对安全
  for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // 3. 由 0 号线程算出这一行的 rsqrt 值，并共享给全块
  __shared__ float rsqrt_val;
  if (tid == 0) {
    float mean_square = sdata[0] / hidden_dim;
    rsqrt_val = rsqrtf(mean_square + eps); // 使用 CUDA 硬件加速的 rsqrtf 指令
  }
  __syncthreads(); // 等待 rsqrt_val 计算并同步完毕

  // 4. 所有线程再次并行，计算当前行每个元素的最终缩放值并写回
  for (size_t j = tid; j < hidden_dim; j += blockDim.x) {
    float val = to_float(row_input[j]);
    float w = to_float(weight[j]);
    row_output[j] = from_float<T>(val * rsqrt_val * w);
  }
}

/**
 * @brief Computes RMSNorm over the last dimension of a 2D tensor.
 *
 * The input is a row-major matrix with shape [rows, hidden_dim]. For each row
 * i and column j:
 *
 *   output[i, j] = input[i, j] * rsqrt(mean(input[i, :]^2) + eps) * weight[j]
 *
 * The output vector is preallocated with rows * hidden_dim elements.
 *
 * @tparam T Data type of input, weight, and output tensors.
 * @param[in] h_input Flattened input matrix of shape [rows, hidden_dim].
 * @param[in] h_weight Per-column scale vector of shape [hidden_dim].
 * @param[out] h_output Flattened output matrix of shape [rows, hidden_dim].
 * @param[in] rows Number of rows/tokens.
 * @param[in] hidden_dim Size of the normalized dimension.
 * @param[in] eps Numerical stability epsilon.
 */
template <typename T>
void rmsNorm(const std::vector<T> &h_input, const std::vector<T> &h_weight,
             std::vector<T> &h_output, size_t rows, size_t hidden_dim,
             float eps) {
  // 1. 定义 Device 端的裸指针
  T *d_input = nullptr;
  T *d_weight = nullptr;
  T *d_output = nullptr;

  size_t input_size = rows * hidden_dim * sizeof(T);
  size_t weight_size = hidden_dim * sizeof(T);

  // 2. 分配 GPU 显存
  cudaMalloc(&d_input, input_size);
  cudaMalloc(&d_weight, weight_size);
  cudaMalloc(&d_output, input_size);

  // 3. 将数据从 Host (CPU) 拷贝到 Device (GPU)
  cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_weight, h_weight.data(), weight_size, cudaMemcpyHostToDevice);

  // 4. 配置配置网格和线程块尺寸
  // 固定使用 256 线程，它是 2 的幂次，能完美支持 Kernel 内部的折半规约
  unsigned int threads_per_block = 256;
  unsigned int blocks_per_grid = rows; // 有多少行就启动多少个 Block
  size_t shared_mem_size = threads_per_block * sizeof(float);

  // 5. 启动 CUDA Kernel
  rmsNormKernel<T><<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
      d_input, d_weight, d_output, rows, hidden_dim, eps);

  // 6. 将计算结果从 GPU 捞回预先分配好的 h_output 中
  cudaMemcpy(h_output.data(), d_output, input_size, cudaMemcpyDeviceToHost);

  // 7. 善后处理：释放显存防止内存泄漏
  cudaFree(d_input);
  cudaFree(d_weight);
  cudaFree(d_output);
}

// =====================================================================
// Flash Attention CUDA Kernel 实现 (online-softmax / 单遍分块流式)
// =====================================================================
// 每个 block 负责一个输出行 (b, t, h);对 K/V 按 tile(大小 = blockDim)
// 流式扫描,维护 running max(m)/running sum(l)/running 输出累加(acc),
// 用校正因子 alpha = exp(m_old - m_new) 把旧累加量搬到新基准,
// 从而单遍完成且数值稳定(任意时刻 exp 参数 <= 0,不溢出)。
// 线程分工:线程 tid 负责本 tile 的 key j0+tid 的打分;同时若 tid<d
// 则负责输出通道 dd=tid 的累加(acc 存于寄存器)。要求 d <= blockDim。
template <typename T>
__global__ void flashAttentionKernel(const T *q, const T *k, const T *v, T *o,
                                     int tgt_len, int src_len, int q_heads,
                                     int kv_heads, int d, bool is_causal) {
  int b = blockIdx.x;
  int t = blockIdx.y;
  int h = blockIdx.z;
  int hkv = h / (q_heads / kv_heads); // GQA:多个 query head 共享一个 kv head
  float scale = 1.0f / sqrtf((float)d); // 1/sqrt(d)(标准正确舍入版)
  int tid = threadIdx.x, nthreads = blockDim.x;

  const T *q_row = q + (((size_t)b * tgt_len + t) * q_heads + h) * d;
  T *o_row = o + (((size_t)b * tgt_len + t) * q_heads + h) * d;

  // 共享内存布局: sh_q[d] | sh_p[nthreads] | sh_red[nthreads]
  extern __shared__ float smem[];
  float *sh_q = smem;
  float *sh_p = sh_q + d;
  float *sh_red = sh_p + nthreads;

  // 把 query 行缓存到共享内存(每个 tile 都要用,避免重复读全局)
  for (int i = tid; i < d; i += nthreads)
    sh_q[i] = to_float(q_row[i]);
  __syncthreads();

  // running 状态:m/l 每个线程各持一份相同副本;acc 每线程负责通道 dd=tid
  float m = -INFINITY, l = 0.f, acc = 0.f;

  // causal: 只需扫到 j<=t;否则扫到 src_len
  int jend = src_len;
  if (is_causal && t + 1 < jend)
    jend = t + 1;

  for (int j0 = 0; j0 < jend; j0 += nthreads) {
    int j = j0 + tid;

    // (1) 本线程负责 key j 的打分 s = (q·k_j) * scale;越界/被 mask 记 -inf
    float s = -INFINITY;
    if (j < jend) {
      const T *k_row = k + (((size_t)b * src_len + j) * kv_heads + hkv) * d;
      float dot = 0.f;
      for (int dd = 0; dd < d; dd++)
        dot += sh_q[dd] * to_float(k_row[dd]);
      s = dot * scale;
    }

    // (2) tile 内最大值(折半归约)
    sh_red[tid] = s;
    __syncthreads();
    for (int r = nthreads / 2; r > 0; r >>= 1) {
      if (tid < r)
        sh_red[tid] = fmaxf(sh_red[tid], sh_red[tid + r]);
      __syncthreads();
    }
    float tile_max = sh_red[0];
    __syncthreads();

    // (3) 更新 running max,并算校正因子 alpha = exp(m_old - m_new)
    float m_new = fmaxf(m, tile_max);
    float alpha = __expf(m - m_new);

    // (4) 本 key 的 exp 权重(相对新基准 m_new)
    float p = (j < jend) ? __expf(s - m_new) : 0.f;
    sh_p[tid] = p;

    // (5) tile 内权重和(折半归约)
    sh_red[tid] = p;
    __syncthreads();
    for (int r = nthreads / 2; r > 0; r >>= 1) {
      if (tid < r)
        sh_red[tid] += sh_red[tid + r];
      __syncthreads();
    }
    float tile_sum = sh_red[0];
    __syncthreads();

    // (6) 更新归一化分母: l = alpha*l + 本 tile 权重和
    l = alpha * l + tile_sum;

    // (7) 更新输出累加:线程 tid 负责通道 dd=tid
    if (tid < d) {
      float delta = 0.f;
      int cnt = jend - j0;
      if (cnt > nthreads)
        cnt = nthreads;
      for (int jj = 0; jj < cnt; jj++) {
        const T *v_row =
            v + (((size_t)b * src_len + (j0 + jj)) * kv_heads + hkv) * d;
        delta += sh_p[jj] * to_float(v_row[tid]);
      }
      acc = alpha * acc + delta;
    }
    m = m_new;
    __syncthreads(); // 复用 sh_p/sh_red 前同步
  }

  // (8) 归一化写回
  if (tid < d)
    o_row[tid] = from_float<T>(acc / l);
}

// Hidden_dim == num_heads * head_dim.
// query_heads和kv_heads是否相同，则决定了head_dim的大小
/**
 * @brief Computes flash attention for given query, key, and value tensors.
 *
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads,
 * head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads,
 * head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads,
 * head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len,
 * query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query
 * attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T> &h_q, const std::vector<T> &h_k,
                    const std::vector<T> &h_v, std::vector<T> &h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim,
                    bool is_causal) {
  // 和 rmsNorm 一模一样的套路，只是换成 4 个张量：
  T *d_q, *d_k, *d_v, *d_o;
  size_t q_size =
      (size_t)batch_size * target_seq_len * query_heads * head_dim * sizeof(T);
  size_t kv_size =
      (size_t)batch_size * src_seq_len * kv_heads * head_dim * sizeof(T);
  // cudaMalloc x4 → cudaMemcpy q/k/v → launch → memcpy 回 h_o → cudaFree x4
  cudaMalloc(&d_q, q_size);
  cudaMalloc(&d_k, kv_size);
  cudaMalloc(&d_v, kv_size);
  cudaMalloc(&d_o, q_size);

  cudaMemcpy(d_q, h_q.data(), q_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_k, h_k.data(), kv_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, h_v.data(), kv_size, cudaMemcpyHostToDevice);

  // 启动配置：一个 block 负责一个输出行 (b, t, h)
  // 三维网格，天然映射, 剩下的一个就是head dim
  dim3 grid(batch_size, target_seq_len, query_heads);
  int threads = 128;
  // 共享内存: sh_q[head_dim] + sh_p[threads] + sh_red[threads]
  size_t shmem = ((size_t)head_dim + 2 * threads) * sizeof(float);
  flashAttentionKernel<T><<<grid, threads, shmem>>>(
      d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, query_heads, kv_heads,
      head_dim, is_causal);

  cudaMemcpy(h_o.data(), d_o, q_size, cudaMemcpyDeviceToHost);
  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_v);
  cudaFree(d_o);
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template void rmsNorm<float>(const std::vector<float> &,
                             const std::vector<float> &, std::vector<float> &,
                             size_t, size_t, float);
template void rmsNorm<half>(const std::vector<half> &,
                            const std::vector<half> &, std::vector<half> &,
                            size_t, size_t, float);
template void flashAttention<float>(const std::vector<float> &,
                                    const std::vector<float> &,
                                    const std::vector<float> &,
                                    std::vector<float> &, int, int, int, int,
                                    int, int, bool);
template void flashAttention<half>(const std::vector<half> &,
                                   const std::vector<half> &,
                                   const std::vector<half> &,
                                   std::vector<half> &, int, int, int, int, int,
                                   int, bool);
