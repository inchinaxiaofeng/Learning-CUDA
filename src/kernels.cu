#include <cstddef>
#include <cuda_fp16.h>
#include <vector>

#include "../tester/utils.h"

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
  // TODO: Implement the rmsNorm function

  // 遍历 rows(batch_size*seq_len)
  for (size_t i = 0; i < rows; i++) {
    float square_sum = 0.0f;
    // 1. 在第 i 行内，先计算所有元素的平方和 (还原你的 mean([i, :]^2) 逻辑)
    for (size_t j = 0; j < hidden_dim; j++) {
      float val = static_cast<float>(h_input[i * hidden_dim + j]);
      square_sum += val * val; // 自乘代替 ^2
    }
    // 2. 计算均方根的倒数 (还原你的 rsqrt(mean + eps) 逻辑)
    float mean_square = square_sum / hidden_dim;
    float rsqrt_val = 1.0f / std::sqrt(mean_square + eps);
    // 内层循环：更新当前行的每一个元素，应用缩放
    for (size_t j = 0; j < hidden_dim; j++) {
      size_t idx = i * hidden_dim + j;
      // 注意这里是 h_weight[j]；统一转 float 计算，避免 half 重载歧义
      h_output[idx] =
          static_cast<T>(static_cast<float>(h_input[idx]) * rsqrt_val *
                         static_cast<float>(h_weight[j]));
    }
  }
}

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
  // TODO: Implement the flash attention function
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
