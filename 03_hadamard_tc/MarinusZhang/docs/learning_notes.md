# 学习笔记

## 1. Hadamard 矩阵

### Sylvester 构造

```
H_1 = [1]
H_2n = [[H_n,  H_n],
        [H_n, -H_n]]
```

即 `H_d = H_2 ⊗ H_2 ⊗ ... ⊗ H_2`（`log2(d)` 个因子），元素只有 ±1。

### 正交性

`H_d · H_d^T = d · I_d`。因此 `y = H_d x / sqrt(d)` 是正交变换，且
`H_d/sqrt(d)` 的逆就是自身：连续两次变换回到原向量。对拍框架中的
`test_orthogonality` 直接依赖这条性质来发现符号、顺序或归一化错误。

### Kronecker 分解

`H_ab = H_a ⊗ H_b`。这意味着大尺寸变换可以拆成小尺寸分块（例如
`H_128 = H_16 ⊗ H_8`），是 Tensor Core 方案里选择分块粒度的依据之一。

## 2. 蝶形 FWHT

`H_n x` 可以用 `log2(n)` 个阶段实现，每阶段把长度为 `len` 的两个块做
`(a+b, a-b)`：

```
for len in {1, 2, 4, ..., d/2}:
    for each pair (i, i+len) within blocks of size 2*len:
        (y[i], y[i+len]) <- (y[i] + y[i+len], y[i] - y[i+len])
```

复杂度 O(d log d)，无位反转（自然顺序输出与 Sylvester 构造一致）。
GPU 版本把行数据放进 shared memory，每行一个 warp 或按行分块即可。

## 3. 待确认约定

- **归一化因子**：本项目统一用 `1/sqrt(d)`。`fast_hadamard_transform` 等库的默认
  行为需要实测确认（有实现默认不做归一化）。对拍前必须先定下来，否则误差必然超标。
- **输出顺序**：自然顺序 vs 位反转顺序。若参考库输出位反转顺序，需要额外置换。

## 4. Tensor Core 相关

- sm_89 支持 FP16 / BF16 / TF32 的 WMMA mma，`nvcuda::wmma` 的 16×16×16 片段；
  FP8 mma 可用，但 wgmma 是 sm_90a 专有，本平台不可用。
- 主实现思路：把 `[B,S,H,D]` 展平成 `M = B·S·H` 行、每行长度 D，问题变成
  `X (M×D) · H_D (D×D)` 的 GEMM（K = D），H_D 为 ±1 常量，可常驻 shared memory。
- 关键优化点：shared memory staging 布局与 bank conflict、`load_matrix_sync` 的对齐
  要求（ldm 需满足 128-bit 对齐）、tile 尺寸与 occupancy、epilogue 里的 1/sqrt(D) 缩放。

## 5. 量化基础

- FP8 E4M3：最大可表示值 448，per-token scale = amax / 448。
- INT4：对称量化，scale = amax / 7，按 8 个值打包进 int32。
- 融合要点：旋转 + 量化在同一 kernel 完成，中间结果不落显存；一致性验证要求
  融合结果与「先变换后量化」两段式逐元素完全一致。

## 6. 阅读清单

- [ ] Tri Dao, *fast-hadamard-transform*：论文与 CUDA 源码
- [ ] QuaRot、SpinQuant：量化前旋转的动机与实验
- [ ] CUDA Samples `simpleTensorCoreGEMM`：WMMA 用法与 fragment 载入
- [ ] CUDA Programming Guide：shared memory bank conflict、`__syncthreads` 语义
