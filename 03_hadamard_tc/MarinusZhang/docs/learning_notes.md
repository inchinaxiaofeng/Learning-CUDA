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

### GPU 映射：全部留在寄存器

把一行长度 D 交给同一 warp 内的 T 个线程，线程 `lane` 持有连续区间
`[lane·E, lane·E + E)`，其中 `E = D / T`。蝶形第 `s` 级步长 `len = 2^s`：

- `len < E`：`i` 与 `i ^ len` 落在同一线程的寄存器里（线程持有的块按 E 对齐，
  异或不会跨块），直接做寄存器加减；
- `len >= E`：配对线程是 `lane ^ (len / E)`，元素下标不变，用一次
  `__shfl_xor_sync` 交换即可，符号可以折进求和：`v = sign·v + other`。

于是 `log2(D)` 级全部在寄存器与 warp shuffle 内完成：不需要 shared memory，
也不需要 `__syncthreads()`，一行只会过一遍显存（一次读、一次写）。

E 的取值以 8 为优选（fp16/bf16 下正好是一个 128-bit 向量），D > 8·32 时改为
增大 E 而不是 T，因为 shuffle 的分组不能跨 warp。

## 3. 已确认的约定（实测）

对参考实现 `Dao-AILab/fast-hadamard-transform` 做了源码 + 数值两级确认：

- 该库的 kernel 把缩放作为参数（`params.scale`）在写回时统一乘上去；其 Python 封装、
  README 与自带测试统一用 `scale = 1/sqrt(dim)`，并声明等价于
  `F.linear(x, scipy.linalg.hadamard(dim)) * scale`。
- 即 **`y = H_d x / sqrt(d)`，Sylvester 构造，自然顺序输出**，与本项目约定一致。
- 非 2 的幂：该库内部补零到下一个 2 的幂；本项目按题面只支持 2 的幂。
- 数值验证：编译安装该库后，把它消费的同一份 fp16/bf16 输入喂给我们 M2 的 kernel，
  `d = 2/64/128/256/512` 全部逐元素 **bit-exact（max|diff| = 0）**。

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

- [x] Tri Dao, *fast-hadamard-transform*：论文与 CUDA 源码（已读 `csrc/` 下 kernel：
  寄存器内蝶形 + warp shuffle + 跨 warp smem 交换，与本项目 M2/M3 的思路一致）
- [ ] QuaRot、SpinQuant：量化前旋转的动机与实验
- [ ] CUDA Samples `simpleTensorCoreGEMM`：WMMA 用法与 fragment 载入
- [ ] CUDA Programming Guide：shared memory bank conflict、`__syncthreads` 语义
