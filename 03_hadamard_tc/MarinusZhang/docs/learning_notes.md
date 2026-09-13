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

`H_ab = H_a ⊗ H_b`（等价于 `H_n[i,j] = (-1)^popcount(i & j)`，两个因子的下标比特
互不重叠），因此大尺寸变换可以拆成小尺寸分块（例如 `H_128 = H_16 ⊗ H_8`）。

M3 用的是这条性质的一个特例。把下标写成 `i = 16·i1 + i0`、`j = 16·j1 + j0`，则

```
H_d[(i1,i0),(j1,j0)] = (-1)^popcount(i1 & j1) · H_16[i0,j0]
```

也就是**任意 (K tile, N tile) 对应的 B 分块只可能是 ±H_16**，符号由
`(-1)^popcount(k_tile & n_tile)` 给出。这是 Tensor Core 方案可行的前提：smem/寄存器里
只需要常驻一个 16×16 的常量矩阵，而不是 d×d（d = 512 时要 512 KiB，单 SM 装不下）。

落手写 kernel 之前先用仓库外的 host 程序验证过：d = 16…1024 逐元素与递归构造的 `H_d`
比对（1,398,016 次检查全等），再用分块累加复算 `H_d x`，与显式 matmul 完全一致。

> 注：`H_n[i,j] = (-1)^popcount(i & j)` 这个形式对**自然顺序**的 Sylvester 矩阵成立，
> 也是核里直接算符号的依据；换任何行/列置换后要重新推导。

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

### 硬件与 API 能力

- sm_89 支持 FP16 / BF16 / TF32 的 WMMA mma，`nvcuda::wmma` 的 16×16×16 片段；
  FP8 mma 可用，但 wgmma 是 sm_90a 专有，本平台不可用。
- 实测裸吞吐（仓库外的微基准：fragment 常驻寄存器、无访存、4 条独立累加器链）：
  fp16→fp32 为 147.0 TFLOPS，bf16→fp32 为 146.7 TFLOPS，**两种精度相同**。所以 fp16 比
  bf16 慢时不要先怀疑数据格式，去看看代码生成。

### 用到的设计

- 把 `[B,S,H,D]` 展平成 `M = B·S·H` 行、每行长度 D，问题变成 `X (M×D) · H_D (D×D)` 的
  GEMM（K = D）。B 操作数不是真的 d×d 矩阵，而是上一节里的 ±H_16；符号在下标里算。
- block = 16 行 × `min(D, 256)` 列，每个 warp 负责一段列并跑完整的 K 循环。
- A 先 staging 进 shared memory；B 的两个符号变体 preload 进寄存器，K 循环里零访存。
- epilogue：累加器（FP32）→ `store_matrix_sync` 到 FP32 shared memory → 标量缩放 +
  按 16 B 向量写回目标精度。

### wmma 实践要点（踩过的坑）

- `load_matrix_sync` / `store_matrix_sync` 的 `ldm` 必须满足元素个数上的对齐（fp16/bf16 为
  8 的倍数、float 为 4 的倍数），指针要 16 B 对齐。行距取 `D + 8` 而不是 D，可以避免 16 行
  落在同一组 bank 上。
- 累加器片段是 FP32，而 `store_matrix_sync` 只能写与片段同类型的 buffer。输出 fp16/bf16 时
  不能直接存，必须先落 FP32 shared memory 再转换（这块 buffer 可以和 A staging 复用，中间
  用 `__syncthreads()` 隔开）。
- 常量矩阵（如 256 个元素的 H_16）的填充循环要写成跨 `blockDim` 的 strided 循环
  （`for (v = tid; v < 256; v += blockDim.x)`）。写成 `if (tid < 256)` 而 block 不足 256 线程时，
  只有前 `blockDim.x` 个元素被写入，剩下的保持 smem 初值 0，症状是输出出现周期性的错误列。
  诊断这类结构性错误的高效手段是 one-hot 输入：令 `x = e_i`（`rows = 1`），输出就是 kernel
  实际作用的矩阵的第 i 行，一次运行就能看出是哪几行错了。
- **分支 vs 选择**：符号选择写成 `if (sign) mma(neg) else mma(pos)` 时，nvcc 会把两个分支
  都发射成被 `@!P0` 谓词化的 HMMA，并在每次谓词化 mma 后插入 WARPSYNC。实测 fp16 因此比
  bf16 慢 1.5×（SASS 里 fp16 是 16 条谓词化 HMMA + 10 条 WARPSYNC，bf16 只有 8 条无谓词
  HMMA）。改成先按标志选好 fragment（`fb = sign ? fb_pos : fb_neg`）再调一次 mma，两条路径
  就一致了，L2 受限的 shape 提升 1.5~1.7×。
- 写微基准时不要用 fp16/bf16 累加器：整条链会被优化掉（实测报出 1162 TFLOPS 这种明显不可能
  的数字）。用 FP32 累加并把结果写回显存才能得到可信的峰值。
- 排查手段：`nvcc -Xptxas -v` 看寄存器用量与 spill；`cuobjdump -sass` 直接数指令（对上面的
  谓词化问题就是靠这个定位的）。

## 5. 量化基础

- FP8 E4M3：最大可表示值 448，per-token scale = amax / 448。
- INT4：对称量化，scale = amax / 7，按 8 个值打包进 int32。
- 融合要点：旋转 + 量化在同一 kernel 完成，中间结果不落显存；一致性验证要求
  融合结果与「先变换后量化」两段式逐元素完全一致。

## 6. 阅读清单

- [x] Tri Dao, *fast-hadamard-transform*：论文与 CUDA 源码（已读 `csrc/` 下 kernel：
  寄存器内蝶形 + warp shuffle + 跨 warp smem 交换，与本项目 M2/M3 的思路一致）
- [ ] QuaRot、SpinQuant：量化前旋转的动机与实验
- [x] CUDA Samples `simpleTensorCoreGEMM`：WMMA 用法与 fragment 载入（M3 里落地：
  `load_matrix_sync`/`store_matrix_sync`、lead dimension 对齐、累加器只能存 FP32）
- [x] CUDA Programming Guide：shared memory bank conflict、`__syncthreads` 语义（M3 里用到：
  A staging 行距 padding、staging 与 epilogue 复用同一块 buffer 时的同步）
- [x] 工具链实操：`-Xptxas -v` 与 `cuobjdump -sass`（寄存器/spill、指令级排查）
