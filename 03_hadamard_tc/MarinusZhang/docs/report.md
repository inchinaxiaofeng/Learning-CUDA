# 作业 03：Hadamard 变换加速（Tensor Core）实现报告

> 骨架文件，随里程碑推进补全；M2（非 Tensor Core baseline）已落地。

## 1. 任务理解

沿最后一维对 `[B, S, H, D]` 做 Hadamard 变换，D 为 2 的幂；要求 FP16 绝对误差 < 1e-2、
BF16 < 5e-2，并与量化算子融合。详见 `03_hadamard_tc/README.md`。

## 2. 实现思路

### 2.1 变换约定（已与参考实现对齐）

做法：clone `Dao-AILab/fast-hadamard-transform`，先读源码把口径定下来，再编译安装它的 CUDA
扩展做数值对拍。

结论：

- 该库把缩放作为参数传入 kernel（`params.scale`），在写回时统一相乘；其 README 与自带测试
  统一取 `scale = 1/sqrt(dim)`，并声明等价于 `F.linear(x, scipy.linalg.hadamard(dim)) * scale`。
- 所以本项目的 `y = H_d x / sqrt(d)`（Sylvester 构造、自然顺序输出）与参考实现一致，不需要
  额外的归一化修正或输出置换。
- 该库对非 2 的幂尺寸内部补零到下一个 2 的幂；本项目按题面只支持 2 的幂。

### 2.2 非 Tensor Core baseline（M2）

实现位于 `src/hadamard_gpu.cu`，核心是把 `[B,S,H,D]` 展平成 `M = B·S·H` 行、行长 `D` 的二维
数组，再做蝶形 FWHT。

- **线程映射**：一行由同一 warp 内的 T 个线程协作，线程 `lane` 连续持有 `E = D / T` 个元素。
  E 以 8 为优选值（fp16/bf16 下正好一个 128-bit 向量）；`D > 8·32` 时改为增大 E 而不是 T，
  因为 shuffle 的分组不能跨 warp。`d = 2` 退化为 T = 1、E = 2。
- **蝶形**：第 `s` 级步长 `len = 2^s`。`len < E` 时配对的两个下标落在同一线程的寄存器里
  （线程持有的块按 E 对齐，异或不会跨块），直接寄存器加减；`len >= E` 时配对线程为
  `lane ^ (len / E)`、元素下标不变，用一次 `__shfl_xor_sync` 交换即可，符号折进求和
  （`v = sign·v + other`）。
- **零 shared memory**：上面两级合起来覆盖全部 `log2(D)` 级，所以 kernel 不碰 shared memory、
  不需要 `__syncthreads()`，一行只过一遍显存（一次读、一次写），理论上限就是 memcpy。
- **精度**：全程 FP32，仅在写回时按目标精度舍入一次；`1/sqrt(D)` 折进 epilogue，避免每级都
  引入舍入误差（否则半精度中间量的误差会直接吃掉 1e-2 的容差预算）。
- **访存**：按每线程 run 宽度选择 16/8/4/2 字节向量；行内 T 个线程连续访存，天然完全合并。
- **尾部处理**：最后一块中越界的线程不提前 return（否则 shuffle 的 mask 与配对会失效），
  而是钳到最后一个有效行继续计算、只跳过写回。
- **launch 粒度**：以每 SM 约 8 个 block 为目标反推 block 持有多少行，向上取整到整 warp，
  上限 256 线程，兼顾小 shape（行数少）与大 shape 的并行度。
- **支持范围**：`head_dim` 为 2 的幂且不超过 1024；其余尺寸返回 `false`，harness 报 SKIP。

### 2.3 Tensor Core 主实现

（待填：展平为 GEMM 的形式化描述，H_D 常量矩阵的存放与复用，WMMA tile
循环，shared memory staging 与 bank conflict 处理，epilogue 缩放）

### 2.4 量化融合

（待填：FP8 E4M3 per-token 量化的 epilogue 实现，scale 计算与写回，可选 INT4 打包）

## 3. 优化历程与踩坑

### 3.1 设计取舍：直接落地寄存器方案

原计划按「v0 逐级 shared memory 蝶形 → v1 向量化访存 → v2 shuffle 换 len < 32 → v3 每线程多元素」
的阶梯实现。读了参考实现的源码后确认：对本作业的 `D ≤ 512`，一行的全部数据在 32 个线程的
寄存器里就能装下（`32 × 8 × 2 B = 512 B`），于是 `len < E` 用寄存器加减、`len ≥ E` 用 warp
shuffle 的组合可以覆盖全部蝶形级次，根本不需要 shared memory。因此直接落地了这一版，
把 smem 版本留给 M3 里可能出现的跨 warp 交换场景。收益体现在实测上：DRAM 受限的大 shape
达到 943 GB/s（理论峰值的 93.6%），kernel 时间已经贴近 memcpy 这一上限（见第 4 节）。

### 3.2 踩坑 1：参考实现的越界 bug（M1）

CPU 参考里的 `build_sylvester` 用元素个数 `n·n` 当作矩阵维度来判断何时退出展开循环，
`d > 2` 时索引远远越过 buffer：`d = 128` 的最大误差 > 1e17、`d = 256` 直接段错误，而蝶形
参考实现却是对的。定位手法：利用解析性质「全 1 输入经 `y = H x / sqrt(d)` 应只有第 0 个
元素为 `sqrt(d)`、其余为 0」，一次就断定问题出在 matmul 参考而不是蝶形实现。修复后
`d = 2 … 512` 的两套参考实现互相吻合到 1e-6。

### 3.3 踩坑 2：host-only 的转换函数

`TypeTraits::to_float/from_float` 原本只在 host 侧（WMMA 探针）使用，kernel 里调用后报
`calling a __host__ function from a __device__ function`。补上 `__host__ __device__` 后
host/device 两侧共用同一套 fp16/bf16 转换。

### 3.4 踩坑 3：benchmark 带宽超过 HBM 峰值

首次实测小 shape 得到 1383–2694 GB/s，超过 4090 D 的 1008 GB/s 理论峰值。原因是 benchmark
对同一块 0.4–16 MiB 的张量重复 50 次迭代，输入输出全部落在 72 MiB 的 L2 里，测到的是 L2
带宽。核实手段：另写一个仓库外的临时程序，用 256 MiB 流量的大 shape 复测，得到 DRAM 受限的
943 GB/s；随后把该大 shape 加进 `bench_hadamard.cu`，让 benchmark 同时覆盖 L2 与 DRAM 两种
情形，并在输出里注明哪一行是 DRAM-bound。

## 4. 性能指标与分析

RTX 4090 D（sm_89，114 SM，1008 GB/s HBM 峰值），`hw_bench`，50 次迭代取均值，
有效带宽按「一次读 + 一次写」计。

| shape (B,S,H,D) | 流量 | CPU fp32 参考 | M2 fp16 | fp16 带宽 | 相对 CPU | M2 bf16 带宽 |
|---|---|---|---|---|---|---|
| [1,128,12,64] | 0.4 MiB | 0.303 ms | 0.0022 ms | 176 GB/s | 137× | 178 GB/s |
| [2,256,16,128] | 4 MiB | 3.842 ms | 0.0030 ms | 1403 GB/s | 1281× | 1403 GB/s |
| [1,512,8,256] | 4 MiB | 3.318 ms | 0.0032 ms | 1313 GB/s | 1037× | 1313 GB/s |
| [4,512,16,128] | 16 MiB | 12.971 ms | 0.0062 ms | 2695 GB/s | 2092× | 2695 GB/s |
| [1,4096,12,64] | 12 MiB | 9.676 ms | 0.0050 ms | 2518 GB/s | 1935× | 2528 GB/s |
| [8,4096,16,128] | 256 MiB | 227.633 ms | 0.2846 ms | 943 GB/s | 800× | 943 GB/s |

分析：

- 前五行的工作集（≤ 16 MiB）在 50 次迭代中一直驻留在 72 MiB L2，因此报出的是 L2 带宽，
  数值高于 HBM 峰值属于正常现象；真正衡量 kernel 上限的是最后一行。
- 最后一行 256 MiB 流量下为 943 GB/s = 理论峰值的 **93.6%**，说明 kernel 已经是访存受限，
  每元素 `log2(D)` 次加减的计算被访存完全掩盖，达到了「时间 ≈ memcpy」这一非 Tensor Core
  实现的理论上限（参考实现 README 对 fp16/bf16 `dim ≤ 512` 也是这个结论）。
- 最小的 shape（0.4 MiB）只有 176 GB/s：2.2 µs 里大部分是 kernel 启动与尾部延迟，
  不是带宽问题；这类 shape 的优化方向是减少 launch 与尾部开销，而不是改蝶形。
- fp16 与 bf16 的耗时基本相同，符合两者带宽一致、计算都被掩盖的预期。

（待补：M3 Tensor Core 与 M4 融合量化两路对比；ncu 的 DRAM 吞吐 / occupancy / stall 采集）

## 5. 正确性验证

- **对拍口径**：输入先量化到目标精度，参考值在该量化输入上计算，因此误差只反映 kernel 误差，
  不混入输入量化误差。
- **CPU 参考自洽**：Sylvester 显式 matmul 与蝶形 FWHT 两条独立实现互相吻合，
  `d = 2/64/128/256/512` 最大差 2e-6；正交性检查（连续做两次变换应回到原值）对全部尺寸
  通过，可捕获符号、顺序与归一化错误。
- **GPU 对拍**：`hw_tests` 中 10 个 GPU 用例（fp16/bf16 × `d = 2/64/128/256/512`）全部 PASS。
  实测最大绝对误差：fp16 ≤ 7.4e-4（门槛 1e-2），bf16 ≤ 7.6e-3（门槛 5e-2），余量均在 10 倍以上。
- **与官方实现对拍**：把官方 `fast_hadamard_transform` kernel 消费的同一份 fp16/bf16 输入喂给
  我们的 kernel，`d = 2/64/128/256/512` 全部逐元素 **bit-exact（max|diff| = 0）**。这说明两者的
  中间精度与舍入方式也完全一致（双方都是 FP32 累加、只在写回时舍入一次）。

（待补：M4 融合量化 vs 两段式的一致性验证）

## 6. 未来工作

- **M3 Tensor Core**：把问题写成 `X (M×D) · H_D (D×D)` 的 GEMM，`H_D` 为 ±1 常量矩阵，
  用 WMMA（`d = 128/256` 下 K = D 需要拆成多个 16×16×16 片段）；与 M2 对比时要注意两者
  算法不同，比较的是端到端时间与有效带宽。
- **M4 融合量化**：FP8 E4M3 per-token 量化的 epilogue，验证与「先变换后量化」逐元素一致。
- **更大的 head_dim**：超过 1024 后每线程寄存器装不下一行，需要跨 warp 的 shared memory
  交换（参考实现的做法）或分块策略。
- **非 2 的幂**：按参考实现补零到下一个 2 的幂。
- **小 shape 的 launch 开销**：`[1,128,12,64]` 这类只有 2 µs 的场景，可考虑合并多行/多次调用、
  持久化 kernel 或 CUDA Graph。
- **ncu / nsys**：采集大 shape 的 DRAM 吞吐、occupancy 与 stall 原因，为报告补充 roofline 分析。
