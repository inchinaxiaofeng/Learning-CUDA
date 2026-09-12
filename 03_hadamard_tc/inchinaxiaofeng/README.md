# Hadamard 变换加速（Tensor Core）

作业 03 的个人实现工程。目标：在 RTX 4090（sm_89）上实现沿最后一维的 Hadamard 变换，
优先使用 Tensor Core，并把量化 epilogue 融合进同一 kernel。

- 输入：4D 张量 `[batch_size, seq_len, num_heads, head_dim]`，FP16 或 BF16
- `head_dim` 为 2 的幂（64 / 128 / 256 等），非 2 幂尺寸不在范围内
- 输出：变换结果 + kernel 执行时间（ms）
- 正确性门槛：FP16 绝对误差 < 1e-2，BF16 < 5e-2

## 变换约定

统一采用 `y = H_d * x / sqrt(d)`，其中 `H_d` 为 Sylvester 矩阵
（`H_1 = [1]`，`H_2n = [[H_n, H_n], [H_n, -H_n]]`）。`1/sqrt(d)` 使变换正交归一，
因此连续施加两次等于恒等变换——对拍框架直接利用了这条性质。

权重因子必须在实现任何 kernel 之前与参考实现（如 `fast_hadamard_transform`）对齐，
否则误差必然超标。详见 `docs/learning_notes.md` 的「待确认约定」一节。

## 环境

| 项 | 值 |
|---|---|
| GPU | RTX 4090 D（Ada, sm_89, 24 GiB） |
| 驱动 / CUDA | 570.124.06 / CUDA 12.8 |
| nvcc | `/usr/local/cuda/bin/nvcc`（不在默认 PATH） |
| 构建 | cmake >= 3.24、make、g++ |

`nvcc` 不在 PATH 时先导出：

```
export PATH=/usr/local/cuda/bin:$PATH
```

## 构建与运行

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/hw_probe     # M0 探针：设备信息 + hello kernel + WMMA identity 探针
./build/hw_tests     # 对拍：参考实现互校、正交性、精度往返、GPU 实现对比
./build/hw_bench     # benchmark：kernel ms 与有效带宽 GB/s
```

也可以走 ctest：

```
cd build && ctest --output-on-failure
```

覆盖目标架构用 `-DCMAKE_CUDA_ARCHITECTURES=89`（默认已是 89）。

## 目录结构

```
include/hadamard/   common.h（shape/dtype/误差门槛）、cuda_utils.h（错误检查/事件计时）、hadamard.h（公开 API）
src/hadamard_ref.cu CPU FP32 参考：Sylvester 显式 matmul + 蝶形 FWHT + FP16/BF16 打包解包
src/hadamard_gpu.cu GPU 入口：hello kernel、WMMA identity 探针，以及 M2/M3 的接口桩
src/probe_main.cu   M0 探针可执行文件
tests/              对拍框架（GPU 用例在接口桩就绪后自动从 SKIP 变为 PASS/FAIL）
bench/              benchmark 驱动：kernel ms + 有效带宽
docs/               学习笔记与总结报告
```

## 进度

- [x] M0 工程骨架：CMake（sm_89）、clang-format、探针跑通
- [x] M1 CPU FP32 参考实现 + 多 shape 对拍框架（FP16/BF16）
- [ ] M2 共享内存蝶形 FWHT baseline（`launch_fwht_baseline`）
- [ ] M3 WMMA Tensor Core 主实现（`launch_hadamard_tc`）
- [ ] M4 FP8 E4M3 量化融合 epilogue 与一致性验证
- [ ] M5 全 shape benchmark + ncu 分析
- [ ] M6 报告
- [ ] M7 整理与 PR

GPU 实现落地前，`hw_tests` 中相关用例报告 SKIP，`hw_bench` 对应行显示 `n/a`。

## 对拍与 benchmark 约定

- 输入先按目标精度取整，参考实现作用在取整后的输入上，因此误差只反映 kernel 误差，
  不混入输入量化误差。
- 有效带宽按一次读 + 一次写统计（FP16/BF16 为 4 字节/元素）。
- benchmark 使用 CUDA event 计时，GPU 侧 50 次迭代求均值（含 1 次预热）。

## 参考资料

- Tri Dao, *fast-hadamard-transform*（QuaRot 同款 kernel）
- QuaRot / SpinQuant：量化前旋转抑制异常值
- CUDA Samples `simpleTensorCoreGEMM`（WMMA API 用法）
