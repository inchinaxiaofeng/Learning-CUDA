# 作业 03：Hadamard 变换加速（Tensor Core）实现报告

> 骨架文件，随里程碑推进补全。

## 1. 任务理解

沿最后一维对 `[B, S, H, D]` 做 Hadamard 变换，D 为 2 的幂；要求 FP16 绝对误差 < 1e-2、
BF16 < 5e-2，并与量化算子融合。详见 `03_hadamard_tc/README.md`。

## 2. 实现思路

### 2.1 变换约定

（待填：归一化因子与输出顺序如何与参考实现对拍确认）

### 2.2 非 Tensor Core baseline

（待填：shared memory 蝶形 FWHT，行分块策略，向量化访存）

### 2.3 Tensor Core 主实现

（待填：展平为 GEMM 的形式化描述，H_D 常量矩阵的存放与复用，WMMA tile
循环，shared memory staging 与 bank conflict 处理，epilogue 缩放）

### 2.4 量化融合

（待填：FP8 E4M3 per-token 量化的 epilogue 实现，scale 计算与写回，可选 INT4 打包）

## 3. 优化历程与踩坑

（待填：按时间顺序记录每次优化的动机、改动、收益；记录失败尝试与原因）

## 4. 性能指标与分析

（待填：CPU 参考 / 蝶形 baseline / TC / 融合 四路对比表，含 kernel ms 与有效带宽
GB/s；ncu 的 roofline 与瓶颈分析结论）

## 5. 正确性验证

（待填：对拍方法、覆盖的 shape 与 dtype、实测最大绝对误差与门槛对比、
融合 vs 两段式一致性结果）

## 6. 未来工作

（待填）
