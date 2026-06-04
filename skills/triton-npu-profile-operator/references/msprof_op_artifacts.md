---
name: triton-npu-diagnose-operator
description: msprof op 单算子诊断。从 hardware pipe 时间、DMA 指令计数、UB 带宽、管线冲突等硬件计数器出发，定位 Ascend NPU 算子的性能瓶颈根因。
---

# msprof op 算子诊断

## 定位

`msprof` 回答「哪个算子慢」，`msprof op` 回答「算子内部**为什么**慢」。

| 维度 | msprof | msprof op |
|------|--------|-----------|
| 粒度 | 算子级聚合 | **per-block 展开** |
| 输出 | time / ratio / cycles | **+ DMA 指令计数 + UB 带宽 + 管线冲突 + Cache 命中** |
| 使用场景 | 多算子全链路，找 hot kernel | 单算子深度诊断，追指令级根因 |
| 开销 | 低 (Level1) | 高 (CSV 大) |

---

## 什么时候用 msprof op

### 应该用的场景

- `msprof` 报告显示某个 kernel 占比高，但不知道瓶颈在哪条管线上
- 需要区分 **scalar 瓶颈 vs vector 瓶颈 vs DMA 瓶颈**——这是后续优化方向选择的关键分叉口
- 需要验证某个优化是否**减少了指令数**还是只改变了**指令分布**
- 做 ablation 实验时需要硬件计数器级别的证据
- 怀疑存在**管线冲突**或 **bank conflict**，但 `op_statistic` 层面看不出
- 推理场景下算子延迟在微秒级，需要排查**头开销**（核启动、TLB miss、同地址访问冲突）
- 怀疑 Block Dim 没有用满物理核，需要验证 **Tiling 配置是否合理**

### 绝对不要用的场景

- 做多算子横向对比——那用 `msprof` 就足够
- 跑所有 case——选 1–2 个代表性 case 即可
- 在 optimize round 的每一次迭代都跑——只在 IR 证据不够时才用
- 还没有看过 `msprof` 的 `op_statistic` 就直接上 `msprof op`——先确认哪个 kernel 慢再说

---

## 命令

### 基础用法

```bash
msprof op \
  --kernel-name=<KERNEL_NAME> \
  --output=<OUTPUT_DIR> \
  --application=<APP> \
  [app_arguments]
```

具体到 Triton 算子的 benchmark 场景：

```bash
msprof op \
  --kernel-name=<KERNEL_NAME> \
  --output=<OUTPUT_DIR> \
  python3 <BENCH.PY> \
  --operator-file <OPERATOR.py> \
  --bench <BENCH_NUM>
```

### 指定采集指标

msprof op 通过 `--aic-metrics` 控制采集哪些硬件计数器。不同指标对应不同的输出 CSV 文件：

| --aic-metrics 值 | 采集内容 | 对应输出 CSV |
|---|---|---|
| `Default` | 全部默认指标（推荐） | PipeUtilization + ArithmeticUtilization + Memory + MemoryUB + MemoryL0 + L2Cache + ResourceConflictRatio |
| `BasicInfo` | 仅算子基础信息 | OpBasicInfo |
| `PipeUtilization` | 管道利用率（快，数据量小） | PipeUtilization |
| `ArithmeticUtilization` | 算术单元利用率 | ArithmeticUtilization |
| `Memory` | DMA 指令计数 + UB 数据量 | Memory |
| `MemoryUB` | UB 带宽 | MemoryUB |
| `MemoryL0` | L0 带宽 | MemoryL0 |
| `L2Cache` | L2 Cache 命中率 | L2Cache |
| `ResourceConflictRatio` | 管线冲突/等待比例 | ResourceConflictRatio |
| `Roofline` | Roofline 瓶颈分析（含 Default） | 供 MindStudio Insight 生成 Roofline 图 |
| `TimelineDetail` | 指令流水 + 代码热点 | 供 MindStudio Insight 可视化 |
| `Occupancy` | 核间负载分析 | 供 MindStudio Insight 可视化 |
| `MemoryDetail` | L2 Cache-L0 连线 + 动态带宽 | 供 MindStudio Insight 可视化 |
| `Source` | 算子代码热点图 | 供 MindStudio Insight 可视化 |
| `KernelScale` | 指定代码段范围的性能指标 | 按需配合 MetricsProfStart/Stop 接口 |

```bash
# 只看管道利用率（数据量最小，最快）
msprof op --aic-metrics=PipeUtilization ...

# Roofline 分析（自动包含 Default）
msprof op --aic-metrics=Roofline ...

# 多个指标用逗号拼接
msprof op --aic-metrics=PipeUtilization,Memory,ResourceConflictRatio ...
```

### 其他关键参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--kernel-name` | 算子名，支持 `"add\|abs"` 拼接和 `*` 通配 | 第一个被调度的算子 |
| `--launch-count` | 最多采集多少个 kernel | 1 |
| `--warm-up` | 预热次数（提频，避免降频影响数据） | 5 |
| `--replay-mode` | 重放模式：kernel / application / range | kernel |
| `--kill` | 采集完成后自动停止程序 | off |
| `--output` | 性能数据输出路径 | 当前目录 |
| `--core-id` | 指定解析部分逻辑核（如 `"0\|31"`） | 全部核 |

---

## 输出文件详解

每次运行在 `--output/<OPPROF_*/` 下生成：

```
OPPROF_XXXXXXXX/
├── OpBasicInfo.csv           # 算子名、Task Duration、Block Dim、频率
├── PipeUtilization.csv       # ★ 每 block 的管道时间 → 瓶颈管道
├── ArithmeticUtilization.csv # ★ Cube/Vector 算术单元利用率、FP/INT 指令数 → 有效算力
├── Memory.csv                # ★ DMA 指令计数 + UB 数据量 → DMA 固定开销
├── MemoryUB.csv              # ★ UB 带宽(GB/s) → 带宽利用率
├── MemoryL0.csv              # L0 带宽
├── L2Cache.csv               # L2 Cache 命中率
├── ResourceConflictRatio.csv # ★ 管线等待/冲突比例 → 下游瓶颈
├── visualize_data.bin        # 可视化数据（可用 MindStudio Insight 打开）
└── dump/                     # 二进制 dump
```

标注 ★ 的是定位瓶颈必看的 5 个文件。

---

## 优化流程

### 前置：理论值估算

在解读 profiling 数据之前，先算出理论上下限，建立合理的优化目标：

**搬运流水的理论耗时** = 搬运数据量 (Byte) / 理论带宽

例如：910B3 的 GM 峰值带宽约 1.6 TB/s，搬运 4096×4096 的 float 矩阵：
`sizeof(float) × 4096 × 4096 / 1.6 TB/s ≈ 41.9 μs`

> 注意：MTE2/MTE3 同时读写时共享带宽，实际搬运时间 ≈ (MTE2 搬运量 + MTE3 搬运量) / GM 带宽。

**计算流水的理论耗时** = 计算数据量 (Element) / 理论算力

例如：910B3 Vector 的 FP16 理论峰值算力约 280 TFLOPS，处理 32K 个 float element：
`32K / 280 TOPS ≈ 0.0001 μs`（远小于搬运时间）

**Ridge AI（算力/带宽边界）** = 峰值算力 / 峰值带宽

例如 910B3：`280,000 GFLOPS / 1600 GB/s ≈ 175 FLOPs/Byte`

若算子的算术强度 < Ridge AI → 大概率是 **memory-bound**；反之为 **compute-bound**。

---

### Step 1: 看 OpBasicInfo.csv — 宏观判断

```csv
Op Name,Op Type,Task Duration(us),Block Dim,Device Id,Pid,Current Freq,Rated Freq
_gelu_kernel,vector,177.703552,4096,0,920434,1800,1800
```

**关注点**：

| 列 | 诊断问题 |
|---|---|
| `Task Duration` | 单次 kernel 调用的绝对耗时。与理论值对比，差距 > 2× 则有优化空间 |
| `Block Dim` | 是否用满物理核？910B3 有 48 个 Vector 核，Block Dim < 48 意味着有算力浪费 |
| `Op Type` | vector / cube → 决定后续看 aiv 还是 aic 指标 |
| `Current Freq` vs `Rated Freq` | 不一致说明降频了，需要增加 warm-up 次数 |

---

### Step 2: 看 PipeUtilization.csv — 瓶颈在哪条管线

每条 row 是一个 block。对 aiv（vector 核）列聚合：

| 列名 | 含义 | 诊断问题 |
|------|------|---------|
| `aiv_time(us)` | 该 block 总耗时 | — |
| `aiv_vec_time(us)` | 向量管线耗时 | 是否 vec-bound？ |
| `aiv_scalar_time(us)` | 标量管线耗时 | **per-block 是否恒定？** |
| `aiv_mte2_time(us)` | MTE2 DMA 耗时 | DMA 占比多高？ |
| `aiv_mte3_time(us)` | MTE3 DMA 耗时 | DMA 占比多高？ |
| `aiv_total_cycles` | 总周期数 | per-block 差值反映计算密度 |

**核心判断法**：

```
计算 per-block scalar_time = sum(aiv_scalar_time) / blocks

如果 per-block scalar_time 基本不变（< 20% 波动）：
  → 标量开销是「固定入场费」，不是按元素收费
  → 优化方向：增大 BLOCK_SIZE 减少 block 数量

如果 per-block scalar_time 与 BLOCK_SIZE 成正比：
  → 标量计算在 hot path 上被逐元素执行
  → 检查 kernel 里是否有按元素做标量运算（如循环内重复计算 index）
```

**瓶颈类别**：

| per-block 占比 | 瓶颈 | 优化方向 |
|---------------|------|---------|
| scalar > 30% | 标量瓶颈 | 增大 BLOCK_SIZE，或减少 per-element 标量指令 |
| vec < 30% | 向量利用率低 | 检查 DMA 是否卡住向量；考虑数据预取 |
| mte2+mte3 > 30% | DMA 瓶颈 | 增大量每次搬运；检查 burst 长度 |
| vec > 60% + vec_wait_ratio 低 | 向量瓶颈（健康状态） | 已达到最佳，考虑精度降级 |

**来自官方文档的方法一**：通过 `op_summary_*.csv` 分析流水情况

比较最长流水和理论值的差距。如果 MTE2 耗时接近理论值但远小于总 Duration，说明计算和 DMA 没有充分 overlap，优化方向是**流水排布 + Tiling**。如果 MTE2 耗时和总 Duration 持平，说明已是 MTE2-bound，达到上限。

---

### Step 3: 看 ArithmeticUtilization.csv — 算力有没有用起来

PipeUtilization 告诉你时间花在哪条管线上，ArithmeticUtilization 告诉你**这些时间产出了多少有效计算**。两者结合才能判断是 "管线忙但没产出" 还是 "管线忙且充分产出"。

| 列名 | 含义 | 诊断问题 |
|------|------|---------|
| `aic_cube_fops` | Cube 单元浮点运算次数 | Cube 算子是否充分使用算力？ |
| `aiv_vec_fops` | Vector 单元浮点运算次数 | Vector 算子是否充分使用算力？ |
| `aic_cube_ratio` | Cube 单元平均利用率 | 接近 100% 说明 Cube 饱和 |
| `aiv_vec_ratio` | Vector 单元平均利用率 | **最关键的指标**：vec < 60% 说明向量单元在空转 |
| `aic_scalar_ratio` | Scalar 单元利用率 | 与 PipeUtilization 中 scalar_time 交叉验证 |
| `aic_mte1_ratio` / `aic_mte2_ratio` / `aic_mte3_ratio` | 各 DMA 管线利用率 | 与 PipeUtilization 中 MTE 时间交叉验证 |
| `aic_fp16_instructions` / `aic_int32_instructions` | FP16/INT32 指令数 | 按精度分列的指令分布 |
| `aic_total_cycles` | 总周期数 | — |

**核心判断**：

```
如果 PipeUtilization 显示 vec_time 占比高，但 vec_ratio < 60%：
  → 向量管线在"空跑"，可能被 DMA 卡住或存在依赖等停
  → 交叉看 ResourceConflictRatio 中的 vec_wait_ratio

如果 vec_ratio 接近 100%，且 vec_time 是最大占比：
  → 算力已充分利用，瓶颈在计算本身
  → 优化方向：精度降级（FP16→INT8）、减少计算量（算法层面）

如果 cube_ratio 低但 vec_ratio 高：
  → 算子被调度到了 Vector 核但实际 Cube 单元闲置
  → 检查是否有未融合的 Cube 操作，或考虑拆分为 Cube 专用算子
```

**vec_ratio 与优化方向**：

| vec_ratio | 含义 | 方向 |
|-----------|------|------|
| > 85% | 向量单元充分饱和 | compute-bound，考虑降精度或减少计算量 |
| 60–85% | 有提升空间 | 检查 DMA 是否卡住；考虑双缓冲 |
| 30–60% | 向量利用率严重不足 | 大概率是 DMA-bound 或 scalar-bound |
| < 30% | 向量基本闲置 | 标量或 DMA 是绝对瓶颈，应先优化它们 |

---

### Step 4: 看 Memory.csv — DMA 指令效率

| 列名 | 含义 |
|------|------|
| `aiv_mte2_instructions` | 该 block 的 MTE2 DMA **指令条数** |
| `aiv_mte3_instructions` | 该 block 的 MTE3 DMA **指令条数** |
| `GM_to_UB_datas(KB)` | 该 block 从全局内存搬到 UB 的总数据量 |
| `UB_to_GM_datas(KB)` | 该 block 从 UB 写回全局内存的总数据量 |

**关键判断**：

```
如果 per-block mte2_instructions 在不同 BLOCK_SIZE 下保持不变：
  → DMA 指令数是固定的，带宽利用率不足是因为每搬数据量太小
  → 优化方向：增大 BLOCK_SIZE

如果 per-block mte2_instructions 随 BLOCK_SIZE 成比例增长：
  → DMA 被切分成多段，可能是 buffer 不够大或 address 不连续
```

---

### Step 5: 看 ResourceConflictRatio.csv — 管线阻塞

| 列名 | 含义 |
|------|------|
| `aiv_vec_wait_ratio` | 向量管线等待比例 |
| `aiv_vec_total_cflt_ratio` | 向量 bank 冲突比例 |
| `aiv_mte2_wait_ratio` | MTE2 等待比例 |
| `aiv_mte3_wait_ratio` | MTE3 等待比例 |

**诊断组合**：

| 特征 | 根因 | 对策 |
|------|------|------|
| mte2_wait 高 + vec_wait 低 | DMA 是瓶颈，向量在等数据 | 增大单次 DMA 量、使用双缓冲 |
| mte3_wait > 80% | MTE3 成为严格瓶颈 | 减小输出写入量、合并写操作 |
| vec_wait 高 | 向量在等 DMA 或标量完成 | 查上游 mte_wait |
| vec_cflt > 20% | 向量访存 bank 冲突严重 | 调整数据 layout、pad 对齐 |

---

### Step 6: 看 MemoryUB.csv — 带宽利用率

| 列名 | 含义 |
|------|------|
| `aiv_ub_read_bw_vector(GB/s)` | UB 向量读带宽 |
| `aiv_ub_write_bw_vector(GB/s)` | UB 向量写带宽 |

```
如果 UB 带宽 < 峰值 50%：
  → 向量单元在空闲等待
  → 查 vec_wait_ratio 确认是被 DMA 还是标量卡住
```

---

### Step 7: Roofline 分析 — 算力 vs 带宽定位

前面 Step 2–6 从管线、算力利用率、DMA、冲突、UB 带宽各个维度做了微观诊断。Roofline 分析把这些信息合成一张宏观图：**这个算子到底卡在算力还是带宽？**

**数据来源**：

| 指标 | 从哪个 CSV 取 | 对应列 |
|------|-------------|--------|
| 总 FLOPs | ArithmeticUtilization | `aic_cube_fops` + `aiv_vec_fops` |
| 总搬运量 (Byte) | Memory | `GM_to_UB_datas(KB)` + `UB_to_GM_datas(KB)` |
| 执行时间 | OpBasicInfo | `Task Duration(us)` |

**计算公式**：

```
算术强度 AI = 总 FLOPs / 总搬运量 (Byte)
实际 GFLOPS = 总 FLOPs / Task Duration / 1e9
理论 GFLOPS = min(AI × 峰值带宽, 峰值算力)
Roof% = 实际 GFLOPS / 理论 GFLOPS × 100%
```

**瓶颈判定**：

| 条件 | 瓶颈类型 | 验证手段 |
|------|---------|---------|
| AI < Ridge + Roof% > 70% | **Memory-bound**：带宽是瓶颈 | 回看 Step 4 Memory：DMA 指令数和单次搬运量 |
| AI ≥ Ridge + vec_ratio > 75% | **Compute-bound**：算力是瓶颈 | 回看 Step 3：vec_ratio 是否已饱和 |
| Roof% < 50% | **Pipeline latency bound**：既没卡带宽也没卡算力 | 回看 Step 5：管线冲突/等停 |

详细工具见下方「集成工具 — msprof_op_parser.py roofline」。

---

### 来自官方文档的额外诊断方法

**方法二 — Tiling 分析**：通过 `op_summary_*.csv` 的 Block Dim 判断是否用满物理核。Block Dim < 物理核数意味着算力浪费，优先优化 Tiling。

**方法三 — 仿真流水图**：在仿真环境下查看各流水线是否有规律性断流。如果 Vector 核的 MTE2/MTE3 有规则性空白，说明存在数据依赖导致断流，需要优化流水排布。

**方法四 — 头开销排查**：推理场景下算子延迟在微秒级时，头开销（核启动、TLB miss、同地址访问冲突、变量初始化）可能占总延迟的显著比例。Atlas A2 上满核头开销约 20~21 μs。通过采集空 kernel 的 TaskDuration 确认头开销，然后通过调整核数和 Kernel Type 来优化。

---

## 集成工具

### msprof_op_parser.py

对本仓库中 `skills/triton-npu-profile-operator/scripts/msprof_op_parser.py` 的封装，提供 roofline 分析、pipeline 分析和对比能力。

```bash
# Roofline 分析（单个 artifact）
python3 msprof_op_parser.py roofline --artifacts-dir OPPROF_001

# 带图
python3 msprof_op_parser.py roofline --artifacts-dir OPPROF_001 --plot

# 指定硬件
python3 msprof_op_parser.py roofline --artifacts-dir OPPROF_001 --hardware 910c

# 对比两个 artifact
python3 msprof_op_parser.py roofline --compare OPPROF_baseline OPPROF_target

# 批量扫描优化轮次
python3 msprof_op_parser.py roofline --discover ./opt-round-*/

# Pipeline 分析（开发中）
python3 msprof_op_parser.py pipeline --artifacts-dir OPPROF_001

# 查看完整帮助
python3 msprof_op_parser.py help
```

**msprof_op_parser.py roofline 做了什么**：

1. 自动定位 `OpBasicInfo.csv`、`ArithmeticUtilization.csv`、`Memory.csv`（通过 `find_newest_csv`）
2. 计算算术强度 (AI)、实际 GFLOPS、理论 GFLOPS、Roof%
3. 判断 memory-bound vs compute-bound
4. `--compare` 模式输出并排对比表，标注改善/退步方向（✓ / ✗）
5. `--plot` 生成 roofline 图，直观呈现数据点在硬件上限曲线上的位置

### MindStudio Insight

华为官方可视化工具，可用于：

- **Roofline 瓶颈分析图**：配合 `--aic-metrics=Roofline` 采集的数据
- **计算内存热力图**：展示计算和内存访问的时间-空间分布
- **指令流水图**：配合 `--aic-metrics=TimelineDetail`
- **算子代码热点图**：定位 kernel 代码中的热点指令
- **Cache 热力图**：分析 L2 Cache 命中率分布
- **核间负载分析图**：配合 `--aic-metrics=Occupancy`

使用方式：采集数据后，用 MindStudio 打开 `OPPROF_*/visualize_data.bin` 或整个 `OPPROF_*/` 目录。

---

## 产出

一次完整的 msprof op 诊断应产出：

1. **瓶颈定位结论**：明确算子是 memory-bound / compute-bound / pipeline-latency-bound
2. **管线占比数据**：scalar / vector / MTE2 / MTE3 各自占比
3. **DMA 指令效率分析**：per-block 指令数是否随 BLOCK_SIZE 合理变化
4. **管线冲突证据**：wait_ratio 和 conflict_ratio 数据
5. **Roofline 图**（可选）：AI vs GFLOPS 在硬件上限曲线上的位置
6. **优化建议优先级排序**：哪些改动见效最快
