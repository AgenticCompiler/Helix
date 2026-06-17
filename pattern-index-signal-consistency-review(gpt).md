# pattern_index 路由表信号一致性审查

审查范围：`skills/triton-npu-optimize-knowledge/references/pattern_index.md` 中 Step 2 simulation routing 和 Step 3 msprof routing 的信号表述。

审查方法：逐条对比路由表里的 profile / signal 条件，和对应 pattern 卡片里的 `## Use When`、`## Signals`。Step 1 的纯代码结构路由默认不纳入异常，除非它引入了 profile 阈值或和 pattern 卡明显冲突。

结论概览：路由表里确实有不少 AI 生成痕迹，主要问题是把别的 pattern 的阈值搬到了不对应的 pattern 下，或者把 pattern 卡里只有定性描述的内容硬写成了具体阈值。你提到的 `COMPUTE_and_MTE2_over_COMPUTE < 1%` 放在 `static-range-to-range` 下，就是最典型的一条不一致。

## 明确异常项

### 1. `static-range-to-range` 路由表出现 `COMPUTE_and_MTE2_over_COMPUTE < 1%`，但 pattern 里没有这个信号（中）

| 项目 | 内容 |
|---|---|
| 路由表 | `pattern_index.md:139`：`COMPUTE_and_MTE2_over_COMPUTE < 1%` 直接路由到 `static-range-to-range`。 |
| pattern 卡 | `static-range-to-range.md` 的 profile 信号是 MTE2/VECTOR 比例、`%(SCALAR&MTE2/SCALAR) < 50%`、`%(MTE2&VECTOR/VECTOR)` 约 50% 或更低、`%(SCALAR&VECTOR/VECTOR) < 5%`、pipeline flows 和 WAIT/BAR。 |
| 异常原因 | `static-range-to-range` 没有写 `COMPUTE_and_MTE2_over_COMPUTE`，也没有 `< 1%` 这个阈值。这个信号更像 `software-pipeline-dependency-profiling` 里的 compute/MTE2 overlap gate。 |
| 判断 | 路由表有阈值，但对应 pattern 没有。建议优先修。 |

### 2. `static-range-to-range` 路由表使用 `flow_SCALAR_to_VECTOR_avg_ns > 50 AND SCALAR_instr% > 75`，但 pattern 没有对应阈值（半中）

| 项目 | 内容 |
|---|---|
| 路由表 | `pattern_index.md:75`：`flow_SCALAR_to_VECTOR_avg_ns > 50` AND `SCALAR_instr% > 75` 路由到 `static-range-to-range`。 |
| pattern 卡 | 只说 `tl.static_range` 中 per-element scalar `tl.load` 会造成 high SCALARToVECTOR latency，但 profile 部分实际用 overlap 和 ratio 信号。 |
| 异常原因 | `> 50ns` 和 `> 75%` 是 `software-pipeline` / `scalar-latency-traps` 中更明确的 Cat 2 dispatch bottleneck 信号，不是 `static-range-to-range` 自己定义的 profile gate。 |
| 判断 | 路由表硬加具体阈值，pattern 没有。 |

### 3. `static-range-to-range` 路由表给 `SCALAR_and_VECTOR_over_VECTOR < 5%` 额外加了 `SCALAR_cycles% > 10`（中）

| 项目 | 内容 |
|---|---|
| 路由表 | `pattern_index.md:142`：`SCALAR_and_VECTOR_over_VECTOR < 5%` AND `SCALAR_cycles% > 10`。 |
| pattern 卡 | `%(SCALAR&VECTOR/VECTOR) < 5%` 是有的；`%(SCALAR&VECTOR/SCALAR) < 15%` 也是有的；但没有把 `SCALAR_cycles% > 10` 写成条件。 |
| 异常原因 | overlap 部分基本一致，但 `SCALAR_cycles% > 10` 是路由表额外加的硬门槛。pattern 里的 `SCALAR dur% = 13.19%` 只是案例数据，不是阈值定义。 |
| 判断 | 路由表条件比 pattern 更严，可能误拒绝本该匹配的情况。 |

### 4. `discrete_memory_access` 路由表漏掉 `SCALARToVECTOR count > 0`（中）

| 项目 | 内容 |
|---|---|
| 路由表 | `pattern_index.md:73`：`flow_MTE2_to_VECTOR_exists = false` AND code has `tl.load`。 |
| pattern 卡 | `discrete_memory_access.md` 要求 `MTE2ToVECTOR count = 0 AND SCALARToVECTOR count > 0`。 |
| 异常原因 | pattern 要确认数据确实绕过 MTE2、走了 SCALARToVECTOR；路由表只检查 MTE2ToVECTOR 不存在。 |
| 判断 | 路由表条件弱于 pattern，可能误报。 |

### 5. `grid-flatten-and-ub-buffering` 路由表漏掉两个关键信号（中）

| 项目 | 内容 |
|---|---|
| 路由表 | `pattern_index.md:74`：`flow_MTE2_to_VECTOR_exists = false` AND code has `tl.load`，区分栏写 “Per-element scalar load”。 |
| pattern 卡 | `grid-flatten-and-ub-buffering.md` 要求 `MTE2ToVECTOR count = 0 AND SCALARToVECTOR count > 0 AND each program loads a single scalar element`。 |
| 异常原因 | `SCALARToVECTOR count > 0` 没进条件；“each program loads a single scalar element” 也只是写在区分栏，没有作为必需 signal。 |
| 判断 | 路由表条件弱于 pattern。 |

### 6. `compile_hint` / `discrete_memory_access` 被 `MTE2_ProcessBytes_avg < 128 OR MTE2_Data_movers = 0` 路由，但目标 pattern 没有这些阈值

| 项目 | 内容 |
|---|---|
| 路由表 | `pattern_index.md:84`：`MTE2_ProcessBytes_avg < 128 OR MTE2_Data_movers = 0`，作为 stop autotune gate，路由到 `compile_hint` 或 `discrete_memory_access`。 |
| pattern 卡 | `compile_hint` 只有 late-stage qualitative profile evidence；`discrete_memory_access` 说 MTE bandwidth utilization low，但没有 `<128` 或 `Data_movers = 0`。 |
| 异常原因 | 这个 gate 也许是 autotune 逻辑里的诊断，但作为 pattern route，它没有被这两个目标 pattern 卡直接支持。 |
| 判断 | 路由表有具体阈值，pattern 没有。 |

### 7. `padded_row_col_copy` 被 `SCALAR_instr% > 80 AND TRACE_total_events > 10000` 路由，但 pattern 只有定性描述（中）

| 项目 | 内容 |
|---|---|
| 路由表 | `pattern_index.md:97`：`SCALAR_instr% > 80` AND `TRACE_total_events > 10000`。 |
| pattern 卡 | `padded_row_col_copy.md` 只说 scalar/control overhead out of proportion，或 masked load / compare chains dominate pad bounds。 |
| 异常原因 | 这个数值阈值像是从 `flat-index-decode-tiling` / `scalar-latency-traps` 借来的，`padded_row_col_copy` 自己没写。 |
| 判断 | 路由表硬加阈值。 |

### 8. `loop-invariant-hoisting` 被 `SCALAR_instr% > 80` 路由，但 pattern 没有这个数值（中）

| 项目 | 内容 |
|---|---|
| 路由表 | `pattern_index.md:99`：`SCALAR_instr% > 80`。 |
| pattern 卡 | `loop-invariant-hoisting.md` 说 high SCALAR instruction/cycle share、high SCALAR:VECTOR ratio，但没有 `>80%`。 |
| 异常原因 | pattern 是定性/比例描述，路由表把它固定成了一个数值阈值。 |
| 判断 | 路由表硬加阈值。 |

### 9. `program-multiple-rows` 路由表只覆盖了 pattern 的一部分数值条件（半中）

| 项目 | 内容 |
|---|---|
| 路由表 | `pattern_index.md:100`：`SCALAR_instr% >= 70` AND `VECTOR_instr% <= 15`。 |
| pattern 卡 | pattern 允许 `SCALAR instr >= 70%` OR `SCALAR cycles >= 40%`，同时 `VECTOR instr <= 15%` OR `VECTOR cycles <= 15%`；还写了 `SCALAR:VECTOR_instr >= 8:1` 或 `SCALAR:VECTOR_cycles >= 3:1`。 |
| 异常原因 | 路由表只保留 instr 版本，漏掉 cycles 版本和 ratio 版本。 |
| 判断 | 路由表不完整，可能漏报。 |

### 10. `block-pointer-dimensionality` Section 3b 把两个 strong signal 混在一起，并漏掉 MTE3 serialization 条件（半中）

| 项目 | 内容 |
|---|---|
| 路由表 | `pattern_index.md:106`：`SCALAR_cycles% / VECTOR_cycles% > 10` AND `MTE3_cycles% > 10`。 |
| pattern 卡 | strong signals 是三选一：`SCALAR_cycles% / VECTOR_cycles% > 10`；或 `MTE3_cycles% > 10% AND %(SCALAR&MTE3/SCALAR) > 20%`；或 `%(MTE2&MTE3/MTE2) > 50%`。 |
| 异常原因 | 路由表把 SCALAR/VECTOR ratio 和 bare `MTE3_cycles% > 10` 拼成一个 AND，但 pattern 中 MTE3 信号必须带 `%(SCALAR&MTE3/SCALAR) > 20%`。 |
| 判断 | Section scan row 不一致；后面的 composite rule 反而是对的。 |

### 11. `merge-adjacent-stores` 被 `MTE3_cycles% > 10` 路由，但 pattern 没有 profile threshold

| 项目 | 内容 |
|---|---|
| 路由表 | `pattern_index.md:123`：`MTE3_cycles% > 10`。 |
| pattern 卡 | `merge-adjacent-stores.md` 只有 code signals：多个相邻 `tl.store`、循环逐次 store、store 数量明显多于 load。没有 profile section。 |
| 异常原因 | 高 MTE3 可能暗示 store pressure，但目标 pattern 没定义这个阈值。 |
| 判断 | 路由表有，pattern 没有。 |

### 12. `scalar-latency-traps` 被 `SCALAR_to_VECTOR_instr > 10` 路由，但 pattern 没有这个指标（中）

| 项目 | 内容 |
|---|---|
| 路由表 | `pattern_index.md:133`：`SCALAR_to_VECTOR_instr > 10` AND `SCALAR_instr% > 80`。 |
| pattern 卡 | pattern 使用 `SCALAR_instr% > 80 AND TRACE_total_events > 10000`、`SCALARToVECTOR avg > 50ns AND SCALAR instr% > 75%`、`VECTORToSCALAR count > 300`、`VECTOR utilization <30% AND VECTOR instr% <15%`。 |
| 异常原因 | `SCALAR_to_VECTOR_instr > 10` 不是 pattern 卡里的信号。 |
| 判断 | 路由表有，pattern 没有。 |

### 13. `reorder-load` 路由表漏掉 UB conflict 的 utilization gate（中）

| 项目 | 内容 |
|---|---|
| 路由表 | `pattern_index.md:149`：`VECTOR_UB_read_conflict > 100 OR VECTOR_UB_write_conflict > 100`。 |
| pattern 卡 | `reorder-load.md` 有同样的 conflict `>100`，但还说 conflicted instructions 的 utilization 也要 `<50%`，否则 conflict 可能是 benign。 |
| 异常原因 | 路由表只保留了前半个条件，漏掉 `<50%` utilization 确认。 |
| 判断 | 路由表条件弱于 pattern，可能误报。 |

### 14. `vec-cmp` 的 vector-utilization 路由加了 pattern 没写的硬阈值（半中）

| 项目 | 内容 |
|---|---|
| 路由表 | `pattern_index.md:151`：`VECTOR_utilization_avg < 30` AND `VECTOR_instr% < 15`。 |
| pattern 卡 | `vec-cmp.md` 说 low or zero utilization，并且 top VECTOR instructions 是 mask-like，例如 `MOVEMASK`；没有 `<30` 或 `VECTOR_instr% <15`。 |
| 异常原因 | pattern 的核心是 integer mask / `tl.where` 代码信号和 mask-like top instruction，路由表写成了泛化的 utilization 阈值。 |
| 判断 | 路由表硬加阈值，并漏掉上下文。 |

### 15. `classic-matmul` 的 vector-utilization 路由不符合 pattern 的 CUBE/MADD 信号（中）

| 项目 | 内容 |
|---|---|
| 路由表 | `pattern_index.md:152`：`VECTOR_utilization_avg < 30` AND `VECTOR_instr% < 15` AND source is matmul-like。 |
| pattern 卡 | `classic-matmul.md` profile 信号是 no CUBE row / CUBE instr = 0、`[CUBE/MMA]` absent or MMAD = 0、`SCALAR instr% > 50%`、`TRACE MADD > 0`、no CUBE-related flows。 |
| 异常原因 | 低 VECTOR utilization 不是 `classic-matmul` 的 documented trigger。这个 pattern 关注的是 missed CUBE lowering 和 multiply-accumulate 证据。 |
| 判断 | 路由表有，pattern 没有。 |

### 16. `accumulator-layout-alignment` 被 VECTOR 阈值路由，但 pattern 完全没有 profile 信号（中）

| 项目 | 内容 |
|---|---|
| 路由表 | `pattern_index.md:153`：`VECTOR_utilization_avg < 30` AND `VECTOR_instr% < 15`。 |
| pattern 卡 | `accumulator-layout-alignment.md` 只有 accumulator shape 与 output layout mismatch 的代码结构信号。 |
| 异常原因 | 这是很明显的 AI 补阈值：pattern 本身没有任何 profile threshold。 |
| 判断 | 路由表有，pattern 没有。 |

### 17. `program-multiple-rows` 的 vector-utilization 路由额外加了 `VECTOR_utilization_avg < 30`

| 项目 | 内容 |
|---|---|
| 路由表 | `pattern_index.md:154`：`VECTOR_utilization_avg < 30` AND `VECTOR_instr% < 15`。 |
| pattern 卡 | `VECTOR_instr% <= 15%` 是有的；但 `VECTOR_utilization_avg < 30` 没有。pattern 只是说 low utilization、near-zero utilization、only tens of VECTOR instructions、mask/setup-heavy top instructions。 |
| 异常原因 | 一半匹配，一半是路由表额外加的硬阈值。 |
| 判断 | 部分不一致。 |

### 18. `pooling-inner-w-slab-gather` 的 TRACE row 把 pattern 的两个信号重新组合了（半中）

| 项目 | 内容 |
|---|---|
| 路由表 | `pattern_index.md:163`：`TRACE_top_types` contains ADD/MUL/SUB dominant AND `SCALAR_instr% > 80`。 |
| pattern 卡 | 一条信号是 `SCALAR instr% > 80% AND TRACE total events > 10,000`；另一条单独信号是 TRACE top events dominated by ADD/MUL/SUB。 |
| 异常原因 | 路由表漏掉 `TRACE_total_events > 10000`，并把 `SCALAR_instr% >80` 和 ADD/MUL/SUB dominant 组合成了 pattern 里没有的条件。 |
| 判断 | 表述接近，但不是同一个信号。 |

### 19. `a5-force-simt-only-discrete-access` msprof 路由漏掉 A5 和 cube 检查

| 项目 | 内容 |
|---|---|
| 路由表 | `pattern_index.md:205`：`aiv_scalar_ratio ≫ aiv_vec_ratio`。 |
| pattern 卡 | 要求 confirmed A5 hardware，并且 `aiv_scalar_ratio` 明显高于 `aiv_vec_ratio` 和 `cube_utilization`，还要是 discrete/index-driven code。 |
| 异常原因 | 路由条件只保留了一个 profile comparison，把 A5 硬件 gate 和 cube utilization comparison 放丢了。 |
| 判断 | 路由表条件弱于 pattern。 |

### 20. `pooling-inner-w-slab-gather` msprof 路由用 high scalar alone，但 pattern 明确提醒不能只靠这个

| 项目 | 内容 |
|---|---|
| 路由表 | `pattern_index.md:206`：`aiv_scalar_ratio high` 路由到 `pooling-inner-w-slab-gather` / `program-multiple-rows`。 |
| pattern 卡 | `pooling-inner-w-slab-gather.md` 写了 high `aiv_scalar_ratio` alone does not mean “skip slab”；真正信号是 per-`kw` W loads、slab/gather 结构、scalar explosion、ADD/MUL/SUB offset computation、transfer pressure。 |
| 异常原因 | high scalar 只能作为调查入口，不足以直接路由到 pooling W-slab。 |
| 判断 | 路由表过宽。 |

### 21. `stencil-resize-gm-to-ub-staging` msprof 路由加了未定义的 `call_count 100k+`

| 项目 | 内容 |
|---|---|
| 路由表 | `pattern_index.md:212`：`code_exe cycles high + hotspot call_count 100k+`。 |
| pattern 卡 | 只说 msprof shows high ST/LD or inner call_count while VGATHER share stays ~1%。没有 `100k+`。 |
| 异常原因 | `100k+` 是路由表独有阈值；pattern 还要求 VGATHER share 约 1% 这个上下文。 |
| 判断 | 路由表有，pattern 没有。 |

## 低风险但需要统一命名的问题

### `software-pipeline-dependency-profiling` 的 `COMPUTE_and_MTE2_*` 命名与 pattern 中 `%((VECTOR+CUBE)&MTE2/*)` 不一致

路由表 `pattern_index.md:143` 和 composite rule `pattern_index.md:194` 使用：

- `COMPUTE_and_MTE2_over_COMPUTE < 1%`
- `COMPUTE_and_MTE2_over_MTE2 < 1%`

pattern 卡使用：

- `%((VECTOR+CUBE)&MTE2/(VECTOR+CUBE))`
- `%((VECTOR+CUBE)&MTE2/MTE2)`

如果 `COMPUTE = VECTOR + CUBE`，这大概率是同义字段；但是 pattern 卡没有写 `COMPUTE_and_MTE2_*` 这个 canonical feature 名。建议后续统一命名，否则 agent 可能会把它们当成两个不同指标。

## 看起来一致的路由

这些行我认为基本匹配 pattern 卡：

- `software-pipeline`：`SCALARToVECTOR avg > 50ns AND SCALAR instr% > 75%`、`VECTORToSCALAR count > 300`、msprof 中 Cube/Vector gaps while MTE fetches，和 pattern 一致。
- `flat-index-decode-tiling`：`SCALAR_instr% > 80 AND TRACE_total_events > 10000`，和 pattern 一致。
- `scalar-latency-traps`：Section 3a 的 `SCALAR_instr% > 80 AND TRACE_total_events > 10000`，和 pattern 一致。
- `classic-matmul`：composite rule 的 `CUBE_present = false AND TRACE_MADD_count > 0` 加 source gate，和 pattern 一致。
- `block-pointer-dimensionality`：最终 composite rule 基本一致；只是前面的 Section 3b support row 有问题。
- `remove-implicit-transpose`：Step 3 的 `WAIT_FLAG_DEVI` 和 `MOV_OUT_TO_L1_MULTI_ND2NZ` / nd2nz，和 pattern 一致。
- `loop-invariant-hoisting`：Step 3 的 `LD_XD_XN_IMM` / `ST_XD_XN_IMM` / `ADD` / `CMP_IMM`，和 pattern 一致。
- `algebraic-optimization`：Step 3 的 NotEqual/BroadcastTo expanded numel，和 pattern 一致。

## 汇总

| 分类 | 数量 | 主要风险 |
|---|---:|---|
| 路由表有具体阈值 / 信号，但 pattern 没有 | 12 | 误报；agent 进 pattern 后找不到对应证据。 |
| 路由表条件弱于 pattern | 5 | 误报；漏掉必要 gate。 |
| 路由表只覆盖部分 pattern 条件，或把条件重新组合 | 3 | 漏报或证据链混乱。 |
| 命名 / alias 不统一 | 1 | 后续 agent 解析时可能把同义指标当成不同指标。 |

## 建议优先处理顺序

1. 优先处理 `static-range-to-range` 下的 `COMPUTE_and_MTE2_over_COMPUTE < 1%`，这是最明显的错位信号。
2. 给 `discrete_memory_access` / `grid-flatten-and-ub-buffering` 补回 `SCALARToVECTOR count > 0` 和 single-scalar-load gate，或者从路由表去掉不完整条件。
3. 修 `block-pointer-dimensionality` Section 3b support row，让它和 pattern 卡的 strong signals 一致；最终 composite rule 目前是对的。
4. 决定 `merge-adjacent-stores`、`accumulator-layout-alignment`、`compile_hint`、`padded_row_col_copy` 这些 pattern 是否真的要 profile 阈值。如果要，就补到 pattern 卡；如果不要，就从路由表删。
5. 补回遗漏 gate：`reorder-load` 的 conflicted instruction utilization `<50%`、`a5` 的 A5 硬件和 cube utilization 检查、pooling / stencil 的特定上下文。
