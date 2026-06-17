# 路由表信号与 Pattern Card 信号一致性审查报告

审查范围：`pattern_index.md` 中 Step 1/2/3 所有路由行的信号条件，与对应 pattern card 中实际描述的信号条件逐一比对。

---

## A类：阈值不同（路由表和pattern card用了不同的数值或指标）

### A1. `discrete_memory_access` — Section 1 缺少 `SCALARToVECTOR > 0` 前置条件（中）

| | 内容 |
|---|---|
| **路由表** | `flow_MTE2_to_VECTOR_exists = false` AND code has `tl.load` |
| **Pattern card** | `MTE2ToVECTOR count = 0 AND SCALARToVECTOR count > 0`（Cat 5: Missing Memory Engine） |
| **问题** | 路由表只检查了MTE2_to_VECTOR不存在，但pattern card还要求`SCALARToVECTOR count > 0`。缺少这个条件意味着：一个没有任何load的kernel（MTE2和SCALAR流量都为零）也会被路由到这个pattern，产生误匹配。Cat 5的语义是"数据本应走MTE2但被迫走了SCALAR"，如果SCALAR也没走，说明根本就没有load操作，不属于这个pattern。 |

### A2. `grid-flatten-and-ub-buffering` — Section 1 同样缺少 `SCALARToVECTOR > 0` 及语义条件（中）

| | 内容 |
|---|---|
| **路由表** | `flow_MTE2_to_VECTOR_exists = false` AND code has `tl.load` |
| **Pattern card** | `MTE2ToVECTOR count = 0 AND SCALARToVECTOR count > 0 AND each program loads a single scalar element` |
| **问题** | 与A1相同的问题。此外pattern card还有一个语义条件"每个program加载单个scalar element"，路由表完全没体现。这个条件区分了"离散访问导致的scalar load"和"其他原因导致的MTE2缺失"。 |

### A3. `vec-cmp` — Section 3a 用绝对百分比代替了比值

| | 内容 |
|---|---|
| **路由表** | `SCALAR_instr% > 80` AND `TRACE_top_types` SIGNEXT dominant |
| **Pattern card** | `SCALAR_cycles% / VECTOR_cycles% > 10` 和 `SCALAR:VECTOR_instr > 4:1` 作为主要比值指标 |
| **问题** | 路由表用SCALAR绝对占比>80%，但pattern card用的是SCALAR和VECTOR的比值。一个SCALAR_instr% = 60%、VECTOR_instr% = 3%的kernel（比值20:1）应该匹配vec-cmp，但路由表的>80%阈值会漏掉。绝对百分比和比值是不同的度量方式，适用场景不同——比值更能反映"scalar远多于vector"这个语义。 |

### A4. `pooling-inner-w-slab-gather` — Section 7 额外加了 `SCALAR_instr% > 80` 守卫（半中）

| | 内容 |
|---|---|
| **路由表** | `TRACE_top_types` contains ADD/MUL/SUB dominant AND `SCALAR_instr% > 80` |
| **Pattern card** | ADD/MUL/SUB dominant是独立信号；`SCALAR_instr% > 80`是另一个独立信号（Section 3a） |
| **问题** | 路由表把两个信号用AND连接，比pattern card更严格。如果TRACE事件是ADD/MUL/SUB主导但SCALAR_instr%只有75%，路由表不会匹配，但pattern card认为这两个信号可以独立作为证据。这是路由表对pattern card语义的过度收紧。 |

### A5. `block-pointer-dimensionality` — Section 3b 丢失了overlap要求（半中）

| | 内容 |
|---|---|
| **路由表** | `SCALAR_cycles% / VECTOR_cycles% > 10` AND `MTE3_cycles% > 10` |
| **Pattern card** | 这是两个独立的"强信号"；MTE3信号要求 `MTE3_cycles% > 10% AND %(SCALAR&MTE3/SCALAR) > 20%` |
| **问题** | 路由表把SCALAR/VECTOR比值和MTE3%用AND组合，但丢掉了MTE3信号的overlap子条件`%(SCALAR&MTE3/SCALAR) > 20%`。MTE3占比高但和SCALAR没有序列化（overlap低）的场景，路由表会误匹配——MTE3高可能只是因为数据量大，不一定是block pointer问题。 |

### A6. `autotune` — Section 2 用 `= 0` 代替了 `< 2`

| | 内容 |
|---|---|
| **路由表** | `MTE2_ProcessBytes_avg < 128` OR `MTE2_Data_movers = 0` |
| **Pattern card** | `ProcessBytes per MTE load avg < 128 bytes` OR `MTE2 data mover count < 2` |
| **问题** | 路由表要求data movers严格等于0，但pattern card说< 2（即0或1）。有1个data mover的case会被pattern card捕获但被路由表漏掉。实际上1个data mover也是MTE2通路不充分的表现，应该被捕获。 |

---

## B类：阈值仅路由表有（路由表写了具体数字，但pattern card没有提到）

### B1. `scalar-latency-traps` — Section 4 使用 `SCALAR_to_VECTOR_instr > 10`（中）

| | 内容 |
|---|---|
| **路由表** | `SCALAR_to_VECTOR_instr > 10` AND `SCALAR_instr% > 80` |
| **Pattern card** | 没有`SCALAR_to_VECTOR_instr > 10`这个阈值。Card用的是Cat 1（`SCALAR instr% > 80%` + `TRACE total events > 10000`）和Cat 2（`SCALARToVECTOR avg > 50ns AND SCALAR instr% > 75%`） |
| **问题** | 路由表凭空引入了一个pattern card从未提到的指令比值阈值。AI按路由表匹配到这个条件后，去读pattern card却找不到对应描述，无法验证。这个阈值可能是从某个具体case总结的，但没有回写到pattern card中。 |

### B2. `static-range-to-range` — Section 1 使用 `SCALAR_instr% > 75`（半中）

| | 内容 |
|---|---|
| **路由表** | `flow_SCALAR_to_VECTOR_avg_ns > 50` AND `SCALAR_instr% > 75` |
| **Pattern card** | 没有`SCALAR_instr% > 75`阈值。Card用的是overlap比值类指标（`%(SCALAR&MTE2/SCALAR)`、`%(MTE2&VECTOR/VECTOR)`、`%(SCALAR&VECTOR/VECTOR)`） |
| **问题** | 路由表用了一个pattern card不使用的指标（绝对SCALAR百分比）。static-range-to-range的核心问题是循环展开后MTE2和VECTOR/SCALAR之间overlap差，而不是SCALAR占比高。这个条件可能导致误匹配——SCALAR占比高但和range展开无关的kernel也会被路由到。 |

### B3. `static-range-to-range` — Section 5 使用 `COMPUTE_and_MTE2_over_COMPUTE < 1%`（中）

| | 内容 |
|---|---|
| **路由表** | `COMPUTE_and_MTE2_over_COMPUTE < 1%` |
| **Pattern card** | 没有这个指标名。Card用的是`%(SCALAR&MTE2/SCALAR) < 50%`、`%(MTE2&VECTOR/VECTOR) <= 50%`、`%(SCALAR&VECTOR/VECTOR) < 5%` |
| **问题** | `COMPUTE_and_MTE2_over_COMPUTE`是合成指标名（COMPUTE = VECTOR + CUBE），pattern card用的是按pipe拆分的overlap指标。虽然语义相关（都是说MTE2和计算部分没有并行），但指标名和阈值都不同。AI在路由表里看到这个条件后去pattern card里找不到对应描述，需要自己推断等价关系。 |

### B4. `static-range-to-range` — Section 5 额外加 `SCALAR_cycles% > 10` 守卫（中）

| | 内容 |
|---|---|
| **路由表** | `SCALAR_and_VECTOR_over_VECTOR < 5%` AND `SCALAR_cycles% > 10` |
| **Pattern card** | 有`%(SCALAR&VECTOR/VECTOR) < 5%`，但没有`SCALAR_cycles% > 10`作为阈值条件 |
| **问题** | 路由表额外加了`SCALAR_cycles% > 10`守卫，但pattern card没有这个条件。这个守卫的意图是避免在SCALAR占比极低时误匹配（低overlap + 低SCALAR没有意义），但这个判断逻辑没有体现在pattern card中。 |

### B5. `loop-invariant-hoisting` — Section 3a 使用 `SCALAR_instr% > 80`（中）

| | 内容 |
|---|---|
| **路由表** | `SCALAR_instr% > 80` |
| **Pattern card** | "high SCALAR instruction/cycle share"——定性描述，没有具体数字 |
| **问题** | 路由表承诺了>80%，但pattern card只说"高"。AI被路由到这个pattern后，读card发现没有这个阈值，无法确认80%是否合理。这里应该要么在pattern card里补上具体数值，要么在路由表里改为定性描述。 |

### B6. `classic-matmul` — Section 6 使用 `VECTOR_utilization_avg < 30 AND VECTOR_instr% < 15`（中）

| | 内容 |
|---|---|
| **路由表** | `VECTOR_utilization_avg < 30` AND `VECTOR_instr% < 15` AND source is matmul-like |
| **Pattern card** | 没有VECTOR利用率阈值。Card聚焦在CUBE缺失、SCALAR > 50%、MADD > 0等信号 |
| **问题** | 路由表用VECTOR利用率低来路由classic-matmul，但pattern card的信号体系完全不同。一个CUBE存在但利用率低的matmul kernel，路由表会因为VECTOR_instr% < 15匹配到classic-matmul，但pattern card的核心信号（CUBE缺失）不存在，说明这个路由条件不恰当。 |

### B7. `vec-cmp` — Section 6 使用 `VECTOR_utilization_avg < 30 AND VECTOR_instr% < 15`（半中）

| | 内容 |
|---|---|
| **路由表** | `VECTOR_utilization_avg < 30` AND `VECTOR_instr% < 15` |
| **Pattern card** | "low or zero utilization"——定性描述，没有具体数字。也没有`VECTOR_instr% < 15` |
| **问题** | 与B5类似：路由表把定性描述硬编码成了具体数值。VECTOR利用率低确实是vec-cmp的一个信号，但30%和15%这两个阈值在pattern card中没有依据。 |

### B8. `accumulator-layout-alignment` — Section 6 使用 `VECTOR_utilization_avg < 30 AND VECTOR_instr% < 15`（中）

| | 内容 |
|---|---|
| **路由表** | `VECTOR_utilization_avg < 30` AND `VECTOR_instr% < 15` |
| **Pattern card** | 完全没有数值阈值。只有定性代码信号（accumulator形状不匹配、store时转置） |
| **问题** | 这是最严重的一例——pattern card没有任何数值型profile信号，路由表却凭空构造了阈值条件。这个pattern的触发条件应该是代码结构特征（accumulator shape和output layout不匹配），而不是仿真数据的某个数值。 |

### B9. `padded_row_col_copy` — Section 3a 使用 `SCALAR_instr% > 80 AND TRACE_total_events > 10000`（中）

| | 内容 |
|---|---|
| **路由表** | `SCALAR_instr% > 80` AND `TRACE_total_events > 10000` |
| **Pattern card** | "Profiling shows high scalar or tl.load / mask cost on last-dim pad boundaries"——定性描述，没有具体数字 |
| **问题** | 路由表把"high scalar"具体化为>80%，但pattern card没有给出这个数值。这个具体化可能有经验依据，但没有回写到pattern card中，导致路由表和card不一致。 |

### B10. `autotune` — Section 2 缺少确认信号

| | 内容 |
|---|---|
| **路由表** | `MTE2_ProcessBytes_avg < 128` OR `MTE2_Data_movers = 0`——独立gate |
| **Pattern card** | 主触发条件（ProcessBytes < 128 OR Data movers < 2）AND 确认信号（`%(MTE2&VECTOR/VECTOR) < 5%`）必须同时满足 |
| **问题** | 路由表只看ProcessBytes/Data_movers就gate掉autotune，但pattern card要求还有一个确认信号。如果ProcessBytes低但MTE2和VECTOR overlap健康（说明kernel虽然小但流水是正常的），pattern card不会fire，路由表却会提前终止autotune。这可能导致对小型但结构健康的kernel过早放弃调参。 |

### B11. `reorder-load` — Section 6 缺少利用率守卫（中）

| | 内容 |
|---|---|
| **路由表** | `VECTOR_UB_read_conflict > 100` OR `VECTOR_UB_write_conflict > 100` |
| **Pattern card** | 同样的UB conflict阈值，但额外要求："Conflicts without low utilization may be benign; fire when utilization on conflicted instructions is also < 50%." |
| **问题** | 路由表只看UB conflict > 100就fire，但pattern card要求利用率也要低（<50%）。UB conflict高但利用率也高的场景可能是正常的（高负载下的正常冲突），不应该路由到reorder-load。路由表缺少这个守卫会产生误匹配。 |

---

## C类：阈值仅Pattern Card有（pattern card有信号条件，但路由表没有覆盖）

### C1. `program-multiple-rows` — Card提供了 `SCALAR_cycles% >= 40%` 替代条件（半中）

| | 内容 |
|---|---|
| **路由表** | `SCALAR_instr% >= 70` AND `VECTOR_instr% <= 15` |
| **Pattern card** | `%(SCALAR instr) >= 70%` OR `%(SCALAR cycles) >= 40%`；`%(VECTOR instr) <= 15%` OR `%(VECTOR cycles) <= 15%` |
| **问题** | Pattern card提供了基于cycles的替代条件（SCALAR cycles >= 40%），但路由表只有基于instruction的条件。一个SCALAR_instr% = 65%但SCALAR_cycles% = 50%的kernel（说明scalar操作虽然条数不多但每条很慢）应该匹配program-multiple-rows，但路由表会漏掉。cycles和instruction的比例失衡本身就说明scalar path有延迟问题。 |

### C2. `classic-matmul` — Card提到 `SCALAR instr% > 50%` 路由表没有

| | 内容 |
|---|---|
| **路由表** | Section 3a没有classic-matmul的行 |
| **Pattern card** | `SCALAR instr% > 50%`（scalar-heavy address computation typical of manual reduction） |
| **问题** | Pattern card的SCALAR > 50%信号在路由表里没有对应行。Classic-matmul目前只通过composite rule（CUBE缺失 + MADD > 0）路由。SCALAR%信号是一个补充入口，对那些CUBE行存在但SCALAR仍然很高的手动matmul可能有用。不过>50%的阈值很低（很多kernel都超过），所以这个条件可能需要和代码特征组合才有效。 |

### C3. `loop-invariant-hoisting` — Card的TRACE事件列表更广

| | 内容 |
|---|---|
| **路由表** | `TRACE_top_types` contains LD_XD_XN_IMM/ST_XD_XN_IMM/ADD/CMP_IMM dominant |
| **Pattern card** | 还包括MUL、MADD、JUMPCMP、SIGNEXT、ZEROEXT |
| **问题** | 如果TRACE事件中MADD或SIGNEXT主导，pattern card认为应该匹配loop-invariant-hoisting，但路由表不会。路由表列出的4种事件类型覆盖了最常见场景，但pattern card的完整列表更全面。 |

### C4. `vec-cmp` — Card的TRACE事件列表更广

| | 内容 |
|---|---|
| **路由表** | `TRACE_top_types` contains CMP_IMM/JUMPC/MOVEMASK/SIGNEXT/ZEROEXT dominant |
| **Pattern card** | 还包括JUMPCMP、AND |
| **问题** | 与C3类似，路由表的事件类型列表比pattern card窄。JUMPCMP和AND主导的场景会被漏掉。 |

### C5. `remove-implicit-transpose` — Card提到AIV scalar信号路由表没有

| | 内容 |
|---|---|
| **路由表** | Step 3: `WAIT_FLAG_DEVI` dominant; `MOV_OUT_TO_L1_MULTI_ND2NZ`/nd2nz frequent |
| **Pattern card** | 还包括："AIV shows large scalar `LD_XD_XN_IMM` / `ST_XD_XN_IMM` overhead tied to staging/reorder" |
| **问题** | Pattern card提到了一个AIV scalar信号，路由表没有作为路由条件。只看WAIT_FLAG和nd2nz可能漏掉那些等待时间不长但scalar reorder开销大的场景。 |

### C6. `exact-tile-no-boundary-fast-path` — Card有Cat 2信号路由表没有对应行

| | 内容 |
|---|---|
| **路由表** | Section 1没有行把`SCALARToVECTOR avg > 50ns AND SCALAR_instr% > 75%`路由到这个pattern |
| **Pattern card** | Profile signals包括"Cat 2: Dispatch Bottleneck — SCALARToVECTOR avg > 50ns AND SCALAR instr% > 75%" |
| **问题** | Pattern card把`SCALARToVECTOR avg > 50ns AND SCALAR_instr% > 75%`作为Cat 2信号，但路由表Section 1里这个条件只路由到static-range-to-range和software-pipeline，不路由到exact-tile-no-boundary-fast-path。目前这个pattern只能通过Section 6的UB conflict条件路由到，缺少了Pipeline Flows维度的入口。 |

---

## 汇总

| 类型 | 数量 | 风险方向 |
|---|---|---|
| A: 阈值不同 | 6 | 可能误匹配也可能漏匹配 |
| B: 阈值仅路由表有 | 11 | 主要风险是误匹配——路由表fire了但card不认可 |
| C: 阈值仅Pattern Card有 | 6 | 主要风险是漏匹配——有效入口路径缺失 |
| **合计** | **23** | |

### 按严重程度排序

**P0（影响正确性，应优先修复）：**

1. **A1/A2**: `discrete_memory_access`和`grid-flatten-and-ub-buffering`缺少`SCALARToVECTOR > 0`——可能对没有load的kernel误路由
2. **A6**: `autotune` gate用`= 0`而非`< 2`——会漏掉1个data mover的case
3. **B10**: `autotune` gate缺确认信号`%(MTE2&VECTOR/VECTOR) < 5%`——可能过早停止autotune
4. **B11**: `reorder-load`缺利用率`< 50%`守卫——UB conflict高但利用率正常的场景会误触发
5. **B8**: `accumulator-layout-alignment`的card完全无数值信号——路由表的阈值是凭空构造的，这个pattern应该走Step 1代码特征路由

**P1（影响一致性，应尽快修复）：**

6. **A3**: `vec-cmp` Section 3a用绝对百分比而非比值
7. **A5**: `block-pointer-dimensionality`丢失overlap要求
8. **B5/B7/B9**: `loop-invariant-hoisting`、`vec-cmp`、`padded_row_col_copy`的card只有定性描述但路由表硬编码了数值
9. **C1**: `program-multiple-rows`缺cycles替代条件
10. **B6**: `classic-matmul`的VECTOR利用率条件在card中不存在

**P2（完善性，可后续修复）：**

11. **A4**: `pooling-inner-w-slab-gather`的TRACE行多加了AND守卫
12. **B1-B4**: `scalar-latency-traps`和`static-range-to-range`的多个路由条件card未提及
13. **C2-C6**: pattern card更广的信号覆盖（TRACE事件、SCALAR%、AIV信号等）
