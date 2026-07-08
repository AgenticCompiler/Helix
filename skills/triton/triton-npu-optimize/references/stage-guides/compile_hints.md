# Stage: compile_hints — classification guide

**Stage scope**: 如何让编译器 lower 更好。分两组:6A 标记式(给编译器无法从代码推断的事实)和 6B 形态式(换写法,工作量不变,编译器自行 better lower)。

**Compile_hints 只管"不改计算量、不改访存路径、不改 program 分布,只通过给编译器 hint 或换写法让它 lower 更好"**——max_contiguous/multiple_of(compile_hint)、constexpr(scalar-latency-traps)、A5 编译参数(attention-cv-pipeline)、双 Vector Core(parallel)、SIMT 执行模型(a5-force-simt-only-discrete-access)、标量比较链→向量掩码(vec-cmp)、static_range→range(static-range-to-range)、单次大 tile→分步 slice(slice_intermediate)、int64→int32 / ptr+= → base+offs / %→mask(scalar-latency-traps 形态部分)。如果是算法改写(减 FLOPs/pass),那是 algorithmic;如果是 load-compute 时序,那是 pipeline;如果是 tile/参数值选择,那是 parameterization。

---

## compile_hint — max_contiguous / multiple_of / compile_hint()

- **Signal**: 热 kernel 结构已好但 lowering 仍保守;可证明比代码更强的对齐/连续性事实;`tl.dot` 输入稳定只需 padding 指引;parent 对比已接近,小 lowering 改动仍重要。
- **Pattern**: `compile_hint`
- **Classification**: compile_hints(6A 标记式:内存布局事实)。
- **NOT compile_hints if**: 主问题仍结构(wrong tiling/launch/algorithm);对齐假设 shape-conditional 未 dispatch 守卫;用 hint 补偿 invalid pointer/index math;launch hint 在 algorithm/layout/grid 稳定前就改。

## scalar-latency-traps(形态部分)— constexpr / int64→int32 / ptr+= → base+offs / %→mask

- **Signal**: 运行时值(shape 常量)当普通参数传(应 `tl.constexpr`);int64 算术在 vector 路上(应 int32);`ptr += stride` 循环内递推(应 `base + offs`);`(start+arange)%N` 可改 `start+arange; mask`。
- **Pattern**: `scalar-latency-traps`
- **Classification**: compile_hints(6A 标记式 constexpr / 6B 形态式 int64→int32/ptr+=%→mask)。
- **Note**: 此 pattern 也出现在 `algorithmic`(语义部分:cumsum 退化 / 重复 mask)。judge 时:`tl.constexpr` / int64→int32 / ptr+= / %→mask = compile_hints(形态改写,不改计算量);`tl.cumsum` 退化 / 重复 mask 删除 = algorithmic(语义改写,减 FLOPs)。
- **NOT compile_hints if**: shape 常量传成 constexpr 是结构改写的一部分(先做结构 pattern);或问题是地址计算 div/mod(用 `flat-index-decode-tiling`)。

## attention-cv-pipeline(A5 编译参数)

- **Signal**: 目标已知 A5(ascend950PR/DT);`tl.dot` 循环后跟 vector epilogue;profiling 显示 Cube+Vector 接近以致 vector 开销限制重叠。
- **Pattern**: `attention-cv-pipeline`(compile_hints 侧重:A5 门控 compile option)
- **Classification**: compile_hints(6A 标记式:硬件事实——A5 编译参数)。
- **Note**: 此 pattern 也出现在 `boundary`(mask 预计算融合)和 `algorithmic`(scale+mask 合并 / exp2→exp)。
- **NOT compile_hints if**: 目标非 A5;kernel 纯 Vector 无 Cube;memory transfer 是瓶颈不是 vector epilogue。

## parallel — 双 Vector Core 可用

- **Signal**: 两个独立 vector-side 计算串行;bottleneck 非访存(暴露更多 vector-core 并发比改 load 更有前途)。
- **Pattern**: `parallel`
- **Classification**: compile_hints(6A 标记式:资源事实——`tl.parallel` 让两个 task 跑在两个 vector core 上)。
- **NOT compile_hints if**: 两个计算有真实数据依赖;操作主要是 memory loading(共享带宽已是瓶颈);操作极小 `tl.parallel` 开销可能大于收益。

## a5-force-simt-only-discrete-access — SIMT 执行模型

- **Signal**: 目标确认 A5;msprof `op_summary` 行匹配核名且 `aiv_scalar_ratio` 明显高于 `aiv_vec_ratio` 和 `cube_utilization`;核体主要离散/索引驱动访存。
- **Pattern**: `a5-force-simt-only-discrete-access`
- **Classification**: compile_hints(6A 标记式:策略事实——走 SIMT 执行模型)。
- **NOT compile_hints if**: kernel 是 Cube-heavy/matmul-like/vector arithmetic 主导;profiling 不能确信匹配核名;scalar ratio 只是略高;bottleneck 是 host launch/copy/另一个 op;shape 范围不代表性;scalar 代价来自 flat 1D decode(先修 `flat-index-decode-tiling`)。

## vec-cmp — 标量比较链→向量掩码

- **Signal**: explicit i64/i32 比较出现在 hot path(非正常 load/store mask);comparison-heavy control flow 看起来是 vectorization blocker;report.txt SCALAR:VECTOR>10、VECTOR top 指令是 MOVEMASK、TRACE 含大量 CMP_IMM/JUMPCMP/MOVEMASK/SIGNEXT。
- **Pattern**: `vec-cmp`
- **Classification**: compile_hints(6B 形态式:标量比较链→向量掩码)。
- **NOT compile_hints if**: 比较在 `tl.load`/`tl.store` mask 里(已自动优化);已用 fp32 比较;非热路径;report.txt 不支持 scalarized-compare bottleneck;scalar 代价来自 index decode(先做 `flat-index-decode-tiling`)。

## static-range-to-range — static_range → range

- **Signal**: 热循环用 `tl.static_range`(或 `tl.range` 带 constexpr bound 触发全展开);迭代独立;body 轻量;profiling MTE2 占比相对 VECTOR 异常低、SCALAR/MTE2 overlap 差。
- **Pattern**: `static-range-to-range`
- **Classification**: compile_hints(6B 形态式:`tl.static_range`→`tl.range`,保留循环结构让编译器流水)。
- **NOT compile_hints if**: body 计算重/中间量多(range 加 register rename 压力);BLOCK_SIZE≥4096 且 body 复杂;num_warps≥8;迭代数≤4;有跨迭代依赖。

## slice_intermediate — 单次大 tile → 分步 slice

- **Signal**: 中间张量(非仅输入/输出)是 UB 压力主因;算法合理但需分步 slice 处理使临时量留在片内。
- **Pattern**: `slice_intermediate`
- **Classification**: compile_hints(6B 形态式:单次大 tile→分步 slice,工作量不变编译器自行 better lower)。
- **NOT compile_hints if**: UB 压力来自输入/输出而非中间量;或 tile 已经合理。
