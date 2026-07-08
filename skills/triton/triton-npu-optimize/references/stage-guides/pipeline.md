# Stage: pipeline — classification guide

**Stage scope**: 不改任何计算表达式,只调换语句执行顺序,让 MTE(访存)和 Compute(VECTOR/CUBE)重叠。

**Pipeline 只管"load 和 compute 的时序关系"**——load-then-compute 串行改跨迭代预取(software-pipeline)、report.txt overlap 弱的信号驱动版(software-pipeline-dependency-profiling)、同一迭代内独立 load 重排(reorder-load)。如果是算法公式问题(改表达式),那是 algorithmic;如果是 tile/参数选择,那是 parameterization;如果是编译器 hint/写法形态,那是 compile_hints。

---

## software-pipeline — 已分块热循环的 load-compute 串行

- **Signal**: 热循环已有真实分块结构但 load 和 compute 仍太串行;profiling 显示 wait-heavy/overlap-poor;下一问题是流水质量不是基本结构。
- **Pattern**: `software-pipeline`
- **Classification**: pipeline(跨迭代 MTE↔Compute 重叠:block pointer+prefetch+流水循环结构)。
- **NOT pipeline if**: 循环只跑 1-2 次(prefetch 开销 > 节省);tile 太大 UB 放不下两套(current+next);Tile i+1 依赖 Tile i 的计算结果;循环还没分块(先做 `tiling`/`classic-matmul`)。

## software-pipeline-dependency-profiling — report.txt overlap 弱信号驱动

- **Signal**: report.txt `[Pipe Overlap Ratio]` 显示 `%((VECTOR+CUBE)&MTE2/(VECTOR+CUBE))` 和 `/MTE2` 都很低;`[Pipe Distribution]` compute_cycles% 与 MTE2 cycles% 相对均衡;SCALAR 非主导;kernel 含 `tl.load` 且路径规则可构循环。
- **Pattern**: `software-pipeline-dependency-profiling`
- **Classification**: pipeline(同上,信号驱动:构 for/steady-state 循环启 prefetch,num_stages 微调)。
- **NOT pipeline if**: overlap 已经好;compute 和 MTE2 严重失衡(一端是另一端 3x+);SCALAR 主导(先做 scalar 相关);无 `tl.load`;load 是 scalar/index/control 数据(用 `scalar-latency-traps`);UB 放不下额外 live tile。

## reorder-load — 同一迭代内独立 load 重排

- **Signal**: 同一迭代内有多个无数据依赖的 load;memory-bound kernel;loop-carried 依赖存在但有独立 load 可重排;NPU 的 memory execution model 受益于 load 重排。
- **Pattern**: `reorder-load`
- **Classification**: pipeline(同一迭代内 MTE 独立 load 并行发射)。
- **NOT pipeline if**: load 顺序影响语义正确性;kernel 极小开销不值;有复杂依赖图重排会引入 race。
- **Common misclassification**: agent 常把 reorder-load 归到 `memory_access`。**不对**——reorder-load 不改地址/搬运路径/事务宽度(那是 memory_access),只调换 load 顺序改善 MTE 并行(是时序重排,属于 pipeline)。
