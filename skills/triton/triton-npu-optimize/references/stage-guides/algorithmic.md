# Stage: algorithmic — classification guide

**Stage scope**: 换公式 / 表达式 / 数据类型,语义不变,工作量(FLOPs / pass 数 / 中间量)减少。

**Algorithmic 只管"能不能换一个等价的数学表达让计算量更少"**——两次遍历合并为一次(Welford)、broadcast 后比较改为先比较再广播、`exp(g_i-g_j)` 矩阵改向量分解、手写 K 归约改 `tl.dot`、2D mask+reduce 改 1D 索引流、循环不变量外提、窄 where/cumsum 退化/重复 mask/constexpr 等标量陷阱(语义部分)。如果是 kernel 内部访存路径/地址计算,那是 memory_access;如果是循环顺序/流水重叠,那是 pipeline;如果是给编译器 hint/换写法不改计算量,那是 compile_hints。

---

## algebraic-optimization — 冗余 pass / 可合并表达式

- **Signal**: 热路径对同一数据做两次以上全遍历(如先 mean 再 rstd);profiler/IR 显示 duplicate MTE-heavy 阶段只差一个标量统计量;elementwise logical ops 用 broadcasting 后才比较;pairwise gated tiles 的 `exp(g_i-g_j)` 只作乘法因子。
- **Pattern**: `algebraic-optimization`
- **Classification**: algorithmic(换公式:两遍→一遍 / 矩阵分解→向量分解)。
- **NOT algorithmic if**: bottleneck 只是 tile size/UB overflow 无冗余 pass(用 `tiling`);或 graph 层 reorder 更简单更便宜。

## classic-matmul — 手写 K 归约未用 tl.dot

- **Signal**: kernel 计算 M×N 输出 tile + 对 K 规整归约,当前是 `sum_k A[...,k]*B[...,k]`;report.txt 显示 CUBE instr=0 且源含 `tl.sum(a*b)`。
- **Pattern**: `classic-matmul`
- **Classification**: algorithmic(换公式:手写归约→`tl.dot`,映射到 Cube)。
- **NOT algorithmic if**: 纯 elementwise / gather-scatter 主导;shape 极小 tile setup 无法 amortize;`tl.dot` 已在但 tile/lowering 次优(用 `tiling`/`software-pipeline`)。
- **Common misclassification**: agent 常把"需要 autotune"归到这里。**不对**——autotune 在结构正确后才做(parameterization),classic-matmul 是结构改写(algorithmic),先结构后调参。

## simt-clip-window-closed-reduction — per-tap 计数改闭式窗口体积

- **Signal**: SIMT 激活(`force_simt_only=True`);固定仿射窗;normalizer=clipped tap count(排除虚拟 pad);热路径为 `for kd/kh/kw` + per-tap `valid_*`/`safe_*` 和/或 `count+=tl.where`。
- **Pattern**: `simt-clip-window-closed-reduction`
- **Classification**: algorithmic(换公式:per-tap 累加计数→闭式窗口体积)。
- **NOT algorithmic if**: SIMT 未激活(先做 `a5-force-simt-only-discrete-access`);索引值依赖/窗口非仿射;include-pad normalizer 要求(用 coordinate-mask + one-shot);kernel 已被 Cube 主导(scalar 窗口簿记不是瓶颈)。

## shift-2d-mask-to-1d-index-stream — 2D mask+reduce 改 1D 索引流

- **Signal**: shift 关系是"取前一元素/块内前一位置",代码用 `arange[:,None]`+`arange[None,:]`+`tl.where`+`tl.sum(...,axis=...)` 做 2D mask 构造;IR 显示 `tt.broadcast`/`tt.reduce`/为 shift 专设的临时 mask 张量。
- **Pattern**: `shift-2d-mask-to-1d-index-stream`
- **Classification**: algorithmic(换公式:2D mask-reduce→1D 索引流 `base+arange-1`)。
- **NOT algorithmic if**: 依赖不是简单前驱(multi-source gather);边界行为依赖非局部逻辑;路径不热;shifted 中间量被多个后续表达式复用(直接前驱 reload 会加流量)。

## scalar-latency-traps(语义部分)— 标量化构造导致退化

- **Signal**: `tl.cumsum`/`tl.associative_scan` 在末轴标量退化;长一维 `tl.cumsum` 标量退化;边界 mask 重复判定(已被 `boundary_check`/zero-padding 处理过的条件又判一遍)。
- **Pattern**: `scalar-latency-traps`
- **Classification**: algorithmic(换公式:轴重排向量化 / 删除冗余 mask 检查)。
- **Note**: 此 pattern 也出现在 `compile_hints`(形态部分:int64→int32 / ptr+= → base+offs / % → mask / tl.constexpr)——judge 时:`tl.cumsum` 退化 / 重复 mask = algorithmic(语义改写);int64→int32 / ptr+= / %→mask / constexpr = compile_hints(形态改写,不改计算量)。
- **NOT algorithmic if**: 问题是地址计算(flat 1D decode)——用 `flat-index-decode-tiling`;或 loop 不变量——用 `loop-invariant-hoisting`;或比较链——用 `vec-cmp`。

## loop-invariant-hoisting — 循环不变量未外提

- **Signal**: 热 K 循环每次重复指针 math / mask 构造 / type cast / shape 簿记;report.txt 显示 SCALAR instr/cycle 占比高、SCALAR≫VECTOR、SCALAR Instr Types 被 ADD/MUL/MADD/CMP/JUMPCMP/SIGNEXT 主导。
- **Pattern**: `loop-invariant-hoisting`
- **Classification**: algorithmic(换公式:每轮重算→循环外一次)。
- **NOT algorithmic if**: 重复工作真依赖循环变量(不能拆成 invariant base + varying delta);MTE2/MTE3 主导(不是 scalar 簿记);CUBE/VECTOR 利用已高;scalar 代价来自 flat 1D decode(用 `flat-index-decode-tiling` 或 `block-pointer-dimensionality`)。

## attention-cv-pipeline(scale+mask 合并 / exp2→exp)

- **Signal**: `scores*scale; scores+=mask` 分离操作可合并为一次表达式;或 `exp2(x*log2e)` 可用 `tl.exp(x)` 替代。
- **Pattern**: `attention-cv-pipeline`(algorithmic 侧重:表达式合并 / 函数替换)
- **Classification**: algorithmic(换公式:分离操作→合并 / exp2→exp)。
- **Note**: 此 pattern 也出现在 `boundary`(mask 预计算融合)和 `compile_hints`(A5 编译参数)。
- **NOT algorithmic if**: kernel 不是 attention(scale/mask 不适用);或表达式已合并。
