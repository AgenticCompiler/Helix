# Stage: parameterization — classification guide

**Stage scope**: 在 kernel 结构和算法确定后,做参数值选择或 dispatch 级路径决策。7A 参数调优(BLOCK/grid/num_warps 选值)和 7B 路径拆分(基于参数选内核变体)。

**Parameterization 只管"结构已经定型,参数该取什么值/该走哪条路径"**——autotune 搜索(tile/num_stages)、tile-selection-heuristic 主机侧 sweep、tiling 设 BLOCK 上限(UB 容量)、effective-extent-tiling 用实际形状替代固定幂次、a5-simt-sliding-window-tuning 推导 BLOCK+grid+num_warps、exact-tile-no-boundary-fast-path 整除 dispatch 无 mask 变体。如果是结构改写(算法/访存/布局/流水),那是前置 stage;如果是编译器 hint/写法形态,那是 compile_hints。

---

## autotune — 搜索 BLOCK / num_stages / num_warps

- **Signal**: kernel 数学正确稳定过验证;结构已合理;热路径暴露 free `tl.constexpr` 参数未硬编码;bounds masks/循环结构可映射回 runtime shape(shape-keyed cache 可行);report.txt 显示 overlap 弱 / WAIT_FLAG 高 / register pressure 等。
- **Pattern**: `autotune`
- **Classification**: parameterization(7A 参数调优:搜索 BLOCK_M/N/K、num_stages)。
- **NOT parameterization if**: 真问题是结构(手写归约应先 `classic-matmul`);算法/layout 在大改(先稳定结构);全被 GM 带宽限制(compute 已藏于 MTE,autotune 到顶了);所有 `tl.constexpr` 已硬编码无搜索空间;语义约束锁死 grid/tile;kernel correctness-fragile 未加 reset/restore。
- **Common misclassification**: agent 常在结构还没做对时就想 autotune。**不对**——autotune 是 parameterization(P7),要等前置 stage(结构/访存/算法/流水/编译器)都做完才轮到。先结构后调参。

## tile-selection-heuristic — 主机侧 sweep 选最小化 grid 的 tile

- **Signal**: autotune 有限配置在多样 shape 下产生不稳定赢家;算子跨 50+ shape 评估;kernel 已用 2D grid;autotune 开销/key 设计有问题。
- **Pattern**: `tile-selection-heuristic`
- **Classification**: parameterization(7A 参数调优:主机侧 `_choose_tiling` sweep)。
- **NOT parameterization if**: 核心 algorithm/layout 还在改(先 `program-multiple-rows`);kernel 用 1D grid 无 2D tile 分解;搜索空间极小(手动选就够)。

## tiling — 设 BLOCK 上限适配 UB

- **Signal**: block sizes / live intermediates / 多 tensor load 有 UB 溢出或局部性差风险;主问题是 working-set 大小/内存 footprint,非需要完全不同结构。
- **Pattern**: `tiling`
- **Classification**: parameterization(7A 参数调优:设 BLOCK 上限,UB 192KB)。
- **NOT parameterization if**: BLOCK 已小无压力;单 tensor 简单操作 UB 占用极小;已有 sub-blocking;真问题是结构(手写归约应先 `classic-matmul`)。

## effective-extent-tiling — 用实际形状替代固定幂次

- **Signal**: `BLOCK_*` 远大于 mask 保护的有效区间;热路径是 indexed/masked 或 copy-like 其宽度不参与 `tl.dot`/cube 对齐;profiling 显示代价随 tile 宽度而非活元素数增长;host 有 shape 信息可选更小 tile。
- **Pattern**: `effective-extent-tiling`
- **Classification**: parameterization(7A 参数调优:实际形状替代固定幂次)。
- **NOT parameterization if**: 轴参与 `tl.dot`/cube 对齐/reduction-tree(backend 敏感对齐);bottleneck 是 atomic/host overhead;更小 tile 会爆 program 数;runtime extents 变化太大 shape-specialized 不可维护。

## a5-simt-sliding-window-tuning — 推导 BLOCK+grid+num_warps

- **Signal**: kernel 匹配 pattern signature(固定窗口+仿射映射+逐输出归约);A5 确认且热路径 scalar/index-heavy(`a5-force-simt-only-discrete-access`);多 shape harness 可用 geomean 判收。
- **Pattern**: `a5-simt-sliding-window-tuning`
- **Classification**: parameterization(7A 参数调优:从结构特征推导 BLOCK+grid+num_warps)。
- **NOT parameterization if**: 非 A5 SIMT 或 compute-bound dense vector math 主导;不能在 same module/process 混 SIMT 和 non-SIMT。

## exact-tile-no-boundary-fast-path — 整除 dispatch 无 mask 变体

- **Signal**: 主要 benchmark shape 整除 BLOCK(`M % BLOCK_M == 0` and `N % BLOCK_N == 0`);Python dispatch 可守卫 aligned branch 保留 masked fallback;MLIR/profiler 仍显示 boundary check/mask/padding/branch overhead on exact-tile hot path。
- **Pattern**: `exact-tile-no-boundary-fast-path`
- **Classification**: parameterization(7B 路径拆分:基于参数做 dispatch 级决策,选无 mask 内核变体)。
- **NOT parameterization if**: mask 是算法语义不是边界/tail guard;整除不能在 dispatch 证明;tail-heavy/irregular shape 主导;bottleneck 是 random GM/atomic/compute(boundary control 可忽略);fast path 会复制太多复杂逻辑 drift。
- **Common omission**: agent 常不识别这个机会——因为 mask 在 exact-fit shape 上全 true,**没有"错误"**,只是"有更好的路可走"。"所有 shape 整除 BLOCK + mask 全 true"就是 signal。如果 profiling 显示 SCALAR 仍高(83%+,mask construction 贡献),这就是 exact-tile-fast-path 的机会。
