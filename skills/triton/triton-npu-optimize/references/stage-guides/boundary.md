# Stage: boundary — classification guide

**Stage scope**: 改 kernel / wrapper 的职责分界线。外部计算迁入内核(计算融合)或删除 wrapper 搬运(布局拷贝消除)。

**Boundary 只管"wrapper ↔ kernel 之间多了一层不该有的东西"**——要么是 wrapper 里多了一个 Python for-loop 逐轮 launch、多了一个外部辅助算子(aclnn*)、多了一个 `.contiguous()` 物化 copy。如果问题在 kernel 内部的地址计算、访存模式、循环结构,那是 memory_access / algorithmic / pipeline / compile_hints 的活,不是 boundary。

---

## auxiliary-op-fusion — 外部辅助算子未融合

- **Signal**: perf 输出显示在主 Triton kernel 之前有一串辅助算子(scales/masks/clamps/casts/offsets/row stats/broadcasted factors/frequency counts),它们的合计时间在 `total_op_avg_time_us` 里占比有意义。
- **Pattern**: `auxiliary-op-fusion`
- **Classification**: boundary(计算融合:外部计算迁入内核)。
- **NOT boundary if**: 辅助操作是纯布局 copy(`aclnnInplaceCopy`/`Contiguous`/`Transpose`)——那是 `layout-materialization-elision`。如果有复杂的全局语义(sort/topk/unique/nonzero)或辅助输出是 API 可见的主输出——不做融合。
- **Common misclassification**: agent 常把 `.contiguous()` copy 归到这里。**不对**——纯布局 copy 是 `layout-materialization-elision`,不是算术融合。

## layout-materialization-elision — 布局物化 copy 未消除

- **Signal**: wrapper 代码里有 `permute(...).contiguous()` / `transpose(...).contiguous()` / `movedim(...).contiguous()` / `.expand(...).contiguous()` / `clone()` / `copy_()`,下一步立即被 kernel 消费。profiling 显示 `Transpose`/`Contiguous`/`DataCopy`/`Memcpy`/`aclnnInplaceCopy` 占有意义时间。
- **Pattern**: `layout-materialization-elision`
- **Classification**: boundary(布局拷贝消除:kernel 改为直接接受原始布局)。
- **Common misclassification**: agent 常把 `.expand().contiguous()` broadcast 物化归到 `memory_access`("kernel 要处理 broadcast stride")。**不对**——消除物化 copy 是 boundary;stride 处理是消除的**机制**,跟消除 `permute().contiguous()` 时 kernel 处理 strided layout 一样。`.expand().contiguous()` 跟 `permute().contiguous()` 是同类(都是 layout 物化 copy 被 kernel 立即消费)。
- **NOT boundary if**: 物化后的 tensor 被多个后续 kernel 复用(物化一次比重复 strided 访问更便宜);或 consumer 真的要求物理 contiguous(backend fast path);或 tensor 极小 copy 代价可忽略。

## reduce-avoid-transpose-copy — 非末轴运算前的 transpose copy

- **Signal**: 算子沿非末轴做 axis-wise 运算(reduction/cumsum 等),当前用 `movedim(...).contiguous()` 或 `transpose(...).contiguous()` 把目标轴挪到末维再做。输入是 contiguous row-major。
- **Pattern**: `reduce-avoid-transpose-copy`
- **Classification**: boundary(布局拷贝消除的特化:非末轴运算改为 `[outer, scan, inner]` 跨步直算)。
- **NOT boundary if**: 目标轴已经是末轴;是全 tensor reduction(`dim=None`);输入不 contiguous 且 kernel 不处理 real stride;`inner_size` 极小(如 `[B,M,1]` 的 dim=1 → inner_size=1)。

## attention-cv-pipeline(mask 预计算)— mask 预计算融合

- **Signal**: `tl.dot` 循环后跟大量 vector epilogue(scale/mask/softmax/dropout/bias);或循环内重复从 sequence lengths 重建 mask tensor;或 scale 和 mask 是分离操作。
- **Pattern**: `attention-cv-pipeline`(boundary 侧重:mask 预计算融合到 kernel)
- **Classification**: boundary(计算融合:mask 预计算从外部迁入内核)。
- **Note**: 此 pattern 也出现在 `compile_hints`(A5 编译参数)和 `algorithmic`(scale+mask 合并/exp2→exp)——judge 时先判 boundary 侧(mask 预计算/融合),compile_hints 侧(A5 参数)和 algorithmic 侧(表达式合并)在其他 stage 判。
- **NOT boundary if**: kernel 是纯 Vector(无 Cube);bottleneck 是 memory transfer 不是 vector epilogue。
