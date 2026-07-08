# Stage: parallel — classification guide

**Stage scope**: 改 program 间的分布与协作——粒度调整(减 program 数,增每 program 数据量)和争用缓解(program 间对共享资源的争抢)。

**Parallel 只管"program 之间怎么分活/怎么抢资源"**——program 数太多/每 program 太小(program-multiple-rows)、grid 轴太多需展平(grid-flatten)、atomic 写争用(owner-computes)、L2 bank 读争用(diagonal)。如果问题在 kernel 内部的地址计算或访存路径,那是 memory_access;如果在循环表达式/算法公式,那是 algorithmic。

---

## program-multiple-rows — 每 program 处理行数太少

- **Signal**: kernel 是 row-wise(row reduction/row-wise fused epilogue/row-major transform),当前一个 program 处理一行(`BLOCK_M=1`),profiling 显示 many thin programs / scalar-heavy overhead。内维度 N 的 streaming 可以保持单 pass 同时增宽 row count。
- **Pattern**: `program-multiple-rows`
- **Classification**: parallel(粒度调整:增大 BLOCK_M,grid 维度不变)。
- **NOT parallel if**: row count 极小无法 amortize;或增 `BLOCK_M` 引入第二遍/数值不稳;或主 bottleneck 在别处(layout/store shape/algorithm/scalar traps)。

## grid-flatten-and-ub-buffering — grid 轴太多需展平

- **Signal**: 逻辑 grid 远大于物理 AICore/VectorCore 数;按 batch/sequence 桶分且有可见负载不均;grid-to-core 映射后每 program 处理很多小行;gather 代码 dest 行连续却逐行 store;scatter-weight-gradient 代码有可批量的重复行 load。
- **Pattern**: `grid-flatten-and-ub-buffering`
- **Classification**: parallel(粒度调整:合并 grid 轴进 program 内部循环,grid 维度减少)。
- **NOT parallel if**: grid 已经合理;问题是 kernel 内部访存(那是 memory_access)。

## atomic-contention-owner-computes-store — atomic 写争用

- **Signal**: 热 kernel 用 `tl.atomic_add`/`tl.atomic_max` 从很多 program 更新小/中输出域(histogram bins/class buckets/segment IDs/sparse row buckets);profiling/benchmark scaling 显示 atomic/store 端争用贵于 owner 重读输入。
- **Pattern**: `atomic-contention-owner-computes-store`
- **Classification**: parallel(争用缓解:换并行策略,grid 从输入 tile 转置到输出目标,owner-computes 消除 atomic)。
- **NOT parallel if**: 输出域大到 owner-computes 会乘爆 GM 读量;atomic 碰撞已经很少(地址分布均匀);reduction 操作不满足交换律(顺序依赖);bottleneck 是普通连续 load 带宽或 scalar 地址生成(不是 atomic 争用)。
- **Note**: 单程序仿真不能量化多 program atomic 争用——需要 benchmark scaling 或 msprof 证据。

## diagonal — L2 cache bank 读争用

- **Signal**: 大块 tiled matrix-style 工作虽有合理分块却显示局部性差或 bank-conflict-like 行为;很多 program 同时触碰相同 cache 区。
- **Pattern**: `diagonal`
- **Classification**: parallel(争用缓解:换遍历顺序改善 L2 带宽分摊)。
- **NOT parallel if**: 工作量小没有 L2 争用;问题是 kernel 内部访存路径(那是 memory_access 的 block-pointer/staging 类)。
