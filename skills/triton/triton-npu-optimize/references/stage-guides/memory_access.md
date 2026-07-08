# Stage: memory_access — classification guide

**Stage scope**: 改 `tl.load`/`tl.store` 的地址生成方式、搬运路径、事务宽度。分三组:地址计算方式(3A)、批量搬运与片上拣选(3B)、碎片事务融合(3C)。

**Memory_access 只管"kernel 内部数据怎么从 GM 搬到寄存器/UB、地址怎么算、DMA 宽不宽"**——div/mod 坐标解码(flat-index-decode)、block_ptr 替代手算地址(block-pointer-dimensionality)、GM→UB staging+gather(discrete/stencil/sliding-window/slice_coalesce)、窄 store 合并(merge-adjacent-stores)、store 端转置退化(accumulator-layout-alignment)、GEMM 操作数隐式转置(remove-implicit-transpose)。如果问题是 wrapper 里的物化 copy,那是 boundary;如果是循环顺序/表达式公式,那是 algorithmic/pipeline。

---

## flat-index-decode-tiling — flat 1D 索引 + div/mod 坐标解码

- **Signal**: kernel 主要是数据搬运(非稠密算术),launch 在 flat `n_elements`/`out.numel()` 上,每 lane 用 `//`/`%` 恢复多维坐标,输出→输入映射是仿射(strides/axis reorder/fixed offsets/padding/slice)。
- **Pattern**: `flat-index-decode-tiling`
- **Classification**: memory_access(3A 地址计算:SCALAR→VECTOR,消除 `//`/`%`)。
- **NOT memory_access if**: 索引是值依赖的(irregular gather/scatter)——用 `discrete_memory_access`;或物化 copy 可以消除(那是 boundary 的 `layout-materialization-elision`);或计算量大到 index decode 不是瓶颈。

## padded_row_col_copy — copy/pad 类算子的 flat 1D 遍历

- **Signal**: 算子是 constant pad / slice+pad / 其他 per-axis 边界 elementwise map(非 gather),baseline 用 `pid*BLOCK+arange` over `numel(out)` + heavy div/mod 做所有坐标,profiling 显示高 scalar 或 tl.load/mask 代价在 last-dim pad 边界。
- **Pattern**: `padded_row_col_copy`
- **Classification**: memory_access(3A 地址计算:flat-index-decode 的 copy/pad 特化)。
- **NOT memory_access if**: 热路径是 gather/scatter(用 `discrete_memory_access`)。

## block-pointer-dimensionality — 手算地址替代 block_ptr

- **Signal**: 高维连续张量被扁平 1D offset 跨内维访问,内维被显式循环或 program_id 解码;profiling/IR 显示 1D pointer 路径产生 strided/non-coalesced loads;report.txt 显示高 SCALAR-to-VECTOR 比(地址生成主导)。
- **Pattern**: `block-pointer-dimensionality`
- **Classification**: memory_access(3A 地址计算:SCALAR→MTE,DMA 描述符硬件解析)。
- **NOT memory_access if**: 访问已经是 coalesced 的;或 SCALAR 和 VECTOR 已经 well-overlapped。

## discrete_memory_access — 散列 GM 直连改为 UB staging+gather

- **Signal**: 瓶颈是离散访存 `out=x[idx]`,index-driven 全局 load 主导热路径;gather 源 array 小/中到可以 stage 进 UB。
- **Pattern**: `discrete_memory_access`
- **Classification**: memory_access(3B 批量搬运:连续范围 stage 进 UB + `tl.gather` 选取)。
- **NOT memory_access if**: 源范围太大无法 stage;访存已连续;会引入不支持的 Ascend tensor indexing(`vec[0]` on loaded tile)。

## sliding-window-inner-w-slab-gather — per-kw 窄 load 改为 slab+gather

- **Signal**: 固定 `KERNEL_W` 窗口归约沿 W(contiguous NCHW),profiling 显示 `kw` 内 repeated narrow/predicate-heavy global loads on W。
- **Pattern**: `sliding-window-inner-w-slab-gather`
- **Classification**: memory_access(3B 批量搬运:单次 slab wide load + per-kw gather)。
- **NOT memory_access if**: A5 SIMT 固定窗归约(用 `a5-simt-sliding-window-tuning`);layout 非 W-contiguous;max-like 归约带 return_indices/dilation。

## stencil-resize-gm-to-ub-staging — 每样本独立 GM load 改为 UB slab

- **Signal**: kernel 是 memory-bound 2D 采样(resize/gather-stencil/pool),input read count 随 (输出像素×stencil footprint) 增长;IR 显示 per-lane GM load 的 DiscreteMemAccess/ExtractedLoadOrStore。
- **Pattern**: `stencil-resize-gm-to-ub-staging`
- **Classification**: memory_access(3B 批量搬运:GM→UB slab 展平后 gather)。
- **NOT memory_access if**: 一个输入样本对应一个输出且已是 row-coalesced;输入 layout 非 contiguous。

## slice_coalesce — scatter/gather UB 批处理

- **Signal**: scatter/gather 式数据搬运主导(token rearrangement/sparse reordering/index-based movement),UB 批处理可替代多次随机 GM 访问。
- **Pattern**: `slice_coalesce`
- **Classification**: memory_access(3B 批量搬运:`extract_slice`/`insert_slice` 最大化复用)。
- **NOT memory_access if**: 访问已是连续的;scatter/gather 量极小。

## merge-adjacent-stores — 多个窄 store 合并为宽 store

- **Signal**: 多个 `tl.store` 写相邻地址但作为独立小 store;dest 可证连续;profiling 显示 store 粒度限制吞吐。
- **Pattern**: `merge-adjacent-stores`
- **Classification**: memory_access(3C 碎片事务融合:多条窄 DMA→一条宽 DMA)。
- **NOT memory_access if**: 目标地址不连续;mask 在候选 store 间不同;合并会超 UB/register 容量。

## accumulator-layout-alignment — store 端隐式转置退化

- **Signal**: `tl.store` 写转置逻辑张量,profiling/代码暗示退化成标量逐元素写;累加器 shape 与输出内存布局不同,迫使 store 端隐式转置;reduction 产生"错"的 shape 序。
- **Pattern**: `accumulator-layout-alignment`
- **Classification**: memory_access(3C 碎片事务融合:对齐累加器 shape,避免 store 退化)。
- **NOT memory_access if**: store 端没有转置问题(累加器已对齐)。

## remove-implicit-transpose — GEMM 操作数隐式转置

- **Signal**: GEMM/Linear 中操作数存为 `[N,K]` 但数学需 `[K,N]`(`y=x@w.T`);kernel 用 transpose-like strides 访问;`tl.dot` 操作数用 `tl.trans(x).to(dtype)`;profiling 显示高 scalar/control 或大 WAIT_FLAG。
- **Pattern**: `remove-implicit-transpose`
- **Classification**: memory_access(3C 碎片事务融合:wrapper 预处理权重重排,kernel 直接连续加载)。
- **NOT memory_access if**: 操作数没有隐式转置(已经是正确 layout);或 `tl.trans` 不是瓶颈。
- **Common misclassification**: agent 常把 `.contiguous()` 物化 copy 归到这里。**不对**——如果是 wrapper 里 `permute().contiguous()` 在 kernel 之前,那是 boundary 的 `layout-materialization-elision`;如果是 kernel 内 `tl.trans` 喂 `tl.dot`,才是这里的 `remove-implicit-transpose`。
