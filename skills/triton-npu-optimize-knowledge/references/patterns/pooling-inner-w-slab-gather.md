# Spatial pooling — innermost W slab plus gather (2D / 3D and beyond)

## Summary

For **sliding-window spatial pooling** in **NCHW-style** layouts (**W** is the **innermost spatial** dimension with **contiguous** columns), replace the inner **`kw` loop** that does **per-lane masked `tl.load` on scattered input columns** (`start_w + kw` or equivalent) with **one contiguous nominal slab** along **input column index** of length **`W_SLAB_LEN = STRIDE_W * (BLOCK_OW - 1) + KERNEL_W`**, then **`tl.gather`** with lane indices **`STRIDE_W * tl.arange(BLOCK_OW) + kw`** for each **`kw`**, accumulating into a **`BLOCK_OW`-wide** vector. **Depth of outer loops** matches problem rank: **2D pooling** uses **`kh` / `kw`** only; **3D** adds **`kd`** (and optional **D** offsets) the same way—**W-slab geometry does not depend on 2D vs 3D.** Host picks **`BLOCK_OW`** from **`out_w`** (e.g. divisor-based candidates up to a cap). Branch with **`tl.constexpr`**: unmasked **`USE_W_SLAB_LOAD`** when **zero padding**, **full `out_w` tile alignment**, and (for **3D**) **every window fully inside** the unpadded input if required; **`USE_W_MASKED_SLAB`** when **tiles are full** but **slab columns can be OOB** (pad / ceil), using **`tl.load(..., mask=, other=0)`** on the slab then the same **`gather`**; else **`NO_PADDING_FASTPATH`** (vector loads per **`kw`**) or **generic boundary** loads. **Grid** is workload-specific; a common **2D** pattern is **`(batch * channels * out_h, cdiv(out_w, BLOCK_OW))`**; **3D** often folds **`out_d * out_h`** into the row axis. Pair with **`program-multiple-rows`** when slab setup should be amortized across consecutive flat spatial rows; pick grid axis order from measured launch vs reuse on the target NPU.

## Use When

- The kernel is **AvgPool2d / AvgPool3d**, **MaxPool2d / MaxPool3d** (**values only**), or any **fixed `KERNEL_W`** reduction along **W** on a **contiguous** NCHW (or 5D) tensor, with **`kw`** in a constexpr loop and **`BLOCK_OW`** outputs per program.
- IR or profiling shows **many narrow or predicate-heavy global loads** along **`kw`** while **`stride_w`** maps output columns to **regularly strided** input columns.
- **`out_w`** is large enough that **vectorizing along `ow`** matters, and **`cdiv(out_w, BLOCK_OW)`** does not hurt launch scalability in measurement.
- You can prove **semantic equivalence** on the branches you enable (**ceil**, **padding**, **divisor** / **count_include_pad** for average, **numeric identity** for max on **`dtype` / `-inf`** rules).

## Avoid When

- **`W_SLAB_LEN`** exceeds **UB / compiler** or **gather** limits—**reduce `BLOCK_OW`** or stay on a simpler branch.
- **`USE_W_SLAB_LOAD`** host predicates are wrong (windows not fully inside, non-zero pad misclassified)—**compare to PyTorch** reference.
- **Tail `out_w`** (`out_w % BLOCK_OW != 0`) needs correct **`ow_mask`** / **store** semantics; do not assume **full tiles** on slab branches without **tail** handling.
- **MaxPool** with **`return_indices`**, or **complex validity / masking** (e.g. **`seen_valid`**, dilation edge cases)—**re-derive** pooling; **W-slab indexing still applies** to loads but **combine / store** differ.
- **Layout** is not **W-contiguous** per row (e.g. certain **channels-last** views without **contiguous()** on the slice you pool).

## Signals

### Code

- **2D**: nested **`for kh` / `for kw`** with **`tl.load(..., mask=...)`** on **W** per **`kw`**. **3D**: adds **`for kd`** outside **`kh`**.
- **`W_SLAB_LEN`** contiguous **`tl.load`** along **`w`**, then **`tl.gather(slab, STRIDE_W * tl.arange(BLOCK_OW) + kw, axis=0)`** per **`kw`**.
- Host **`constexpr`** toggles: **`USE_W_SLAB_LOAD`**, **`USE_W_MASKED_SLAB`**, **`NO_PADDING_FASTPATH`**, **`BLOCK_OW`**, **`W_SLAB_LEN`**, **`FULL_W_TILE`**.

### Profile

- **`high-transfer-pressure`** or **discrete GM reads** on the pooling kernel **name** improve after the rewrite (**`op_statistic` Avg**, **Count**).

### IR

- Baseline: repeated **small loads** or **per-lane** predicates on **W** inside lowered loops. Optimized: **`tensor<W_SLAB_LEN x dtype>`** (or masked equivalent), then **`hfusion.gather` / `triton_gather`** into **`BLOCK_OW`** and **`arith.addf`** (**average**) or **`maxnumf` / max** (**max**).

## Related Patterns

- `gather-load`
- `discrete_memory_access`
- `constexpr-tile-discrete-access`
- `program-multiple-rows`

## What To Verify After Applying

- **Average**: **`torch.nn.functional.avg_pool2d` / `avg_pool3d`** — **`ceil_mode`**, **`count_include_pad`**, **`divisor_override`**, tails.
- **Max**: **`max_pool2d` / `max_pool3d`** (**values**), dtypes, padding / dilation if enabled.
- **Latency** vs **`BLOCK_OW`**, grid shape, and **physical core** count (launch knees).
