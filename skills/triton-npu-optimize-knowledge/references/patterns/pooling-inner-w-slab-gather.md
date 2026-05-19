# Spatial pooling — innermost W slab plus gather (2D / 3D and beyond)

## Summary

For **sliding-window spatial pooling** in **NCHW-style** layouts where **W** is the **innermost contiguous** dimension, load one contiguous **W slab** that covers a whole **`BLOCK_OW`** tile of output windows, then use **`tl.gather`** to pick the per-`kw` values each output column needs. This rewrites many overlapping, predicate-heavy loads inside the inner **`kw`** loop into **one reusable contiguous load plus local indexed selection**; the **W-slab geometry is the same in 2D and 3D**, only the outer **`kh`** / **`kd`** loops differ.

## Use When

- The kernel is **AvgPool2d / AvgPool3d**, **MaxPool2d / MaxPool3d** (**values only**), or any **fixed `KERNEL_W`** reduction along **W** on a **contiguous** NCHW (or 5D) tensor, with **`kw`** in a constexpr loop and **`BLOCK_OW`** outputs per program.
- IR or profiling shows **many narrow or predicate-heavy global loads** along **`kw`** while **`stride_w`** maps output columns to **regularly strided** input columns.
- **`out_w`** is large enough that **vectorizing along `ow`** matters, and **`cdiv(out_w, BLOCK_OW)`** does not hurt launch scalability in measurement.
- You can prove **semantic equivalence** on the branches you enable (**ceil**, **padding**, **divisor** / **count_include_pad** for average, **numeric identity** for max on **`dtype` / `-inf`** rules).

## Detail

### Intuition

The problem this pattern targets is **overlap**. Adjacent output windows along **W** often reuse many of the same input columns, but a naive pooling loop still reloads those columns inside every **`kw`** step.

- Baseline mental model: for each **`kw`**, every lane does something like **`tl.load(input_ptr + start_w + ow * STRIDE_W + kw, mask=...)`**.
- Optimized mental model: first load the smallest contiguous W range that covers all **`BLOCK_OW`** output windows owned by the program, then reuse that staged slab for every **`kw`** with **`tl.gather`**.

The slab length is:

`W_SLAB_LEN = STRIDE_W * (BLOCK_OW - 1) + KERNEL_W`

That is exactly the span from the first input column needed by the tile to the last one.

### Example

Suppose:

- **`BLOCK_OW = 4`**
- **`STRIDE_W = 2`**
- **`KERNEL_W = 3`**

Then:

- **`W_SLAB_LEN = 2 * (4 - 1) + 3 = 9`**
- one output tile needs input columns **`[0..8]`**

The four output windows consume:

- **`ow = 0`** -> **`[0, 1, 2]`**
- **`ow = 1`** -> **`[2, 3, 4]`**
- **`ow = 2`** -> **`[4, 5, 6]`**
- **`ow = 3`** -> **`[6, 7, 8]`**

So the program can load one contiguous slab **`[0, 1, 2, 3, 4, 5, 6, 7, 8]`** once, then gather:

- **`kw = 0`** -> indices **`[0, 2, 4, 6]`**
- **`kw = 1`** -> indices **`[1, 3, 5, 7]`**
- **`kw = 2`** -> indices **`[2, 4, 6, 8]`**

The overlapping columns (**`2`**, **`4`**, **`6`**) are loaded once from global memory and then reused locally.

### Implementation Shape

- One program usually owns **`BLOCK_OW`** adjacent output columns.
- For each **`kh`** (and **`kd`** in 3D), load a contiguous slab along **W**.
- For each **`kw`**, gather with **`STRIDE_W * tl.arange(BLOCK_OW) + kw`** and reduce into a **`BLOCK_OW`**-wide accumulator vector.
- **2D pooling** uses **`kh` / `kw`** only; **3D** adds **`kd`** outside them. The slab logic itself does not change.

### Fast-Path Branches

- **`USE_W_SLAB_LOAD`**: use an unmasked slab load when padding is zero, the tile is a full **`out_w`** tile, and the entire slab is known in-bounds.
- **`USE_W_MASKED_SLAB`**: still use the slab rewrite when the tile is full but slab columns may cross padding or **`ceil_mode`** boundaries; load the slab with **`mask=`** / **`other=0`**, then gather the same way.
- Otherwise, fall back to a simpler branch such as **`NO_PADDING_FASTPATH`** or generic boundary handling.

### Launch Notes

- Host-side tuning usually picks **`BLOCK_OW`** from **`out_w`** using a bounded candidate set.
- A common **2D** grid is **`(batch * channels * out_h, cdiv(out_w, BLOCK_OW))`**.
- A common **3D** variant folds **`out_d * out_h`** into the row axis.
- Pair with **`program-multiple-rows`** when consecutive flat spatial rows can share enough setup to amortize slab overhead.

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


## What To Verify After Applying

- **Average**: **`torch.nn.functional.avg_pool2d` / `avg_pool3d`** — **`ceil_mode`**, **`count_include_pad`**, **`divisor_override`**, tails.
- **Max**: **`max_pool2d` / `max_pool3d`** (**values**), dtypes, padding / dilation if enabled.
- **Latency** vs **`BLOCK_OW`**, grid shape, and **physical core** count (launch knees).
