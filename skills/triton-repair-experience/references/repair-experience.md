# Ascend Triton operator repair experience

Heuristic fixes for **Triton Ascend** kernels. These are **not guaranteed optimal**; match the compiler error or symptom, apply a **minimal** change, and re-run validation. Extend this list as new patterns appear.

---

## 1. Prefer `tl.math.*` over `libdevice.*` for elementwise math

The Ascend toolchain often routes “libdevice-style” math through symbols that are **unsupported or missing** compared to CUDA-oriented references. When the error names `libdevice.<name>`, try the **`tl.math`** equivalent first.

| Message / clue | Try |
|----------------|-----|
| `libdevice.tanh` (or similar) | `tl.math.tanh` |
| `libdevice.erf` (or similar) | `tl.math.erf` |
| Other `libdevice.<op>` | `tl.math.<op>` if it exists in your Triton version |

Keep semantics (dtype, masking) identical aside from the call target.

---

## 2. `allow_tf32` deprecated

Older flags such as **`allow_tf32`** may be rejected or deprecated on Ascend Triton.

- Prefer the documented replacement (e.g. **`input_precision='hf32'`** where the API supports it).
- Ensure operands that participate in that path use **float32** as expected by the compiler (cast or construct tensors/parameters in **fp32** so lowering matches intent).

---

## 3. Infinity literals and `float32`

If lowering fails with **dtype / inference** issues around special values, ensure **infinity** is visible to the compiler as **float32** (e.g. scalars feeding `tl.full`, masks, or constants). Integer or ambiguous dtypes for `inf` can break inference.

---

## 4. UB overflow → reduce block size

Errors indicating **UB (unified buffer) overflow** or on-chip scratch exhaustion:

- **Reduce** `BLOCK_SIZE`, tile size, or other per-block parameters so each block uses less local/UB budget.
- Re-tune for performance only after correctness is stable.

---

## 5. `TypeError: 0d block_type is forbidden` (Ascend)

The Ascend Triton compiler may reject certain **zero-dimensional `block_type`** forms.

**Symptom:** compile-time `TypeError` mentioning **`0d`** and **`block_type`**.

**Direction:**

- Avoid shapes that lower to a forbidden **scalar block** in that position. A pattern that has worked in practice is to materialize the value with an explicit dtype and a **defined** rank, for example:

  ```python
  sxn = tl.broadcast_to(tl.full((), sxn, tl.int64), ())
  ```

  Adapt the **variable name**, **dtype** (`tl.int64` vs `tl.float32`, etc.), and surrounding uses to your kernel. The intent is to replace a naïve 0-D form the compiler rejects with an explicit `tl.full` + `tl.broadcast_to` (or equivalent) so the block type is acceptable.

- This is **context-dependent**; if one reshape does not fix it, simplify how scalars enter loads/stores so they always pass through explicit `tl.full` / broadcasts with clear dtypes.

---