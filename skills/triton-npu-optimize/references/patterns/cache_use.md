## Summary

Analyze memory access patterns, try to make use of cache and UB as much as possible. Make note of L2
cache (96MB, shared by all cores) and size of L1 and UB (512KB, 256KB, respectively).

## Use When

- The bottleneck looks memory-hierarchy bound rather than purely compute bound.
- Repeated reloads, weak reuse, or poor locality suggest that L2, L1, or UB usage can be improved through better data placement or tile sizing.

## Detail

Make use of information about sizes of caches and UB to optimize parameters by computation.
Take note that the UB size is 192KB (used for most operations), and the L1 cache for the Cube core
is 512KB (used for both input matrices of a matrix multiplication only).
