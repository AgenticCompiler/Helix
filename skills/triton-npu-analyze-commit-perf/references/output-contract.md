# Workspace Plan Contract

## Purpose

`workspace-plan.json` describes changed operators so `scaffold_operators.py` can create
operator workspace directories for downstream optimization.

## Schema

```json
{
  "repo": "/absolute/path/to/repo",
  "base_revision": "origin/main",
  "operators": [
    {
      "launch_function": "add_kernel",
      "source_path": "op_impl/add.py",
      "kernels": ["add_kernel_impl", "add_kernel_impl_v2"]
    }
  ]
}
```

## Fields

- **repo** (string, required): Absolute path to the Git repository root.
- **base_revision** (string, required): The base revision (branch or commit) to compare against.
- **operators** (list, required): One entry per changed operator.

Each operator entry:

- **launch_function** (string, required): The top-level launch function name.
- **source_path** (string, required): Repository-relative path to the source file.
- **kernels** (list of strings): Kernel function names called by the launch function.
  Used for transitive dependency extraction during scaffolding.
