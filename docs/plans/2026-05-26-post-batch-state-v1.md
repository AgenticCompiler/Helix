---
name: post batch state minimal v1
overview: 精简版 P0：定义 post-batch-state.json 最小必要 schema，第一版先把批跑后的核心统计跑通。
todos:
  - id: implement-collector-minimal
    content: 实现 post-batch collector 最小版：扫描 batch-root、抽取核心字段、写入 post-batch-state.json
    status: pending
  - id: implement-report-renderer-minimal
    content: 从 post-batch-state.json 渲染 post-batch-report.md
    status: pending
  - id: collector-tests
    content: 为 summary 聚合补充单元测试与 fixture
    status: pending
isProject: false
---

# Post-Batch State（V1）

> 第一版只覆盖 **核心统计**：optimize / verify / check 状态、skills 用到的 patterns。resume_state、discovery 等留到后续版本。

## 目标

把 `optimize-batch` 结束后每个 workspace 的关键结果归一为 `post-batch-state.json`，再渲染 `post-batch-report.md`。第一版不求全，先把批跑后最需要回答的问题覆盖：

- 每个 workspace 跑完了没？结果如何？
- verify 复验通过率如何？
- check 各项检查通过率如何？

## 职责边界


| 产物                      | 角色                           |
| ----------------------- | ---------------------------- |
| `post-batch-state.json` | 机器事实源：核心字段归一化 + 批级聚合         |
| `post-batch-report.md`  | 人读报告：渲染 summary、workspace 列表 |


原则：

- 报告**不**直接扫描 workspace；扫描与判断全部在 collector。
- V1 **只读**：不自动 rerun、reset。

## 可复用代码入口

- [src/triton_agent/optimize/batch.py](../../src/triton_agent/optimize/batch.py)：`optimize-batch-status.json`
- [src/triton_agent/status/core.py](../../src/triton_agent/status/core.py)：`inspect_optimize_status_workspace`、best round、speedup
- [src/triton_agent/verification/core.py](../../src/triton_agent/verification/core.py)：`verify-state.json`
- [src/triton_agent/log_check/batch.py](../../src/triton_agent/log_check/batch.py)：`log_check_result.json` 解析
- [src/triton_agent/log_check/check_json.py](../../src/triton_agent/log_check/check_json.py)：JSON schema 校验与修复（计划中）
- [src/triton_agent/log_check/check_markdown.py](../../src/triton_agent/log_check/check_markdown.py)：从 JSON 渲染 markdown（计划中）

---

## 1. 字段对齐原则

- 路径：相对 `<batch-root>` 的 POSIX 路径
- 状态枚举：小写短横线 `completed`、`incomplete`、`passed`、`failed`、`skipped`、`unknown`
- `check.checks[].result` 直接取 `log_check_result.json` 中 `checks[].result` 的原值（小写 `pass` / `fail`），不做大小写转换
- speedup：JSON 存 float 倍数（如 `1.23`）
- `best_round`：统一 `round-N`

---

## 2. 核心字段来源

### workspace 级


| 字段                              | 来源                                            | 说明                                                         |
| ------------------------------- | --------------------------------------------- | ---------------------------------------------------------- |
| `workspace`                     | 目录扫描                                          | 相对 batch-root                                              |
| `operator_file`                 | `optimize-batch-status.json`                  | 相对路径                                                       |
| `status`                        | `optimize-batch-status.json` → `status` + collector 推导 | `completed` / `incomplete` / `skipped`。`optimize-batch-status.json` 有记录则取其值；无记录且目录存在则推导为 `incomplete`
| `optimize.status`               | `status.core.OptimizeStatusWorkspace.state`   | `ok` / `warning` / `no-session`                            |
| `optimize.round_count`          | `opt-round-`* 目录计数                            | int                                                        |
| `optimize.best_round`           | `status.core.best_round`                      | `round-N` 或 null                                           |
| `optimize.best_geomean_speedup` | `status.core.geomean_speedup`                 | float 或 null                                               |
| `verify.status`                 | `verify-state.json` 归一化                       | 归一化规则见下方"verify.status 归一化"
| `verify.geomean_speedup`        | `verify-result.speedup.geomean_speedup`       | float 或 null                                               |
| `check.status`                  | `log_check_result.json` → `overall` + collector 推导 | `"PASS"` → `passed`；`"FAIL"` → `failed`；JSON 文件不存在 → `skipped` |
| `check.checks[].id`             | `log_check_result.json` → `checks[].id`          | `check-1` ~ `check-9`（无 check-5）                           |
| `check.checks[].name`           | `log_check_result.json` → `checks[].name`        | 如 `distinct strategies per round`                          |
| `check.checks[].result`         | `log_check_result.json` → `checks[].result`      | `pass` / `fail` / `skip`                                   |
| `check.checks[].detail`         | `log_check_result.json` → `checks[].detail`      | 字符串，通过时 null                                               |
| `pattern.known`                 | `pattern_analysis.json` → `summary.known`         | `{name, rounds[], evidence}`                               |
| `pattern.new`                   | `pattern_analysis.json` → `summary.new`           | `{name, rounds[]}`                                         |
| `pattern.extended`              | `pattern_analysis.json` → `summary.extended`      | `{name, rounds[], from}`                                   |


### 批级聚合 `summary`


| 字段                                                      | 推导依据                         |
| ------------------------------------------------------- | ---------------------------- |
| `total_workspaces`                                      | batch-root 下 workspace 目录数   |
| `optimize.process.completed/incomplete/skipped`          | `status` 计数（进程是否跑完）          |
| `optimize.health.ok/warning/no_session`                 | `optimize.status` 计数（产物是否健康） |
| `verify.passed/failed/skipped`                          | `verify.status` 计数           |
| `check.passed/failed/skipped`                           | `check.status` 计数            |


### verify.status 归一化

从 `verify-state.json` → `verify-result` 归一化到 `post-batch-state.json` 的 `verify.status`：

| 条件 | `verify.status` 取值 |
|---|---|
| `verify-state.json` 不存在（`opt-verify/` 目录为空或不存在） | `skipped` |
| 存在，且 `test.status` / `rerun_baseline_bench.status` / `rerun_best_bench.status` / `compare_perf.status` 全部为 `"passed"` | `passed` |
| 存在，但任一 status 不是 `"passed"` | `failed` |

### input_sources 说明

`collector.input_sources` 列出所有参与数据采集的文件，其中：

- **直接解析**（collector 直接 `json.load` 或 `path.read_text`）：`optimize-batch-status.json`、`log_check_result.json`、`pattern_analysis.json`
- **间接依赖**（通过 status.core Python API 内部读取，collector 不直接解析）：`opt-note.md`、`opt-round-*/*_perf.txt`、`opt-round-*/round-state.json`、`opt-verify/verify-*/verify-state.json`

### 字段 required / nullable

| 字段路径 | required | nullable | 说明 |
|---|---|---|---|
| `workspace` | 是 | 否 | 始终存在 |
| `operator_file` | 是 | 是 | 从 `optimize-batch-status.json` 获取，无记录时为 `null` |
| `status` | 是 | 否 | — |
| `optimize.status` | 是 | 否 | `no-session` 时仍为 `"no-session"` |
| `optimize.round_count` | 是 | 否 | 无 round 目录时为 `0` |
| `optimize.best_round` | 否 | 是 | 无可比较 round 时为 `null` |
| `optimize.best_geomean_speedup` | 否 | 是 | 无可比较 round 时为 `null` |
| `verify.status` | 是 | 否 | 未跑 verify 时为 `"skipped"` |
| `verify.geomean_speedup` | 否 | 是 | verify 未通过或无 speedup 数据时为 `null` |
| `check.status` | 是 | 否 | 未跑 log-check 时为 `"skipped"` |
| `check.checks[]` | 否 | 是 | `check.status` 为 `"skipped"` 时数组为空 `[]` |
| `check.checks[].id/name/result` | 是（条目存在时） | 否 | — |
| `check.checks[].detail` | 否 | 是 | result 为 `"pass"` 时为 `null` |
| `pattern.known/new/extended` | 否 | 否 | 未跑 log-check 时为空数组 `[]` |

> **规则**：`skipped` 表示"未执行该步骤"，对应的子字段用空数组或 `null` 占位；`required=否` 的字段在 JSON 中始终输出，值为 `null` 时表示数据缺失。


---

## 3. `post-batch-state.json` schema v1（最小版）

```json
{
  "schema_version": 1,
  "generated_at": "2026-05-26T14:54:00+08:00",
  "batch_root": "path/to/batch-root",
  "collector": {
    "name": "post-batch",
    "input_sources": [
      "optimize-batch-status.json",
      "opt-note.md",
      "opt-round-*/*_perf.txt",
      "opt-round-*/round-state.json",
      "opt-verify/verify-*/verify-state.json",
      "log_check_result.json",
      "pattern_analysis.json"
    ]
  },
  "summary": {
    "total_workspaces": 0,
    "optimize": {
      "process": {
        "completed": 0,
        "incomplete": 0,
        "skipped": 0
      },
      "health": {
        "ok": 0,
        "warning": 0,
        "no_session": 0
      }
    },
    "verify": {
      "passed": 0,
      "failed": 0,
      "skipped": 0
    },
    "check": {
      "passed": 0,
      "failed": 0,
      "skipped": 0
    }
  },
  "workspaces": [
    {
      "workspace": "operator-a",
      "operator_file": "operator-a/kernel.py",
      "status": "completed",
      "optimize": {
        "status": "ok",
        "round_count": 3,
        "best_round": "round-2",
        "best_geomean_speedup": 1.23
      },
      "verify": {
        "status": "passed",
        "geomean_speedup": 1.21
      },
      "check": {
        "status": "failed",
        "checks": [
          {
            "id": "check-1",
            "name": "distinct strategies per round",
            "result": "pass",
            "detail": null
          },
          {
            "id": "check-2",
            "name": "strategy novelty beyond patterns",
            "result": "fail",
            "detail": "仅使用了 references/patterns/ 中已有的 tile-optimize 和 fuse-kernel，未发现新策略"
          },
          {
            "id": "check-3",
            "name": "autotune instead of manual param tuning",
            "result": "pass",
            "detail": null
          },
          {
            "id": "check-4",
            "name": "no code duplication or regression",
            "result": "fail",
            "detail": "round-3/optimized_kernel.py 与 round-1 重复，仅进行了无意义的格式变动"
          },
          {
            "id": "check-6",
            "name": "Triton invocation preserved",
            "result": "pass",
            "detail": null
          },
          {
            "id": "check-7",
            "name": "baseline correctness and benchmark valid",
            "result": "pass",
            "detail": null
          },
          {
            "id": "check-8",
            "name": "best version valid and verified",
            "result": "pass",
            "detail": null
          },
          {
            "id": "check-9",
            "name": "round logs and evidence complete",
            "result": "pass",
            "detail": null
          }
        ]
      },
      "pattern": {
        "known": [
          { "name": "layout-store-and-block-pointers", "rounds": [1], "evidence": "explicit" },
          { "name": "tiling",                          "rounds": [1], "evidence": "inferred" },
          { "name": "program-multiple-rows",           "rounds": [2], "evidence": "explicit" },
          { "name": "algebraic-optimization",          "rounds": [2], "evidence": "inferred" }
        ],
        "new": [
          { "name": "host-side shape dispatch",        "rounds": [3] }
        ],
        "extended": [
          { "name": "dispatch threshold refinement",   "rounds": [4, 7, 9], "from": "host-side shape dispatch" },
          { "name": "tile budget tuning",              "rounds": [5], "from": "tiling" },
          { "name": "conditional tile gating",         "rounds": [6, 8, 10], "from": "host-side shape dispatch" }
        ]
      }
    }
  ]
}
```

---

## 4. 实现顺序

1. 只读 collector：扫描 batch-root，抽取上述核心字段，写入 `post-batch-state.json`。
2. report renderer：从 state 生成 `post-batch-report.md`（批级概览 + workspace 表）。
3. 单测：fixture workspace + summary 计数。

## 非目标（本版本）

- discovery、resume_state 不做
- 不做详细的 artifacts 路径索引
- 不自动 rerun

