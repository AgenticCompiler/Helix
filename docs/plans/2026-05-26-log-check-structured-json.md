# Plan: 为 log_check_result 和 pattern_analysis 产出结构化 JSON

## Context

当前 `triton-agent log-check` 命令通过 LLM agent 生成两个 markdown 文件：
- `log_check_result.md` — 9 项优化质量检查结果
- `pattern_analysis.md` — 优化策略模式分析（check-10）

这两个文件由 LLM 自由格式输出，结构不稳定，导致后续 `post-batch-state.json` collector 难以可靠解析。

**目标**：在产出 md 的同时，产出对应的结构化 JSON 文件。后续 `post-batch-state.json` 从 JSON 读取而非解析 md。

## 核心设计决策

**方案：Agent 产出 JSON → Python 渲染 MD**

- Agent prompt 改为要求输出结构化 JSON（`log_check_result.json` + `pattern_analysis.json`）
- 新增 Python 模块从 JSON 渲染 markdown（格式稳定、可复现）
- JSON 是机器事实源，MD 是人读视图
- 兼容旧 workspace：若 JSON 缺失，回退解析 markdown

选择此方案的理由：
- LLM 生成的 markdown 格式不稳定 → 让 LLM 输出 JSON（schema 约束更强）
- JSON 更容易做程序化校验和修复
- markdown 由 Python 渲染，格式完全可控
- 单一事实源（JSON），避免 JSON 和 MD 内容不一致

## JSON Schema 设计

### `log_check_result.json` (v1)

完全对应 `post-batch-state.json` 中 `workspaces[].check` 的结构：

```json
{
  "schema_version": 1,
  "overall": "PASS",
  "failed_checks": "check-2: strategy novelty beyond patterns, check-4: no code duplication or regression",
  "overview_detail": "大部分检查通过，但策略新颖性不足，且 round-3 存在代码重复。",
  "checks": [
    {
      "id": "check-1",
      "name": "distinct strategies per round",
      "result": "pass",
      "detail": "每轮使用了不同策略：round-1 tiling, round-2 algebraic-optimization, round-3 autotune。"
    },
    {
      "id": "check-2",
      "name": "strategy novelty beyond patterns",
      "result": "fail",
      "detail": "仅使用了 references/patterns/ 中已有的 tile-optimize 和 fuse-kernel，未发现新策略。"
    }
  ]
}
```

字段说明：
- `overall`: `"PASS"` | `"FAIL"` — 所有 check 都 pass 才为 PASS
- `failed_checks`: 字符串，PASS 时为 `"none"`，FAIL 时列出失败项
- `checks[]`: 8 项（check-1 ~ check-4, check-6 ~ check-9，无 check-5）
- `checks[].result`: `"pass"` | `"fail"`（小写，与 post-batch-state 对齐）
- `checks[].detail`: 字符串，通过时描述具体情况，失败时说明原因

### `pattern_analysis.json` (v1)

完全对应 `post-batch-state.json` 中 `workspaces[].pattern` 的结构，加上 per-round 详情：

```json
{
  "schema_version": 1,
  "rounds": [
    {
      "round": "round-1",
      "patterns": [
        {
          "name": "layout-store-and-block-pointers",
          "evidence": "explicit",
          "source": "round-1/attempts.md: 'applied layout-store-and-block-pointers pattern'"
        },
        {
          "name": "tiling",
          "evidence": "inferred",
          "source": "diff vs baseline: added tl.arange tile index computation, matched signal 'tile-based access'"
        }
      ]
    }
  ],
  "summary": {
    "known": [
      { "name": "layout-store-and-block-pointers", "rounds": [1], "evidence": "explicit" },
      { "name": "tiling", "rounds": [1, 2], "evidence": "inferred" }
    ],
    "new": [
      { "name": "host-side shape dispatch", "rounds": [3] }
    ],
    "extended": [
      { "name": "dispatch threshold refinement", "rounds": [4, 7, 9], "from": "host-side shape dispatch" }
    ]
  }
}
```

字段说明：
- `rounds[]`: 每轮的 pattern 使用详情（供 markdown 渲染用）
- `rounds[].patterns[].evidence`: `"explicit"` | `"inferred"`
- `summary.known[]`: 匹配到已知 pattern 的汇总，含 evidence 级别
- `summary.new[]`: 全新策略（不匹配任何已知 pattern）
- `summary.extended[]`: 对已知 pattern 的增量扩展，`from` 指向来源 pattern

## 实现步骤

### Step 1: 新增 JSON schema 定义与校验模块

**新文件**: `src/triton_agent/log_check/check_json.py`

职责：
- 定义 `LOG_CHECK_JSON_SCHEMA` 和 `PATTERN_ANALYSIS_JSON_SCHEMA` 常量
- `validate_log_check_json(data: dict) -> list[str]` — 校验并返回错误列表
- `validate_pattern_analysis_json(data: dict) -> list[str]`
- `repair_json(text: str) -> dict | None` — 尝试修复常见 LLM JSON 错误（尾部逗号、未转义换行等）

### Step 2: 新增 markdown 渲染模块

**新文件**: `src/triton_agent/log_check/check_markdown.py`

职责：
- `render_log_check_markdown(data: dict) -> str` — 从 JSON 渲染 `log_check_result.md`
- `render_pattern_analysis_markdown(data: dict) -> str` — 从 JSON 渲染 `pattern_analysis.md`
- MD 格式与当前 prompt 定义的格式保持一致，但由 Python 保证结构稳定

### Step 3: 修改 agent prompt

**修改文件**: `src/triton_agent/log_check/log_check_launcher.py`

- 修改 `build_log_check_prompt()`：
  - 将 check-1 ~ check-9 的输出指令改为写入 `log_check_result.json`（而非 `.md`）
  - 将 check-10 的输出指令改为写入 `pattern_analysis.json`（而非 `.md`）
  - prompt 中内嵌 JSON schema 示例，要求 agent 严格遵循
  - 强调 JSON 字符串字段中的特殊字符必须正确转义

### Step 4: 修改 launcher 后处理逻辑

**修改文件**: `src/triton_agent/log_check/log_check_launcher.py`

在 `run_log_check()` 中 agent 执行完成后：
1. 读取 agent 产出的 JSON 文件
2. 调用 `validate_log_check_json()` / `validate_pattern_analysis_json()` 校验
3. 若校验失败，尝试 `repair_json()` 修复
4. 若修复仍失败，记录 warning 并尝试从历史 markdown 文件回退解析
5. 校验通过后，调用 markdown 渲染器写入对应的 `.md` 文件
6. 若 JSON 完全缺失（agent 未产出），标记为失败

### Step 5: 更新 batch.py 读取逻辑

**修改文件**: `src/triton_agent/log_check/batch.py`

- 修改 `summarize_log_check_output()`：
  - 优先读取 `log_check_result.json`，从中提取 `overall` 和 `failed_checks`
  - 若 JSON 不存在，回退到当前的 markdown 解析逻辑（兼容旧 workspace）

### Step 6: 更新 log_check_launcher.py 入口参数

**修改文件**: `src/triton_agent/log_check/log_check_launcher.py`

- `build_log_check_request()` 和 `run_log_check()` 增加 `output_json` 参数
- `build_parser()` 增加 `--output-json` CLI 选项

## 受影响的文件

| 文件 | 变更类型 | 说明 |
|---|---|---|
| `src/triton_agent/log_check/check_json.py` | **新增** | JSON schema 定义、校验、修复 |
| `src/triton_agent/log_check/check_markdown.py` | **新增** | 从 JSON 渲染 markdown |
| `src/triton_agent/log_check/log_check_launcher.py` | **修改** | prompt 改为输出 JSON；增加后处理逻辑 |
| `src/triton_agent/log_check/batch.py` | **修改** | `summarize_log_check_output()` 优先读 JSON |
| `src/triton_agent/log_check/__init__.py` | **修改** | 导出新模块的公开 API |
| `tests/` | **新增** | JSON 校验、MD 渲染、batch 解析的单元测试 |

## 兼容性

- **旧 workspace**：若 `log_check_result.json` 不存在，batch.py 回退到解析 markdown
- **新 workspace**：JSON 为主，MD 由 Python 渲染
- **CLI 接口**：保持 `--output-file` 参数不变，新增 `--output-json` 控制 JSON 文件名

## 验证方式

1. 单元测试：用 fixture JSON 数据验证 markdown 渲染输出与预期一致
2. 单元测试：用合法/非法 JSON 验证校验和修复逻辑
3. 集成测试：运行 `triton-agent log-check -i <fixture-workspace>` 验证端到端流程
4. 回归测试：用旧 workspace（只有 MD 无 JSON）验证 batch 回退逻辑
5. 检查 `post-batch-state.json` 的 check/pattern 字段能从新 JSON 文件中正确读取
