# Channel 集成实现计划

**目标：** 新增 `triton-agent channel serve` 子命令，启动一个常驻服务，通过飞书等 IM 平台接收消息、桥接到 triton-agent CLI 操作。支持结构化斜杠命令（快速通道，零 LLM 成本）和自然语言交互（通过受限 Agent 自主决策）。Agent 可执行只读查询和写操作，写操作需用户通过交互式卡片二次确认。

**平台扩展：** 采用「平台无关核心 + 平台相关适配器」架构。一期实现飞书集成，核心管线预留多平台扩展能力。新增微信/钉钉/Slack 等平台只需实现运输层和格式化层两个适配器。

**技术栈：** Python 3.12+，`aiohttp` / `FastAPI`（webhook 传输），飞书 `lark-oapi` SDK，企业微信 `wechatpy` SDK，现有 triton-agent 编排模块，`asyncio` 任务跟踪。

---

## 设计决策

| 决策 | 理由 |
|---|---|
| **斜杠命令跳过 LLM** | 对于 `/status` 等结构化查询，零 token 成本、亚 100ms 响应。 |
| **自然语言走受限 Agent** | 处理「最近一批里哪些 workspace 性能回退了？」这类规则无法覆盖的多步推理。 |
| **Agent 无 bash、无文件系统写入权限** | 安全性底线。Agent 只能通过预设的 triton-agent 领域工具操作，不得触碰文件系统或执行任意命令。 |
| **Agent 只读工具即时执行，写操作需二次确认** | Agent 可自主查询状态、分析数据；当它建议跑 optimize/verify 时，dispatcher 拦截并返回确认卡片，用户点击「确认」后才真正执行。斜杠命令 `/optimize` 无需此流程（用户已明确表达意图）。 |
| **进程内调用 handler** | 直接 import `commands/status.py`、`optimize/orchestration.py` 等现有模块，无子进程开销。 |
| **文件系统即状态存储，不加数据库** | 所有可查询数据已存在于 `batch-status.json`、`round-state.json`、`verify-state.json` 等文件中。channel 服务只需读取。 |
| **平台无关核心 + 平台相关适配器** | 命令解析、ACL、分发、状态读取、Agent 运行器等核心逻辑完全平台无关。新增平台只需实现 Transport + Formatter + Auth 三个适配器。 |

---

## 多平台扩展架构

Channel 系统参考 openclaw 的 `ChannelPlugin` 模式，核心管线与平台实现严格分离：

```
┌──────────────────────────────────────────────────────────────────────┐
│                     平台无关核心（通用逻辑，一次实现）                 │
│                                                                      │
│  消息入口                                                             │
│      │                                                               │
│   commands.py  ── 消息分类（斜杠命令 vs 自然语言）                    │
│      │                                                               │
│   acl.py       ── 权限校验（operators / viewers）                    │
│      │                                                               │
│   dispatcher.py ── 任务分发（快速通道 / 异步 job / Agent 路由）      │
│      │                                                               │
│  ┌───┼──────────┬──────────────┬──────────────┐                      │
│  │   │          │              │              │                      │
│  ▼   ▼          ▼              ▼              ▼                      │
│ state   optimize    agent_     agent_     formatter.py               │
│ _reader /verify     tools.py   runner.py  (平台无关格式化基类)       │
│                                                                      │
│  所有平台共享。新增平台无需修改任何核心代码。                          │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┬────────────────┐
          ▼                ▼                ▼                ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │  飞书    │    │ 企业微信  │    │  钉钉    │    │  Slack   │
    │  Plugin  │    │  Plugin   │    │  Plugin  │    │  Plugin  │
    ├──────────┤    ├──────────┤    ├──────────┤    ├──────────┤
    │ transport│    │ transport │    │ transport│    │ transport│
    │ formatter│    │ formatter │    │ formatter│    │ formatter│
    │ auth     │    │ auth      │    │ auth     │    │ auth     │
    └──────────┘    └──────────┘    └──────────┘    └──────────┘
```

### 平台适配器需要实现的三个接口

| 接口 | 职责 | 飞书实现 | 企业微信实现 |
|---|---|---|---|
| **Transport** | 接收消息、发送回复 | WebSocket 长连接 + REST API | 回调 URL（HTTP POST）+ 企业微信 API |
| **Formatter** | 将 `DispatchResult` 转为平台消息格式 | 飞书交互式卡片 JSON | 企业微信 Markdown / 模板卡片 |
| **Auth** | 签名校验、发送者身份解析 | SHA256 签名、提取 `open_id` | AES 解密 + SHA1 签名、提取 `UserId` |

### 新增平台的步骤（以企业微信为例）

1. 在 `src/triton_agent/channel/transports/` 下新增 `wecom.py`，实现 `TransportAdapter` 接口：
   - 接收企业微信回调 URL 的 POST 请求（消息解密、签名校验）
   - 将原始消息转为平台无关的 `InboundMessage`（sender_id, chat_id, text, platform）
   - 调用共享管线 `commands.parse_message()` → `acl.check()` → `dispatcher.dispatch()`
   - 将 `DispatchResult` 通过企业微信 API 发送回复

2. 在 `src/triton_agent/channel/formatter.py` 中为微信新增格式化函数：
   - `format_wecom_markdown(summary)` — 企业微信支持的 Markdown 子集
   - `format_wecom_template_card(summary)` — 企业微信模板卡片
   - `format_wecom_confirm_card(action)` — 写操作确认卡片

3. 在 `config.py` 的 `ChannelConfig` 中新增 `wecom` 字段，在 `server.py` 中按 `enabled` 标记启动对应 transport。核心管线代码零修改。

### 发送者身份抽象

为支持多平台，发送者身份使用「平台前缀 + 平台内 ID」的统一格式：

| 平台 | sender_id 格式 | 示例 |
|---|---|---|
| 飞书 | `feishu:ou_xxx` | `feishu:ou_abc123` |
| 企业微信 | `wecom:UserId` | `wecom:zhangsan` |
| 钉钉 | `dingtalk:UserId` | `dingtalk:manager01` |

ACL 配置中的 `operators` 和 `viewers` 使用此格式。`"*"` 表示所有平台的用户。

---

## 架构总览

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        triton-agent channel serve                        │
│                                                                          │
│   平台接入层（按配置启动多个 transport）                                    │
│                                                                          │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                 │
│   │ feishu       │   │ wecom        │   │ ...          │                 │
│   │ transport    │   │ transport    │   │              │                 │
│   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘                 │
│          │                  │                  │                          │
│          └──────────────────┼──────────────────┘                          │
│                             │ InboundMessage                             │
│                      ┌──────▼───────────────┐                            │
│                      │   auth (平台签名校验) │                            │
│                      └──────┬───────────────┘                            │
│                             │                                            │
│                      ┌──────▼───────────────┐                            │
│                      │   commands.py        │                            │
│                      │   (消息分类)          │                            │
│                      └──┬──────────────┬────┘                            │
│                         │              │                                  │
│              斜杠命令   │              │  自然语言                        │
│          (/status 等)   │              │                                  │
│                     ┌───▼───┐      ┌───▼──────────────┐                  │
│                     │ 快速   │      │ 受限 Agent        │                  │
│                     │ 通道   │      │ agent_runner.py   │                  │
│                     │       │      │ agent_tools.py    │                  │
│                     │       │      │                   │                  │
│                     │       │      │ 只读工具（即时）   │                  │
│                     │       │      │ 写操作（需确认）   │                  │
│                     └───┬───┘      └───┬───────────────┘                  │
│                         │              │                                  │
│                     ┌───▼──────────────▼───┐                              │
│                     │   acl.py             │                              │
│                     │   (权限校验)          │                              │
│                     └───┬──────────────────┘                              │
│                         │                                                │
│                     ┌───▼──────────────────┐                              │
│                     │   dispatcher.py      │                              │
│                     │   (任务分发)          │                              │
│                     └───┬──────────────────┘                              │
│                         │                                                │
│              ┌──────────┼──────────┐                                     │
│              │          │          │                                     │
│         ┌────▼────┐ ┌───▼────┐ ┌───▼─────────┐                           │
│         │ state_  │ │ optimize│ │ optimize    │                           │
│         │ reader  │ │ /orch.. │ │ /batch.py   │                           │
│         │ (只读)  │ │ (写操作)│ │ (写操作)    │                           │
│         └────┬────┘ └───┬────┘ └───┬─────────┘                           │
│              │          │          │                                     │
│              └──────────┼──────────┘                                     │
│                         │                                                │
│                     ┌───▼──────────────────┐                              │
│                     │   formatter.py       │                              │
│                     │   (按平台格式化)      │                              │
│                     └───┬──────────────────┘                              │
│                         │                                                │
│          ┌──────────────┼──────────────┐                                  │
│          ▼              ▼              ▼                                  │
│     回复飞书        回复企微       回复钉钉                                │
│     (REST API)     (REST API)     (REST API)                              │
└──────────────────────────────────────────────────────────────────────────┘
```

### 消息处理流程

```
IM 平台 ──消息──▶ triton-agent channel serve
                     │
                     ▼
              auth (平台签名校验)
                     │
                     ▼
              commands.py (消息分类)
                     │
         ┌───────────┼───────────┐
         ▼                       ▼
   斜杠命令 (/xxx)          自然语言消息
         │                       │
         ▼                       ▼
    ACL 校验               ACL 校验 (viewer)
         │                       │
         ▼                       ▼
   直接调用 handler        受限 Agent 自主决策
   (state_reader /            │
    optimize 等)              ├─ 只读工具（即时执行）
         │                    │   query_workspace_status
         │                    │   list_batch_workspaces
         │                    │   read_round_detail
         │                    │   read_opt_note
         │                    │   compare_baseline_vs_round
         │                    │
         │                    ├─ 写操作工具（建议确认）
         │                    │   propose_optimize
         │                    │   propose_verify
         │                    │   propose_gen_test
         │                    │   propose_upload
         │                    │
         │                    │   写操作被 dispatcher 拦截
         │                    │   → 返回确认卡片给用户
         │                    │   → 用户点击确认 → ACL(operator)
         │                    │   → 执行真实 handler
         │                    │
         ▼                    ▼
   formatter (按平台格式化)
         │
         ▼
   回复 IM 消息
```

---

## 文件清单

- 新建：`src/triton_agent/channel/__init__.py`
- 新建：`src/triton_agent/channel/models.py`（平台无关消息模型）
- 新建：`src/triton_agent/channel/config.py`
- 新建：`src/triton_agent/channel/server.py`
- 新建：`src/triton_agent/channel/transports/__init__.py`
- 新建：`src/triton_agent/channel/transports/base.py`（TransportAdapter 抽象接口）
- 新建：`src/triton_agent/channel/transports/feishu.py`（飞书 WebSocket + webhook）
- 新建：`src/triton_agent/channel/auth.py`
- 新建：`src/triton_agent/channel/commands.py`
- 新建：`src/triton_agent/channel/dispatcher.py`
- 新建：`src/triton_agent/channel/formatter.py`
- 新建：`src/triton_agent/channel/state_reader.py`
- 新建：`src/triton_agent/channel/acl.py`
- 新建：`src/triton_agent/channel/agent_tools.py`
- 新建：`src/triton_agent/channel/agent_runner.py`
- 新建：`src/triton_agent/commands/channel.py`
- 新建：`tests/test_channel_models.py`
- 新建：`tests/test_channel_config.py`
- 新建：`tests/test_channel_commands.py`
- 新建：`tests/test_channel_state_reader.py`
- 新建：`tests/test_channel_acl.py`
- 新建：`tests/test_channel_dispatcher.py`
- 新建：`tests/test_channel_formatter.py`
- 修改：`src/triton_agent/cli.py`
- 修改：`README.md`

---

## 任务一：配置体系、消息模型与 CLI 接口

**涉及文件：**
- 新建：`src/triton_agent/channel/__init__.py`
- 新建：`src/triton_agent/channel/models.py`
- 新建：`src/triton_agent/channel/config.py`
- 新建：`src/triton_agent/channel/auth.py`
- 新建：`src/triton_agent/channel/acl.py`
- 修改：`src/triton_agent/cli.py`
- 新建：`src/triton_agent/commands/channel.py`
- 新建：`tests/test_channel_models.py`
- 新建：`tests/test_channel_config.py`
- 新建：`tests/test_channel_acl.py`

- [ ] **步骤 1：编写平台无关消息模型的失败测试**

  `tests/test_channel_models.py`：
  - 测试 `InboundMessage` 包含 `sender_id`（格式 `{platform}:{platform_id}`）、`chat_id`、`text`、`platform`
  - 测试 `DispatchResult` 支持 `text` 和可选的 `card`（平台无关的抽象卡片）
  - 测试 `ConfirmAction` 包含 `action_id`（用于追踪确认/取消）、`tool_name`、`tool_args`、`summary`

- [ ] **步骤 2：编写 channel 配置加载的失败测试**

  `tests/test_channel_config.py`：
  - 测试最小飞书配置（`channels.feishu.enabled: true` + 凭证）能正常加载
  - 测试飞书 + 企微双平台配置能同时加载，各自独立校验
  - 测试 `disabled` 的平台配置被跳过
  - 测试缺少必填字段时抛出清晰的校验错误
  - 测试 `${ENV_VAR}` 占位符从环境变量正确解析
  - 测试 ACL 中 `operators` 和 `viewers` 使用 `{platform}:{id}` 格式

- [ ] **步骤 3：编写 ACL 规则的失败测试**

  `tests/test_channel_acl.py`：
  - 测试 `operators` 列表中的用户有权执行 `/optimize`
  - 测试不在 `operators` 中的用户不能执行 `/optimize`，但能执行 `/status`（viewer 权限）
  - 测试 `viewers: ["*"]` 允许任何人查询状态
  - 测试跨平台 sender_id：`feishu:ou_001` 在 operators 列表则有权，`wecom:user_001` 不在则无权
  - 测试未知命令对所有用户均拒绝

- [ ] **步骤 4：编写 CLI 解析失败测试**

  `tests/test_cli.py`：
  ```python
  def test_channel_serve_解析_config_和_port(self):
      parser = build_parser()
      args = parser.parse_args(["channel", "serve", "--config", "channels.yaml", "--port", "8080"])
      self.assertEqual(args.config, "channels.yaml")
      self.assertEqual(args.port, 8080)
  ```

- [ ] **步骤 5：运行测试确认失败**

  `uv run python -m unittest tests.test_channel_models tests.test_channel_config tests.test_channel_acl tests.test_cli -v`
  预期：FAIL

- [ ] **步骤 6：实现平台无关消息模型**

  `src/triton_agent/channel/models.py`：
  ```python
  @dataclass(frozen=True)
  class InboundMessage:
      """Transport 层将平台消息统一转为此格式。"""
      sender_id: str          # "feishu:ou_xxx"
      chat_id: str            # "feishu:oc_xxx"
      text: str               # 原始消息文本
      platform: str           # "feishu" | "wecom" | "dingtalk"

  @dataclass(frozen=True)
  class DispatchResult:
      """dispatcher 返回此结构，formatter 按平台转换。"""
      text: str
      card: AbstractCard | None = None

  @dataclass(frozen=True)
  class AbstractCard:
      """平台无关的卡片抽象，formatter 负责转为具体平台的卡片 JSON。"""
      kind: str               # "status" | "batch" | "error" | "confirm" | "help" | "job"
      title: str
      fields: dict[str, str]  # key-value 展示数据
      actions: tuple[str, ...] = ()  # "confirm" | "cancel" 等

  @dataclass(frozen=True)
  class ConfirmAction:
      action_id: str          # UUID，dispatcher 用它追踪确认/取消
      tool_name: str          # "propose_optimize" | "propose_verify" 等
      tool_args: dict         # 传给实际 handler 的参数
      summary: str            # 给用户看的中文摘要
  ```

- [ ] **步骤 7：实现多平台配置加载器**

  `src/triton_agent/channel/config.py`：
  ```python
  @dataclass(frozen=True)
  class FeishuConfig:
      enabled: bool = False
      app_id: str = ""
      app_secret: str = ""
      encrypt_key: str = ""
      verification_token: str = ""
      connection_mode: Literal["websocket", "webhook"] = "websocket"
      webhook_port: int = 8080

  @dataclass(frozen=True)
  class MultiChannelConfig:
      batch_root: Path | None = None
      dm_policy: Literal["open", "allowlist"] = "allowlist"
      allow_from: tuple[str, ...] = ()
      group_policy: Literal["open", "allowlist", "disabled"] = "allowlist"
      group_allow_from: tuple[str, ...] = ()
      operators: tuple[str, ...] = ()     # 格式: "feishu:ou_xxx", "wecom:zhangsan"
      viewers: tuple[str, ...] = ("*",)
      feishu: FeishuConfig | None = None

  def load_channel_config(path: Path) -> MultiChannelConfig: ...
  ```

  支持 YAML 开箱即用，`${ENV_VAR}` 占位符从环境变量解析。对 `enabled: true` 的平台校验必填字段。

- [ ] **步骤 8：实现 ACL 模块**

  `src/triton_agent/channel/acl.py`：
  ```python
  def check_command_permission(
      config: MultiChannelConfig,
      sender_id: str,        # "feishu:ou_xxx"
      command: str,
  ) -> bool: ...
  ```
  - `/status`、`/results`、`/report-batch` → `viewers` 列表控制（`"*"` 表示所有人）
  - `/optimize`、`/verify`、`/gen-test`、`/upload` → `operators` 列表控制

- [ ] **步骤 9：实现飞书签名校验（一期）**

  `src/triton_agent/channel/auth.py`：
  - 为飞书提供签名校验：`SHA256(timestamp + nonce + encryptKey + raw_body)`，常量时间比较
  - 预留平台扩展点：不同平台的 auth 会作为 transport 的一部分调用，而非核心管线的一部分
  - 后续企微等平台各自在 transport 内部实现签名/解密

- [ ] **步骤 10：接入 CLI 子命令**

  `src/triton_agent/cli.py`：新增 `channel` 子命令组，含 `serve` 子命令。注册 `CommandKind.CHANNEL_SERVE`。

  `src/triton_agent/commands/channel.py`：轻量 handler，加载配置并调用 server 入口。当前阶段先抛出 `NotImplementedError`。

- [ ] **步骤 11：重新运行测试，确认通过**

  `uv run python -m unittest tests.test_channel_models tests.test_channel_config tests.test_channel_acl tests.test_cli -v`
  预期：PASS

- [ ] **步骤 12：提交 CLI 接口与配置层**

  ```bash
  git add src/triton_agent/channel/__init__.py src/triton_agent/channel/models.py src/triton_agent/channel/config.py src/triton_agent/channel/auth.py src/triton_agent/channel/acl.py src/triton_agent/cli.py src/triton_agent/commands/channel.py tests/test_channel_models.py tests/test_channel_config.py tests/test_channel_acl.py
  git commit -m "feat: add channel cli surface, config, and message model"
  ```

---

## 任务二：状态读取器 — Workspace 查询层

**涉及文件：**
- 新建：`src/triton_agent/channel/state_reader.py`
- 新建：`tests/test_channel_state_reader.py`

- [ ] **步骤 1：编写状态读取器的失败测试**

  `tests/test_channel_state_reader.py`：
  - 测试 `read_workspace_summary()` 在含 `baseline/`、`opt-round-1/`、`opt-round-2/` 的临时 workspace 上，返回 `best_round`、`geomean_speedup`、`state`
  - 测试无 optimize 产物的 workspace → `state: "no-session"`
  - 测试已验证的 workspace → `verified: true`，`verified_geomean_speedup` 有值
  - 测试 `read_batch_summary()` 扫描 batch root，聚合 `total/completed/failed/in_progress`
  - 测试 `read_round_detail()` 返回 hypothesis、perf、correctness
  - 测试 `read_latest_results()` 返回最近一次 batch run 的摘要

- [ ] **步骤 2：运行测试确认失败**

  `uv run python -m unittest tests.test_channel_state_reader -v`
  预期：FAIL

- [ ] **步骤 3：实现状态读取器**

  `src/triton_agent/channel/state_reader.py`：

  复用现有 status 基础设施：
  ```python
  from triton_agent.commands.status.core import read_optimize_status as _read_status

  @dataclass(frozen=True)
  class WorkspaceSummary:
      workspace_name: str
      operator_file: str | None
      state: Literal["ok", "warning", "no-session"]
      best_round: str | None
      geomean_speedup: float | None
      round_count: int
      verified: bool
      verified_geomean_speedup: float | None
      warnings: tuple[str, ...]

  def read_workspace_summary(workspace: Path) -> WorkspaceSummary: ...
  def read_batch_summary(batch_root: Path) -> BatchSummary: ...
  def read_round_detail(workspace: Path, round_name: str) -> RoundDetail: ...
  def read_opt_note(workspace: Path) -> str | None: ...
  def compare_perf(baseline: Path, round_perf: Path) -> PerfComparison: ...
  ```

- [ ] **步骤 4：重新运行测试，确认通过**

  `uv run python -m unittest tests.test_channel_state_reader -v`
  预期：PASS

- [ ] **步骤 5：提交状态读取器**

  ```bash
  git add src/triton_agent/channel/state_reader.py tests/test_channel_state_reader.py
  git commit -m "feat: add channel state reader"
  ```

---

## 任务三：命令解析器与任务分发器

**涉及文件：**
- 新建：`src/triton_agent/channel/commands.py`
- 新建：`src/triton_agent/channel/dispatcher.py`
- 新建：`tests/test_channel_commands.py`
- 新建：`tests/test_channel_dispatcher.py`

- [ ] **步骤 1：编写命令解析的失败测试**

  `tests/test_channel_commands.py`：
  - `/status matmul` → `ParsedCommand(kind="status", args={"workspace": "matmul"})`
  - `/optimize kernel.py --target npu --mode supervise` → `ParsedCommand(kind="optimize", args={"operator": "kernel.py", "target": "npu", "mode": "supervise"})`
  - `/results matmul round-3` → `ParsedCommand(kind="results", args={"workspace": "matmul", "round": "round-3"})`
  - `帮我看下matmul优化的最新进展` → `ParsedCommand(kind="natural_language", text="帮我看下matmul优化的最新进展")`
  - `hello` → `None`
  - `/help` → `ParsedCommand(kind="help")`

- [ ] **步骤 2：编写任务分发器的失败测试**

  `tests/test_channel_dispatcher.py`：
  - 测试分发 `/status` 返回结果文本
  - 测试分发 `/optimize` 创建异步 job，返回 job ID + 状态消息
  - 测试通过 job ID 查询进度
  - 测试不存在的 workspace 返回友好错误
  - 测试并发 optimize 任务被阻塞（每 NPU 一个）
  - 测试 `AgentResponse(kind="confirm")` 被正确拦截，生成 `AbstractCard(kind="confirm")`

- [ ] **步骤 3：运行测试确认失败**

  `uv run python -m unittest tests.test_channel_commands tests.test_channel_dispatcher -v`
  预期：FAIL

- [ ] **步骤 4：实现命令解析器**

  `src/triton_agent/channel/commands.py`：
  ```python
  @dataclass(frozen=True)
  class ParsedCommand:
      kind: Literal["status", "status-batch", "optimize", "gen-test",
                    "verify", "upload", "results", "report-batch",
                    "help", "natural_language"]
      args: dict[str, str] | None = None
      text: str | None = None  # 自然语言的原始文本

  def parse_message(text: str) -> ParsedCommand | None: ...
  ```
  - 以 `/` 开头 → 斜杠命令，`shlex` 风格解析参数
  - 非 `/` 开头或无法识别 → `natural_language`
  - 空消息 → `None`

- [ ] **步骤 5：实现任务分发器（含 Agent 确认拦截）**

  `src/triton_agent/channel/dispatcher.py`：
  ```python
  @dataclass(frozen=True)
  class DispatchResult:
      text: str
      card: AbstractCard | None = None

  @dataclass(frozen=True)
  class JobHandle:
      job_id: str
      status: Literal["queued", "running", "completed", "failed"]
      result: DispatchResult | None

  async def dispatch_command(
      cmd: ParsedCommand,
      msg: InboundMessage,
      config: MultiChannelConfig,
  ) -> DispatchResult: ...

  async def dispatch_natural_language(
      text: str,
      msg: InboundMessage,
      config: MultiChannelConfig,
  ) -> DispatchResult: ...
  ```
  - 快速通道命令（`/status`、`/results`、`/help`、`/report-batch`）：同步调用 state reader
  - 长任务命令（`/optimize`、`/verify`、`/gen-test`、`/upload`）：创建异步 job
  - 自然语言消息 → `dispatch_natural_language()` → `run_agent_query()`
  - Agent 调用 `propose_*` 写操作工具时 → 拦截为 `AbstractCard(kind="confirm")` → 注册 `action_id → ConfirmAction` 映射 → 返回确认卡片

- [ ] **步骤 6：重新运行测试，确认通过**

  `uv run python -m unittest tests.test_channel_commands tests.test_channel_dispatcher -v`
  预期：PASS

- [ ] **步骤 7：提交命令解析器与分发器**

  ```bash
  git add src/triton_agent/channel/commands.py src/triton_agent/channel/dispatcher.py tests/test_channel_commands.py tests/test_channel_dispatcher.py
  git commit -m "feat: add channel command parser and dispatcher"
  ```

---

## 任务四：Transport 抽象层与飞书传输实现

**涉及文件：**
- 新建：`src/triton_agent/channel/transports/__init__.py`
- 新建：`src/triton_agent/channel/transports/base.py`
- 新建：`src/triton_agent/channel/transports/feishu.py`
- 新建：`src/triton_agent/channel/formatter.py`
- 新建：`src/triton_agent/channel/server.py`
- 新建：`tests/test_channel_formatter.py`

- [ ] **步骤 1：编写 Transport 抽象接口**

  `src/triton_agent/channel/transports/base.py`：
  ```python
  class TransportAdapter(ABC):
      """平台传输适配器抽象。新增平台继承此类实现三个核心方法。"""

      @abstractmethod
      async def start(self, on_message: Callable[[InboundMessage], Awaitable[DispatchResult]]) -> None:
          """启动传输，收到消息后调用 on_message 回调，拿到 DispatchResult 后发送回复。"""
          ...

      @abstractmethod
      async def send_message(self, chat_id: str, result: DispatchResult) -> None:
          """向指定 chat 发送消息（用于异步 job 完成后主动推送）。"""
          ...

      @abstractmethod
      async def stop(self) -> None:
          """优雅关闭传输。"""
          ...
  ```

- [ ] **步骤 2：编写消息格式化的失败测试**

  `tests/test_channel_formatter.py`：
  - 测试 `format_feishu_status_card(summary)` 生成合法的飞书卡片 JSON
  - 测试 `format_feishu_confirm_card(action)` 生成确认卡片（含「确认」「取消」按钮）
  - 测试 `format_feishu_error_card(message)` 生成红色错误卡片
  - 测试 `format_feishu_help_card()` 列出所有命令
  - 测试文本超过飞书 4000 字符限制时正确分片

- [ ] **步骤 3：运行格式化测试确认失败**

  `uv run python -m unittest tests.test_channel_formatter -v`
  预期：FAIL

- [ ] **步骤 4：实现 formatter（按平台分函数）**

  `src/triton_agent/channel/formatter.py`：
  ```python
  def format_message(platform: str, result: DispatchResult) -> dict | str:
      """根据平台选择对应的格式化函数。"""
      ...

  # 飞书格式化函数
  def format_feishu_status_card(summary: WorkspaceSummary) -> dict: ...
  def format_feishu_batch_card(summary: BatchSummary) -> dict: ...
  def format_feishu_confirm_card(action: ConfirmAction) -> dict: ...
  def format_feishu_error_card(message: str) -> dict: ...
  def format_feishu_help_card() -> dict: ...
  def format_feishu_job_card(job: JobHandle) -> dict: ...
  def split_feishu_text(text: str, max_chars: int = 4000) -> list[str]: ...

  # 企微格式化函数（预留）
  def format_wecom_markdown(summary: WorkspaceSummary) -> str: ...
  def format_wecom_confirm_card(action: ConfirmAction) -> dict: ...
  ```

- [ ] **步骤 5：实现飞书 TransportAdapter**

  `src/triton_agent/channel/transports/feishu.py`：
  ```python
  class FeishuTransport(TransportAdapter):
      def __init__(self, config: FeishuConfig, channel_config: MultiChannelConfig): ...

      async def start(self, on_message): ...
      async def send_message(self, chat_id, result): ...
      async def stop(self): ...
  ```
  - **WebSocket 模式**：使用 `lark-oapi` WSClient，指数退避重连（1s 初始，30s 上限）
  - **Webhook 模式**：FastAPI HTTP 服务，`POST /feishu/events`，签名校验、challenge 握手
  - `on_message` 回调内部：解析飞书事件 → 构造 `InboundMessage(sender_id="feishu:ou_xxx", ...)` → 调用核心管线
  - 发送回复：通过飞书 REST API `im/v1/messages`，支持文本和交互式卡片
  - 监听 `card.action.trigger` 事件处理确认/取消按钮

- [ ] **步骤 6：实现 server 入口（多 transport 管理）**

  `src/triton_agent/channel/server.py`：
  ```python
  async def serve_channel(config: MultiChannelConfig) -> None:
      transports: list[TransportAdapter] = []

      if config.feishu and config.feishu.enabled:
          transports.append(FeishuTransport(config.feishu, config))

      # 未来扩展：
      # if config.wecom and config.wecom.enabled:
      #     transports.append(WecomTransport(config.wecom, config))

      async def handle_message(msg: InboundMessage) -> DispatchResult:
          cmd = parse_message(msg.text)
          if cmd is None:
              return DispatchResult(text="")  # 空消息不回复
          if not check_command_permission(config, msg.sender_id, cmd.kind):
              return DispatchResult(
                  text=f"权限不足：{msg.sender_id} 无权执行 {cmd.kind}"
              )
          return await dispatch_command(cmd, msg, config)

      await asyncio.gather(
          *(t.start(handle_message) for t in transports)
      )
  ```
  - `on_message` 回调组装完整管线：解析 → ACL → 分发 → 格式化 → 回复
  - 多个 transport 并发运行，共享同一个 `handle_message`

- [ ] **步骤 7：重新运行格式化测试，确认通过**

  `uv run python -m unittest tests.test_channel_formatter -v`
  预期：PASS

- [ ] **步骤 8：提交传输层与 server 层**

  ```bash
  git add src/triton_agent/channel/transports/ src/triton_agent/channel/formatter.py src/triton_agent/channel/server.py tests/test_channel_formatter.py
  git commit -m "feat: add channel transport abstraction and feishu implementation"
  ```

---

## 任务五：自然语言 Agent（受限运行 + 写操作二次确认）

**涉及文件：**
- 新建：`src/triton_agent/channel/agent_tools.py`
- 新建：`src/triton_agent/channel/agent_runner.py`

- [ ] **步骤 1：定义受限 Agent 工具集（只读即时 + 写操作建议确认）**

  `src/triton_agent/channel/agent_tools.py`：

  **即时执行工具（只读查询）：**

  | 工具名称 | 功能描述 | 底层调用 |
  |---|---|---|
  | `query_workspace_status` | 查询某个 workspace 的优化状态 | `state_reader.read_workspace_summary()` |
  | `list_batch_workspaces` | 列出 batch root 下所有 workspace | `state_reader.read_batch_summary()` |
  | `read_round_detail` | 读取指定轮次的 hypothesis、evidence、perf | `state_reader.read_round_detail()` |
  | `read_opt_note` | 读取优化会话日志 | `state_reader.read_opt_note()` |
  | `compare_baseline_vs_round` | 对比基线与某轮次性能 | `state_reader.compare_perf()` |

  **建议确认工具（写操作，需用户二次确认）：**

  | 工具名称 | 功能描述 | 执行方式 |
  |---|---|---|
  | `propose_optimize` | 建议对某算子启动优化 | dispatcher 拦截 → 返回确认卡片 → 确认后调用 `optimize/orchestration.py` |
  | `propose_verify` | 建议对某 workspace 执行验证 | dispatcher 拦截 → 返回确认卡片 → 确认后调用 `verify` handler |
  | `propose_gen_test` | 建议为某算子生成测试 harness | dispatcher 拦截 → 返回确认卡片 → 确认后调用 `gen-test` handler |
  | `propose_upload` | 建议上传某 workspace 到分析服务器 | dispatcher 拦截 → 返回确认卡片 → 确认后调用 `upload-optimize` handler |

  每个工具定义为带类型参数和 docstring 的 Python 函数，工具描述注入 Agent system prompt。

- [ ] **步骤 2：实现确认流程**

  ```
  用户: "matmul 收敛了，跑一下验证"
         │
         ▼
  Agent 调用 propose_verify(workspace="matmul")
         │
         ▼
  dispatcher 拦截 → 注册 action_id → 返回 AbstractCard(kind="confirm")
         │
         ▼
  formatter 按平台输出确认卡片:
  ┌────────────────────────────────────┐
  │ 即将对 matmul 执行验证             │
  │                                    │
  │ 最佳轮次: round-3                  │
  │ 几何加速比: 1.42x                  │
  │                                    │
  │ ┌──────────┐  ┌──────────┐        │
  │ │ ✅ 确认   │  │ ❌ 取消   │        │
  │ └──────────┘  └──────────┘        │
  └────────────────────────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
  确认      取消
    │         │
    ▼         ▼
  ACL(operator)   回复 "已取消"
    │
    ▼
  执行 handler → 回复结果卡片
  ```

- [ ] **步骤 3：实现 Agent runner**

  `src/triton_agent/channel/agent_runner.py`：
  ```python
  @dataclass(frozen=True)
  class AgentResponse:
      kind: Literal["text", "confirm"]
      text: str | None = None
      confirm_action: ConfirmAction | None = None

  async def run_agent_query(
      text: str,
      msg: InboundMessage,
      config: MultiChannelConfig,
  ) -> AgentResponse: ...
  ```
  - 构建 system prompt，描述可用工具、workspace 上下文和回复格式
  - 以非交互模式启动轻量 Agent（codex 或 opencode），传入用户查询
  - Agent 调用只读工具 → 返回 `AgentResponse(kind="text")`
  - Agent 调用 `propose_*` → dispatcher 拦截，返回 `AgentResponse(kind="confirm")`，**不实际执行**
  - 超时限制 60s

- [ ] **步骤 4：集成确认卡片回调到 Transport**

  在 `FeishuTransport` 中监听飞书 `card.action.trigger` 事件：
  - 解析 `action_id` → 查找已注册的 `ConfirmAction`
  - 「确认」→ 校验 sender 的 ACL（operator 权限）→ 调用实际 handler → 格式化回复
  - 「取消」→ 回复 "已取消"

  未来其他平台（企微等）在各自 transport 中实现等价的按钮回调处理。

- [ ] **步骤 5：提交 Agent 层**

  ```bash
  git add src/triton_agent/channel/agent_tools.py src/triton_agent/channel/agent_runner.py
  git commit -m "feat: add channel constrained agent with confirmable write operations"
  ```

---

## 任务六：端到端集成与文档

**涉及文件：**
- 修改：`src/triton_agent/commands/channel.py`
- 修改：`README.md`

- [ ] **步骤 1：接入 `channel serve` handler**

  `src/triton_agent/commands/channel.py`：
  ```python
  def handle_channel_serve(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
      config_path = Path(args.config)
      if not config_path.exists():
          print(f"错误：配置文件不存在: {config_path}", file=sys.stderr)
          return 1
      config = load_channel_config(config_path)
      asyncio.run(serve_channel(config))
      return 0
  ```

- [ ] **步骤 2：运行针对性测试**

  `uv run python -m unittest tests.test_channel_models tests.test_channel_config tests.test_channel_acl tests.test_channel_commands tests.test_channel_dispatcher tests.test_channel_state_reader tests.test_channel_formatter -v`
  预期：PASS

- [ ] **步骤 3：运行仓库验证命令**

  ```bash
  uv run --group dev ruff check
  uv run pyright
  uv run python -m unittest discover -s tests -v
  ```

- [ ] **步骤 4：更新 README**

  - `triton-agent channel serve` 命令及参数
  - 多平台配置文件格式，以飞书为例说明
  - 飞书应用创建前置准备
  - 可用斜杠命令及用法
  - Agent 模式说明：自然语言交互 + 写操作二次确认流程
  - ACL 权限模型：operators vs viewers，sender_id 格式
  - 部署说明（systemd unit 示例）
  - 扩展新平台的步骤指引

- [ ] **步骤 5：最终提交**

  ```bash
  git add src/triton_agent/commands/channel.py README.md docs/plans/2026-05-28-channel-feishu-integration-plan.md
  git commit -m "feat: integrate channel serve end-to-end"
  ```

---

## 配置文件参考

```yaml
# channels.yaml — 多平台 channel 配置

channels:
  feishu:
    enabled: true
    app_id: "cli_xxxxxxxxxxxx"
    app_secret: "${FEISHU_APP_SECRET}"
    encrypt_key: "${FEISHU_ENCRYPT_KEY}"
    verification_token: "${FEISHU_VERIFICATION_TOKEN}"
    connection_mode: "websocket"        # websocket | webhook
    webhook_port: 8080                  # 仅 webhook 模式

  # wecom:                              # 未来扩展：企业微信
  #   enabled: false
  #   corp_id: "wwxxx"
  #   agent_id: 1000002
  #   secret: "${WECOM_SECRET}"
  #   token: "${WECOM_TOKEN}"
  #   encoding_aes_key: "${WECOM_AES_KEY}"

workspaces:
  batch_root: "/data/optimize-workspaces"

access_control:
  operators:                            # 可执行 /optimize、/verify、/gen-test、/upload
    - "feishu:ou_admin_001"
    - "feishu:ou_admin_002"
    # - "wecom:zhangsan"                # 未来企微用户
  viewers: ["*"]                        # "*" 表示所有平台的用户均可查询状态

dm_policy: "allowlist"                   # open | allowlist
allow_from:                              # 仅 dm_policy: allowlist 时有效
  - "feishu:ou_xxx"
```

---

## 可用命令（快速通道）

| 命令 | 参数 | 说明 |
|---|---|---|
| `/status <workspace>` | workspace 名称或路径 | 查询优化状态：最佳轮次、加速比、是否已验证 |
| `/status-batch [root]` | 可选 batch root 路径 | 聚合所有 batch workspace 的状态 |
| `/results <workspace> [round]` | workspace，可选轮次名 | 读取指定轮次的详细结果 |
| `/optimize <operator> [--target npu] [--mode supervise]` | 算子路径及参数 | 启动优化任务，返回 job ID 跟踪进度 |
| `/gen-test <operator>` | 算子路径 | 为算子生成测试 harness |
| `/verify <workspace>` | workspace 路径 | 对最佳轮次重新执行验证 |
| `/upload <workspace>` | workspace 路径 | 上传优化 workspace 到分析服务器 |
| `/report-batch [root]` | 可选 batch root 路径 | 生成 batch 优化报告 |
| `/help` | — | 显示可用命令 |

---

## 部署说明

### systemd unit 示例

```ini
[Unit]
Description=Triton Agent Channel Service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=triton-agent
WorkingDirectory=/opt/triton-agent
Environment="FEISHU_APP_SECRET=xxx"
Environment="FEISHU_ENCRYPT_KEY=xxx"
Environment="FEISHU_VERIFICATION_TOKEN=xxx"
ExecStart=/opt/triton-agent/.venv/bin/triton-agent channel serve --config /etc/triton-agent/channels.yaml
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### 飞书应用配置要点

1. 在飞书开放平台创建企业自建应用
2. 开启「机器人」能力
3. 在「事件订阅」中配置 WebSocket（由 SDK 自动管理，无需手动填写 URL）
4. 订阅事件：`im.message.receive_v1`（接收消息）
5. 权限管理：`im:message`、`im:message:send_as_bot`

---

## 未来迭代方向

1. **企业微信 / 钉钉 / Slack 集成**：按「多平台扩展架构」章节方案，实现对应 `TransportAdapter` + formatter 函数。核心管线零修改。

2. **定时通知**：集成 `apscheduler`，定时扫描 batch 状态，将汇总推送到指定群聊。支持 cron 表达式配置推送频率。

3. **告警集成**：workspace 出现 correctness 失败或 performance 回退时主动推送警告。

4. **交互式卡片增强**：`/optimize` 卡片支持参数下拉选择（target、mode），用户直接在卡片上配置后启动优化。

5. **多轮对话上下文**：Agent 模式下维护 per-chat 对话历史，支持追问和澄清。

6. **Agent 工具扩展**：增加 `propose_optimize_batch`（对 batch 内未完成的 workspace 启动优化）、`propose_repair`（修复 correctness 失败的 workspace）等高级写操作工具。
