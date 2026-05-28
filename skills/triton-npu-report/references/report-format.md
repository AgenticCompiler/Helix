# 优化报告

**生成时间**：<YYYY-MM-DD HH:MM:SS>
**workspace**：<workspace_path>

---

## 1、优化概览

| 指标 | 初始版本 | Round N | ... |
| ---- | -------- | ------- | --- |
| 平均延迟 | <baseline latency> | <round latency> | ... |

## 2、优化历程

| Round | Parent | Round目录 | Theme | Analysis Level | Latency | Speedup | Des |
| ----- | ------ | --------- | ----- | -------------- | ------- | ------- | --- |
| N     | M      | opt-round-N/ | <theme> | <level> | <latency> | <speedup> | <description> |

## 3、各轮优化详解

### 3.1 Round N (opt-round-N/)

<summary.md content for this round>

## 4、最终代码结构

在最优轮次的源码中，通过 `# -- <注释> --` 格式的行内注释标注每个关键优化点的修改位置和原因。每个注释块说明：哪个 Round 引入了该修改、修改了什么、为什么这样改。

```python
# -- Round N: <优化主题> --
# 原因: <为什么这样改，对应的性能问题或优化方向>
# 效果: <性能提升或行为变化>
<修改后的代码片段>

# -- Round M: <优化主题> --
# 原因: <为什么这样改>
# 效果: <性能提升或行为变化>
<修改后的代码片段>
```

## 5、性能提升分解

| 优化点 | 提升 | Round | 说明 |
| ------ | ---- | ----- | ---- |
| <point> | <improvement> | N | <explanation> |

## 6、验证信息

**目标芯片**：<target_chip>
**芯片型号**：<chip_name>
**CANN 版本**：<cann_version>
**驱动版本**：<driver_version>
**精度验证**：<correctness_status>
**验证模式**：性能优化模式

## 7、优化文件

- 优化后源码：opt-round-N/opt_<operator>.py
- 最终性能报告：opt-round-N/<operator>_perf.txt
- 优化笔记：opt-note.md
- 优化知识：learned_lessons.md
