import fs from "node:fs/promises";
import crypto from "node:crypto";
import path from "node:path";

// ---------------------------------------------------------------------------
// Static policy constants
// ---------------------------------------------------------------------------

// Unique tool-use ID counter shared across all invocations in a session.
const toolLifecycleCache = new Map();
let _toolUseIdCounter = 0;
function generateToolUseId() {
  _toolUseIdCounter += 1;
  return `opencode-hook-${Date.now()}-${_toolUseIdCounter}`;
}

const READ_COMMANDS = new Set([
  "awk",
  "cat",
  "find",
  "grep",
  "head",
  "less",
  "ls",
  "more",
  "rg",
  "sed",
  "stat",
  "tail",
  "tree",
]);
const PROTECTED_RELATIVE_PATH_PREFIXES = [".triton-agent/", "triton-agent-logs/"];

const PATH_FRAGMENT_RE =
  /(?:^|[^A-Za-z0-9_./-])(?<path>(?:\/|\.\.?\/|\.triton-agent\/|\.opencode\/|triton-agent-logs\/)[A-Za-z0-9_./*?{}+@%:,=-]+)/g;
const WINDOWS_PATH_FRAGMENT_RE = /[A-Za-z]:[\\/][A-Za-z0-9_ .\\/(){}+@%:,=-]+/g;
const EDIT_TOOLS = new Set(["edit", "write", "patch", "update", "multiedit", "multi_edit"]);
const WORKFLOW_STATE_RELATIVE_PATH = ".triton-agent/state.json";

// ---------------------------------------------------------------------------
// Hook entrypoints
// ---------------------------------------------------------------------------

export async function TritonAgentHookGuard(context) {
  const policy = await loadPolicy(context);
  return {
    "tool.execute.before": async (input, output) => {
      if (!policy) {
        return;
      }
      const toolUseId = output?.meta?.tool_use_id || generateToolUseId();
      toolLifecycleCache.set(toolUseId, {
        startTime: Date.now(),
        tool: input?.tool || "unknown",
        args: output?.args ?? {},
      });

      await appendTraceEvents(policy, input, output);
      const reason = await denyReasonForToolUse(policy, input, output);
      if (reason) {
        throw new Error(reason);
      }
    },

    "tool.execute.after": async (input, output) => {
      try {
        await handleToolAfter(policy, input, output);
      } catch (_err) {
        // Fail open: trace write failure must not interrupt workflow
      }
    },
  };
}

export default TritonAgentHookGuard;

// ---------------------------------------------------------------------------
// Policy loading
// ---------------------------------------------------------------------------

async function loadPolicy(context) {
  for (const root of policySearchRoots(context)) {
    try {
      const policyPath = path.join(root, ".opencode", "triton-agent-hooks", "policy.json");
      return JSON.parse(await fs.readFile(policyPath, "utf8"));
    } catch {
      continue;
    }
  }
  return null;
}

function policySearchRoots(context) {
  const roots = [];
  appendPolicySearchRoot(roots, context?.directory);
  appendPolicySearchRoot(roots, context?.experimental_workspace?.root);
  appendPolicySearchRoot(roots, context?.project?.path?.root);
  appendPolicySearchRoot(roots, context?.project?.root);
  return roots;
}

function appendPolicySearchRoot(roots, value) {
  if (typeof value !== "string" || value.length === 0) {
    return;
  }
  const resolved = path.resolve(value);
  if (!roots.includes(resolved)) {
    roots.push(resolved);
  }
}

// ---------------------------------------------------------------------------
// Tool-use denial checks
// ---------------------------------------------------------------------------

async function denyReasonForToolUse(policy, input, output) {
  const guardPolicy = guardPolicyFor(policy);
  if (guardPolicy.enabled === false) {
    return null;
  }
  if (!input || !output) {
    return null;
  }

  const workspaceRoot = resolvePolicyPath(policy.workspace_root);
  if (!workspaceRoot) {
    return null;
  }

  const cwd = resolveToolCwd(output.args?.cwd ?? input.cwd, workspaceRoot);
  const allowReadRoots = allowReadRootsForPolicy(guardPolicy, workspaceRoot);
  const denyReadGlobs = Array.isArray(guardPolicy.deny_read_globs)
    ? guardPolicy.deny_read_globs.filter((item) => typeof item === "string")
    : [];
  const denyMessage =
    typeof guardPolicy.deny_message === "string" && guardPolicy.deny_message.length > 0
      ? guardPolicy.deny_message
      : "This read is blocked by workspace policy.";

  // Built-in edit tools (write, edit, patch, …) are checked against
  // .triton-agent/state.json for phase enforcement before falling
  // through to the read/Bash deny_globs layer.
  if (EDIT_TOOLS.has(String(input.tool).toLowerCase())) {
    const filePath = firstToolPath(output.args);
    if (typeof filePath !== "string") {
      return null;
    }
    return denyReasonForBuiltInEditPath(filePath, cwd, workspaceRoot);
  }

  if (input.tool === "read") {
    const filePath = output.args?.filePath;
    if (typeof filePath !== "string") {
      return null;
    }
    return denyReasonForPathAccess(filePath, cwd, allowReadRoots, denyReadGlobs, denyMessage);
  }

  if (input.tool !== "bash") {
    return null;
  }
  const command = output.args?.command;
  if (typeof command !== "string") {
    return null;
  }

  const tokens = splitCommand(command);
  if (!containsReadCommand(tokens)) {
    return null;
  }

  // We scan path-like references from read-oriented shell commands instead of
  // trying to model the full shell grammar.
  for (const pathText of collectCommandPathReferences(command, tokens)) {
    const reason = await denyReasonForPathAccess(
      pathText,
      cwd,
      allowReadRoots,
      denyReadGlobs,
      denyMessage,
    );
    if (reason) {
      return reason;
    }
  }

  return null;
}

// ---------------------------------------------------------------------------
// Trace output
// ---------------------------------------------------------------------------

async function appendTraceEvents(policy, input, output) {
  const tracePolicy = tracePolicyFor(policy);
  if (tracePolicy.enabled !== true || typeof tracePolicy.path !== "string" || tracePolicy.path.length === 0) {
    return;
  }
  const workspaceRoot = resolvePolicyPath(policy.workspace_root);
  if (!workspaceRoot) {
    return;
  }

  const timestamp = new Date().toISOString();
  const tool = typeof input?.tool === "string" ? input.tool : "unknown";
  const args = output?.args ?? {};
  const toolCallStartEvent = {
    schema_version: 1,
    timestamp,
    run_id: typeof tracePolicy.run_id === "string" ? tracePolicy.run_id : "",
    type: "tool_call",
    phase: "start",
    tool,
    status: "started",
    summary: toolSummary(tool, args),
  };
  await appendTraceEvent(tracePolicy.path, toolCallStartEvent);

  if (tool === "bash" && typeof args.command === "string") {
    const command = args.command;
    await appendTraceEvent(tracePolicy.path, {
      schema_version: 1,
      timestamp,
      run_id: toolCallStartEvent.run_id,
      type: "command",
      phase: "start",
      command_kind: classifyCommand(command),
      command,
      remote: extractRemote(command),
      status: "started",
    });

    const tokens = splitCommand(command);
    if (containsReadCommand(tokens)) {
      const cwd = resolveToolCwd(args.cwd ?? input?.cwd, workspaceRoot);
      for (const pathText of collectCommandPathReferences(command, tokens)) {
        const resolvedPath = await resolvePathText(pathText, cwd);
        if (!resolvedPath) {
          continue;
        }
        await appendFileAccessTrace(tracePolicy.path, toolCallStartEvent, workspaceRoot, resolvedPath, "read");
      }
    }
  }

  if (tool === "read") {
    const filePath = args.filePath;
    if (typeof filePath === "string") {
      const cwd = resolveToolCwd(input?.cwd, workspaceRoot);
      const resolvedPath = await resolvePathText(filePath, cwd);
      if (resolvedPath) {
        await appendFileAccessTrace(tracePolicy.path, toolCallStartEvent, workspaceRoot, resolvedPath, "read");
      }
    }
  }

  if (EDIT_TOOLS.has(tool.toLowerCase())) {
    const filePath = firstToolPath(args);
    if (typeof filePath === "string") {
      const cwd = resolveToolCwd(args.cwd ?? input?.cwd, workspaceRoot);
      const resolvedPath = await resolvePathText(filePath, cwd);
      if (resolvedPath) {
        const stats = editStats(tool, args);
        await appendTraceEvent(tracePolicy.path, {
          schema_version: 1,
          timestamp,
          run_id: toolCallStartEvent.run_id,
          type: "edit",
          phase: "instant",
          path: displayPath(resolvedPath, workspaceRoot),
          edit_kind: classifyEditPath(resolvedPath),
          added_lines: stats.addedLines,
          removed_lines: stats.removedLines,
          diff_digest: stats.diffDigest,
          status: "started",
        });
      }
    }
  }
}

async function appendFileAccessTrace(tracePath, toolCallStartEvent, workspaceRoot, resolvedPath, action) {
  await appendTraceEvent(tracePath, {
    schema_version: 1,
    timestamp: toolCallStartEvent.timestamp,
    run_id: toolCallStartEvent.run_id,
    type: "file_access",
    phase: "instant",
    action,
    path: displayPath(resolvedPath, workspaceRoot),
    status: "started",
  });
}

async function appendTraceEvent(tracePath, event) {
  await fs.mkdir(path.dirname(tracePath), { recursive: true });
  await fs.appendFile(tracePath, `${JSON.stringify(event)}\n`, "utf8");
}

// ---------------------------------------------------------------------------
// Shared policy and trace metadata helpers
// ---------------------------------------------------------------------------

function guardPolicyFor(policy) {
  return policy && typeof policy.guard === "object" && policy.guard !== null ? policy.guard : policy;
}

function tracePolicyFor(policy) {
  return policy && typeof policy.trace === "object" && policy.trace !== null ? policy.trace : { enabled: false };
}

function toolSummary(tool, args) {
  if ((tool === "bash" || tool === "shell") && typeof args.command === "string") {
    return `${tool}: ${classifyCommand(args.command)}`;
  }
  if (typeof args.filePath === "string") {
    return args.filePath;
  }
  return tool;
}

function firstToolPath(args) {
  for (const key of ["filePath", "file_path", "path", "notebookPath", "notebook_path"]) {
    if (typeof args?.[key] === "string" && args[key].length > 0) {
      return args[key];
    }
  }
  return null;
}

function editStats(tool, args) {
  const parts = [tool];
  let addedLines = 0;
  let removedLines = 0;
  if (Array.isArray(args?.edits)) {
    for (const edit of args.edits) {
      if (!edit || typeof edit !== "object") {
        continue;
      }
      if (typeof edit.old_string === "string") {
        removedLines += lineCount(edit.old_string);
        parts.push(edit.old_string);
      }
      if (typeof edit.new_string === "string") {
        addedLines += lineCount(edit.new_string);
        parts.push(edit.new_string);
      }
    }
  }
  for (const key of ["old_string", "oldString"]) {
    if (typeof args?.[key] === "string") {
      removedLines += lineCount(args[key]);
      parts.push(args[key]);
    }
  }
  for (const key of ["new_string", "newString", "content"]) {
    if (typeof args?.[key] === "string") {
      addedLines += lineCount(args[key]);
      parts.push(args[key]);
    }
  }
  return {
    addedLines,
    removedLines,
    diffDigest: `sha256:${sha256(parts.join("\\n"))}`,
  };
}

function lineCount(text) {
  if (text.length === 0) {
    return 0;
  }
  const matches = text.match(/\n/g);
  return matches ? matches.length + (text.endsWith("\n") ? 0 : 1) : 1;
}

function sha256(text) {
  return crypto.createHash("sha256").update(text, "utf8").digest("hex");
}

function classifyEditPath(filePath) {
  const normalized = filePath.split(path.sep).join("/").toLowerCase();
  const name = path.basename(filePath).toLowerCase();
  if (normalized.includes("/opt-round-")) {
    return "round_artifact";
  }
  if (name.startsWith("test_") || name.startsWith("differential_test_")) {
    return "test_harness";
  }
  if (name.startsWith("bench_")) {
    return "bench_harness";
  }
  if (/\.(json|ya?ml|toml)$/.test(name)) {
    return "metadata";
  }
  if (/\.(md|txt)$/.test(name)) {
    return "documentation";
  }
  if (name.endsWith(".py")) {
    return "operator";
  }
  return "unknown";
}

function classifyCommand(command) {
  const normalized = command.toLowerCase();
  if (normalized.includes("compare-perf")) {
    return "compare_perf";
  }
  if (normalized.includes("compare-result")) {
    return "compare_result";
  }
  if (normalized.includes("check-baseline")) {
    return "check_baseline";
  }
  if (normalized.includes("check-round")) {
    return "check_round";
  }
  if (normalized.includes("run-test") || normalized.includes("pytest") || normalized.includes("differential_test_")) {
    return "correctness_test";
  }
  if (normalized.includes("run-bench") || normalized.includes("bench_")) {
    return normalized.includes("ssh") ? "remote_bench" : "benchmark";
  }
  if (normalized.includes("msprof") || normalized.includes("profile export")) {
    return "profile";
  }
  return extractRemote(command) ? "remote_command" : "local_command";
}

function extractRemote(command) {
  const tokens = splitCommand(command);
  for (let index = 0; index < tokens.length; index += 1) {
    if (path.basename(tokens[index]) !== "ssh") {
      continue;
    }
    return tokens[index + 1] ?? "ssh";
  }
  return null;
}

function displayPath(resolvedPath, workspaceRoot) {
  const relative = path.relative(workspaceRoot, resolvedPath);
  if (relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative))) {
    return relative.split(path.sep).join("/");
  }
  return resolvedPath.split(path.sep).join("/");
}

async function denyReasonForPathAccess(
  pathText,
  cwd,
  allowReadRoots,
  denyReadGlobs,
  denyMessage,
) {
  const resolvedPath = await resolvePathText(pathText, cwd);
  if (!resolvedPath) {
    return null;
  }
  if (!isUnderAnyRoot(resolvedPath, allowReadRoots)) {
    return denyMessage;
  }
  if (matchesAnyGlob(resolvedPath, denyReadGlobs)) {
    return denyMessage;
  }
  return null;
}

async function denyReasonForBuiltInEditPath(pathText, cwd, workspaceRoot) {
  const resolvedPath = await resolvePathText(pathText, cwd);
  if (!resolvedPath) {
    return null;
  }
  if (!isUnderRoot(resolvedPath, workspaceRoot)) {
    return builtInEditOutsideWorkspaceDenial();
  }

  const state = await loadWorkflowState(workspaceRoot);
  if (!state) {
    return builtInEditMissingStateDenial();
  }

  const workspaceRelativePath = displayPath(resolvedPath, workspaceRoot);
  if (state.phase === "baseline") {
    if (isAllowedBaselineEditPath(workspaceRelativePath, state.source_operator)) {
      return null;
    }
    return baselinePhaseBuiltInEditDenial();
  }
  if (state.phase === "awaiting_round_start") {
    return awaitingRoundStartBuiltInEditDenial();
  }
  if (state.phase === "round_active") {
    const roundDir = activeRoundDir(state);
    if (!roundDir) {
      return builtInEditMissingStateDenial();
    }
    if (workspaceRelativePath === roundDir || workspaceRelativePath.startsWith(`${roundDir}/`)) {
      return null;
    }
    return roundActiveBuiltInEditDenial(roundDir);
  }

  return builtInEditMissingStateDenial();
}

// ---------------------------------------------------------------------------
// Built-in edit workflow-state helpers
// ---------------------------------------------------------------------------

async function loadWorkflowState(workspaceRoot) {
  const statePath = path.join(workspaceRoot, WORKFLOW_STATE_RELATIVE_PATH);
  try {
    const state = JSON.parse(await fs.readFile(statePath, "utf8"));
    if (!state || typeof state !== "object") {
      return null;
    }
    if (typeof state.phase !== "string" || state.phase.length === 0) {
      return null;
    }
    if (typeof state.source_operator !== "string" || state.source_operator.length === 0) {
      return null;
    }
    return state;
  } catch {
    return null;
  }
}

function isAllowedBaselineEditPath(relativePath, sourceOperator) {
  if (relativePath === "baseline" || relativePath.startsWith("baseline/")) {
    return true;
  }
  if (relativePath === sourceOperator) {
    return true;
  }
  if (relativePath.includes("/")) {
    return false;
  }
  return (
    relativePath.startsWith("test_") ||
    relativePath.startsWith("differential_test_") ||
    relativePath.startsWith("bench_")
  );
}

function activeRoundDir(state) {
  if (!Number.isInteger(state.current_round) || !state.rounds || typeof state.rounds !== "object") {
    return null;
  }
  const roundEntry = state.rounds[String(state.current_round)];
  if (!roundEntry || typeof roundEntry !== "object") {
    return null;
  }
  if (roundEntry.status !== "active" || typeof roundEntry.round_dir !== "string" || roundEntry.round_dir.length === 0) {
    return null;
  }
  return roundEntry.round_dir;
}

function builtInEditMissingStateDenial() {
  return (
    "Built-in edit tool blocked by optimize workflow policy. " +
    "The temporary optimize workflow state is missing or invalid. " +
    "Ask the runner to restart the optimize session so workflow state can be rebuilt."
  );
}

function builtInEditOutsideWorkspaceDenial() {
  return (
    "Built-in edit tool blocked by optimize workflow policy. " +
    "Keep built-in edits inside the current optimize workspace."
  );
}

function baselinePhaseBuiltInEditDenial() {
  return (
    "Built-in edit tool blocked by optimize workflow policy. " +
    "Current phase is baseline. During baseline, built-in edits are limited to the baseline-minimal file set: " +
    "the source operator, root-level test/bench harness files, and `baseline/` artifacts. " +
    "Finish or repair baseline, then submit it through `triton-npu-optimize-submit-baseline` before opening a round."
  );
}

function awaitingRoundStartBuiltInEditDenial() {
  return (
    "Built-in edit tool blocked by optimize workflow policy. " +
    "Current phase is awaiting_round_start, so no optimize round is active yet. " +
    "Use `triton-npu-optimize-start-round` to open the next `opt-round-N/` before editing."
  );
}

function roundActiveBuiltInEditDenial(roundDir) {
  return (
    "Built-in edit tool blocked by optimize workflow policy. " +
    `Current active round is ${roundDir}. Built-in edits must stay inside \`${roundDir}/\`. ` +
    "Edit the round-local snapshot and round artifacts instead of top-level workspace files. " +
    "When this round is ready, use `triton-npu-optimize-submit-round` to submit it before moving on."
  );
}

// ---------------------------------------------------------------------------
// Read-path extraction from shell commands
// ---------------------------------------------------------------------------

function splitCommand(command) {
  const tokens = [];
  let current = "";
  let quote = null;
  let escaped = false;

  for (const char of command) {
    if (escaped) {
      current += char;
      escaped = false;
      continue;
    }
    if (char === "\\") {
      escaped = true;
      continue;
    }
    if (quote) {
      if (char === quote) {
        quote = null;
      } else {
        current += char;
      }
      continue;
    }
    if (char === "'" || char === '"') {
      quote = char;
      continue;
    }
    if (/\s/.test(char)) {
      if (current.length > 0) {
        tokens.push(current);
        current = "";
      }
      continue;
    }
    current += char;
  }

  if (current.length > 0) {
    tokens.push(current);
  }
  return tokens;
}

function containsReadCommand(tokens) {
  return tokens.some((token) => isReadCommandToken(token));
}

function isReadCommandToken(token) {
  return READ_COMMANDS.has(path.basename(token));
}

function collectCommandPathReferences(command, tokens) {
  const scanCommand = stripHeredocPayload(command);
  const scanTokens = filterTokensForReadScan(splitCommand(scanCommand));
  const scanText = scanTokens.join(" ");
  const pathTexts = [];
  const explicitPathTokens = new Set(scanTokens.filter((token) => looksLikePath(token)));

  for (const token of scanTokens) {
    if (isReadCommandToken(token)) {
      continue;
    }
    if (looksLikePath(token)) {
      pathTexts.push(token);
    }
  }

  for (const match of scanText.matchAll(PATH_FRAGMENT_RE)) {
    const pathText = match.groups?.path;
    if (typeof pathText !== "string") {
      continue;
    }
    if (!isReadCommandToken(pathText) && !isNestedPathFragment(pathText, explicitPathTokens)) {
      pathTexts.push(pathText);
    }
  }
  for (const match of scanText.matchAll(WINDOWS_PATH_FRAGMENT_RE)) {
    const pathText = match[0].replace(/['"),]+$/g, "");
    if (!isReadCommandToken(pathText) && !isNestedPathFragment(pathText, explicitPathTokens)) {
      pathTexts.push(pathText);
    }
  }

  return pathTexts;
}

function stripHeredocPayload(command) {
  if (!command.includes("<<") || !command.includes("\n")) {
    return command;
  }
  return command.split(/\r?\n/, 1)[0];
}

function filterTokensForReadScan(tokens) {
  const filtered = [];
  for (let index = 0; index < tokens.length;) {
    const token = tokens[index];
    if (isHeredocOperatorToken(token)) {
      index += 1;
      if ((token === "<<" || token === "<<-") && index < tokens.length) {
        index += 1;
      }
      continue;
    }

    const outputTarget = outputRedirectionTarget(token);
    if (outputTarget !== null) {
      index += outputTarget.length === 0 ? 2 : 1;
      continue;
    }

    const inputTarget = inputRedirectionTarget(token);
    if (inputTarget !== null) {
      if (inputTarget.length === 0) {
        index += 1;
        if (index < tokens.length) {
          filtered.push(tokens[index]);
          index += 1;
        }
        continue;
      }
      filtered.push(inputTarget);
      index += 1;
      continue;
    }

    filtered.push(token);
    index += 1;
  }
  return filtered;
}

function isHeredocOperatorToken(token) {
  return token === "<<" || token === "<<-" || token.startsWith("<<");
}

function outputRedirectionTarget(token) {
  const match = token.match(/^(?:(?:\d+)?>>|(?:\d+)?>\||(?:\d+)?>|&>>|&>)(.*)$/);
  return match ? match[1] : null;
}

function inputRedirectionTarget(token) {
  if (token.startsWith("<<")) {
    return null;
  }
  const match = token.match(/^(?:(?:\d+)?<>|(?:\d+)?<)(.*)$/);
  return match ? match[1] : null;
}

function looksLikePath(token) {
  return (
    token === ".triton-agent" ||
    token.startsWith("/") ||
    token.startsWith("./") ||
    token.startsWith("../") ||
    token.startsWith(".triton-agent/") ||
    token.startsWith(".opencode/") ||
    PROTECTED_RELATIVE_PATH_PREFIXES.some((prefix) => token.startsWith(prefix)) ||
    token.includes("\\") ||
    path.extname(token).length > 0
  );
}

function isNestedPathFragment(pathText, explicitPathTokens) {
  for (const token of explicitPathTokens) {
    if (pathText !== token && token.includes(pathText)) {
      return true;
    }
  }
  return false;
}

// ---------------------------------------------------------------------------
// Low-level path and policy helpers
// ---------------------------------------------------------------------------

function resolvePolicyPath(value) {
  if (typeof value !== "string" || value.length === 0) {
    return null;
  }
  return path.resolve(value);
}

function resolveCwd(value, workspaceRoot) {
  if (typeof value !== "string" || value.length === 0) {
    return workspaceRoot;
  }
  return path.resolve(workspaceRoot, value);
}

function resolveToolCwd(value, workspaceRoot) {
  return resolveCwd(value, workspaceRoot);
}

function allowReadRootsForPolicy(policy, workspaceRoot) {
  const roots = [workspaceRoot];
  if (!Array.isArray(policy.allow_read_roots)) {
    return roots;
  }
  for (const rawRoot of policy.allow_read_roots) {
    const root = resolvePolicyPath(rawRoot);
    if (root && !roots.includes(root)) {
      roots.push(root);
    }
  }
  return roots;
}

async function resolvePathText(pathText, cwd) {
  if (pathText.includes("*") || pathText.includes("?") || pathText.includes("{") || pathText.includes("}")) {
    return null;
  }
  const resolvedPath = path.resolve(cwd, pathText);
  try {
    return await fs.realpath(resolvedPath);
  } catch {
    try {
      const realParent = await fs.realpath(path.dirname(resolvedPath));
      return path.join(realParent, path.basename(resolvedPath));
    } catch {
      return resolvedPath;
    }
  }
}

function isUnderAnyRoot(resolvedPath, roots) {
  return roots.some((root) => isUnderRoot(resolvedPath, root));
}

function isUnderRoot(resolvedPath, root) {
  const relative = path.relative(root, resolvedPath);
  return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative));
}

function matchesAnyGlob(resolvedPath, patterns) {
  return patterns.some((pattern) => globMatches(resolvedPath, pattern));
}

function globMatches(resolvedPath, pattern) {
  return new RegExp(`^${globToRegexSource(pattern)}$`).test(resolvedPath);
}

function globToRegexSource(pattern) {
  let source = "";
  for (let index = 0; index < pattern.length; index += 1) {
    const char = pattern[index];
    if (char === "*") {
      if (pattern[index + 1] === "*") {
        source += ".*";
        index += 1;
      } else {
        source += "[^/]*";
      }
      continue;
    }
    if (".+^${}()|[]\\".includes(char)) {
      source += `\\${char}`;
    } else {
      source += char;
    }
  }
  return source;
}

// ---------------------------------------------------------------------------
// Tool lifecycle completion trace
// ---------------------------------------------------------------------------

async function handleToolAfter(policy, input, output) {
  const tracePolicy = tracePolicyFor(policy);
  if (tracePolicy.enabled !== true || typeof tracePolicy.path !== "string" || tracePolicy.path.length === 0) {
    return;
  }

  const toolUseId = output?.meta?.tool_use_id;
  const toolLifecycleRecord = toolUseId ? toolLifecycleCache.get(toolUseId) : undefined;
  const endTime = Date.now();
  const durationMs = toolLifecycleRecord ? endTime - toolLifecycleRecord.startTime : 0;
  const tool = input?.tool || toolLifecycleRecord?.tool || "unknown";
  const isError = output?.error != null;
  const status = isError ? "error" : "ok";

  const timestamp = new Date().toISOString();
  const runId = typeof tracePolicy.run_id === "string" ? tracePolicy.run_id : "";
  await appendTraceEvent(tracePolicy.path, {
    schema_version: 1,
    timestamp,
    run_id: runId,
    type: "tool_call",
    phase: "end",
    tool,
    tool_use_id: toolUseId,
    duration_ms: durationMs,
    status,
  });

  if (tool === "bash" || tool === "shell") {
    const command = toolLifecycleRecord?.args?.command ?? output?.args?.command ?? "";
    await appendTraceEvent(tracePolicy.path, {
      schema_version: 1,
      timestamp,
      run_id: runId,
      type: "command",
      phase: "end",
      tool_use_id: toolUseId,
      command_kind: command ? classifyCommand(command) : "unknown",
      command,
      remote: command ? extractRemote(command) : null,
      duration_ms: durationMs,
      status,
    });
  }

  if (toolUseId) {
    toolLifecycleCache.delete(toolUseId);
  }
}
