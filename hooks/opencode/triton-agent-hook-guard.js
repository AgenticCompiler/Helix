import fs from "node:fs/promises";
import crypto from "node:crypto";
import path from "node:path";

const toolLifecycleCache = new Map();
let _toolUseIdCounter = 0;
function generateToolUseId() {
  _toolUseIdCounter += 1;
  return `opencode-hook-${Date.now()}-${_toolUseIdCounter}`;
}

const READ_COMMANDS = new Set([
  "awk",
  "cat",
  "head",
  "less",
  "more",
  "python",
  "python3",
  "rg",
  "sed",
  "tail",
]);

const PATH_FRAGMENT_RE = /(?:\/|\.\.?\/|\.opencode\/)[A-Za-z0-9_./*?{}+@%:,=-]+/g;
const WINDOWS_PATH_FRAGMENT_RE = /[A-Za-z]:[\\/][A-Za-z0-9_ .\\/(){}+@%:,=-]+/g;
const EDIT_TOOLS = new Set(["edit", "write", "patch", "update", "multiedit", "multi_edit"]);

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
      const reason = await evaluateOutput(policy, input, output);
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

async function loadPolicy(context) {
  for (const root of contextRoots(context)) {
    try {
      const policyPath = path.join(root, ".opencode", "triton-agent-hooks", "policy.json");
      return JSON.parse(await fs.readFile(policyPath, "utf8"));
    } catch {
      continue;
    }
  }
  return null;
}

function contextRoots(context) {
  const roots = [];
  addContextRoot(roots, context?.directory);
  addContextRoot(roots, context?.experimental_workspace?.root);
  addContextRoot(roots, context?.project?.path?.root);
  addContextRoot(roots, context?.project?.root);
  return roots;
}

function addContextRoot(roots, value) {
  if (typeof value !== "string" || value.length === 0) {
    return;
  }
  const resolved = path.resolve(value);
  if (!roots.includes(resolved)) {
    roots.push(resolved);
  }
}

async function evaluateOutput(policy, input, output) {
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
  const allowRoots = allowRootsForPolicy(guardPolicy, workspaceRoot);
  const denyGlobs = Array.isArray(guardPolicy.deny_read_globs)
    ? guardPolicy.deny_read_globs.filter((item) => typeof item === "string")
    : [];
  const denyMessage =
    typeof guardPolicy.deny_message === "string" && guardPolicy.deny_message.length > 0
      ? guardPolicy.deny_message
      : "This read is blocked by workspace policy.";

  if (input.tool === "read") {
    const filePath = output.args?.filePath;
    if (typeof filePath !== "string") {
      return null;
    }
    return evaluateCandidate(filePath, cwd, allowRoots, denyGlobs, denyMessage);
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

  for (const candidate of candidatePaths(command, tokens)) {
    const reason = await evaluateCandidate(candidate, cwd, allowRoots, denyGlobs, denyMessage);
    if (reason) {
      return reason;
    }
  }

  return null;
}

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
  const baseEvent = {
    schema_version: 1,
    timestamp,
    run_id: typeof tracePolicy.run_id === "string" ? tracePolicy.run_id : "",
    role: typeof tracePolicy.role === "string" ? tracePolicy.role : "",
    type: "tool_call",
    phase: "start",
    tool,
    status: "started",
    summary: toolSummary(tool, args),
    source: "opencode_hook",
    confidence: "high",
  };
  await appendTraceEvent(tracePolicy.path, baseEvent);

  if (tool === "bash" && typeof args.command === "string") {
    const command = args.command;
    await appendTraceEvent(tracePolicy.path, {
      schema_version: 1,
      timestamp,
      run_id: baseEvent.run_id,
      role: baseEvent.role,
      type: "command",
      phase: "start",
      command_kind: classifyCommand(command),
      command,
      remote: extractRemote(command),
      status: "started",
      source: "opencode_hook",
      confidence: "high",
    });

    const tokens = splitCommand(command);
    if (containsReadCommand(tokens)) {
      const cwd = resolveToolCwd(args.cwd ?? input?.cwd, workspaceRoot);
      for (const candidate of candidatePaths(command, tokens)) {
        const resolved = await resolveCandidate(candidate, cwd);
        if (!resolved) {
          continue;
        }
        await appendFileAccessTrace(tracePolicy.path, baseEvent, workspaceRoot, resolved, "read");
      }
    }
  }

  if (tool === "read") {
    const filePath = args.filePath;
    if (typeof filePath === "string") {
      const cwd = resolveToolCwd(input?.cwd, workspaceRoot);
      const resolved = await resolveCandidate(filePath, cwd);
      if (resolved) {
        await appendFileAccessTrace(tracePolicy.path, baseEvent, workspaceRoot, resolved, "read");
      }
    }
  }

  if (EDIT_TOOLS.has(tool.toLowerCase())) {
    const filePath = firstToolPath(args);
    if (typeof filePath === "string") {
      const cwd = resolveToolCwd(args.cwd ?? input?.cwd, workspaceRoot);
      const resolved = await resolveCandidate(filePath, cwd);
      if (resolved) {
        const stats = editStats(tool, args);
        await appendTraceEvent(tracePolicy.path, {
          schema_version: 1,
          timestamp,
          run_id: baseEvent.run_id,
          role: baseEvent.role,
          type: "edit",
          phase: "instant",
          path: displayPath(resolved, workspaceRoot),
          edit_kind: classifyEditPath(resolved),
          added_lines: stats.addedLines,
          removed_lines: stats.removedLines,
          diff_digest: stats.diffDigest,
          status: "started",
          source: "opencode_hook",
          confidence: "high",
        });
      }
    }
  }
}

async function appendFileAccessTrace(tracePath, baseEvent, workspaceRoot, resolved, action) {
  await appendTraceEvent(tracePath, {
    schema_version: 1,
    timestamp: baseEvent.timestamp,
    run_id: baseEvent.run_id,
    role: baseEvent.role,
    type: "file_access",
    phase: "instant",
    action,
    path: displayPath(resolved, workspaceRoot),
    status: "started",
    source: "opencode_hook",
    confidence: "high",
  });
}

async function appendTraceEvent(tracePath, event) {
  await fs.mkdir(path.dirname(tracePath), { recursive: true });
  await fs.appendFile(tracePath, `${JSON.stringify(event)}\n`, "utf8");
}

function guardPolicyFor(policy) {
  return policy && typeof policy.guard === "object" && policy.guard !== null ? policy.guard : policy;
}

function tracePolicyFor(policy) {
  return policy && typeof policy.trace === "object" && policy.trace !== null ? policy.trace : { enabled: false };
}

function toolSummary(tool, args) {
  if (tool === "bash" && typeof args.command === "string") {
    return args.command;
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

function displayPath(candidate, workspaceRoot) {
  const relative = path.relative(workspaceRoot, candidate);
  if (relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative))) {
    return relative.split(path.sep).join("/");
  }
  return candidate.split(path.sep).join("/");
}

async function evaluateCandidate(candidate, cwd, allowRoots, denyGlobs, denyMessage) {
  const resolved = await resolveCandidate(candidate, cwd);
  if (!resolved) {
    return null;
  }
  if (!isUnderAnyRoot(resolved, allowRoots)) {
    return denyMessage;
  }
  if (matchesAnyGlob(resolved, denyGlobs)) {
    return denyMessage;
  }
  return null;
}

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
    if (char === "\\" && process.platform !== "win32") {
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

function candidatePaths(command, tokens) {
  const candidates = [];
  for (const token of tokens) {
    if (isReadCommandToken(token)) {
      continue;
    }
    if (looksLikePath(token)) {
      candidates.push(token);
    }
  }

  for (const match of command.matchAll(PATH_FRAGMENT_RE)) {
    const candidate = match[0];
    if (!isReadCommandToken(candidate)) {
      candidates.push(candidate);
    }
  }
  for (const match of command.matchAll(WINDOWS_PATH_FRAGMENT_RE)) {
    const candidate = match[0].replace(/['"),]+$/g, "");
    if (!isReadCommandToken(candidate)) {
      candidates.push(candidate);
    }
  }

  return candidates;
}

function looksLikePath(token) {
  return (
    path.isAbsolute(token) ||
    token.startsWith("/") ||
    token.startsWith("./") ||
    token.startsWith("../") ||
    token.startsWith(".opencode/") ||
    token.includes("\\") ||
    path.extname(token).length > 0
  );
}

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

function allowRootsForPolicy(policy, workspaceRoot) {
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

async function resolveCandidate(candidate, cwd) {
  if (candidate.includes("*") || candidate.includes("?") || candidate.includes("{") || candidate.includes("}")) {
    return null;
  }
  const resolved = path.resolve(cwd, candidate);
  try {
    return await fs.realpath(resolved);
  } catch {
    return resolved;
  }
}

function isUnderAnyRoot(candidate, roots) {
  return roots.some((root) => isUnderRoot(candidate, root));
}

function isUnderRoot(candidate, root) {
  const relative = path.relative(root, candidate);
  return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative));
}

function matchesAnyGlob(candidate, patterns) {
  return patterns.some((pattern) => globMatches(candidate, pattern));
}

function globMatches(candidate, pattern) {
  return new RegExp(`^${globToRegexSource(pattern)}$`).test(candidate);
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

async function handleToolAfter(policy, input, output) {
  const tracePolicy = tracePolicyFor(policy);
  if (tracePolicy.enabled !== true || typeof tracePolicy.path !== "string" || tracePolicy.path.length === 0) {
    return;
  }

  const toolUseId = output?.meta?.tool_use_id;
  const cached = toolUseId ? toolLifecycleCache.get(toolUseId) : undefined;
  const endTime = Date.now();
  const durationMs = cached ? endTime - cached.startTime : 0;
  const tool = input?.tool || cached?.tool || "unknown";
  const isError = output?.error != null;
  const status = isError ? "error" : "ok";

  const timestamp = new Date().toISOString();
  const runId = typeof tracePolicy.run_id === "string" ? tracePolicy.run_id : "";
  const role = typeof tracePolicy.role === "string" ? tracePolicy.role : "";

  await appendTraceEvent(tracePolicy.path, {
    schema_version: 1,
    timestamp,
    run_id: runId,
    role,
    type: "tool_call",
    phase: "end",
    tool,
    tool_use_id: toolUseId,
    duration_ms: durationMs,
    duration_source: "hook_clock_join",
    status,
    source: "opencode_hook",
    confidence: "high",
  });

  if (tool === "bash" || tool === "shell") {
    const command = cached?.args?.command ?? output?.args?.command ?? "";
    await appendTraceEvent(tracePolicy.path, {
      schema_version: 1,
      timestamp,
      run_id: runId,
      role,
      type: "command",
      phase: "end",
      tool_use_id: toolUseId,
      command_kind: command ? classifyCommand(command) : "unknown",
      command,
      remote: command ? extractRemote(command) : null,
      duration_ms: durationMs,
      duration_source: "hook_clock_join",
      status,
      source: "opencode_hook",
      confidence: "high",
    });
  }

  if (toolUseId) {
    toolLifecycleCache.delete(toolUseId);
  }
}
