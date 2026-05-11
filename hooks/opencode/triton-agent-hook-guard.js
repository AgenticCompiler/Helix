import fs from "node:fs/promises";
import path from "node:path";

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

export async function TritonAgentHookGuard(context) {
  const policy = await loadPolicy(context);
  return {
    "tool.execute.before": async (input, output) => {
      if (!policy) {
        return;
      }
      const reason = await evaluateOutput(policy, input, output);
      if (reason) {
        throw new Error(reason);
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
  if (!input || !output) {
    return null;
  }

  const workspaceRoot = resolvePolicyPath(policy.workspace_root);
  if (!workspaceRoot) {
    return null;
  }

  const cwd = resolveToolCwd(output.args?.cwd ?? input.cwd, workspaceRoot);
  const allowRoots = allowRootsForPolicy(policy, workspaceRoot);
  const denyGlobs = Array.isArray(policy.deny_read_globs)
    ? policy.deny_read_globs.filter((item) => typeof item === "string")
    : [];
  const denyMessage =
    typeof policy.deny_message === "string" && policy.deny_message.length > 0
      ? policy.deny_message
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

  return candidates;
}

function looksLikePath(token) {
  return (
    token.startsWith("/") ||
    token.startsWith("./") ||
    token.startsWith("../") ||
    token.startsWith(".opencode/")
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
