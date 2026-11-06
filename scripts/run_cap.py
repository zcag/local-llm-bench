#!/usr/bin/env python3
"""Capability run on a Mac: real Claude Code against one model on a local Rapid-MLX server.

Same 10 polyglot tasks and suffix as L2 (bench/run_l2.py), test-blind, 900 s cap, plus an MCP task.
Appends one JSON line per task to --out. Usage: run_cap.py MODEL --tasks DIR --out FILE
"""
import argparse, json, os, re, shutil, signal, subprocess, tempfile, time, urllib.request

PORT = 8102
TASK_LIMIT = 10
TASK_TIMEOUT = 900
SUFFIX = ("\n\n---\nImplement the solution by editing {files} in the current "
          "working directory. Edit the file(s) directly; do not create new files.")
CODING_TOOLS = "Read,Write,Edit,Glob,Grep,Bash(python3 *),Bash(pytest *),Bash(ls *),Bash(cat *)"
MCP_REPO = os.path.expanduser("~/proj/reviewaudit")
MCP_PROMPT = ("Read only, change nothing. Do each step as its own separate tool call, in order, waiting for each "
              "result. Step 1: call the atc MCP status tool. Step 2: run git log -1 --oneline with Bash. Step 3: read "
              "the first 3 lines of README.md. Step 4: call the openwa MCP tool SessionFindAll. Then reply with one "
              "line per step saying what you got.")
MCP_TOOLS = "mcp__atc__status,mcp__openwa__SessionFindAll,Read,Bash(git log *)"


def settings(model):
    env = {"ANTHROPIC_BASE_URL": f"http://127.0.0.1:{PORT}", "ANTHROPIC_AUTH_TOKEN": "local",
           "ANTHROPIC_DEFAULT_OPUS_MODEL": model, "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
           "ANTHROPIC_DEFAULT_HAIKU_MODEL": "not-served", "ANTHROPIC_SMALL_FAST_MODEL": "not-served",
           "CLAUDE_CODE_SUBAGENT_MODEL": model,
           "CLAUDE_CODE_ATTRIBUTION_HEADER": "0", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
           "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "16384", "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "70000",
           "ENABLE_TOOL_SEARCH": "false"}
    return json.dumps({"env": env, "permissions": {"defaultMode": "default"}})


def wait_for_power():
    while True:
        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
        pct = int(re.search(r"(\d+)%", out).group(1)) if re.search(r"(\d+)%", out) else 100
        if "AC Power" in out or pct >= 25:
            return pct
        print(f"battery {pct}% on battery, waiting", flush=True)
        time.sleep(300)


def run_claude(model, prompt, cwd, tools, timeout, mcp):
    cmd = ["claude", "-p", prompt, "--settings", settings(model), "--model", model, "--output-format", "stream-json",
           "--verbose", "--allowedTools", tools, "--exclude-dynamic-system-prompt-sections"]
    if not mcp:
        cmd += ["--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}']
    start = time.time()
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, _ = proc.communicate()
        timed_out = True
    tool_calls, tool_errors, result, usage = [], 0, "", {}
    for line in out.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        message = row.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        for block in content if isinstance(content, list) else []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                tool_calls.append(block["name"])
            elif block.get("type") == "tool_result" and block.get("is_error"):
                tool_errors += 1
        if row.get("type") == "result":
            result, usage = str(row.get("result") or ""), row.get("usage") or {}
    return dict(seconds=round(time.time() - start), timed_out=timed_out, tool_calls=tool_calls, tool_errors=tool_errors,
                result=result[:1500], input=usage.get("input_tokens"), cached=usage.get("cache_read_input_tokens"),
                output=usage.get("output_tokens"))


def coding_task(model, task, tasks_dir, out):
    ex = os.path.join(tasks_dir, task)
    cfg = json.load(open(os.path.join(ex, ".meta", "config.json")))
    sol, tst = cfg["files"]["solution"], cfg["files"]["test"]
    work = os.path.join(tempfile.gettempdir(), "capwork")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    for rel in sol:
        shutil.copy(os.path.join(ex, rel), os.path.join(work, rel))
    prompt = open(os.path.join(ex, ".docs", "instructions.md")).read() + SUFFIX.format(files=", ".join(sol))
    row = run_claude(model, prompt, work, CODING_TOOLS, TASK_TIMEOUT, mcp=False)
    for rel in tst:
        shutil.copy(os.path.join(ex, rel), os.path.join(work, rel))
    try:
        graded = subprocess.run(["uv", "run", "--quiet", "--with", "pytest", "python", "-m", "pytest", "-q", "--no-header", *tst],
                                cwd=work, capture_output=True, text=True, timeout=120)
        passed, tail = graded.returncode == 0, (graded.stdout + graded.stderr)[-200:]
    except subprocess.TimeoutExpired:
        passed, tail = False, "pytest timeout"
    row.update(kind="coding", task=task, passed=passed, tail=tail)
    return row


def mcp_task(model):
    head = subprocess.run(["git", "log", "-1", "--format=%h"], cwd=MCP_REPO, capture_output=True, text=True).stdout.strip()
    row = run_claude(model, MCP_PROMPT, MCP_REPO, MCP_TOOLS, TASK_TIMEOUT, mcp=True)
    calls = row["tool_calls"]
    checks = {"atc": "mcp__atc__status" in calls, "openwa": "mcp__openwa__SessionFindAll" in calls,
              "bash": "Bash" in calls, "read": "Read" in calls, "hash_in_reply": bool(head) and head in row["result"]}
    row.update(kind="mcp", task="mcp-4-tools", passed=all(checks.values()), checks=checks)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-prompt-limit", action="store_true", help="for models where the server cannot count prompt tokens")
    args = ap.parse_args()
    limit = [] if args.no_prompt_limit else ["--max-prompt-tokens", "80000"]
    logpath = f"/tmp/cap-{re.sub(r'[^A-Za-z0-9._-]', '_', args.model)}.log"
    server = subprocess.Popen(["rapid-mlx", "--no-banner", "--no-telemetry", "--disable-version-check", "serve", args.model,
                               "--port", str(PORT), "--cache-memory-mb", "8192", *limit],
                              stdout=open(logpath, "w"), stderr=subprocess.STDOUT, start_new_session=True,
                              env={**os.environ, "RAPID_MLX_PREFIX_CACHE_SHUTDOWN_BUDGET": "60"})
    try:
        for _ in range(1800):
            if server.poll() is not None:
                raise SystemExit(f"server exited, see {logpath}")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2)
                break
            except OSError:
                time.sleep(2)
        tasks = sorted(d for d in os.listdir(args.tasks) if os.path.isdir(os.path.join(args.tasks, d)))[:TASK_LIMIT]
        done = set()
        if os.path.exists(args.out):
            rows = [json.loads(line) for line in open(args.out)]
            done = {"mcp" if r["kind"] == "mcp" else r["task"] for r in rows if r["model"] == args.model}
        with open(args.out, "a") as out:
            for task in ["mcp", *tasks]:
                if task in done:
                    continue
                battery = wait_for_power()
                row = mcp_task(args.model) if task == "mcp" else coding_task(args.model, task, args.tasks, out)
                if row["result"].strip().startswith("API Error"):
                    raise SystemExit(f"invalid run, {task}: {row['result'][:200]}")
                row.update(model=args.model, battery=battery, at=time.strftime("%H:%M:%S"))
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(row["task"], "PASS" if row["passed"] else "fail", row["seconds"], "s", flush=True)
    finally:
        os.killpg(server.pid, signal.SIGTERM)
        try:
            server.wait(60)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)


if __name__ == "__main__":
    main()
