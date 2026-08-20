# ZS32 Unlimited Round Confirmation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the default 120-second operator-confirmation limit while retaining an optional explicit timeout.

**Architecture:** Represent an unlimited confirmation wait as `timeout_seconds=None`. Pass that value from the CLI by default; terminal confirmation delegates it to `select.select`, while Dashboard polling uses an open-ended loop.

**Tech Stack:** Python, argparse, pytest

## Global Constraints

- Do not change camera-frame `--timeout-ms`.
- Keep `q`, closed stdin, Ctrl+C, and explicit positive confirmation timeouts working.
- Preserve current uncommitted timing-related changes in the affected files.

---

### Task 1: Make operator confirmation unlimited by default

**Files:**
- Modify: `tests/unit/zs32_refactor/capture_data/test_bootstrap_capture_cli.py`
- Modify: `src/zs32_inspection/cli/bootstrap_capture.py`
- Modify: `src/zs32_inspection/capture/bootstrap.py`
- Modify: `AGENTS_MEMORY.md`

**Interfaces:**
- Consumes: CLI option `--round-confirmation-timeout SECONDS`
- Produces: `BootstrapRoundCoordinator.timeout_seconds: float | None` and `DashboardRoundCoordinator.timeout_seconds: float | None`

- [x] **Step 1: Write failing tests**

Add assertions that the CLI default passes `None`, an explicit timeout remains a float, and both coordinators can wait with `None`.

- [x] **Step 2: Verify RED**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run --no-sync pytest \
  tests/unit/zs32_refactor/capture_data/test_bootstrap_capture_cli.py -q
```

Expected: failure because the current default is `120.0`.

- [x] **Step 3: Implement the minimum change**

Change the CLI default and coordinator annotations to `None`; validate only non-`None` values; use `select.select(..., None)` for terminal waits and an open-ended Dashboard polling loop.

- [x] **Step 4: Verify GREEN**

Run the focused CLI tests plus coordinator tests, then compile the changed modules and run `git diff --check`.

- [x] **Step 5: Record the behavior**

Append the confirmed unlimited-default contract and unchanged `--timeout-ms` meaning to `AGENTS_MEMORY.md`.
