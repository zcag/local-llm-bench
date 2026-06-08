"""Make tardis single-tenant for a run, then restore — so 'external things'
never touch the numbers. Takes down the live MLX + embed launchd units and the
open-webui container; restores them on exit no matter what. Run ON tardis.
"""
from __future__ import annotations
import contextlib
import os
import shutil
import subprocess
import time
import httpx
from . import config

# OrbStack's docker isn't on the non-interactive ssh PATH — resolve it explicitly.
_DOCKER = shutil.which("docker") or next(
    (p for p in ("/usr/local/bin/docker", os.path.expanduser("~/.orbstack/bin/docker"),
                 "/opt/homebrew/bin/docker") if os.path.exists(p)), None)


def _uid() -> int:
    return os.getuid()


def _plist(unit: str) -> str:
    return os.path.expanduser(f"~/Library/LaunchAgents/{unit}.plist")


def _sh(cmd: list[str]) -> tuple[int, str]:
    """Run a command, tolerating a missing binary (never aborts the caller)."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
        return p.returncode, (p.stdout + p.stderr).strip()
    except FileNotFoundError as e:
        return 127, str(e)


def bootout(unit: str) -> None:
    _sh(["launchctl", "bootout", f"gui/{_uid()}/{unit}"])


def bootstrap(unit: str) -> None:
    """Reload via the proven bootout->settle->bootstrap dance (avoids launchctl
    '5: I/O error' on immediate reload — same as the deploy script)."""
    pl = _plist(unit)
    if not os.path.exists(pl):
        return
    _sh(["launchctl", "bootout", f"gui/{_uid()}/{unit}"])
    time.sleep(2)
    _sh(["launchctl", "bootstrap", f"gui/{_uid()}", pl])


def _webui(action: str) -> None:
    compose = os.path.join(config.OPENWEBUI_DIR, "compose.yml")
    if _DOCKER and os.path.exists(compose):
        args = [_DOCKER, "compose", "-f", compose]
        args += ["down"] if action == "down" else ["up", "-d"]
        _sh(args)


def port_free(port: int) -> bool:
    rc, out = _sh(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"])
    return not out.strip()


def assert_idle(warn=print) -> dict:
    """Report load + free benchmark ports; raise if a launchd unit is still up."""
    la1, la5, la15 = os.getloadavg()
    busy = [p for p in (config.MLX_PORT, config.EMBED_PORT, config.LLAMACPP_PORT,
                        config.LMSTUDIO_PORT, config.OLLAMA_CHAT_PORT) if not port_free(p)]
    if la1 > 2.0:
        warn(f"[idle] load1={la1:.2f} is high — something is running")
    if busy:
        raise RuntimeError(f"ports still in use: {busy}")
    return {"load1": la1, "load5": la5}


@contextlib.contextmanager
def quiesced(restore=True, settle=4.0):
    """Take the box single-tenant; always restore on exit."""
    units = config.LAUNCHD_UNITS
    try:
        for u in units:
            bootout(u)
        _webui("down")
        time.sleep(settle)
        # wait for the served ports to actually free
        for _ in range(20):
            if port_free(config.MLX_PORT) and port_free(config.EMBED_PORT):
                break
            time.sleep(1)
        yield
    finally:
        if restore:
            for u in units:
                bootstrap(u)
            _webui("up")
            _wait_up(config.base_url(config.MLX_PORT), 120)


def _wait_up(base: str, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with contextlib.suppress(Exception):
            if httpx.get(f"{base}/models", timeout=4).status_code == 200:
                return True
        time.sleep(2)
    return False
