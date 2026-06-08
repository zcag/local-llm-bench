"""Make evalplus's sandbox memory guard tolerant of macOS, where setrlimit(
RLIMIT_AS/RLIMIT_DATA) is unsupported and otherwise makes EVERY solution error
out (pass@1 collapses to 0). Upstream already skips RLIMIT_STACK on Darwin but
not these two — same root cause. Idempotent; run after creating the venv.

This only relaxes the *memory cap* on macOS; the rest of reliability_guard
(disabling os.system, builtins, etc.) is untouched, and the OS still isolates
the subprocess. On Linux nothing changes.
"""
import os
import sys

TARGET_REL = "lib/python3.14/site-packages/evalplus/eval/utils.py"

ORIG = """        resource.setrlimit(
            resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes)
        )
        resource.setrlimit(
            resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes)
        )"""

PATCHED = """        for _rl in (resource.RLIMIT_AS, resource.RLIMIT_DATA):
            try:  # macOS rejects these; skip rather than fail every solution
                resource.setrlimit(_rl, (maximum_memory_bytes, maximum_memory_bytes))
            except (ValueError, OSError):
                pass"""


def find_target() -> str:
    venv = os.environ.get("VIRTUAL_ENV") or os.path.join(os.path.dirname(__file__), "..", ".venv")
    # tolerate any python3.x dir
    import glob
    hits = glob.glob(os.path.join(venv, "lib", "python3.*", "site-packages", "evalplus", "eval", "utils.py"))
    return hits[0] if hits else os.path.join(venv, TARGET_REL)


def main():
    path = find_target()
    if not os.path.exists(path):
        print(f"!! evalplus utils.py not found at {path}")
        sys.exit(1)
    src = open(path).read()
    if "skip rather than fail every solution" in src:
        print("already patched")
        return
    if ORIG not in src:
        print("!! expected block not found — evalplus version changed; inspect manually")
        sys.exit(2)
    open(path, "w").write(src.replace(ORIG, PATCHED))
    print(f"patched {path}")


if __name__ == "__main__":
    main()
