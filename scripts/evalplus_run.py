"""Run evalplus.evaluate with multiprocessing forced to 'fork'. On macOS the
3.12+ default is 'spawn', which re-imports per evaluation and blows evalplus's
per-test timeout (every solution -> 'timeout', pass@1 collapses to 0). fork is
fast and correct here. Usage mirrors `python -m evalplus.evaluate ...`.
"""
import multiprocessing as mp
import os

os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
try:
    mp.set_start_method("fork", force=True)
except RuntimeError:
    pass

from evalplus.evaluate import evaluate  # noqa: E402
import fire  # noqa: E402

if __name__ == "__main__":
    fire.Fire(evaluate)
