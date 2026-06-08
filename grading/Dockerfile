# evalplus grading sandbox — native arm64 Linux (OrbStack), where evalplus's
# setrlimit/signal-based execution sandbox works correctly. macOS can't run it
# (every solution -> timeout). Generation happens on the host; only grading runs
# here. Build: docker build -t llmbench-evalplus grading/
FROM python:3.12-slim
RUN pip install --no-cache-dir "evalplus>=0.3.1"
# pre-fetch the HumanEval+ / MBPP+ datasets into the image
RUN python -c "from evalplus.data import get_human_eval_plus, get_mbpp_plus; get_human_eval_plus(); get_mbpp_plus()"
WORKDIR /data
