"""The L1 model zoo — MLX repos, all verified to fit under the 56 GB wired ceiling.
Excluded (don't fit, documented): gpt-oss-120b (62 GB) and GLM-4.5-Air (60 GB+).
`model_tag` is the short label used in results/charts.
"""

ZOO = [
    # tag,                     hf repo,                                                  size_gb, note
    ("coder-next-mxfp4",      "mlx-community/Qwen3-Coder-Next-mxfp4",                    42.4, "flagship 80B-A3B MoE"),
    ("30b-a3b-4bit",          "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",        17.2, "quant sweep"),
    ("30b-a3b-6bit",          "mlx-community/Qwen3-Coder-30B-A3B-Instruct-6bit",        24.8, "quant sweep"),
    ("30b-a3b-8bit",          "mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit",        32.4, "quant sweep"),
    ("30b-a3b-4bit-dwq",      "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit-DWQ",     17.2, "DWQ vs vanilla 4bit"),
    ("gpt-oss-20b",           "mlx-community/gpt-oss-20b-MXFP4-Q8",                      12.1, "small/fast challenger"),
    ("devstral-2507-8bit",    "mlx-community/Devstral-Small-2507-8bit",                  19.2, "agentic 24B dense"),
    ("qwen2.5-coder-32b-8bit","mlx-community/Qwen2.5-Coder-32B-Instruct-8bit",          34.8, "dense baseline"),
]

EXCLUDED = [
    ("gpt-oss-120b", "mlx-community/gpt-oss-120b-MXFP4-Q4", 62.3, "exceeds 56 GB wired ceiling"),
    ("glm-4.5-air",  "mlx-community/GLM-4.5-Air-4bit",      60.1, "exceeds 56 GB wired ceiling"),
]

if __name__ == "__main__":
    for tag, repo, sz, note in ZOO:
        print(f"{tag:24} {sz:5.1f}G  {repo}   # {note}")
    print("\nexcluded:")
    for tag, repo, sz, note in EXCLUDED:
        print(f"{tag:24} {sz:5.1f}G  {repo}   # {note}")
