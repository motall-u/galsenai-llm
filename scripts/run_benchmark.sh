#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/benchmark/sample_benchmark.yaml}"

uv run galsenai benchmark run --config "$CONFIG_PATH"
