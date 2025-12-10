#!/usr/bin/env bash
# File: run_experiments.sh
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT_DIR/results/logs"
METRICS_DIR="$ROOT_DIR/results/metrics"

mkdir -p "$LOG_DIR" "$METRICS_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/hybrid_layer_test_$TIMESTAMP.log"
JUNIT_FILE="$METRICS_DIR/hybrid_layer_test_$TIMESTAMP.xml"

echo "Running hybrid layer tests (tests/test_unit_hybrid_model.py)"
pytest -q tests/test_unit_hybrid_model.py --junitxml="$JUNIT_FILE" 2>&1 | tee "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}

echo "Log: $LOG_FILE"
echo "JUnit XML: $JUNIT_FILE"

exit $EXIT_CODE