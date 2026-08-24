#!/bin/bash
set -euo pipefail
exec python /workspace/scripts/run_drift_check.py "$@"
