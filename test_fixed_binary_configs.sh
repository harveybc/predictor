#!/usr/bin/env bash
# Test all 32 FIXED binary configs — run each to completion (10 epochs).
set +e  # don't exit on individual config failures

CONFIG_DIR="examples/config/phase_1b_binary"
LOG_DIR="/tmp/binary_fixed_tests"
mkdir -p "$LOG_DIR"

PASS=0
FAIL=0
ERRORS=()

echo "============================================"
echo " Testing all 32 fixed binary configs"
echo "============================================"

for cfg in "$CONFIG_DIR"/phase_1b_binary_*_1d_config.json; do
    name=$(basename "$cfg" .json)
    log="$LOG_DIR/${name}.log"

    echo ""
    echo ">>> [$name] starting..."
    
    # Run to completion
    bash predictor.sh --load_config "$cfg" > "$log" 2>&1
    EXIT_CODE=$?

    # Check for errors
    if grep -qi "traceback\|TypeError\|KeyError\|ValueError\|RuntimeError\|ImportError\|CUDA_ERROR_OUT_OF_MEMORY" "$log" 2>/dev/null; then
        echo ">>> [$name] FAIL"
        FAIL=$((FAIL + 1))
        ERRORS+=("$name")
        grep -i "traceback\|TypeError\|KeyError\|ValueError\|RuntimeError\|CUDA_ERROR" "$log" | tail -3
    elif [[ $EXIT_CODE -ne 0 ]]; then
        echo ">>> [$name] FAIL (exit code $EXIT_CODE)"
        FAIL=$((FAIL + 1))
        ERRORS+=("$name")
    else
        echo ">>> [$name] PASS"
        PASS=$((PASS + 1))
    fi
done

echo ""
echo "============================================"
echo " RESULTS: $PASS passed, $FAIL failed / $((PASS + FAIL)) total"
echo "============================================"
if [[ ${#ERRORS[@]} -gt 0 ]]; then
    echo "FAILED configs:"
    for e in "${ERRORS[@]}"; do
        echo "  - $e"
    done
fi
