#!/usr/bin/env bash
# Test all 32 binary configs — run each for TIMEOUT seconds, check for errors, kill, move on.
set +e  # don't exit on individual config failures

TIMEOUT=${1:-90}   # seconds per config (default 90)
CONFIG_DIR="examples/config/phase_1b_binary"
LOG_DIR="/tmp/binary_config_tests"
mkdir -p "$LOG_DIR"

PASS=0
FAIL=0
ERRORS=()

echo "============================================"
echo " Testing all binary configs (${TIMEOUT}s each)"
echo "============================================"

for cfg in "$CONFIG_DIR"/phase_1b_binary_*_1d_config.json "$CONFIG_DIR"/optimization/phase_1b_binary_*_config.json; do
    name=$(basename "$cfg" .json)
    log="$LOG_DIR/${name}.log"

    echo ""
    echo ">>> [$name] starting..."
    
    # Run in background, capture PID
    bash predictor.sh --load_config "$cfg" > "$log" 2>&1 &
    PID=$!
    
    # Wait up to TIMEOUT seconds
    sleep "$TIMEOUT"
    
    # Check if process is still running (good = training started)
    if kill -0 "$PID" 2>/dev/null; then
        # Still running = no crash, training started OK
        kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null || true
        
        # Double-check log for errors
        if grep -qi "traceback\|error.*exception\|TypeError\|KeyError\|ValueError\|RuntimeError\|ImportError" "$log" 2>/dev/null; then
            echo ">>> [$name] FAIL (error in log despite running)"
            FAIL=$((FAIL + 1))
            ERRORS+=("$name")
            grep -i "traceback\|error.*exception\|TypeError\|KeyError\|ValueError\|RuntimeError" "$log" | tail -3
        else
            echo ">>> [$name] PASS (trained ${TIMEOUT}s, no errors)"
            PASS=$((PASS + 1))
        fi
    else
        # Process exited — check if it was an error
        EXIT_CODE=$?
        if grep -qi "traceback\|TypeError\|KeyError\|ValueError\|RuntimeError\|ImportError" "$log" 2>/dev/null; then
            echo ">>> [$name] FAIL (crashed)"
            FAIL=$((FAIL + 1))
            ERRORS+=("$name")
            grep -i "TypeError\|KeyError\|ValueError\|RuntimeError\|ImportError" "$log" | tail -3
        else
            echo ">>> [$name] PASS (completed quickly or early-stopped)"
            PASS=$((PASS + 1))
        fi
    fi
done

echo ""
echo "============================================"
echo " RESULTS: ${PASS} passed, ${FAIL} failed / $((PASS + FAIL)) total"
echo "============================================"
if [ ${#ERRORS[@]} -gt 0 ]; then
    echo "FAILED configs:"
    for e in "${ERRORS[@]}"; do
        echo "  - $e"
    done
fi
