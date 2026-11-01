#!/usr/bin/env bash
# All 25 alignment conditions from Kobayashi et al. (2014), Clin. Biomech. 29, 47-56:
#   Angle       2, 4, 6 deg   (flexion/extension -> socket_ori Z, abduction/adduction -> socket_ori X)
#   Translation 5, 10, 15 mm  (anterior/posterior -> socket_pos X, lateral/medial -> socket_pos Z)
#   + nominal alignment (all offsets 0)
# Both signs are run for every condition; which one is called "anterior" vs. "posterior" etc.
# depends on the convention in Fig. 2 of the paper.
#
# Modes:
#   eval_esr.py commented out -> Only run main_eval.py on existing CSV files
#   eval_esr.py uncommented   -> Run simulation first, then main_eval.py
#
# Usage:
#   screen -S align
#   bash run_paper_alignments.sh
#   (Ctrl+A, D to detach; screen -r align to reattach)

CK="../outputs/2026-10-27/07-14-55/PPOJax_saved.pkl"
N_STEPS=1000
MASS=86.6

DATE="2026-10-27"
TIME="07-14-55"
BASE="plots/ESR/linear/$DATE"

run() {
    local name="$1"
    local folder="$2"
    shift 2

    local raw="$BASE/${TIME}_${folder}/raw_joint_data.csv"

    echo
    echo "=========================================="
    echo "=== $name ($(date +%H:%M:%S)) ==="
    echo "=========================================="

    # Run simulation (uncomment if new simulation data is needed)
    # python eval_esr.py --path "$CK" --n_steps "$N_STEPS" --body_mass_kg "$MASS" "$@"

    # Check for existing raw data
    if [[ -f "$raw" ]]; then
        echo "Using raw data: $raw"

        python main_eval.py --path "$raw"

        if [ $? -eq 0 ]; then
            echo "SUCCESS: $name"
        else
            echo "ERROR: main_eval.py failed for $name"
        fi
    else
        echo "WARNING: Missing raw_joint_data.csv for $name"
        echo "Expected: $raw"
    fi
}

# Uncomment if nominal alignment should also be evaluated
# run "nominal" "pos0_0_ori0_0_0"

# Translation: 5, 10, 15 mm
for p in 0.005 0.01 0.015; do
    for s in "" "-"; do
        v="${s}${p}"

        run "pos_x_${v}" "pos${v}_0_ori0_0_0" \
            --socket_pos "${v}" 0

        run "pos_z_${v}" "pos0_${v}_ori0_0_0" \
            --socket_pos 0 "${v}"
    done
done

# Rotation: 2, 4, 6 degrees (in radians)
for a in 0.0349 0.0698 0.1047; do
    for s in "" "-"; do
        v="${s}${a}"

        run "ori_x_${v}" "pos0_0_ori${v}_0_0" \
            --socket_ori "${v}" 0 0

        run "ori_z_${v}" "pos0_0_ori0_0_${v}" \
            --socket_ori 0 0 "${v}"
    done
done

echo
echo "=========================================="
echo "All alignment evaluations finished."
echo "=========================================="