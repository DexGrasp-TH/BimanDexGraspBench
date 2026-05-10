#!/bin/bash

# Process all tabletop grasp types exported by BimanBODex.

set -e

BODEX_PATH="../BimanBODex/src/curobo/content/assets/output"
HAND_NAME=""
RUN_NAME=""

usage() {
    echo "Usage: $0 --hand <shadow|leap|leap_sp> --run_name <run_name> [--bodex_path <path>]"
    echo "Example: $0 --hand shadow --run_name minitest"
    echo "Example: $0 --hand leap_sp --run_name minitest --bodex_path ../BimanBODex/src/curobo/content/assets/output"
}

# Map the requested hand family to BimanDexGraspBench hand configs and BimanBODex output folders.
# Args:
#   $1: Hand family requested by --hand.
# Return:
#   Sets SINGLE_HAND, DUAL_HAND, SINGLE_DATASET, and DUAL_DATASET; exits on unsupported input.
configure_hand() {
    case "$1" in
        shadow)
            SINGLE_HAND="shadow"
            DUAL_HAND="dual_dummy_arm_shadow"
            SINGLE_DATASET="sim_shadow"
            DUAL_DATASET="sim_dual_dummy_arm_shadow"
            ;;
        leap)
            SINGLE_HAND="leap"
            DUAL_HAND="dual_dummy_arm_leap"
            SINGLE_DATASET="sim_leap"
            DUAL_DATASET="sim_dual_dummy_arm_leap"
            ;;
        leap_sp)
            SINGLE_HAND="leap_sp"
            DUAL_HAND="dual_dummy_arm_leap_sp"
            SINGLE_DATASET="sim_leap_sp"
            DUAL_DATASET="sim_dual_dummy_arm_leap_sp"
            ;;
        *)
            echo "Unsupported hand: $1"
            usage
            exit 1
            ;;
    esac
}

# Run format, eval, and collect for one grasp type.
# Args:
#   $1: 1-based task index used in progress text.
#   $2: Total number of tasks.
#   $3: Output suffix appended to RUN_NAME for this grasp type.
#   $4: BimanDexGraspBench hand config name.
#   $5: BimanBODex dataset folder name.
#   $6: BimanBODex tabletop split folder name.
# Return:
#   Runs the three processing commands; exits immediately if any command fails.
process_grasp_type() {
    local index="$1"
    local total="$2"
    local suffix="$3"
    local hand="$4"
    local dataset="$5"
    local tabletop_split="$6"
    local exp_name="${RUN_NAME}_${suffix}"
    local data_path="${BODEX_PATH}/${dataset}/${tabletop_split}/${RUN_NAME}/graspdata"

    echo -e "\n[${index}/${total}] Processing ${suffix}..."
    python src/main.py task=format exp_name="${exp_name}" hand="${hand}" \
        task.data_name=BimanBODex task.max_num=-1 task.data_path="${data_path}"
    python src/main.py task=eval exp_name="${exp_name}" hand="${hand}" \
        task.debug_viewer=False task.max_num=-1
    python src/main.py task=collect exp_name="${exp_name}" hand="${hand}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --hand)
            HAND_NAME="$2"
            shift 2
            ;;
        --run_name)
            RUN_NAME="$2"
            shift 2
            ;;
        --bodex_path)
            BODEX_PATH="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

if [ -z "$HAND_NAME" ] || [ -z "$RUN_NAME" ]; then
    usage
    exit 1
fi

configure_hand "$HAND_NAME"

echo "Processing all grasp types for hand: $HAND_NAME, run: $RUN_NAME"
echo "BimanBODex path: $BODEX_PATH"
echo "================================================"

TOTAL_TASKS=5
process_grasp_type 1 "$TOTAL_TASKS" "right_two" "$SINGLE_HAND" "$SINGLE_DATASET" "tabletop_two"
process_grasp_type 2 "$TOTAL_TASKS" "right_three" "$SINGLE_HAND" "$SINGLE_DATASET" "tabletop_three"
process_grasp_type 3 "$TOTAL_TASKS" "right_full" "$SINGLE_HAND" "$SINGLE_DATASET" "tabletop_full"
process_grasp_type 4 "$TOTAL_TASKS" "both_three" "$DUAL_HAND" "$DUAL_DATASET" "tabletop_three"
process_grasp_type 5 "$TOTAL_TASKS" "both_full" "$DUAL_HAND" "$DUAL_DATASET" "tabletop_full"

echo -e "\n================================================"
echo "All grasp types processed successfully!"
