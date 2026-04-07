#!/bin/bash

# Process all grasp types for BimanBODex data
# Usage: ./script/process_all_grasp_types_bodex.sh --hand <hand> --run_name <run_name> [--bodex_path <path>]

set -e

BODEX_PATH="../BimanBODex/src/curobo/content/assets/output"
HAND_NAME=""
RUN_NAME=""

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
            echo "Usage: $0 --hand <hand> --run_name <run_name> [--bodex_path <path>]"
            echo "Example: $0 --hand shadow --run_name minitest"
            echo "Example: $0 --hand leap --run_name minitest --bodex_path ../BimanBODex/src/curobo/content/assets/output"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --hand <hand> --run_name <run_name> [--bodex_path <path>]"
            exit 1
            ;;
    esac
done

if [ -z "$HAND_NAME" ] || [ -z "$RUN_NAME" ]; then
    echo "Usage: $0 --hand <hand> --run_name <run_name> [--bodex_path <path>]"
    echo "Example: $0 --hand shadow --run_name minitest"
    echo "Example: $0 --hand leap --run_name minitest"
    exit 1
fi

case "$HAND_NAME" in
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
    *)
        echo "Unsupported hand: $HAND_NAME"
        echo "Supported hands: shadow, leap"
        exit 1
        ;;
esac

echo "Processing all grasp types for hand: $HAND_NAME, run: $RUN_NAME"
echo "BimanBODex path: $BODEX_PATH"
echo "================================================"

# Right-two
echo -e "\n[1/5] Processing right_two..."
python src/main.py task=format exp_name=${RUN_NAME}_right_two hand=${SINGLE_HAND} \
    task.data_name=BimanBODex task.max_num=-1 \
    task.data_path=${BODEX_PATH}/${SINGLE_DATASET}/tabletop_two/${RUN_NAME}/graspdata
python src/main.py task=eval exp_name=${RUN_NAME}_right_two hand=${SINGLE_HAND} \
    task.debug_viewer=False task.max_num=-1
python src/main.py hand=${SINGLE_HAND} task=collect exp_name=${RUN_NAME}_right_two

# Right-three
echo -e "\n[2/5] Processing right_three..."
python src/main.py task=format exp_name=${RUN_NAME}_right_three hand=${SINGLE_HAND} \
    task.data_name=BimanBODex task.max_num=-1 \
    task.data_path=${BODEX_PATH}/${SINGLE_DATASET}/tabletop_three/${RUN_NAME}/graspdata
python src/main.py task=eval exp_name=${RUN_NAME}_right_three hand=${SINGLE_HAND} \
    task.debug_viewer=False task.max_num=-1
python src/main.py hand=${SINGLE_HAND} task=collect exp_name=${RUN_NAME}_right_three

# Right-full
echo -e "\n[3/5] Processing right_full..."
python src/main.py task=format exp_name=${RUN_NAME}_right_full hand=${SINGLE_HAND} \
    task.data_name=BimanBODex task.max_num=-1 \
    task.data_path=${BODEX_PATH}/${SINGLE_DATASET}/tabletop_full/${RUN_NAME}/graspdata
python src/main.py task=eval exp_name=${RUN_NAME}_right_full hand=${SINGLE_HAND} \
    task.debug_viewer=False task.max_num=-1
python src/main.py hand=${SINGLE_HAND} task=collect exp_name=${RUN_NAME}_right_full

# Both-three
echo -e "\n[4/5] Processing both_three..."
python src/main.py task=format exp_name=${RUN_NAME}_both_three hand=${DUAL_HAND} \
    task.data_name=BimanBODex task.max_num=-1 \
    task.data_path=${BODEX_PATH}/${DUAL_DATASET}/tabletop_three/${RUN_NAME}/graspdata
python src/main.py task=eval exp_name=${RUN_NAME}_both_three hand=${DUAL_HAND} \
    task.debug_viewer=False task.max_num=-1
python src/main.py hand=${DUAL_HAND} task=collect exp_name=${RUN_NAME}_both_three

# Both-full
echo -e "\n[5/5] Processing both_full..."
python src/main.py task=format exp_name=${RUN_NAME}_both_full hand=${DUAL_HAND} \
    task.data_name=BimanBODex task.max_num=-1 \
    task.data_path=${BODEX_PATH}/${DUAL_DATASET}/tabletop_full/${RUN_NAME}/graspdata
python src/main.py task=eval exp_name=${RUN_NAME}_both_full hand=${DUAL_HAND} \
    task.debug_viewer=False task.max_num=-1
python src/main.py hand=${DUAL_HAND} task=collect exp_name=${RUN_NAME}_both_full

echo -e "\n================================================"
echo "All grasp types processed successfully!"
