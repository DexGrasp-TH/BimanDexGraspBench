#!/bin/bash

# Process all grasp types for BimanBODex data
# Usage: ./script/process_all_grasp_types_shadow.sh <run_name>

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <run_name>"
    echo "Example: $0 minitest"
    exit 1
fi

RUN_NAME=$1
BODEX_PATH="../BimanBODex/src/curobo/content/assets/output"

echo "Processing all grasp types for run: $RUN_NAME"
echo "================================================"

# Right-two
echo -e "\n[1/5] Processing right_two..."
python src/main.py task=format exp_name=${RUN_NAME}_right_two hand=shadow \
    task.data_name=BimanBODex task.max_num=-1 \
    task.data_path=${BODEX_PATH}/sim_shadow/tabletop_two/${RUN_NAME}/graspdata
python src/main.py task=eval exp_name=${RUN_NAME}_right_two hand=shadow \
    task.debug_viewer=False task.max_num=-1
python src/main.py hand=shadow task=collect exp_name=${RUN_NAME}_right_two

# Right-three
echo -e "\n[2/5] Processing right_three..."
python src/main.py task=format exp_name=${RUN_NAME}_right_three hand=shadow \
    task.data_name=BimanBODex task.max_num=-1 \
    task.data_path=${BODEX_PATH}/sim_shadow/tabletop_three/${RUN_NAME}/graspdata
python src/main.py task=eval exp_name=${RUN_NAME}_right_three hand=shadow \
    task.debug_viewer=False task.max_num=-1
python src/main.py hand=shadow task=collect exp_name=${RUN_NAME}_right_three

# Right-full
echo -e "\n[3/5] Processing right_full..."
python src/main.py task=format exp_name=${RUN_NAME}_right_full hand=shadow \
    task.data_name=BimanBODex task.max_num=-1 \
    task.data_path=${BODEX_PATH}/sim_shadow/tabletop_full/${RUN_NAME}/graspdata
python src/main.py task=eval exp_name=${RUN_NAME}_right_full hand=shadow \
    task.debug_viewer=False task.max_num=-1
python src/main.py hand=shadow task=collect exp_name=${RUN_NAME}_right_full

# Both-three
echo -e "\n[4/5] Processing both_three..."
python src/main.py task=format exp_name=${RUN_NAME}_both_three hand=dual_dummy_arm_shadow \
    task.data_name=BimanBODex task.max_num=-1 \
    task.data_path=${BODEX_PATH}/sim_dual_dummy_arm_shadow/tabletop_three/${RUN_NAME}/graspdata
python src/main.py task=eval exp_name=${RUN_NAME}_both_three hand=dual_dummy_arm_shadow \
    task.debug_viewer=False task.max_num=-1
python src/main.py hand=dual_dummy_arm_shadow task=collect exp_name=${RUN_NAME}_both_three

# Both-full
echo -e "\n[5/5] Processing both_full..."
python src/main.py task=format exp_name=${RUN_NAME}_both_full hand=dual_dummy_arm_shadow \
    task.data_name=BimanBODex task.max_num=-1 \
    task.data_path=${BODEX_PATH}/sim_dual_dummy_arm_shadow/tabletop_full/${RUN_NAME}/graspdata
python src/main.py task=eval exp_name=${RUN_NAME}_both_full hand=dual_dummy_arm_shadow \
    task.debug_viewer=False task.max_num=-1
python src/main.py hand=dual_dummy_arm_shadow task=collect exp_name=${RUN_NAME}_both_full

echo -e "\n================================================"
echo "All grasp types processed successfully!"
