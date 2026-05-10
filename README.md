# BimanDexGraspBench

Simulation-based filtering of synthesized dual-arm-hand grasps from [BimanGrasp-Generation](https://github.com/DexGrasp-TH/BimanGrasp-Generation).

Modified from [DexGraspBench](https://github.com/JYChen18/DexGraspBench) to enable bimanual grasp simulation and filtering. It may be useful to overview the original README first.

## Getting Started

### Installation
1. Clone this repo and place it alongside the `BimanGrasp-Generation` repository as below.
    ```
    BimanBODex/
    |_ ...
    BimanDexGraspBench/
    |_ ...
    ```
1. Clone the third-party library:
    ```bash
    git submodule update --init --recursive --progress
    ```
1. Install the python environment via [Anaconda](https://www.anaconda.com/). 
    ```bash
    conda create -n BiDGBench python=3.10 
    conda activate BiDGBench
    pip install numpy==1.26.4
    conda install pytorch==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia 
    pip install mkl==2024.0.0
    pip install mujoco==3.3.2
    pip install trimesh
    pip install hydra-core
    pip install transforms3d
    pip install matplotlib
    pip install scikit-learn
    pip install usd-core
    pip install imageio
    pip install tqdm
    pip install 'qpsolvers[clarabel]'
    pip install -e ./third_party/pytorch_kinematics
    pip install -e ./third_party/utils_python
    ```
1. Export the dataset path:
    ```bash
    # on local
    export AnyScaleGraspDataset=/data/dataset/AnyScaleGrasp
    # on server
    export AnyScaleGraspDataset=/data/mingrui/dataset/AnyScaleGrasp
    ```
1. Create the object symbolic link in `./assets`:
    ```bash
    ln -s ${AnyScaleGraspDataset}/object ./assets/object
    ```

## USage

### Robot Preparation

The MJCF file of the dual-arm-hand robots should be provided in `assets/hand/`. The configuration file should be provided in `config/hand/`.

Before running, check the consistency between URDF and MJCF:
```bash
$ python script/check_urdf_mjcf.py --urdf <path_to_urdf> --mjcf <path_to_mjcf>
```

### Filtering of BimanBODex

```bash
# convert data format
$ python src/main.py task=format exp_name=<EXP_NAME> hand=<HAND> task.data_name=BimanBODex task.max_num=-1 task.data_path=../BimanBODex/src/curobo/content/assets/output/<PATH>/graspdata
```

```bash
# simulation-based evaluation and filtering
$ python src/main.py task=eval exp_name=<EXP_NAME> hand=<HAND> task.debug_viewer=False task.max_num=-1 n_worker=6
```

Examples:

```bash
# Right-full
$ python src/main.py task=format exp_name=minitest_right_full hand=shadow task.data_name=BimanBODex task.max_num=-1 task.data_path=../BimanBODex/src/curobo/content/assets/output/sim_shadow/tabletop_full/minitest/graspdata
$ python src/main.py task=eval exp_name=minitest_right_full hand=shadow task.debug_viewer=False task.max_num=-1 n_worker=6

# Both-full
$ python src/main.py task=format exp_name=minitest_both_full hand=dual_dummy_arm_shadow task.data_name=BimanBODex task.max_num=-1 task.data_path=../BimanBODex/src/curobo/content/assets/output/sim_dual_dummy_arm_shadow/tabletop_full/minitest/graspdata
$ python src/main.py task=eval exp_name=minitest_both_full hand=dual_dummy_arm_shadow task.debug_viewer=False task.max_num=-1 n_worker=6
```

**Batch Processing**: process all BimanBODex tabletop grasp types (`right_two`, `right_three`, `right_full`,
`both_three`, `both_full`) with `script/process_all_grasp_types.py`.

```bash
# Run format, eval, and collect for all five grasp types.
$ python script/process_all_grasp_types.py --hand <shadow|leap|leap_sp> --run_name <RUN_NAME> --max_num 100

# Use a custom BimanBODex output root.
$ python script/process_all_grasp_types.py --hand shadow --run_name minitest \
    --bodex_path ../BimanBODex/src/curobo/content/assets/output --max_num 100
```

The script maps the selected hand family to the correct single-hand and dual-dummy-arm Bench configs and BimanBODex
output folders. It creates experiment names in the form `<RUN_NAME>_<GRASP_TYPE>`.

Useful options:

```bash
# Inspect generated commands without running them.
$ python script/process_all_grasp_types.py --hand shadow --run_name minitest --dry-run

# Run only one grasp type. Repeat --grasp-type to select multiple.
$ python script/process_all_grasp_types.py --hand shadow --run_name minitest --grasp-type right_two

# Run only one stage. Repeat --stage to select multiple.
$ python script/process_all_grasp_types.py --hand shadow --run_name minitest --stage eval
$ python script/process_all_grasp_types.py --hand shadow --run_name minitest --stage eval --stage collect
```

Before processing, the script asks whether to delete existing outputs for the selected jobs. The deletion scope depends on
the selected stage: `format` deletes the whole job output folder, `eval` deletes `evaluation/`, `succgrasp/`, `debug/`,
and `succ_collect/`, and `collect` deletes only `succ_collect/`.

Eval-only per-grasp-type Hydra overrides are defined near the top of `script/process_all_grasp_types.py` in
`ADDITIONAL_EVAL_HYDRA_ARGS`.

**Concatenate Dataset**: combine successful grasps from all grasp types into a unified dataset saved in `${AnyScaleGraspDataset}/<DATASET_NAME>` for NN training:

```bash
$ python script/concatenate_dataset.py --hand_name <HAND_NAME> --run_name <RUN_NAME> --dataset_name <DATASET_NAME>
# e.g.: $ python script/concatenate_dataset.py --hand_name leap --run_name minitest --dataset_name BimanBODex
```


### Evaluation of AnyScaleDexLearn

Convert data format.
```bash
python src/main.py hand=<HAND> task=format exp_name=<EXP_NAME>_<GRASP_TYPE> task.max_num=-1 task.data_name=Learning task.data_path=<PATH>

# E.g., python src/main.py task=format hand=leap exp_name=learn_right_full task.max_num=100 task.data_name=Learning task.data_path=../AnyScaleDexLearn/output/leapMulti_robotMultiHierar_dataset_full_1/tests/step_050000/leapMulti

# E.g., python src/main.py task=format hand=dual_dummy_arm_leap exp_name=learn_both_full task.max_num=100 task.data_name=Learning task.data_path=../AnyScaleDexLearn/output/leapMulti_robotMultiHierar_dataset_full_1/tests/step_050000/leapMulti
```

Evaluation.
```bash
python src/main.py task=eval hand=<HAND> exp_name=<EXP_NAME>_<GRASP_TYPE> task.max_num=-1 task.debug_viewer=False n_worker=96 

# E.g., python src/main.py task=eval hand=dual_dummy_arm_leap exp_name=learn_both_full task.max_num=-1 task.debug_viewer=False 
```

