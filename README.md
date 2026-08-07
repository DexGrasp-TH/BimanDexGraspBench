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
    pip install mujoco==3.6.0
    pip install mjviser==0.0.14 viser==1.0.27 pillow==12.2.0
    pip install trimesh==4.11.5
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

### Debug Visualization on a Server

Eval visualization is disabled by default. When `task.debug_viewer=True`, the default backend is
[mjviser](https://github.com/mujocolab/mjviser), which serves a read-only browser visualization while the existing
Bench loop remains responsible for every MuJoCo simulation step.

Viewer mode accepts exactly one grasp per run by default. Use `task.start` and `task.end` to select it:

```bash
python src/main.py task=eval hand=<HAND> exp_name=<EXP_NAME> \
    task.debug_viewer=True task.start=<INDEX> task.end=<INDEX+1>
```

By default, mjviser listens only on `127.0.0.1:8080`, waits for a browser connection before starting the motion, and
keeps the final frame visible until the browser disconnects or Ctrl+C is pressed. Forward the port from your local
machine before opening `http://127.0.0.1:8080`:

```bash
ssh -L 8080:127.0.0.1:8080 <server>
```

Use another port when 8080 is occupied:

```bash
python src/main.py task=eval hand=<HAND> exp_name=<EXP_NAME> \
    task.debug_viewer=True task.viewer.port=8081 \
    task.start=<INDEX> task.end=<INDEX+1>
```

Do not bind `task.viewer.host=0.0.0.0` on a shared server unless network access is independently restricted. The Viser
server does not provide project-level authentication.

The native MuJoCo GUI remains available as a fallback:

```bash
python src/main.py task=eval hand=<HAND> exp_name=<EXP_NAME> \
    task.debug_viewer=True task.viewer.backend=mujoco \
    task.start=<INDEX> task.end=<INDEX+1>
```

For automated headless smoke checks, disable the interactive waits explicitly:

```bash
python src/main.py task=eval hand=<HAND> exp_name=<EXP_NAME> \
    task.debug_viewer=True task.viewer.wait_for_client=False task.viewer.hold_on_finish=False \
    task.start=<INDEX> task.end=<INDEX+1>
```

To play several grasps continuously in one browser connection, enable the mjviser playlist and select an index range:

```bash
python src/main.py task=eval hand=<HAND> exp_name=<EXP_NAME> \
    task.debug_viewer=True task.viewer.playlist.enabled=True \
    task.viewer.playlist.interval_seconds=1.0 \
    task.start=<START> task.end=<END>
```

The Viser server stays on the same host and port for the whole playlist. Each grasp still gets an independent MuJoCo
model, data, and Bench evaluator; only the browser scene is replaced. The server waits for a client before the first
grasp, keeps each intermediate final frame for `interval_seconds`, and applies `hold_on_finish` only after the last
grasp. Playlist mode currently supports only `task.viewer.backend=mjviser`; the native MuJoCo GUI remains single-grasp.

`mjviser==0.0.14` requires `mujoco>=3.6.0`. Keep the versions in the installation instructions pinned and record the
MuJoCo version in experiment evidence; installing an unpinned mjviser may silently select a newer physics engine.

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

Before processing, the script asks once whether to delete existing outputs for all selected run names and jobs. The
deletion scope depends on the selected stage: `format` deletes the whole job output folder, `eval` deletes
`evaluation/`, `succgrasp/`, `debug/`, and `succ_collect/`, and `collect` deletes only `succ_collect/`.

Eval-only Hydra overrides are defined near the top of `script/process_all_grasp_types.py` in
`ADDITIONAL_EVAL_HYDRA_ARGS`. The `default` table is used for every hand unless a hand-specific table, such as
`shadow`, `leap`, or `leap_sp`, overrides a grasp type.

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

**Batch Processing**: process AnyScaleDexLearn samples for all five grasp types without modifying the BimanBODex
workflow:

```bash
python script/process_learning_grasp_types.py \
    --hand leap_sp \
    --run-name leapspMulti_robotMultiHierar_debug0_step_050000 \
    --learning-path ../AnyScaleDexLearn/output/leapspMulti_robotMultiHierar_debug0/tests/step_050000/leapspMulti \
    --max-num 100 \
    --dry-run

python script/process_learning_grasp_types.py \
    --hand shadow \
    --run-name shadowMulti_robotMultiHierar_human_0_step_050000 \
    --learning-path ../AnyScaleDexLearn/output/shadowMulti_robotMultiHierar_human_0/tests/step_050000/shadowMulti
```

The script runs only `format` and `eval`. It sends every format job to the same AnyScaleDexLearn sample root with
`task.data_name=Learning`; the grasp type suffix in `exp_name` lets the Learning formatter keep only matching
`pred_grasp_type_id` samples. Single-hand types use `<hand>`, and bimanual types use the corresponding
`dual_dummy_arm_<hand>` config.
