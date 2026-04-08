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
    pip install 'qpsolvers[clarabel]'
    pip install -e ./third_party/pytorch_kinematics
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
$ python src/main.py task=eval exp_name=<EXP_NAME> hand=<HAND> task.debug_viewer=False task.max_num=-1
```

Examples:

```bash
# Right-full
$ python src/main.py task=format exp_name=minitest_right_full hand=shadow task.data_name=BimanBODex task.max_num=-1 task.data_path=../BimanBODex/src/curobo/content/assets/output/sim_shadow/tabletop_full/minitest/graspdata
$ python src/main.py task=eval exp_name=minitest_right_full hand=shadow task.debug_viewer=False task.max_num=-1

# Both-full
$ python src/main.py task=format exp_name=minitest_both_full hand=dual_dummy_arm_shadow task.data_name=BimanBODex task.max_num=-1 task.data_path=../BimanBODex/src/curobo/content/assets/output/sim_dual_dummy_arm_shadow/tabletop_full/minitest/graspdata
$ python src/main.py task=eval exp_name=minitest_both_full hand=dual_dummy_arm_shadow task.debug_viewer=False task.max_num=-1
```

**Batch Processing**: process all grasp types (right_two, right_three, right_full, both_three, both_full) at once:

```bash
$ ./script/process_all_grasp_types.sh --hand <HAND> --run_name <RUN_NAME> [--bodex_path <BODEX_OUTPUT_PATH>]
# e.g.: $ ./script/process_all_grasp_types.sh --hand shadow --run_name minitest
# e.g.: $ ./script/process_all_grasp_types.sh --hand leap --run_name minitest
```

**Concatenate Dataset**: combine successful grasps from all grasp types into a unified dataset saved in `${AnyScaleGraspDataset}/<DATASET_NAME>` for NN training:

```bash
$ python script/concatenate_dataset.py --hand_name <HAND_NAME> --run_name <RUN_NAME> --dataset_name <DATASET_NAME>
# e.g.: $ python script/concatenate_dataset.py --run_name minitest --dataset_name BimanBODex
```
