# BimanDexGraspBench

Simulation-based filtering of synthesized dual-arm-hand grasps from [BimanGrasp-Generation](https://github.com/DexGrasp-TH/BimanGrasp-Generation).

Modified from [DexGraspBench](https://github.com/JYChen18/DexGraspBench) to enable bimanual grasp simulation and filtering. It may be useful to overview the original README first.

## Introduction

### Main Usage
- Replay and test **open-loop** grasping poses/trajectories in parallel.

- Each grasping data point only needs to include:
  - Object (must be pre-processed by [MeshProcess](https://github.com/JYChen18/MeshProcess)): `obj_scale`, `obj_pose`, `obj_path`.
  - Hand: `approach_qpos` (optional), `pregrasp_qpos`, `grasp_qpos`, `squeeze_qpos`.


## Getting Started

### Installation
1. Clone this repo and place it alongside the `BimanGrasp-Generation` repository as below.
    ```
    BimanGrasp-Generation/
    |_ ...
    BimanDexGraspBench/
    |_ ...
    ```
1. (Optional) Clone the third-party library [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).
    ```bash
    git submodule update --init --recursive --progress
    ```
1. Install the python environment via [Anaconda](https://www.anaconda.com/). 
    ```bash
    conda create -n DGBench python=3.10 
    conda activate DGBench
    pip install numpy==1.26.4
    conda install pytorch==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia 
    pip install mujoco==3.3.2
    pip install trimesh
    pip install hydra-core
    pip install transforms3d
    pip install matplotlib
    pip install scikit-learn
    pip install usd-core
    pip install imageio
    pip install 'qpsolvers[clarabel]'
    ```
1. You may need to run the following command to avoid potential errors related to MKL such as `undefined symbol: iJIT_NotifyEvent`
    ```bash
    conda install -c conda-forge mkl=2020.2 -y
    ```

### Robot Preparation

The MJCF file of the dual-arm-hand robots should be provided in `assets/hand/`. The configuration file should be provided in `config/hand/`.

### Running

```bash
# convert data format
# e.g.
$ python src/main.py task=format exp_name=<EXP_NAME> hand=dual_ur5_shadow task.data_name=BimanSynthesis task.max_num=-1 task.data_path=../BimanGrasp-Generation/data/experiments/<SYN_EXP_NAME>/arm_filtered

# simulation-based evaluation and filtering
# e.g.
$ python src/main.py task=eval exp_name=<EXP_NAME> hand=dual_ur5_shadow task.debug_viewer=False task.max_num=-1
```

The filtered successful grasp files are located in `output/<EXP_NAME>_dual_ur5_shadow/succgrasp`.


#### BimanBODEx
```bash
# convert data format
# e.g.
$ python src/main.py task=format exp_name=<EXP_NAME> hand=dual_dummy_arm_shadow task.data_name=BimanBODex task.max_num=-1 task.data_path=../../project_any_scale_grasp/BimanBODex/src/curobo/content/assets/output/sim_dual_dummy_arm_shadow/fc/data_0/graspdata

# simulation-based evaluation and filtering
# e.g.
$ python src/main.py task=eval exp_name=<EXP_NAME> hand=dual_dummy_arm_shadow task.debug_viewer=False task.max_num=-1
```