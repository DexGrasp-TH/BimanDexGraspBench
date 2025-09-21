exp=dexonomy_learning
python src/main.py hand=shadow exp_name=${exp} task=format task.max_num=-1 task.data_name=Learning task.data_path=../DexLearn/output/dexonomy_shadow_nflow_cond/tests/step_050000
python src/main.py hand=shadow task=eval exp_name=${exp} task.max_num=-1

