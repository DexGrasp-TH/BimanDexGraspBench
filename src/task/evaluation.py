import os
import multiprocessing
import logging
from glob import glob
import traceback

import numpy as np
from tqdm import tqdm

from .convert_format import calculate_object_world_height
from .eval_func import *


def normalize_eval_index_range(input_items, start_index, end_index):
    """Slice evaluation inputs by configured start and end indices.

    Args:
        input_items: Evaluation items after optional random sampling and max_num truncation.
        start_index: Inclusive start index. Negative values are clamped to 0.
        end_index: Exclusive end index. Negative values mean the end of the list.

    Returns:
        Tuple `(sliced_items, normalized_start, normalized_end)` used by evaluation and logs.
    """

    total_num = len(input_items)
    normalized_start = max(int(start_index), 0)
    normalized_end = total_num if int(end_index) < 0 else min(int(end_index), total_num)
    normalized_start = min(normalized_start, total_num)
    normalized_end = max(normalized_start, normalized_end)
    return input_items[normalized_start:normalized_end], normalized_start, normalized_end


def calculate_converted_grasp_object_height(input_npy_path):
    """Calculate posed object height for one converted grasp file.

    Args:
        input_npy_path: Converted grasp `.npy` file containing object metadata.

    Returns:
        World-z object height in meters.
    """

    grasp_data = np.load(input_npy_path, allow_pickle=True).item()
    return calculate_object_world_height(grasp_data["obj_path"], grasp_data["obj_scale"], grasp_data["obj_pose"])


def safe_eval_one(params):
    input_npy_path, configs = params[0], params[1]
    try:
        if configs.hand.mocap:
            eval_func_name = f"{configs.setting}MocapEval"
        else:
            eval_func_name = f"{configs.setting}ArmEval"

        eval(eval_func_name)(input_npy_path, configs).run()
        return
    except Exception:
        error_traceback = traceback.format_exc()
        logging.warning(f"Failed to evaluate {input_npy_path}\n{error_traceback}")
        return


def task_eval(configs):
    assert (
        configs.task.simulation_metrics is not None
        or configs.task.analytic_fc_metrics is not None
        or configs.task.pene_contact_metrics is not None
    ), "You should at least evaluate one kind of metrics"
    input_path_lst = glob(os.path.join(configs.grasp_dir, "**/*.npy"), recursive=True)
    init_num = len(input_path_lst)

    if configs.skip:
        eval_path_lst = glob(os.path.join(configs.eval_dir, "**/*.npy"), recursive=True)
        eval_path_lst = [p.replace(configs.eval_dir, configs.grasp_dir) for p in eval_path_lst]
        input_path_lst = list(set(input_path_lst).difference(set(eval_path_lst)))
    skip_num = init_num - len(input_path_lst)
    input_path_lst = sorted(input_path_lst)
    start_index = int(getattr(configs.task, "start", 0))
    end_index = int(getattr(configs.task, "end", -1))
    if configs.task.max_num > 0:
        input_path_lst = np.random.permutation(input_path_lst)[: configs.task.max_num].tolist()
    sampled_input_num = len(input_path_lst)
    indexed_input_path_lst, normalized_start, normalized_end = normalize_eval_index_range(
        list(enumerate(input_path_lst)),
        start_index,
        end_index,
    )
    input_index_lst = [sampled_index for sampled_index, _ in indexed_input_path_lst]
    input_path_lst = [input_path for _, input_path in indexed_input_path_lst]

    logging.info(
        f"Find {init_num} grasp data in {configs.grasp_dir}, skip {skip_num} already evaluated, "
        f"apply max_num to get {sampled_input_num}, select index range "
        f"[{normalized_start}, {normalized_end}), and use {len(input_path_lst)}."
    )

    if len(input_path_lst) == 0:
        return

    enable_tqdm = bool(getattr(configs.task, "tqdm", True))
    iterable_params = zip(input_path_lst, [configs] * len(input_path_lst))
    progress_desc = "Evaluating grasps"
    if configs.task.debug_viewer or configs.task.debug_render:
        enable_tqdm = enable_tqdm and not bool(configs.task.debug_viewer)
        # Debug rendering runs serially. Viewer mode prints each grasp explicitly because tqdm is disabled.
        iterator = tqdm(iterable_params, total=len(input_path_lst), desc=progress_desc, disable=not enable_tqdm)
        for selected_index, ip in enumerate(iterator):
            if configs.task.debug_viewer:
                try:
                    object_height = calculate_converted_grasp_object_height(ip[0])
                    height_text = f", object_height={object_height:.6f}m"
                except Exception:
                    height_text = ", object_height=unknown"
                print(f"Evaluate grasp index {input_index_lst[selected_index]}: {ip[0]}{height_text}", flush=True)
            safe_eval_one(ip)
    else:
        with multiprocessing.Pool(processes=configs.n_worker) as pool:
            result_iter = pool.imap_unordered(safe_eval_one, iterable_params)
            # Multiprocessing progress advances when worker jobs finish.
            results = list(tqdm(result_iter, total=len(input_path_lst), desc=progress_desc, disable=not enable_tqdm))

    grasp_lst = glob(os.path.join(configs.grasp_dir, "**/*.npy"), recursive=True)
    succ_lst = glob(os.path.join(configs.succ_dir, "**/*.npy"), recursive=True)
    eval_lst = glob(os.path.join(configs.eval_dir, "**/*.npy"), recursive=True)
    logging.info(
        f"Get {len(grasp_lst)} grasp data, {len(eval_lst)} evaluated, and {len(succ_lst)} succeeded in {configs.save_dir}"
    )
    logging.info("Finish evaluation")

    return
