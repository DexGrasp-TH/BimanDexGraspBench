import os
import sys
import types
import unittest
from unittest import mock


sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from task import evaluation
from task.eval_func.base import BaseEval


def make_configs(input_count=1, backend="mjviser"):
    task = types.SimpleNamespace(
        simulation_metrics=types.SimpleNamespace(),
        analytic_fc_metrics=None,
        pene_contact_metrics=None,
        debug_viewer=True,
        debug_render=False,
        viewer={
            "backend": backend,
            "host": "127.0.0.1",
            "port": 8080,
            "wait_for_client": False,
            "hold_on_finish": False,
        },
        obj_scale=None,
        start=0,
        end=-1,
        max_num=-1,
        tqdm=False,
    )
    return types.SimpleNamespace(
        task=task,
        skip=False,
        grasp_dir="/input/graspdata",
        eval_dir="/output/evaluation",
        succ_dir="/output/succgrasp",
        save_dir="/output",
        input_count=input_count,
    )


class EvaluationViewerContractTest(unittest.TestCase):
    def test_invalid_backend_fails_before_reading_inputs(self):
        configs = make_configs(backend="unknown")

        with mock.patch.object(evaluation, "glob") as glob:
            with self.assertRaisesRegex(ValueError, "Unsupported debug viewer backend"):
                evaluation.task_eval(configs)
        glob.assert_not_called()

    def test_debug_viewer_rejects_multiple_grasps(self):
        configs = make_configs(input_count=2)

        with mock.patch.object(evaluation, "glob", return_value=["a.npy", "b.npy"]):
            with self.assertRaisesRegex(ValueError, "exactly one grasp"):
                evaluation.task_eval(configs)

    def test_debug_viewer_runs_one_grasp_serially(self):
        configs = make_configs(input_count=1)
        glob_results = [
            ["a.npy"],
            ["a.npy"],
            [],
            ["a.npy"],
        ]

        with mock.patch.object(evaluation, "glob", side_effect=glob_results), mock.patch.object(
            evaluation,
            "safe_eval_one",
            return_value={
                "tiny_convex_skipped_case": False,
                "tiny_convex_skipped_mesh_count": 0,
                "tiny_convex_skipped_meshes": [],
            },
        ) as safe_eval_one, mock.patch.object(evaluation.multiprocessing, "Pool") as pool:
            evaluation.task_eval(configs)

        safe_eval_one.assert_called_once()
        pool.assert_not_called()


class BaseEvalViewerCleanupTest(unittest.TestCase):
    def test_viewer_is_closed_when_evaluation_raises(self):
        runner = object.__new__(BaseEval)
        runner.original_grasp_data = {"obj_path": "/object"}
        runner.grasp_data = {"obj_path": "/object"}
        runner.nonfinite_qpos_fields = []
        runner.configs = types.SimpleNamespace(
            task=types.SimpleNamespace(
                pene_contact_metrics=None,
                analytic_fc_metrics=None,
                simulation_metrics=types.SimpleNamespace(),
            )
        )
        runner.mj_ho = types.SimpleNamespace(
            _init_viewer_and_render=mock.Mock(),
            wait_for_viewer_client=mock.Mock(),
            close_view_and_render=mock.Mock(),
        )
        runner._copy_object_physics_to_eval_results = mock.Mock()
        runner._eval_simulate_under_extforce = mock.Mock(side_effect=RuntimeError("simulation failed"))

        with self.assertRaisesRegex(RuntimeError, "simulation failed"):
            runner.run()

        runner.mj_ho._init_viewer_and_render.assert_called_once_with()
        runner.mj_ho.wait_for_viewer_client.assert_called_once_with()
        runner.mj_ho.close_view_and_render.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
