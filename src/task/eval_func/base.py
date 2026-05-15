import os
import sys
from copy import deepcopy
import logging

import numpy as np
import imageio

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from util.rot_util import (
    np_get_delta_qpos,
    np_normalize_vector,
)
from util.hand_util import MjHO
from util.file_util import load_json
from .fc_metric import *


class BaseEval:
    def __init__(self, input_npy_path, configs):
        self.input_npy_path = input_npy_path
        self.configs = configs
        self.grasp_data = np.load(input_npy_path, allow_pickle=True).item()
        self.original_grasp_data = deepcopy(self.grasp_data)
        self.mj_joint_names = None
        self.data2mj_indices = None
        self.nonfinite_qpos_fields = self._find_nonfinite_qpos_fields(self.grasp_data)
        self.mj_ho = None
        if len(self.nonfinite_qpos_fields) > 0:
            if self.configs.task.debug_viewer or self.configs.task.debug_render:
                print(f"Skip non-finite qpos fields: {self.nonfinite_qpos_fields}")
            return

        # Fix object mass by setting density
        obj_info = load_json(os.path.join(self.grasp_data["obj_path"], "info/simplified.json"))
        obj_coef = obj_info["mass"] / (obj_info["density"] * (obj_info["scale"] ** 3))
        new_obj_density = configs.task.obj_mass / (obj_coef * (self.grasp_data["obj_scale"] ** 3))

        # Build mj_spec
        self.mj_ho = MjHO(
            obj_path=self.grasp_data["obj_path"],
            obj_scale=self.grasp_data["obj_scale"],
            has_floor_z0=configs.setting == "tabletop",
            obj_density=new_obj_density,
            hand_xml_path=configs.hand.xml_path,
            hand_mocap=configs.hand.mocap,
            exclude_table_contact=configs.hand.exclude_table_contact,
            friction_coef=getattr(configs.task, "sim_friction_coef", configs.task.miu_coef),
            debug_render=configs.task.debug_render,
            debug_viewer=configs.task.debug_viewer,
        )

        # convert the qpos to mujoco order
        if "joint_names" in self.grasp_data:
            joint_names = self.grasp_data["joint_names"]
            mj_joint_names = self.mj_ho.get_joint_names()
            if self.configs.hand.mocap:
                mj_joint_names = mj_joint_names[1:]  # skip free_joint
            data2mj_indices = [joint_names.index(name) for name in mj_joint_names]
            self.mj_joint_names = mj_joint_names
            self.data2mj_indices = data2mj_indices
            for key in ["pregrasp_qpos", "grasp_qpos", "squeeze_qpos"]:
                qpos = np.asarray(self.grasp_data[key])
                if self.configs.hand.mocap:
                    qpos[7:] = qpos[7:][data2mj_indices]
                else:
                    qpos = qpos[data2mj_indices]
                self.grasp_data[key] = qpos
            if "approach_qpos" in self.grasp_data:
                approach_qpos = np.asarray(self.grasp_data["approach_qpos"])
                if self.configs.hand.mocap:
                    approach_qpos[:, 7:] = approach_qpos[:, 7:][:, data2mj_indices]
                else:
                    approach_qpos = approach_qpos[:, data2mj_indices]
                self.grasp_data["approach_qpos"] = approach_qpos
        else:
            return NotImplementedError("The input npy file should contain joint_names to specify the order of qpos.")

        self._apply_eval_pose_adjustments()

        self.mj_ho.reset_pose_qpos(self.grasp_data["pregrasp_qpos"], self.grasp_data["obj_pose"], set_ctrl=False)
        self.mj_ho._init_after_first_fk()  # for some property that needs to be initialized after first FK
        self.mj_ho._init_viewer_and_render()

        if self.configs.task.debug_viewer or self.configs.task.debug_render:
            with open("debug.xml", "w") as f:
                f.write(self.mj_ho.spec.to_xml())

        return

    def _find_nonfinite_qpos_fields(self, grasp_data):
        """Find qpos fields that contain NaN or Inf values before MuJoCo evaluation.

        Args:
            grasp_data: Loaded grasp dictionary before any MuJoCo scene is built.

        Returns:
            A list of qpos field names containing at least one non-finite value.
        """

        invalid_fields = []
        for key in ["approach_qpos", "pregrasp_qpos", "grasp_qpos", "squeeze_qpos", "lift_qpos"]:
            if key not in grasp_data:
                continue
            qpos = np.asarray(grasp_data[key], dtype=np.float64)
            if not np.isfinite(qpos).all():
                invalid_fields.append(key)
        return invalid_fields

    def _get_eval_pose_adjustment_config(self):
        """Read optional evaluation-time pose adjustment values.

        Args:
            None.

        Returns:
            Tuple `(z_offset, pregrasp_ratio, squeeze_ratio)` where z offset is in meters
            and both ratios use 1.0 as the no-op extrapolation value.
        """

        pose_config = getattr(self.configs.task, "pose_adjustment", None)
        if pose_config is None:
            return 0.0, 1.0, 1.0

        z_offset = float(getattr(pose_config, "grasp_global_z_offset", 0.0))
        pregrasp_ratio = float(getattr(pose_config, "pregrasp_extrapolate_ratio", 1.0))
        squeeze_ratio = float(getattr(pose_config, "squeeze_extrapolate_ratio", 1.0))
        return z_offset, pregrasp_ratio, squeeze_ratio

    def _apply_eval_pose_adjustments(self):
        """Apply configured evaluation-only qpos adjustments.

        Args:
            None.

        Returns:
            None. The method updates `self.grasp_data` in-place after qpos has been
            converted to MuJoCo joint order.
        """

        z_offset, pregrasp_ratio, squeeze_ratio = self._get_eval_pose_adjustment_config()
        if z_offset == 0.0 and pregrasp_ratio == 1.0 and squeeze_ratio == 1.0:
            return

        if self.configs.hand.mocap:
            self._apply_mocap_pose_adjustments(z_offset, pregrasp_ratio, squeeze_ratio)
        elif "dummy_arm" in self.configs.hand_name:
            self._apply_dummy_arm_pose_adjustments(z_offset, pregrasp_ratio, squeeze_ratio)
        else:
            raise NotImplementedError(
                "Evaluation pose adjustment currently supports mocap hands and dummy_arm hand configs."
            )

    def _apply_mocap_pose_adjustments(self, z_offset, pregrasp_ratio, squeeze_ratio):
        """Adjust mocap wrist pose and hand joints for eval-time pose perturbation.

        Args:
            z_offset: World-frame z translation added to the grasp wrist pose in meters.
            pregrasp_ratio: Extrapolation ratio from grasp to pregrasp qpos.
            squeeze_ratio: Extrapolation ratio from grasp to squeeze qpos.

        Returns:
            None. The stage qpos arrays are updated in-place.
        """

        original_grasp_qpos = self.grasp_data["grasp_qpos"].copy()
        adjusted_grasp_qpos = original_grasp_qpos.copy()
        adjusted_grasp_qpos[:3] += np.array([0.0, 0.0, z_offset])

        self.grasp_data["grasp_qpos"] = adjusted_grasp_qpos
        self.grasp_data["pregrasp_qpos"] = self._extrapolate_mocap_qpos_from_grasp(
            self.grasp_data["pregrasp_qpos"],
            original_grasp_qpos,
            adjusted_grasp_qpos,
            pregrasp_ratio,
        )
        self.grasp_data["squeeze_qpos"] = self._extrapolate_mocap_qpos_from_grasp(
            self.grasp_data["squeeze_qpos"],
            original_grasp_qpos,
            adjusted_grasp_qpos,
            squeeze_ratio,
        )
        if "approach_qpos" in self.grasp_data:
            # Approach waypoints are shifted by the same z offset so approach-phase evaluation remains continuous.
            self.grasp_data["approach_qpos"][:, 2] += z_offset

    def _apply_dummy_arm_pose_adjustments(self, z_offset, pregrasp_ratio, squeeze_ratio):
        """Adjust dummy-arm translation and finger joints for eval-time pose perturbation.

        Args:
            z_offset: World-frame z translation added to each grasp-side dummy arm in meters.
            pregrasp_ratio: Linear extrapolation ratio from grasp to pregrasp qpos.
            squeeze_ratio: Linear extrapolation ratio from grasp to squeeze qpos.

        Returns:
            None. Dummy-arm translation and finger joints are updated in-place, while
            dummy-arm rotation joints keep their original pregrasp/squeeze values.
        """

        original_grasp_qpos = self.grasp_data["grasp_qpos"].copy()
        adjusted_grasp_qpos = original_grasp_qpos.copy()
        z_indices = []
        arm_rotation_indices = []
        for prefix in ["ra", "la"]:
            trans_indices = self._get_dummy_arm_translation_indices(prefix)
            if trans_indices is None:
                continue
            adjusted_grasp_qpos[trans_indices[2]] += z_offset
            z_indices.append(trans_indices[2])
            arm_rotation_indices.extend(self._get_dummy_arm_rotation_indices(prefix))

        self.grasp_data["grasp_qpos"] = adjusted_grasp_qpos
        self.grasp_data["pregrasp_qpos"] = self._extrapolate_linear_qpos_from_grasp(
            self.grasp_data["pregrasp_qpos"],
            original_grasp_qpos,
            adjusted_grasp_qpos,
            pregrasp_ratio,
        )
        self.grasp_data["squeeze_qpos"] = self._extrapolate_linear_qpos_from_grasp(
            self.grasp_data["squeeze_qpos"],
            original_grasp_qpos,
            adjusted_grasp_qpos,
            squeeze_ratio,
        )
        # Dummy-arm rotation joints encode wrist orientation, so keep stage orientations unchanged like mocap quaternions.
        self.grasp_data["pregrasp_qpos"][arm_rotation_indices] = np.asarray(
            self.original_grasp_data["pregrasp_qpos"]
        )[self.data2mj_indices][arm_rotation_indices]
        self.grasp_data["squeeze_qpos"][arm_rotation_indices] = np.asarray(
            self.original_grasp_data["squeeze_qpos"]
        )[self.data2mj_indices][arm_rotation_indices]
        if "approach_qpos" in self.grasp_data:
            # Apply only the global z offset to approach waypoints; extrapolation is defined for pre/squeeze stages.
            for z_idx in z_indices:
                self.grasp_data["approach_qpos"][:, z_idx] += z_offset

    def _get_dummy_arm_translation_indices(self, prefix):
        """Find xyz translation joint indices for one dummy arm.

        Args:
            prefix: Dummy-arm side prefix, usually `ra` for right arm or `la` for left arm.

        Returns:
            A list of three indices `[x_idx, y_idx, z_idx]`, or None when that side is absent.
        """

        trans_joint_names = [f"{prefix}_TransXJ", f"{prefix}_TransYJ", f"{prefix}_TransZJ"]
        if not all(name in self.mj_joint_names for name in trans_joint_names):
            return None
        return [self.mj_joint_names.index(name) for name in trans_joint_names]

    def _get_dummy_arm_rotation_indices(self, prefix):
        """Find xyz rotation joint indices for one dummy arm.

        Args:
            prefix: Dummy-arm side prefix, usually `ra` for right arm or `la` for left arm.

        Returns:
            A list of existing rotation indices `[rx_idx, ry_idx, rz_idx]`.
        """

        rot_joint_names = [f"{prefix}_RotXJ", f"{prefix}_RotYJ", f"{prefix}_RotZJ"]
        return [self.mj_joint_names.index(name) for name in rot_joint_names if name in self.mj_joint_names]

    def _extrapolate_linear_qpos_from_grasp(self, stage_qpos, original_grasp_qpos, adjusted_grasp_qpos, ratio):
        """Linearly extrapolate one stage qpos from the grasp qpos.

        Args:
            stage_qpos: Original stage qpos vector.
            original_grasp_qpos: Original grasp qpos vector before z offset.
            adjusted_grasp_qpos: Grasp qpos vector after z offset.
            ratio: Extrapolation ratio; 1.0 preserves the original offset from grasp.

        Returns:
            Adjusted stage qpos vector.
        """

        return adjusted_grasp_qpos + ratio * (stage_qpos - original_grasp_qpos)

    def _extrapolate_mocap_qpos_from_grasp(self, stage_qpos, original_grasp_qpos, adjusted_grasp_qpos, ratio):
        """Extrapolate mocap qpos while preserving the stage wrist orientation.

        Args:
            stage_qpos: Original mocap stage qpos with `[pos, quat, joints...]`.
            original_grasp_qpos: Original grasp qpos before z offset.
            adjusted_grasp_qpos: Grasp qpos after z offset.
            ratio: Extrapolation ratio; 1.0 preserves the original offset from grasp.

        Returns:
            Adjusted mocap qpos with linearly extrapolated translation/joints and
            original stage wrist quaternion.
        """

        adjusted_stage_qpos = self._extrapolate_linear_qpos_from_grasp(
            stage_qpos,
            original_grasp_qpos,
            adjusted_grasp_qpos,
            ratio,
        )
        # Keep the original pregrasp/squeeze orientation; only translation and finger joints are extrapolated.
        adjusted_stage_qpos[3:7] = stage_qpos[3:7]
        return adjusted_stage_qpos

    def _copy_adjusted_qpos_to_eval_results(self, eval_results):
        """Copy adjusted MuJoCo-order qpos back to the saved input joint order.

        Args:
            eval_results: Evaluation result dictionary copied from the original input data.

        Returns:
            None. The qpos fields in `eval_results` are updated in-place so saved eval
            files match the actually evaluated wrist translations.
        """

        for key in ["pregrasp_qpos", "grasp_qpos", "squeeze_qpos"]:
            eval_results[key] = self._qpos_from_mj_to_data_order(self.grasp_data[key])
        if "approach_qpos" in self.grasp_data:
            eval_results["approach_qpos"] = np.stack(
                [self._qpos_from_mj_to_data_order(qpos) for qpos in self.grasp_data["approach_qpos"]], axis=0
            )

    def _qpos_from_mj_to_data_order(self, mj_qpos):
        """Convert one adjusted qpos from MuJoCo order back to the input data order.

        Args:
            mj_qpos: One qpos array after MuJoCo-order conversion and pose adjustment.

        Returns:
            A qpos array aligned with `self.original_grasp_data["joint_names"]`.
        """

        mj_qpos = np.asarray(mj_qpos)
        data_qpos = np.asarray(self.original_grasp_data["grasp_qpos"]).copy()
        if self.configs.hand.mocap:
            data_qpos[:7] = mj_qpos[:7]
            for mj_idx, data_idx in enumerate(self.data2mj_indices):
                data_qpos[7 + data_idx] = mj_qpos[7 + mj_idx]
        else:
            for mj_idx, data_idx in enumerate(self.data2mj_indices):
                data_qpos[data_idx] = mj_qpos[mj_idx]
        return data_qpos

    def _simulate_under_extforce_details(self, pre_obj_qpos):
        raise NotImplementedError

    def _eval_pene_and_contact(self):
        eval_config = self.configs.task.pene_contact_metrics

        ho_contact, hh_contact = self.mj_ho.get_contact_info(
            self.grasp_data["grasp_qpos"],
            self.grasp_data["obj_pose"],
            obj_margin=eval_config.contact_margin,
        )

        contact_link_set = set()
        contact_dist_dict = {name: eval_config.contact_margin for name in self.configs.hand.finger_prefix}
        for c in ho_contact:
            hand_body_name = c["body1_name"]
            # Update the distance between the finger and the object
            for finger_prefix in contact_dist_dict:
                if hand_body_name.startswith(finger_prefix):
                    contact_dist_dict[finger_prefix] = min(contact_dist_dict[finger_prefix], c["contact_dist"])
                    break
            # Update the name set of hand bodies in contact with the object
            if (
                np.abs(c["contact_dist"]) < eval_config.contact_threshold
                and hand_body_name in self.configs.hand.valid_body_name
            ):
                contact_link_set.add(hand_body_name)
        contact_dist_lst = list(contact_dist_dict.values())
        contact_distance = np.mean([max(i, 0.0) for i in contact_dist_lst])
        contact_consistency = np.max(contact_dist_lst) - np.min(contact_dist_lst)
        contact_number = len(contact_link_set)

        ho_pene = -min([c["contact_dist"] for c in ho_contact]) if len(ho_contact) > 0 else 0
        ho_pene = max(ho_pene, 0)
        self_pene = -min([c["contact_dist"] for c in hh_contact]) if len(hh_contact) > 0 else 0
        self_pene = max(self_pene, 0)

        return ho_pene, self_pene, contact_number, contact_distance, contact_consistency

    def _eval_simulate_under_extforce(self):
        eval_config = self.configs.task.simulation_metrics

        # Reset to init hand qpos and check contact
        # init_qpos = self.grasp_data["pregrasp_qpos"] if self.configs.hand.mocap else self.grasp_data["approach_qpos"][0]
        init_qpos = (
            self.grasp_data["approach_qpos"]
            if self.configs.task.simulation_metrics.approach_phase
            else self.grasp_data["pregrasp_qpos"]
        )

        ho_contact, hh_contact = self.mj_ho.get_contact_info(init_qpos, self.grasp_data["obj_pose"])

        # Filter out bad initialization with severe penetration
        ho_dist = min([c["contact_dist"] for c in ho_contact]) if len(ho_contact) > 0 else 0
        hh_dist = min([c["contact_dist"] for c in hh_contact]) if len(hh_contact) > 0 else 0
        if ho_dist < -eval_config.max_pene or hh_dist < -eval_config.max_pene:
            if self.configs.task.debug_viewer or self.configs.task.debug_render:
                print(f"Severe penetration larger than {eval_config.max_pene}")
            return False, 100, 100

        # Record initial object pose
        pre_obj_qpos = deepcopy(self.mj_ho.get_obj_pose())

        if self.configs.setting == "tabletop":
            lift_height = 0.1
            pre_obj_qpos[2] += lift_height
            lift_qpos = deepcopy(self.grasp_data["squeeze_qpos"])
            if self.configs.hand.mocap:
                lift_qpos[2] += lift_height
            else:
                if "dummy_arm" in self.configs.hand_name:
                    joint_names = self.mj_ho.get_joint_names()
                    if "ra_TransZJ" in joint_names:
                        lift_qpos[joint_names.index("ra_TransZJ")] += lift_height
                    if "la_TransZJ" in joint_names:
                        lift_qpos[joint_names.index("la_TransZJ")] += lift_height
                else:
                    raise NotImplementedError("Please specify how to lift the hand for non-mocap arm setting.")
            self.grasp_data["lift_qpos"] = lift_qpos

        # Detailed simulation methods for testing
        self._simulate_under_extforce_details(pre_obj_qpos)

        # Compare the resulted object pose
        latter_obj_qpos = self.mj_ho.get_obj_pose()
        delta_pos, delta_angle = np_get_delta_qpos(pre_obj_qpos, latter_obj_qpos)
        succ_flag = (delta_pos < eval_config.trans_thre) & (delta_angle < eval_config.angle_thre)

        if self.configs.task.debug_viewer or self.configs.task.debug_render:
            print(succ_flag, delta_pos, delta_angle)
            if self.configs.task.debug_render:
                debug_path = self.input_npy_path.replace(self.configs.grasp_dir, self.configs.task.debug_dir).replace(
                    ".npy", ".gif"
                )
                os.makedirs(os.path.dirname(debug_path), exist_ok=True)
                imageio.mimsave(debug_path, self.mj_ho.debug_images)
                print("Save GIF to ", debug_path)

        return succ_flag, delta_pos, delta_angle

    def _eval_analytic_fc_metric(self):
        eval_config = self.configs.task.analytic_fc_metrics

        ho_contact, _ = self.mj_ho.get_contact_info(
            self.grasp_data["grasp_qpos"],
            self.grasp_data["obj_pose"],
            obj_margin=eval_config.contact_threshold,
        )

        contact_point_dict = {}
        contact_normal_dict = {}
        for c in ho_contact:
            hand_body_name = c["body1_name"]
            # Check whether the hand contact body name is needed
            if (
                (hand_body_name not in self.configs.hand.valid_body_name)
                or (eval_config.contact_tip_only and hand_body_name not in self.configs.hand.tip_body_name)
                or (np.abs(c["contact_dist"]) > eval_config.contact_threshold)
            ):
                continue

            # Record valid contact point and normal
            if hand_body_name not in contact_point_dict:
                contact_point_dict[hand_body_name] = []
                contact_normal_dict[hand_body_name] = []
            contact_point_dict[hand_body_name].append(c["contact_pos"])
            contact_normal_dict[hand_body_name].append(c["contact_normal"])

        # If no contact, directly set a bad value as metric
        fc_metric_results = {}
        if len(contact_point_dict) == 0:
            # logging.warning("No contact when calculate fc metric!")
            for metric_name in eval_config.type:
                fc_metric_results[f"{metric_name}_metric"] = 2
            return fc_metric_results
        else:
            # Average all contacts on the same hand body
            contact_points = np.stack([np.mean(np.array(v), axis=0) for v in contact_point_dict.values()])
            contact_normals = np.stack(
                [np_normalize_vector(np.mean(np.array(v), axis=0)) for v in contact_normal_dict.values()]
            )
            # Use a smaller friction to leave some room to adjust
            miu_coef = 0.5 * np.array(self.configs.task.miu_coef)

            # Calculate analytic force closure metrics
            for metric_name in eval_config.type:
                fc_metric_results[f"{metric_name}_metric"] = eval(f"calcu_{metric_name}_metric")(
                    contact_points, contact_normals, miu_coef
                )

        return fc_metric_results

    def _determine_grasp_type(self):
        """Extract grasp_type from exp_name (format: {run_name}_{grasp_type})"""
        exp_name = self.configs.exp_name
        grasp_types = ["right_two", "right_three", "right_full", "both_three", "both_full"]
        matched_type = None
        for grasp_type in grasp_types:
            if exp_name.endswith(f"_{grasp_type}"):
                if matched_type is not None:
                    raise ValueError(f"Multiple grasp types matched in exp_name: {exp_name}")
                matched_type = grasp_type

        if matched_type is None:
            raise ValueError(f"No valid grasp_type found in exp_name: {exp_name}")

        return matched_type

    def _extract_robot_poses(self, eval_results):
        """Extract wrist poses and joint angles from qpos data.

        Computes wrist poses in world frame via forward kinematics and separates
        them from joint angles. Returns dictionary with keys:
        - {prefix}_global_pose: 7-dim (mocap) or 14-dim (dual arm) wrist pose(s)
        - {prefix}_joint_pos: joint angles without wrist pose components

        Args:
            eval_results: Dictionary containing qpos data and joint_names

        Returns:
            Dictionary with extracted global poses and joint positions
        """

        if not hasattr(self.configs.hand, "wrist_body_names"):
            raise ValueError("wrist_body_names not defined in hand config")

        result = {"wrist_body_names": list(self.configs.hand.wrist_body_names)}
        joint_names = list(eval_results.get("joint_names", []))

        for qpos_key in ["pregrasp_qpos", "grasp_qpos", "squeeze_qpos"]:
            # Set the hand and object pose for FK
            self.mj_ho.reset_pose_qpos(self.grasp_data[qpos_key], self.grasp_data["obj_pose"])

            # Compute wrist poses
            global_poses = []
            for wrist_name in self.configs.hand.wrist_body_names:
                body_id = self.mj_ho.model.body(f"child-{wrist_name}").id
                pose = self.mj_ho.data.xpos[body_id].copy()
                quat = self.mj_ho.data.xquat[body_id].copy()
                global_poses.append(np.concatenate([pose, quat]))
            # Extract joint positions
            qpos = list(eval_results[qpos_key].copy())

            if self.configs.hand.mocap:
                # For mocap: remove first 7 dims (wrist pose)
                joint_pos = qpos[7:]
            elif "dummy_arm" in self.configs.hand_name:
                # For dummy_arm: remove dummy arm joints
                indices_to_remove = set()
                for i, _ in enumerate(self.configs.hand.wrist_body_names):
                    prefix = "ra" if i == 0 else "la"
                    arm_joints = [
                        f"{prefix}_TransXJ",
                        f"{prefix}_TransYJ",
                        f"{prefix}_TransZJ",
                        f"{prefix}_RotXJ",
                        f"{prefix}_RotYJ",
                        f"{prefix}_RotZJ",
                    ]

                    for joint in arm_joints:
                        if joint in joint_names:
                            indices_to_remove.add(joint_names.index(joint))

                joint_pos = [value for idx, value in enumerate(qpos) if idx not in indices_to_remove]
            else:
                raise NotImplementedError("Please specify how to extract joint positions for non-dummy arm setting.")

            # Save results
            prefix = qpos_key.replace("_qpos", "")
            result[f"{prefix}_global_pose"] = np.concatenate(global_poses)
            result[f"{prefix}_joint_pos"] = np.array(joint_pos)

        return result

    def run(self):

        eval_results = deepcopy(self.original_grasp_data)

        if len(self.nonfinite_qpos_fields) > 0:
            eval_results["succ_flag"] = False
            eval_results["delta_pos"] = np.nan
            eval_results["delta_angle"] = np.nan
            eval_results["eval_failure_reason"] = "nonfinite_qpos"
            eval_results["nonfinite_qpos_fields"] = list(self.nonfinite_qpos_fields)
            eval_results["grasp_type"] = self._determine_grasp_type()
            self._save_eval_results(eval_results)
            return

        if self.configs.task.pene_contact_metrics is not None:
            (
                eval_results["ho_pene"],
                eval_results["self_pene"],
                eval_results["contact_num"],
                eval_results["contact_dist"],
                eval_results["contact_consis"],
            ) = self._eval_pene_and_contact()

        if self.configs.task.analytic_fc_metrics is not None:
            fc_metric_results = self._eval_analytic_fc_metric()
            for k, v in fc_metric_results.items():
                eval_results[k] = v

        if self.configs.task.simulation_metrics is not None:
            (
                eval_results["succ_flag"],
                eval_results["delta_pos"],
                eval_results["delta_angle"],
            ) = self._eval_simulate_under_extforce()

        self.mj_ho.close_view_and_render()

        # Determine grasp_type before saving
        eval_results["grasp_type"] = self._determine_grasp_type()
        # Keep saved qpos consistent with the poses that were actually evaluated.
        self._copy_adjusted_qpos_to_eval_results(eval_results)
        # Extract and combine robot poses
        robot_poses = self._extract_robot_poses(eval_results)
        eval_results.update(robot_poses)

        self._save_eval_results(eval_results)

        # Save success data symlink after eval_results is saved
        if self.configs.task.simulation_metrics is not None and eval_results["succ_flag"]:
            succ_npy_path = self.input_npy_path.replace(self.configs.grasp_dir, self.configs.succ_dir)
            if not os.path.exists(succ_npy_path):
                eval_npy_path = self.input_npy_path.replace(self.configs.grasp_dir, self.configs.eval_dir)
                os.makedirs(os.path.dirname(succ_npy_path), exist_ok=True)
                os.system(f"ln -s {os.path.relpath(eval_npy_path, os.path.dirname(succ_npy_path))} {succ_npy_path}")

        return

    def _save_eval_results(self, eval_results):
        """Save one evaluation result dictionary to the configured eval directory.

        Args:
            eval_results: Evaluation result dictionary to persist.

        Returns:
            None. The method writes the `.npy` file and optionally logs the save path.
        """

        eval_npy_path = self.input_npy_path.replace(self.configs.grasp_dir, self.configs.eval_dir)
        os.makedirs(os.path.dirname(eval_npy_path), exist_ok=True)
        np.save(eval_npy_path, eval_results)
        if not bool(getattr(self.configs.task, "tqdm", True)):
            logging.info(f"Save eval file to {eval_npy_path}.")
        return
