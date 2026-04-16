#!/usr/bin/env python3
# Copyright 2025 TIER IV, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Autoware ROS 2 node for Alpamayo end-to-end trajectory planning.

Subscribes to camera images and odometry, runs Alpamayo VLM + Expert diffusion,
and publishes Autoware trajectories.

Optimized: PyTorch BF16 VLM + TRT FP16 Expert + greedy decode + 5-step diffusion.
Interface matches alpamayo_ros Python node from alpamayo-autoware.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
import math
import os
from pathlib import Path
import sys
import time
from typing import Dict
from typing import Optional

from autoware_planning_msgs.msg import Trajectory
from autoware_planning_msgs.msg import TrajectoryPoint
from builtin_interfaces.msg import Duration
import cv2
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import ColorRGBA
from std_msgs.msg import String
import torch
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

# Add alpamayo-autoware/src to path for model imports
_ALPAMAYO_SRC = os.environ.get(
    "ALPAMAYO_SRC_DIR",
    str(Path.home() / "work" / "research" / "alpamayo" / "alpamayo-autoware" / "src"),
)
if _ALPAMAYO_SRC not in sys.path:
    sys.path.insert(0, _ALPAMAYO_SRC)

os.environ.setdefault("HF_HOME", "/media/binwang/COMLOPS/.cache/huggingface")

from alpamayo_r1 import helper  # noqa: E402
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1  # noqa: E402
from alpamayo_r1.trt.expert_runtime import TrtExpertEngine  # noqa: E402


class AlpamayoTrtNode(Node):
    """Autoware ROS 2 node for Alpamayo E2E trajectory planning with TRT optimization."""

    def __init__(self) -> None:
        super().__init__("alpamayo_trt_node")

        # ── Parameters ──
        self.declare_parameter("model_id", "nvidia/Alpamayo-R1-10B")
        self.declare_parameter("expert_onnx_path", "")
        self.declare_parameter("num_diffusion_steps", 10)  # 10=best quality, 5=fast (~40ms faster)
        self.declare_parameter("use_greedy_decode", False)  # True=fastest, False=sampling (default)
        self.declare_parameter("top_p", 0.98)
        self.declare_parameter("temperature", 0.6)
        self.declare_parameter("trajectory_topic", "/alpamayo/predicted_trajectory")
        self.declare_parameter("cot_topic", "/alpamayo/reasoning")
        self.declare_parameter("odometry_topic", "/localization/kinematic_state")
        self.declare_parameter("inference_period_sec", 1.0)
        self.declare_parameter("camera_topics", [""])
        self.declare_parameter("frame_id", "base_link")
        self.declare_parameter("num_frames", 4)
        self.declare_parameter("num_history_steps", 16)

        self._frame_id = self.get_parameter("frame_id").value
        self._num_frames = self.get_parameter("num_frames").as_int()
        self._num_history_steps = self.get_parameter("num_history_steps").as_int()

        KINEMATIC_STATE_HZ = 50.0
        ALPAMAYO_INPUT_HZ = 1.0
        self._skip_num = int(KINEMATIC_STATE_HZ / ALPAMAYO_INPUT_HZ)

        self._device = torch.device("cuda")
        self._dtype = torch.bfloat16

        # ── Publishers ──
        traj_topic = self.get_parameter("trajectory_topic").value
        self._trajectory_pub = self.create_publisher(Trajectory, traj_topic, 10)
        cot_topic = self.get_parameter("cot_topic").value
        self._cot_pub = self.create_publisher(String, cot_topic, 10)
        self._marker_pub = self.create_publisher(MarkerArray, traj_topic + "_markers", 10)

        # ── Camera subscriptions ──
        camera_topics = list(
            self.get_parameter("camera_topics").get_parameter_value().string_array_value
        )
        self._camera_topics = [t for t in camera_topics if t]
        self._camera_buffers: Dict[str, deque] = {
            topic: deque(maxlen=self._num_frames * 3) for topic in self._camera_topics
        }
        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10
        )
        for topic in self._camera_topics:
            self.create_subscription(
                CompressedImage,
                topic,
                lambda msg, t=topic: self._image_callback(t, msg),
                camera_qos,
            )
            self.get_logger().info(f"Subscribed to camera: {topic}")

        # ── Odometry subscription ──
        odom_topic = self.get_parameter("odometry_topic").value
        self._odometry_buffer: deque[Odometry] = deque(
            maxlen=self._num_history_steps * self._skip_num + 10
        )
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=50
        )
        self.create_subscription(Odometry, odom_topic, self._odometry_callback, odom_qos)

        # ── Inference executor ──
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._active_future: Optional[Future] = None

        # ── Load model ──
        model_id = self.get_parameter("model_id").value
        self.get_logger().info(f"Loading Alpamayo model: {model_id}")
        self._model = AlpamayoR1.from_pretrained(model_id, dtype=self._dtype).to(self._device)
        self._model.eval()

        # TRT Expert
        expert_onnx = self.get_parameter("expert_onnx_path").value
        if expert_onnx and Path(expert_onnx).exists():
            engine = TrtExpertEngine(
                onnx_model_path=expert_onnx,
                engine_cache_dir=str(Path(expert_onnx).parent / "engine_cache"),
                enable_int8=False,
                enable_fp16=True,
            )
            self._model.set_expert_step_runner(engine)
            self.get_logger().info(f"TRT Expert loaded: {expert_onnx}")

        # Diffusion steps
        num_steps = self.get_parameter("num_diffusion_steps").as_int()
        self._model.diffusion.num_inference_steps = num_steps

        self._processor = helper.get_processor(self._model.tokenizer)

        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        # ── Timer ──
        period = self.get_parameter("inference_period_sec").as_double()
        self._timer = self.create_timer(period, self._timer_callback)
        self.get_logger().info(
            f"Alpamayo TRT node ready ({num_steps}-step diffusion, period={period}s)"
        )

    def destroy_node(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        super().destroy_node()

    # ── Callbacks (identical to alpamayo_ros) ──

    def _image_callback(self, topic: str, msg: CompressedImage) -> None:
        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            return
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        self._camera_buffers[topic].append((msg.header.stamp, tensor))

    def _odometry_callback(self, msg: Odometry) -> None:
        self._odometry_buffer.append(msg)

    def _timer_callback(self) -> None:
        if self._active_future and not self._active_future.done():
            return
        payload = self._prepare_payload()
        if payload is None:
            return
        self._active_future = self._executor.submit(self._run_inference, payload)
        self._active_future.add_done_callback(self._on_future_done)

    # ── Data preparation (identical to alpamayo_ros) ──

    def _prepare_payload(self) -> Optional[dict]:
        if not all(len(buf) >= self._num_frames for buf in self._camera_buffers.values()):
            return None

        camera_tensors = []
        for topic in self._camera_topics:
            frames = list(self._camera_buffers[topic])[-self._num_frames :]
            camera_tensors.append(torch.stack([f for _, f in frames], dim=0))
        image_frames = torch.stack(camera_tensors, dim=0)

        if len(self._odometry_buffer) < self._num_history_steps * self._skip_num:
            return None

        odom_history = list(self._odometry_buffer)[
            -self._num_history_steps * self._skip_num :: self._skip_num
        ]

        positions, rotations = [], []
        for msg in odom_history:
            pose = msg.pose.pose
            positions.append([pose.position.x, pose.position.y, pose.position.z])
            quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
            rotations.append(Rotation.from_quat(quat).as_matrix())

        positions_np = np.asarray(positions, dtype=np.float32)
        rotations_np = np.asarray(rotations, dtype=np.float32)
        t0_rot_inv = np.linalg.inv(rotations_np[-1])
        history_xyz = (positions_np - positions_np[-1]) @ t0_rot_inv.T
        history_rot = np.einsum("ij,njk->nik", t0_rot_inv, rotations_np)

        return {
            "image_frames": image_frames,
            "ego_history_xyz": torch.from_numpy(history_xyz).unsqueeze(0).unsqueeze(0),
            "ego_history_rot": torch.from_numpy(history_rot).unsqueeze(0).unsqueeze(0),
        }

    # ── Inference (optimized: greedy + TRT Expert) ──

    def _run_inference(self, payload: dict) -> dict:
        start = time.time()
        frames = payload["image_frames"]
        messages = helper.create_message(frames.flatten(0, 1))
        processor_inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_inputs = {
            "tokenized_data": processor_inputs,
            "ego_history_xyz": payload["ego_history_xyz"],
            "ego_history_rot": payload["ego_history_rot"],
        }
        model_inputs = helper.to_device(model_inputs, device=self._device)

        use_greedy = self.get_parameter("use_greedy_decode").value
        top_p = 1.0 if use_greedy else self.get_parameter("top_p").as_double()
        temperature = 1.0 if use_greedy else self.get_parameter("temperature").as_double()

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=self._dtype):
            pred_xyz, pred_rot, extra = self._model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=top_p,
                temperature=temperature,
                num_traj_samples=1,
                num_traj_sets=1,
                max_generation_length=64,
                return_extra=True,
            )

        trajectory = pred_xyz[0, 0, 0].detach().cpu()
        rotation = pred_rot[0, 0, 0].detach().cpu()

        # Publish trajectory
        traj_msg = self._to_trajectory(trajectory, rotation)
        self._trajectory_pub.publish(traj_msg)

        # Publish markers
        self._marker_pub.publish(self._to_markers(trajectory))

        # Publish CoT
        cot_text = self._extract_cot(extra)
        if cot_text:
            cot_msg = String()
            cot_msg.data = cot_text
            self._cot_pub.publish(cot_msg)

        return {"duration_sec": time.time() - start, "num_poses": len(traj_msg.points)}

    # ── Output conversion (identical to alpamayo_ros) ──

    def _to_trajectory(self, trajectory: torch.Tensor, rotations: torch.Tensor) -> Trajectory:
        traj_np = trajectory.numpy()
        rot_np = rotations.numpy()
        now = self.get_clock().now().to_msg()
        dt = 0.1

        traj_msg = Trajectory()
        traj_msg.header.stamp = now
        traj_msg.header.frame_id = self._frame_id

        prev_xy = None
        for idx, point in enumerate(traj_np):
            pt = TrajectoryPoint()
            pt.pose.position.x = float(point[0])
            pt.pose.position.y = float(point[1])
            pt.pose.position.z = float(point[2])

            quat = Rotation.from_matrix(rot_np[idx]).as_quat()
            pt.pose.orientation.x = float(quat[0])
            pt.pose.orientation.y = float(quat[1])
            pt.pose.orientation.z = float(quat[2])
            pt.pose.orientation.w = float(quat[3])

            if prev_xy is not None:
                dist = math.hypot(float(point[0] - prev_xy[0]), float(point[1] - prev_xy[1]))
                pt.longitudinal_velocity_mps = float(dist / dt)

            seconds_float = idx * dt
            pt.time_from_start = Duration(
                sec=int(seconds_float), nanosec=int((seconds_float - int(seconds_float)) * 1e9)
            )
            traj_msg.points.append(pt)
            prev_xy = (point[0], point[1])

        return traj_msg

    def _to_markers(self, trajectory: torch.Tensor) -> MarkerArray:
        traj_np = trajectory.numpy()
        now = self.get_clock().now().to_msg()

        marker = Marker()
        marker.header.stamp = now
        marker.header.frame_id = self._frame_id
        marker.ns = "alpamayo_trajectory"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 1.0
        marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
        marker.pose.orientation.w = 1.0

        for point in traj_np:
            p = Point()
            p.x, p.y, p.z = float(point[0]), float(point[1]), float(point[2])
            marker.points.append(p)

        return MarkerArray(markers=[marker])

    def _extract_cot(self, extra: dict) -> Optional[str]:
        if not extra or "cot" not in extra:
            return None
        try:
            text = str(extra["cot"][0, 0, 0]).strip()
        except Exception:
            return None
        return text or None

    def _on_future_done(self, future: Future) -> None:
        try:
            metrics = future.result()
        except Exception as exc:
            self.get_logger().error(f"Inference failed: {exc}")
            return
        if metrics:
            self.get_logger().info(
                f"Inference: {metrics['duration_sec']:.2f}s ({metrics['num_poses']} poses)"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AlpamayoTrtNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
