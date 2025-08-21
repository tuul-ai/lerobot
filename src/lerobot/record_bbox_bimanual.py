# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Records a bimanual dataset with red/white arm affordance detection.

Example:

```shell
python -m lerobot.record_bbox_bimanual \
    --robot.type=bimanual_follower \
    --robot.left_arm_port=/dev/tty.usbmodem58760431541 \
    --robot.right_arm_port=/dev/tty.usbmodem58760431542 \
    --robot.cameras="{
    top: {type: "opencv", "index_or_path": 3, "width": 640, "height": 480, "fps": 30}, 
    front: {type: "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30},
    left: {type: "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
    right: {type: "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}
    }" \
    --dataset.repo_id=username/bimanual-lego-dataset \
    --dataset.num_episodes=10 \
    --dataset.single_task="Bimanual LEGO manipulation" \
    --teleop.type=bimanual_leader \
    --teleop.left_arm_port=/dev/tty.usbmodem58760431551 \
    --teleop.right_arm_port=/dev/tty.usbmodem58760431552
```
"""

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

import numpy as np
import rerun as rr
import torch
import json

from lerobot.cameras import (  # noqa: F401
    CameraConfig,  # noqa: F401
)
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.policies.factory import make_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so100_follower,
    make_robot_from_config,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    bi_so100_leader,
    TeleoperatorConfig,
    make_teleoperator_from_config,
)
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    predict_action,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
    log_say,
)
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.gemini_perception_bimanual import (
    get_2D_bbox_bimanual,
    reconcile_multiview_affordances,
    create_bimanual_action_plan,
    plot_bimanual_bbox,
    parse_json,
)

# Bimanual robot configurations are imported through general robot imports above


@dataclass
class DatasetRecordConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/bimanual-lego`).
    repo_id: str
    # A short but accurate description of the bimanual task performed during recording (e.g. "Red and white arms collaboratively disassemble LEGO structures.")
    single_task: str
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | Path | None = None
    # Limit the frames per second.
    fps: int = 30
    # Number of seconds for data recording for each episode.
    episode_time_s: int | float = 60
    # Number of seconds for resetting the environment after each episode.
    reset_time_s: int | float = 60
    # Number of episodes to record.
    num_episodes: int = 50
    # Encode frames in the dataset into video
    video: bool = True
    # Upload dataset to Hugging Face hub.
    push_to_hub: bool = True
    # Upload on private repository on the Hugging Face hub.
    private: bool = False
    # Add tags to your dataset on the hub.
    tags: list[str] | None = None
    # Number of subprocesses handling the saving of frames as PNG.
    num_image_writer_processes: int = 0
    # Number of threads writing the frames as png images on disk, per camera.
    num_image_writer_threads_per_camera: int = 4
    # Number of episodes to record before batch encoding videos
    # Set to 1 for immediate encoding (default behavior), or higher for batched encoding
    video_encoding_batch_size: int = 1

    def __post_init__(self):
        if self.single_task is None:
            raise ValueError("You need to provide a task as argument in `single_task`.")


@dataclass
class RecordConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
    # Whether to control the robot with a teleoperator
    teleop: TeleoperatorConfig | None = None
    # Whether to control the robot with a policy
    policy: PreTrainedConfig | None = None
    # Display all cameras on screen
    display_data: bool = False
    # Use vocal synthesis to read events.
    play_sounds: bool = True
    # Resume recording on an existing dataset.
    resume: bool = False

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        if self.teleop is None and self.policy is None:
            raise ValueError("Choose a policy, a teleoperator or both to control the robot")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


def get_bimanual_affordances_multi_view(top_image, front_image, task_type=None):
    """
    Get bimanual affordances from multiple camera views and reconcile them.
    
    Args:
        top_image: Top view camera image
        front_image: Front view camera image  
        task_type: Optional task type hint
        
    Returns:
        dict: Reconciled bimanual action plan with red/white arm assignments
    """
    try:
        # Get affordances from both views
        top_result = get_2D_bbox_bimanual(top_image, task_type=task_type)
        front_result = get_2D_bbox_bimanual(front_image, task_type=task_type)
        
        # Reconcile multi-view results
        reconciled_affordances = reconcile_multiview_affordances(top_result, front_result)
        
        # Convert to structured action plan
        action_plan = create_bimanual_action_plan(reconciled_affordances)
        
        return action_plan, reconciled_affordances
        
    except Exception as e:
        logging.warning(f"Affordance detection failed: {e}. Using default parallel sort configuration.")
        # Return default configuration for parallel sorting
        default_action_plan = {
            'task_type': 'parallel_sort',
            'coordination_required': False,
            'red_arm': {'targets': [], 'actions': [], 'priority_order': []},
            'white_arm': {'targets': [], 'actions': [], 'priority_order': []},
            'coordination': {'sync_points': [], 'handoff_zones': [], 'sequence_constraints': []}
        }
        return default_action_plan, {}


def extract_arm_targets(action_plan, view_type="top"):
    """
    Extract red and white arm targets from action plan for specific view.
    
    Args:
        action_plan: Structured action plan from bimanual affordance detection
        view_type: "top" or "front" for camera view
        
    Returns:
        tuple: (red_targets, white_targets, handoff_zones, task_info)
    """
    if not action_plan:
        # Return default empty targets
        return [], [], [], {"task_type": "parallel_sort", "coordination_required": False}
    
    red_targets = []
    white_targets = []
    handoff_zones = []
    
    # Extract red arm targets
    for target in action_plan.get('red_arm', {}).get('targets', []):
        red_targets.append({
            'label': target.get('label', ''),
            'bbox': target.get('bbox', [0, 0, 0, 0]),
            'action': target.get('action', 'pick'),
            'priority': target.get('priority', 2)
        })
    
    # Extract white arm targets  
    for target in action_plan.get('white_arm', {}).get('targets', []):
        white_targets.append({
            'label': target.get('label', ''),
            'bbox': target.get('bbox', [0, 0, 0, 0]),
            'action': target.get('action', 'pick'),
            'priority': target.get('priority', 2)
        })
    
    # Extract handoff zones
    for zone in action_plan.get('coordination', {}).get('handoff_zones', []):
        handoff_zones.append({
            'label': zone.get('label', 'handoff_zone'),
            'bbox': zone.get('bbox', [0, 0, 0, 0]),
            'action': 'handoff'
        })
    
    task_info = {
        'task_type': action_plan.get('task_type', 'parallel_sort'),
        'coordination_required': action_plan.get('coordination_required', False)
    }
    
    return red_targets, white_targets, handoff_zones, task_info


def normalize_bbox_coordinates(bbox_list):
    """
    Normalize bounding box coordinates to [0, 1] range.
    
    Args:
        bbox_list: List of [y1, x1, y2, x2] coordinates in 0-1000 range
        
    Returns:
        list: Normalized coordinates in [0, 1] range
    """
    if not bbox_list or len(bbox_list) != 4:
        return [0.0, 0.0, 0.0, 0.0]
    
    return [coord / 1000.0 for coord in bbox_list]


def get_primary_target_bbox(targets):
    """
    Get the primary (highest priority) target bbox from a list of targets.
    
    Args:
        targets: List of target dictionaries with bbox and priority
        
    Returns:
        list: Normalized bbox coordinates [y1, x1, y2, x2]
    """
    if not targets:
        return [0.0, 0.0, 0.0, 0.0]
    
    # Sort by priority (1 = highest priority)
    sorted_targets = sorted(targets, key=lambda x: x.get('priority', 3))
    primary_target = sorted_targets[0]
    
    return normalize_bbox_coordinates(primary_target.get('bbox', [0, 0, 0, 0]))


@safe_stop_image_writer
def record_loop(
    robot: Robot,
    events: dict,
    fps: int,
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | None = None,
    policy: PreTrainedPolicy | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
):
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    # if policy is given it needs cleaning up
    if policy is not None:
        policy.reset()

    timestamp = 0
    start_episode_t = time.perf_counter()
    
    observation = robot.get_observation()
    print("Available cameras:", [key for key in observation.keys() if isinstance(observation[key], np.ndarray) and len(observation[key].shape) == 3])

    # Get initial bimanual affordances from multi-view cameras
    # Assuming the robot has both 'top' and 'front' cameras
    camera_keys = [key for key in observation.keys() if isinstance(observation[key], np.ndarray) and len(observation[key].shape) == 3]
    
    # Check for required cameras: top and front
    required_cameras = ["top", "front"]
    missing_cameras = [cam for cam in required_cameras if cam not in camera_keys]
    if missing_cameras:
        raise ValueError(f"Bimanual recording requires 'top' and 'front' cameras. Missing: {missing_cameras}. Available: {camera_keys}")
    
    # Use specific camera keys for top and front views
    top_camera = "top"
    front_camera = "front"
    print(f"Using cameras: {top_camera} (top view), {front_camera} (front view)")
    
    # Get initial bimanual affordances
    action_plan, affordance_result = get_bimanual_affordances_multi_view(
        observation[top_camera], 
        observation[front_camera]
    )
    
    # Extract targets for both views
    red_targets_top, white_targets_top, handoff_zones_top, task_info = extract_arm_targets(action_plan, "top")
    red_targets_front, white_targets_front, handoff_zones_front, _ = extract_arm_targets(action_plan, "front")
    
    # Get primary target bboxes for each arm and view
    red_bbox_top = get_primary_target_bbox(red_targets_top)
    white_bbox_top = get_primary_target_bbox(white_targets_top)
    red_bbox_front = get_primary_target_bbox(red_targets_front)
    white_bbox_front = get_primary_target_bbox(white_targets_front)
    
    # Set initial task description
    if red_targets_top and white_targets_top:
        red_target_label = red_targets_top[0].get('label', 'object')
        white_target_label = white_targets_top[0].get('label', 'object')
        task_type = task_info.get('task_type', 'parallel_sort')
        
        if task_type == 'disassembly':
            single_task = f"Red arm holds {red_target_label}, white arm pulls {white_target_label}"
        elif task_type == 'handoff':
            single_task = f"Red arm picks {red_target_label}, hands off to white arm"
        else:
            single_task = f"Red arm manipulates {red_target_label}, white arm manipulates {white_target_label}"
    
    print(f"Initial task: {single_task}")
    print(f"Task type: {task_info.get('task_type')}")
    print(f"Coordination required: {task_info.get('coordination_required')}")
                
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break
            
        if events["select_new_bbox"]:
            events["select_new_bbox"] = False
            
            # Get new bimanual affordances
            observation_temp = robot.get_observation()
            action_plan, affordance_result = get_bimanual_affordances_multi_view(
                observation_temp[top_camera], 
                observation_temp[front_camera]
            )
            
            # Extract new targets
            red_targets_top, white_targets_top, handoff_zones_top, task_info = extract_arm_targets(action_plan, "top")
            red_targets_front, white_targets_front, handoff_zones_front, _ = extract_arm_targets(action_plan, "front")
            
            # Update primary target bboxes
            red_bbox_top = get_primary_target_bbox(red_targets_top)
            white_bbox_top = get_primary_target_bbox(white_targets_top)
            red_bbox_front = get_primary_target_bbox(red_targets_front)
            white_bbox_front = get_primary_target_bbox(white_targets_front)
            
            # Update task description
            if red_targets_top and white_targets_top:
                red_target_label = red_targets_top[0].get('label', 'object')
                white_target_label = white_targets_top[0].get('label', 'object')
                task_type = task_info.get('task_type', 'parallel_sort')
                
                if task_type == 'disassembly':
                    single_task = f"Red arm holds {red_target_label}, white arm pulls {white_target_label}"
                elif task_type == 'handoff':
                    single_task = f"Red arm picks {red_target_label}, hands off to white arm"
                else:
                    single_task = f"Red arm manipulates {red_target_label}, white arm manipulates {white_target_label}"
            
            print(f"New task: {single_task}")
        
        observation = robot.get_observation()
        
        # Add bimanual affordance observations (25 additional features)
        task_type_encoded = {'disassembly': 0.0, 'handoff': 1.0, 'parallel_sort': 2.0}.get(
            task_info.get('task_type', 'parallel_sort'), 2.0
        )
        
        observation.update({
            # Task context (3 features)
            'task_type': task_type_encoded,
            'coordination_required': 1.0 if task_info.get('coordination_required', False) else 0.0,
            'primary_arm': 1.0 if red_targets_top else 0.0,  # 1.0 if red arm has priority
            
            # Red arm target bbox coordinates (8 features)
            'red_target_y1_top': red_bbox_top[0],
            'red_target_x1_top': red_bbox_top[1],
            'red_target_y2_top': red_bbox_top[2],
            'red_target_x2_top': red_bbox_top[3],
            'red_target_y1_front': red_bbox_front[0],
            'red_target_x1_front': red_bbox_front[1],
            'red_target_y2_front': red_bbox_front[2],
            'red_target_x2_front': red_bbox_front[3],
            
            # White arm target bbox coordinates (8 features)
            'white_target_y1_top': white_bbox_top[0],
            'white_target_x1_top': white_bbox_top[1],
            'white_target_y2_top': white_bbox_top[2],
            'white_target_x2_top': white_bbox_top[3],
            'white_target_y1_front': white_bbox_front[0],
            'white_target_x1_front': white_bbox_front[1],
            'white_target_y2_front': white_bbox_front[2],
            'white_target_x2_front': white_bbox_front[3],
            
            # Action type encodings (6 features - one-hot encoded)
            'red_action_pick': 1.0 if red_targets_top and red_targets_top[0].get('action') == 'pick' else 0.0,
            'red_action_hold': 1.0 if red_targets_top and red_targets_top[0].get('action') == 'hold' else 0.0,
            'red_action_place': 1.0 if red_targets_top and red_targets_top[0].get('action') == 'place' else 0.0,
            'white_action_pick': 1.0 if white_targets_top and white_targets_top[0].get('action') == 'pick' else 0.0,
            'white_action_pull': 1.0 if white_targets_top and white_targets_top[0].get('action') == 'pull' else 0.0,
            'white_action_place': 1.0 if white_targets_top and white_targets_top[0].get('action') == 'place' else 0.0,
        })

        if policy is not None or dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, observation, prefix="observation")
    
        if policy is not None:
            action_values = predict_action(
                observation_frame,
                policy,
                get_safe_torch_device(policy.config.device),
                policy.config.use_amp,
                task=single_task,
                robot_type=robot.robot_type,
            )
            action = {key: action_values[i].item() for i, key in enumerate(robot.action_features)}
        elif policy is None and teleop is not None:
            action = teleop.get_action()
        else:
            logging.info(
                "No policy or teleoperator provided, skipping action generation."
                "This is likely to happen when resetting the environment without a teleop device."
                "The robot won't be at its rest position at the start of the next episode."
            )
            continue

        # Action can eventually be clipped using `max_relative_target`,
        # so action actually sent is saved in the dataset.
        sent_action = robot.send_action(action)

        if dataset is not None:
            action_frame = build_dataset_frame(dataset.features, sent_action, prefix="action")
            frame = {**observation_frame, **action_frame}
            dataset.add_frame(frame, task=single_task)

        if display_data:
            # Log bimanual affordance data to rerun for detailed visualization
            for obs, val in observation.items():
                if isinstance(val, float):
                    rr.log(f"observation.{obs}", rr.Scalar(val))
                elif isinstance(val, np.ndarray) and len(val.shape) == 3:
                    # Plot bimanual bboxes on camera images
                    if top_camera in obs:
                        enhanced_val = np.array(plot_bimanual_bbox(
                            observation[top_camera], 
                            red_targets=red_targets_top,
                            white_targets=white_targets_top,
                            handoff_zones=handoff_zones_top
                        ))
                        rr.log(f"observation.{obs}", rr.Image(enhanced_val))
                        
                        # Log individual target bboxes as separate entities for better visualization
                        for i, target in enumerate(red_targets_top):
                            bbox = target.get('bbox', [0, 0, 0, 0])
                            label = f"RED: {target.get('label', '')} ({target.get('action', '')})"
                            rr.log(f"targets/red_targets_top/target_{i}", rr.Boxes2D(
                                array=[[bbox[1], bbox[0], bbox[3], bbox[2]]],  # Convert [y1,x1,y2,x2] to [x1,y1,x2,y2]
                                labels=[label],
                                colors=[[255, 0, 0]]  # Red color
                            ))
                            
                        for i, target in enumerate(white_targets_top):
                            bbox = target.get('bbox', [0, 0, 0, 0])
                            label = f"WHITE: {target.get('label', '')} ({target.get('action', '')})"
                            rr.log(f"targets/white_targets_top/target_{i}", rr.Boxes2D(
                                array=[[bbox[1], bbox[0], bbox[3], bbox[2]]],  # Convert [y1,x1,y2,x2] to [x1,y1,x2,y2]
                                labels=[label],
                                colors=[[255, 255, 255]]  # White color
                            ))
                            
                    elif front_camera in obs:
                        enhanced_val = np.array(plot_bimanual_bbox(
                            observation[front_camera],
                            red_targets=red_targets_front, 
                            white_targets=white_targets_front,
                            handoff_zones=handoff_zones_front
                        ))
                        rr.log(f"observation.{obs}", rr.Image(enhanced_val))
                        
                        # Log individual target bboxes for front view
                        for i, target in enumerate(red_targets_front):
                            bbox = target.get('bbox', [0, 0, 0, 0])
                            label = f"RED: {target.get('label', '')} ({target.get('action', '')})"
                            rr.log(f"targets/red_targets_front/target_{i}", rr.Boxes2D(
                                array=[[bbox[1], bbox[0], bbox[3], bbox[2]]],
                                labels=[label],
                                colors=[[255, 0, 0]]
                            ))
                            
                        for i, target in enumerate(white_targets_front):
                            bbox = target.get('bbox', [0, 0, 0, 0])
                            label = f"WHITE: {target.get('label', '')} ({target.get('action', '')})"
                            rr.log(f"targets/white_targets_front/target_{i}", rr.Boxes2D(
                                array=[[bbox[1], bbox[0], bbox[3], bbox[2]]],
                                labels=[label],
                                colors=[[255, 255, 255]]
                            ))
                    else:
                        # Regular camera without bimanual overlay
                        rr.log(f"observation.{obs}", rr.Image(val))
            
            # Log action values
            for act, val in action.items():
                if isinstance(val, float):
                    rr.log(f"action.{act}", rr.Scalar(val))
                    
            # Log bimanual task information
            rr.log("bimanual/task_type", rr.TextLog(task_info.get('task_type', 'parallel_sort')))
            rr.log("bimanual/coordination_required", rr.Scalar(1.0 if task_info.get('coordination_required', False) else 0.0))
            rr.log("bimanual/red_targets_count", rr.Scalar(len(red_targets_top)))
            rr.log("bimanual/white_targets_count", rr.Scalar(len(white_targets_top)))
            rr.log("bimanual/current_task", rr.TextLog(single_task))

        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / fps - dt_s)

        timestamp = time.perf_counter() - start_episode_t


@parser.wrap()
def record(cfg: RecordConfig) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        _init_rerun(session_name="bimanual_recording")

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    # Verify this is a bimanual robot
    if not hasattr(robot, 'left_arm') or not hasattr(robot, 'right_arm'):
        raise ValueError("This script requires a bimanual robot with left_arm and right_arm attributes")

    action_features = hw_to_dataset_features(robot.action_features, "action", cfg.dataset.video)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", cfg.dataset.video)
    dataset_features = {**action_features, **obs_features}

    if cfg.resume:
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )

        if hasattr(robot, "cameras") and len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=cfg.dataset.num_image_writer_processes,
                num_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            )
        sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
    else:
        # Create empty dataset or load existing saved episodes
        sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.dataset.fps,
            root=cfg.dataset.root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=cfg.dataset.video,
            image_writer_processes=cfg.dataset.num_image_writer_processes,
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )

    # Load pretrained policy
    policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)

    if policy is not None:
        policy.eval()
    
    robot.connect(calibrate=False)
    if teleop is not None:
        teleop.connect()

    listener, events = init_keyboard_listener()

    with VideoEncodingManager(dataset):
        recorded_episodes = 0
        while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
            log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
            record_loop(
                robot=robot,
                events=events,
                fps=cfg.dataset.fps,
                teleop=teleop,
                policy=policy,
                dataset=dataset,
                control_time_s=cfg.dataset.episode_time_s,
                single_task=cfg.dataset.single_task,
                display_data=cfg.display_data,
            )

            # Execute a few seconds without recording to give time to manually reset the environment
            # Skip reset for the last episode to be recorded
            if not events["stop_recording"] and (
                (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
            ):
                log_say("Reset the environment", cfg.play_sounds)
                record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop=teleop,
                    control_time_s=cfg.dataset.reset_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                )

            if events["rerecord_episode"]:
                log_say("Re-record episode", cfg.play_sounds)
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            dataset.save_episode()
            recorded_episodes += 1

    log_say("Stop recording", cfg.play_sounds, blocking=True)

    robot.disconnect()
    if teleop is not None:
        teleop.disconnect()

    if not is_headless() and listener is not None:
        listener.stop()

    if cfg.dataset.push_to_hub:
        dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

    log_say("Exiting", cfg.play_sounds)

    return dataset


if __name__ == "__main__":
    record()