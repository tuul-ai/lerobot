import os
from google import genai
from google.genai import types
from PIL import Image, ImageDraw
import torch
import numpy as np
import json
import time

GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_ID = "gemini-2.0-flash"  # Use Gemini 2.0 Flash for 3D capabilities
PRO_MODEL_ID = 'gemini-2.5-pro'

def tensor_to_pil(tensor: torch.Tensor | np.ndarray) -> Image.Image:
    """Convert a PyTorch tensor or numpy array to PIL Image"""
    # Handle both torch tensors and numpy arrays
    if isinstance(tensor, torch.Tensor):
        if len(tensor.shape) == 4:  # Remove batch dimension if present
            tensor = tensor[0]
        
        # Convert from CxHxW to HxWxC
        if len(tensor.shape) == 3 and tensor.shape[0] == 3:
            img_array = tensor.permute(1, 2, 0).cpu().numpy()
        else:
            img_array = tensor.cpu().numpy()
    else:  # numpy array
        if len(tensor.shape) == 4:  # Remove batch dimension if present
            tensor = tensor[0]
        
        # Convert from CxHxW to HxWxC
        if len(tensor.shape) == 3 and tensor.shape[0] == 3:
            img_array = np.transpose(tensor, (1, 2, 0))
        else:
            img_array = tensor
    
    # Scale to 0-255 if needed
    if img_array.max() <= 1.0:
        img_array = (img_array * 255).astype(np.uint8)
    
    return Image.fromarray(img_array)

def parse_json(json_output: str):
    """Parse JSON output from Gemini, handling markdown fencing"""
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output

def get_disassembly_prompt():
    """Prompt for LEGO disassembly tasks"""
    return """Identify LEGO assemblies that can be disassembled. 
    Mark the larger/base piece as "hold_target" for the red arm to hold steady.
    Mark the piece to be removed as "pull_target" for the white arm to pull apart.
    Also identify any loose pieces as "pick_red" or "pick_white" based on their position relative to the workspace.
    Mark any bins or containers as "place_target".
    Focus on identifying pieces that are actually connected and can be separated."""

def get_handoff_prompt(): 
    """Prompt for handoff tasks between red and white arms"""
    return """Identify objects for bimanual handoff based on size and weight.
    Mark large pieces (4x4 LEGO bricks or bigger objects) as "pick_red" (red arm picks first).
    Mark the handoff zone (empty space between arms where transfer occurs) as "handoff_zone".
    Mark the final bins or placement areas as "place_target".
    Consider which arm is better positioned to initially grasp each object."""

def get_parallel_sort_prompt():
    """Prompt for parallel sorting between red and white arms"""
    return """Identify objects for parallel sorting between red and white arms.
    Assign objects on the left side of workspace as "pick_red" for the red arm.
    Assign objects on the right side of workspace as "pick_white" for the white arm.
    Mark sorting bins clearly for each arm as "place_target".
    Ensure both arms can work simultaneously without conflicts."""

def detect_task_type(scene_objects):
    """
    Automatically detect the most appropriate bimanual task based on scene analysis.
    
    Args:
        scene_objects: List of detected objects from initial scene analysis
        
    Returns:
        str: Task type ('disassembly', 'handoff', 'parallel_sort')
    """
    # Look for assembled/connected pieces (indicates disassembly task)
    assembled_indicators = ['assembly', 'connected', 'attached', 'tower', 'stack', 'built']
    if any(indicator in str(scene_objects).lower() for indicator in assembled_indicators):
        return 'disassembly'
    
    # Look for large objects (indicates handoff task)
    large_object_indicators = ['large', 'big', 'heavy', '4x4', '6x6', '8x8']
    if any(indicator in str(scene_objects).lower() for indicator in large_object_indicators):
        return 'handoff'
    
    # Default to parallel sorting
    return 'parallel_sort'

def get_2D_bbox_bimanual_single(image, task_type=None):
    """
    Enhanced object detection for bimanual manipulation using single top view camera.
    
    Args:
        image: Input image (numpy array, torch tensor, or PIL Image)
        task_type: Optional specific task type hint
    
    Returns:
        dict: Detection results with bounding boxes and bimanual assignments
    """
    if isinstance(image, (torch.Tensor, np.ndarray)):
        pil_image = tensor_to_pil(image)
    else:
        pil_image = image
        
    try:
        # Initial scene analysis to understand context
        scene_prompt = """Analyze this image for bimanual robot manipulation.
        
        Describe what you see focusing on:
        1. LEGO pieces or other objects that need manipulation
        2. Whether objects appear connected/assembled or separate
        3. Object sizes and relative positions
        4. Any bins, containers, or workspace areas
        5. Spatial layout that suggests coordination between two robot arms
        
        Be specific about object types, colors, and spatial relationships."""
        
        scene_response = client.models.generate_content(
            model=MODEL_ID,
            contents=[scene_prompt, pil_image]
        )
        
        scene_description = scene_response.candidates[0].content.parts[0].text
        
        # Auto-detect task type if not provided
        if task_type is None:
            detected_task_type = detect_task_type(scene_description)
            print(f"Auto-detected task type: {detected_task_type}")
        else:
            detected_task_type = task_type
            
        # Get appropriate prompt based on task type
        if detected_task_type == 'disassembly':
            task_prompt = get_disassembly_prompt()
        elif detected_task_type == 'handoff':
            task_prompt = get_handoff_prompt()
        else:
            task_prompt = get_parallel_sort_prompt()
        
        # Enhanced bimanual detection prompt
        detection_prompt = f"""You are a bimanual robot manipulation system. Analyze this workspace image from a top-down view and identify objects for coordinated red and white arm manipulation.

TASK CONTEXT: {detected_task_type}
{task_prompt}

OBJECT CLASSIFICATION:
For the red arm (left manipulator):
- pick_red: Objects for red arm to pick up
- hold_target: Objects for red arm to hold steady during disassembly

For the white arm (right manipulator):
- pick_white: Objects for white arm to pick up  
- pull_target: Objects for white arm to pull/remove during disassembly

Shared/coordination zones:
- place_target: Bins, containers, or placement areas
- handoff_zone: Space between arms for object transfer

DETECTION REQUIREMENTS:
1. Provide precise bounding boxes in [y1, x1, y2, x2] format (0-1000 scale)
2. Consider arm reachability from typical bimanual robot setup
3. Prioritize objects that enable coordinated manipulation
4. Ensure actions don't cause collisions between arms

CURRENT SCENE: {scene_description}

Return only a JSON array of detected objects:
```json
[
  {{
    "label": "object_description",
    "category": "pick_red|pick_white|hold_target|pull_target|place_target|handoff_zone", 
    "bbox": [y1, x1, y2, x2],
    "confidence": 0.9,
    "priority": 1,
    "manipulation_notes": "specific_action_description"
  }}
]
```"""

        response = client.models.generate_content(
            model=PRO_MODEL_ID,
            contents=[detection_prompt, pil_image]
        )
        
        result_text = response.candidates[0].content.parts[0].text
        result_text = parse_json(result_text)
        
        try:
            detections = json.loads(result_text)
            
            # Add task type and scene context to results
            result = {
                'detections': detections,
                'task_type': detected_task_type,
                'scene_description': scene_description,
                'coordination_required': detected_task_type in ['disassembly', 'handoff']
            }
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Raw response: {result_text}")
            return {
                'detections': [],
                'task_type': detected_task_type,
                'scene_description': scene_description,
                'coordination_required': False,
                'error': 'Failed to parse detection results'
            }
            
    except Exception as e:
        print(f"Error in bimanual detection: {e}")
        return {
            'detections': [],
            'task_type': 'parallel_sort',
            'scene_description': '',
            'coordination_required': False,
            'error': str(e)
        }

def create_bimanual_action_plan(detection_result):
    """
    Convert detection results into structured bimanual action plan.
    
    Args:
        detection_result: Output from get_2D_bbox_bimanual_single
        
    Returns:
        dict: Structured action plan for bimanual coordination
    """
    if not detection_result or 'detections' not in detection_result:
        return {}
        
    detections = detection_result['detections']
    task_type = detection_result.get('task_type', 'parallel_sort')
    
    # Initialize action plan structure
    action_plan = {
        'task_type': task_type,
        'coordination_required': detection_result.get('coordination_required', False),
        'red_arm': {
            'targets': [],
            'actions': [],
            'priority_order': []
        },
        'white_arm': {
            'targets': [],
            'actions': [],
            'priority_order': []
        },
        'coordination': {
            'sync_points': [],
            'handoff_zones': [],
            'sequence_constraints': []
        }
    }
    
    # Process detections and assign to appropriate arms
    for detection in detections:
        category = detection.get('category', '')
        target_info = {
            'label': detection.get('label', ''),
            'bbox': detection.get('bbox', [0, 0, 0, 0]),
            'confidence': detection.get('confidence', 0.5),
            'priority': detection.get('priority', 2),
            'action': '',
            'manipulation_notes': detection.get('manipulation_notes', '')
        }
        
        # Assign targets based on category
        if category in ['pick_red', 'hold_target']:
            if category == 'pick_red':
                target_info['action'] = 'pick'
            else:
                target_info['action'] = 'hold'
            action_plan['red_arm']['targets'].append(target_info)
            
        elif category in ['pick_white', 'pull_target']:
            if category == 'pick_white':
                target_info['action'] = 'pick'
            else:
                target_info['action'] = 'pull'
            action_plan['white_arm']['targets'].append(target_info)
            
        elif category == 'handoff_zone':
            action_plan['coordination']['handoff_zones'].append({
                'label': target_info['label'],
                'bbox': target_info['bbox'],
                'action': 'handoff'
            })
            
        elif category == 'place_target':
            # Place targets can be used by both arms
            place_target = target_info.copy()
            place_target['action'] = 'place'
            action_plan['red_arm']['targets'].append(place_target)
            action_plan['white_arm']['targets'].append(place_target.copy())
    
    # Sort targets by priority (1 = highest priority)
    action_plan['red_arm']['targets'].sort(key=lambda x: x.get('priority', 3))
    action_plan['white_arm']['targets'].sort(key=lambda x: x.get('priority', 3))
    
    return action_plan

def plot_bimanual_bbox(im_tensor, red_targets=None, white_targets=None, handoff_zones=None):
    """
    Draw bounding boxes on image for bimanual targets - compatible with old plot_bbox format.
    
    Args:
        im_tensor: Input image tensor/array
        red_targets: List of red arm targets
        white_targets: List of white arm targets  
        handoff_zones: List of handoff zones
        
    Returns:
        PIL Image with drawn bounding boxes
    """
    # Convert tensor to PIL image (same as old script)
    im = tensor_to_pil(im_tensor)
    img = im.copy()
    width, height = img.size
    draw = ImageDraw.Draw(img)
    
    # Helper function to draw bbox with proper coordinate conversion (same as old script)
    def draw_bbox_with_label(bbox, color, label_text=None):
        # Convert normalized coordinates to absolute coordinates (0-1000 scale to pixel)
        abs_y1 = int(bbox[0]/1000 * height)
        abs_x1 = int(bbox[1]/1000 * width)
        abs_y2 = int(bbox[2]/1000 * height)
        abs_x2 = int(bbox[3]/1000 * width)

        # Ensure correct coordinate order
        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1
        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1

        # Draw the bounding box
        draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4)
        
        # Draw label if provided
        if label_text:
            try:
                from PIL import ImageFont
                font = ImageFont.load_default()
            except ImportError:
                font = None
            
            # Position text above the box
            text_x = abs_x1
            text_y = max(0, abs_y1 - 20)
            draw.text((text_x, text_y), label_text, fill=color, font=font)
    
    # Draw red arm targets in red
    if red_targets:
        for i, target in enumerate(red_targets):
            bbox = target.get('bbox', [0, 0, 0, 0])
            label = f"RED: {target.get('label', '')} ({target.get('action', '')})"
            draw_bbox_with_label(bbox, 'red', label)
    
    # Draw white arm targets in white
    if white_targets:
        for i, target in enumerate(white_targets):
            bbox = target.get('bbox', [0, 0, 0, 0])
            label = f"WHITE: {target.get('label', '')} ({target.get('action', '')})"
            draw_bbox_with_label(bbox, 'white', label)
    
    # Draw handoff zones in blue
    if handoff_zones:
        for i, zone in enumerate(handoff_zones):
            bbox = zone.get('bbox', [0, 0, 0, 0])
            label = f"HANDOFF: {zone.get('label', '')}"
            draw_bbox_with_label(bbox, 'blue', label)
    
    return img