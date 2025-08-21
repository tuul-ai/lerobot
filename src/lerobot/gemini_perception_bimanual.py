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

def get_2D_bbox_bimanual(img, task_type=None, arm_positions=None) -> str:
    """
    Detects objects and assigns them to red/white arms based on bimanual task requirements.
    
    Args:
        img: PIL Image or tensor of the scene
        task_type: Optional task type ('disassembly', 'handoff', 'parallel_sort'). 
                  If None, will be auto-detected.
        arm_positions: Optional dict with red/white arm positions for workspace awareness
        
    Returns:
        str: JSON string with bimanual affordance detection results
    """
    
    # System instructions for bimanual affordance detection
    bimanual_system_instructions = """
    You are an expert at analyzing images for bimanual robot manipulation with a red arm and white arm. 
    Return bounding boxes as a JSON object with this exact structure:
    {
        "task_type": "disassembly|handoff|parallel_sort",
        "coordination_required": true|false,
        "objects": [
            {
                "label": "descriptive object name",
                "box_2d": [y1, x1, y2, x2],
                "arm_assignment": "red|white|both|either",
                "action_type": "pick|hold|pull|place|handoff_receive|handoff_give",
                "priority": 1|2|3,
                "coordination_frame": "simultaneous|sequential|leader_follower"
            }
        ],
        "workspace_constraints": {
            "red_reachable_zone": [y1, x1, y2, x2],
            "white_reachable_zone": [y1, x1, y2, x2],
            "handoff_zone": [y1, x1, y2, x2]
        }
    }
    
    The "box_2d" coordinates must be [ymin, xmin, ymax, xmax], normalized to a 0-1000 scale.
    Possible object types based on task:
    - "pick_red": Objects the red arm should pick
    - "pick_white": Objects the white arm should pick  
    - "hold_target": Objects that need to be held steady (for disassembly)
    - "pull_target": Objects that need to be pulled (for disassembly)
    - "handoff_zone": Area where arms should meet for handoffs
    - "place_target": Final placement locations
    
    Never return code fencing around the JSON. Only output the raw JSON object.
    """
    
    # If no task type specified, do initial analysis to detect task
    if task_type is None:
        initial_prompt = """Briefly analyze this scene and identify the objects present. 
        Look for: assembled LEGO pieces, large objects, multiple small objects, bins/containers.
        List what you see without bounding boxes."""
        
        initial_response = client.models.generate_content(
            model=MODEL_ID,
            contents=[img, initial_prompt],
            config=types.GenerateContentConfig(temperature=0.3),
        )
        
        task_type = detect_task_type(initial_response.text)
    
    # Get task-specific prompt
    if task_type == 'disassembly':
        task_prompt = get_disassembly_prompt()
    elif task_type == 'handoff':
        task_prompt = get_handoff_prompt()
    elif task_type == 'parallel_sort':
        task_prompt = get_parallel_sort_prompt()
    else:
        task_prompt = get_parallel_sort_prompt()  # Default fallback
    
    # Combine with detailed analysis prompt
    detailed_prompt = f"""
    {task_prompt}
    
    Analyze the provided image for bimanual manipulation. The task type is: {task_type}
    
    Detect all relevant objects and assign them to the red arm or white arm based on:
    1. Spatial positioning relative to each arm's workspace
    2. Task requirements (holding vs pulling for disassembly, size for handoffs, etc.)
    3. Coordination needs between the two arms
    
    Return your findings as a JSON object following the exact structure specified in the system instructions.
    Include workspace constraints showing the reachable zones for each arm.
    
    For coordination_required: set to true if both arms must work together simultaneously.
    For priority: 1 = highest priority (do first), 2 = medium, 3 = lowest.
    """
    
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[img, detailed_prompt],
        config=types.GenerateContentConfig(
            system_instruction=bimanual_system_instructions, 
            temperature=0.5
        ),
    )
    
    return response.text

def reconcile_multiview_affordances(top_view_result, front_view_result):
    """
    Merge affordance detections from multiple camera views.
    Prioritizes spatial consistency and arm reachability.
    
    Args:
        top_view_result: JSON string from top camera affordance detection
        front_view_result: JSON string from front camera affordance detection
        
    Returns:
        dict: Reconciled bimanual plan with consistent arm assignments
    """
    try:
        top_data = json.loads(parse_json(top_view_result))
        front_data = json.loads(parse_json(front_view_result))
    except json.JSONDecodeError as e:
        print(f"Error parsing multiview results: {e}")
        # Fallback to top view if parsing fails
        return json.loads(parse_json(top_view_result))
    
    # Use top view as primary, validate with front view
    reconciled = top_data.copy()
    
    # Cross-validate arm assignments between views
    top_objects = {obj['label']: obj for obj in top_data.get('objects', [])}
    front_objects = {obj['label']: obj for obj in front_data.get('objects', [])}
    
    # Check for conflicts in arm assignments
    for label, top_obj in top_objects.items():
        if label in front_objects:
            front_obj = front_objects[label]
            if top_obj['arm_assignment'] != front_obj['arm_assignment']:
                # Conflict detected - use spatial reasoning to resolve
                # For now, prioritize top view but add confidence flag
                top_obj['assignment_confidence'] = 'medium'
    
    # Merge workspace constraints (use intersection for safety)
    if 'workspace_constraints' in front_data:
        front_constraints = front_data['workspace_constraints']
        top_constraints = reconciled.get('workspace_constraints', {})
        
        # Take more conservative (smaller) workspace bounds
        for zone in ['red_reachable_zone', 'white_reachable_zone', 'handoff_zone']:
            both_constraints = [top_constraints.get(zone), front_constraints.get(zone)]
            if all(constraint is not None for constraint in both_constraints):
                # Use intersection of both zones for safety
                reconciled['workspace_constraints'][zone] = [
                    max(both_constraints[0][0], both_constraints[1][0]),  # max y1
                    max(both_constraints[0][1], both_constraints[1][1]),  # max x1  
                    min(both_constraints[0][2], both_constraints[1][2]),  # min y2
                    min(both_constraints[0][3], both_constraints[1][3])   # min x2
                ]
    
    return reconciled

def create_bimanual_action_plan(affordance_result):
    """
    Convert affordance detection results into structured action plan for red/white arms.
    
    Args:
        affordance_result: Dict from affordance detection or JSON string
        
    Returns:
        dict: Structured action plan with arm-specific targets and coordination info
    """
    if isinstance(affordance_result, str):
        try:
            affordance_data = json.loads(parse_json(affordance_result))
        except json.JSONDecodeError:
            return None
    else:
        affordance_data = affordance_result
    
    action_plan = {
        'task_type': affordance_data.get('task_type', 'parallel_sort'),
        'coordination_required': affordance_data.get('coordination_required', False),
        'red_arm': {'targets': [], 'actions': [], 'priority_order': []},
        'white_arm': {'targets': [], 'actions': [], 'priority_order': []},
        'coordination': {
            'sync_points': [],
            'handoff_zones': [],
            'sequence_constraints': []
        }
    }
    
    # Process each detected object
    for obj in affordance_data.get('objects', []):
        arm_assignment = obj.get('arm_assignment', 'either')
        action_type = obj.get('action_type', 'pick')
        priority = obj.get('priority', 2)
        
        target_info = {
            'label': obj.get('label', ''),
            'bbox': obj.get('box_2d', [0, 0, 0, 0]),
            'action': action_type,
            'priority': priority,
            'coordination_frame': obj.get('coordination_frame', 'sequential')
        }
        
        # Assign to appropriate arm(s)
        if arm_assignment in ['red', 'pick_red']:
            action_plan['red_arm']['targets'].append(target_info)
            action_plan['red_arm']['actions'].append(action_type)
            action_plan['red_arm']['priority_order'].append(priority)
            
        elif arm_assignment in ['white', 'pick_white']:
            action_plan['white_arm']['targets'].append(target_info)
            action_plan['white_arm']['actions'].append(action_type)
            action_plan['white_arm']['priority_order'].append(priority)
            
        elif arm_assignment == 'both':
            # Add to coordination requirements
            action_plan['coordination']['sync_points'].append(target_info)
            
        elif action_type == 'handoff_zone':
            action_plan['coordination']['handoff_zones'].append(target_info)
    
    # Sort by priority (1 = highest priority, 3 = lowest)
    for arm in ['red_arm', 'white_arm']:
        arm_data = action_plan[arm]
        if arm_data['targets']:
            # Sort targets by priority
            sorted_indices = sorted(range(len(arm_data['targets'])), 
                                  key=lambda i: arm_data['targets'][i]['priority'])
            arm_data['targets'] = [arm_data['targets'][i] for i in sorted_indices]
            arm_data['actions'] = [arm_data['actions'][i] for i in sorted_indices]
            arm_data['priority_order'] = [arm_data['priority_order'][i] for i in sorted_indices]
    
    return action_plan

def plot_bimanual_bbox(im_tensor, red_targets=None, white_targets=None, handoff_zones=None):
    """
    Plots red and white arm targets with different colors and styles.
    
    Args:
        im_tensor: The image tensor
        red_targets: List of red arm targets with bbox coordinates
        white_targets: List of white arm targets with bbox coordinates  
        handoff_zones: List of handoff zones
    """
    im = tensor_to_pil(im_tensor)
    img = im.copy()
    width, height = img.size
    draw = ImageDraw.Draw(img)
    
    # Define colors and styles
    red_color = 'red'
    white_color = 'white'
    handoff_color = 'yellow'
    
    def draw_bbox_with_label(bbox, color, label_text=None, line_width=3):
        # Convert normalized coordinates (0-1000) to absolute coordinates
        y1, x1, y2, x2 = bbox
        x1_abs = int((x1 / 1000) * width)
        y1_abs = int((y1 / 1000) * height)
        x2_abs = int((x2 / 1000) * width)
        y2_abs = int((y2 / 1000) * height)
        
        # Draw bounding box
        for i in range(line_width):
            draw.rectangle([x1_abs - i, y1_abs - i, x2_abs + i, y2_abs + i], 
                          outline=color, fill=None)
        
        # Add label if provided
        if label_text:
            # Position label above the bounding box
            label_y = max(0, y1_abs - 20)
            draw.text((x1_abs, label_y), f"{label_text}", fill=color)
    
    # Draw red arm targets
    if red_targets:
        for target in red_targets:
            bbox = target.get('bbox', target.get('box_2d', [0, 0, 0, 0]))
            label = f"RED: {target.get('label', '')} ({target.get('action', '')})"
            draw_bbox_with_label(bbox, red_color, label)
    
    # Draw white arm targets  
    if white_targets:
        for target in white_targets:
            bbox = target.get('bbox', target.get('box_2d', [0, 0, 0, 0]))
            label = f"WHITE: {target.get('label', '')} ({target.get('action', '')})"
            draw_bbox_with_label(bbox, white_color, label)
    
    # Draw handoff zones
    if handoff_zones:
        for zone in handoff_zones:
            bbox = zone.get('bbox', zone.get('box_2d', [0, 0, 0, 0]))
            label = f"HANDOFF: {zone.get('label', '')}"
            draw_bbox_with_label(bbox, handoff_color, label)
    
    return img

if __name__ == "__main__":
    # Example usage and testing
    print("Bimanual Affordance Detection System Initialized")
    print("Available functions:")
    print("- get_2D_bbox_bimanual(): Main affordance detection")
    print("- reconcile_multiview_affordances(): Merge multi-camera results")
    print("- create_bimanual_action_plan(): Convert to action plan")
    print("- plot_bimanual_bbox(): Visualize results")