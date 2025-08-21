
Until now I was using single robotic arm. And i built my polycies using affordances. 
- gemini_percepttion.py has all the logic for bbox detection.
- record_bbox.py has the logic for recording the bbox for single camera.
- record_bbox_multi.py has the logic for recording the bbox for two cameras.
- datasets/utils.py has the logic for creating the features for the dataset.

Now i want to extend the bimanual tasks. Help me design the affordances for bimanual tasks. Then update the code to support bimanual tasks. Create a new file called record_bbox_bimanual.py to support bimanual tasks for two cameras. And if necessary update the other files to support bimanual tasks. If necessary update the datasets/utils.py to support bimanual tasks and update the gemini_perception.py to support bimanual tasks (or create a new gemini_perception_bimanual.py for bimanual tasks).

My main goal is lego disassembly, handoff and sorting. I want to create a dataset for these tasks. So i need to design the affordances for these tasks. 
Now think step by step, come up with a plan, show me the plan and once I approve the plan, then update the code to support bimanual tasks.

Here is a rough draft, but critique it and improve it.

**1. Extend the bbox detection to support multiple action types:**

```python
def get_2D_bbox_bimanual(img, prompt=None) -> str:
    """Prompts Gemini for bimanual task affordances."""
    bimanual_system_instructions = """
    You are an expert at analyzing images for bimanual robot manipulation. 
    Return bounding boxes as a JSON array with these possible object types:
    - "pick_left": Objects the left arm should pick
    - "pick_right": Objects the right arm should pick  
    - "hold_target": Objects that need to be held steady (for disassembly)
    - "pull_target": Objects that need to be pulled (for disassembly)
    - "handoff_zone": Area where arms should meet for handoffs
    - "place_target": Final placement locations
    
    Each object should have "type" (string), "label" (string), and "box_2d" (array of 4 numbers).
    Example: [{"type": "hold_target", "label": "blue 2x4 brick", "box_2d": [100, 200, 150, 280]}]
    """
```

**2. Add task-specific prompts:**

```python
def get_disassembly_prompt():
    return """Identify LEGO assemblies that can be disassembled. 
    Mark the larger/base piece as "hold_target" and the piece to be removed as "pull_target".
    Also identify any loose pieces as "pick_left" or "pick_right" based on their position."""

def get_handoff_prompt(): 
    return """Identify objects for bimanual handoff based on size.
    Mark large pieces (4x4 or bigger) as "pick_left" (left arm picks first).
    Mark the handoff zone (empty space between arms) as "handoff_zone".
    Mark the final bins as "place_target"."""
```

**3. Modify the data structure for bimanual coordination:**

```python
def create_bimanual_action_lists(objects_list):
    actions = {
        'left_arm': {'pick': [], 'hold': [], 'place': []},
        'right_arm': {'pick': [], 'pull': [], 'place': []},
        'coordination': {'handoff_zones': [], 'sync_points': []}
    }
    
    for obj in objects_list:
        obj_type = obj.get("type", "")
        if obj_type == "pick_left":
            actions['left_arm']['pick'].append(obj)
        elif obj_type == "pick_right":
            actions['right_arm']['pick'].append(obj)
        elif obj_type == "hold_target":
            actions['left_arm']['hold'].append(obj)
        elif obj_type == "pull_target":
            actions['right_arm']['pull'].append(obj)
        elif obj_type == "handoff_zone":
            actions['coordination']['handoff_zones'].append(obj)
            
    return actions
```

**4. Add coordination detection between views:**

```python
def get_bimanual_coordination(top_view_actions, front_view_actions):
    """Determine which arm should do what based on multi-view analysis."""
    coordination_plan = {
        'sequence': [],  # Order of operations
        'constraints': [],  # Which arm must act first
        'sync_required': []  # When both arms must move together
    }
    
    # For disassembly: left holds, right pulls
    # For handoffs: check object positions relative to arms
    # For sorting: one arm holds container, other picks
    
    return coordination_plan
```

**5. Update the observation space in record_bbox_multi.py:**

```python
observation.update({
    # Left arm targets
    'left_pick_y1_top': left_pick_bbox[0],
    'left_pick_x1_top': left_pick_bbox[1],
    # ... (all left arm bbox coords)
    
    # Right arm targets  
    'right_pick_y1_top': right_pick_bbox[0],
    'right_pick_x1_top': right_pick_bbox[1],
    # ... (all right arm bbox coords)
    
    # Coordination info
    'action_type': action_type,  # 'disassemble', 'handoff', 'sort'
    'primary_arm': primary_arm,  # 'left' or 'right'
})
```

**6. Smart task selection based on what's visible:**

```python
def select_bimanual_task(objects_detected):
    """Choose appropriate bimanual task based on scene."""
    
    # Check for assembled pieces -> disassembly
    if has_connected_pieces(objects_detected):
        return "disassembly", get_disassembly_targets(objects_detected)
    
    # Check for size variety -> handoff task
    elif has_size_variety(objects_detected):
        return "handoff", get_handoff_targets(objects_detected)
    
    # Default to parallel sorting
    else:
        return "parallel_sort", get_parallel_sort_targets(objects_detected)
```

**7. Update the feature creation in datasets/utils.py:**

Make necessary changes to robot states on line 411:
Update this section to include the new bbox coordinates for the left and right arms, such that it should be uniform for different tasks. Consider for two views, the bbox coordinates should be the same for both views.
   ```python
   def hw_to_dataset_features():
         .....
         ....
      if joint_fts and prefix == "observation":
         # Comment this out when using multi-view or no bbox augmentation
        additional_states = ['pick_y1', 'pick_x1', 'pick_y2', 'pick_x2', 'place_y1', 'place_x1', 'place_y2', 'place_x2']
        # Comment this out when using single-view and no bbox augmentation
        # additional_states = ['pick_y1_top', 'pick_x1_top', 'pick_y2_top', 'pick_x2_top', 'place_y1_top', 'place_x1_top', 'place_y2_top', 'place_x2_top', 'pick_y1_front', 'pick_x1_front', 'pick_y2_front', 'pick_x2_front', 'place_y1_front', 'place_x1_front', 'place_y2_front', 'place_x2_front']
        # Comment this out when using no bbox augmentation
        joint_fts = {**joint_fts, **{name: float for name in additional_states}}
        features[f"{prefix}.state"] = {
            "dtype": "float32",
            "shape": (len(joint_fts),),
            "names": list(joint_fts),
        }
         .....
         ....
   ```


These changes should give rich bimanual affordances that can handle the complex coordination required for two-arm tasks. The key insight is that bimanual isn't just "two single arms" - it's about coordination and task-specific roles for each arm.