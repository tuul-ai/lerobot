Your draft is solid but needs some crucial improvements. Here's an enhanced version:

```markdown
# Bimanual Robot Affordances Implementation

## Overview
Extend the existing single-arm affordance system to support bimanual (two-arm) robot tasks, specifically:
1. **Disassembly**: One arm holds, other pulls apart LEGO pieces
2. **Handoff**: Transfer objects between arms based on size/weight
3. **Parallel Sorting**: Both arms sort simultaneously into different bins

## Current Files Structure
- `gemini_perception.py`: Single-arm bbox detection
- `record_bbox_multi.py`: Multi-camera recording for single arm
- `datasets/utils.py`: Dataset feature creation

## Implementation Plan

### 1. Create `gemini_perception_bimanual.py`

**Key Design Decisions:**
- Each arm needs its own set of bboxes
- Some objects require coordination (e.g., assembled pieces need both arms)
- Task type should be auto-detected from scene

```python
def get_2D_bbox_bimanual(img, task_type=None) -> str:
    """
    Detects objects and assigns them to appropriate arms based on task.
    
    Returns JSON with structure:
    {
        "task_type": "disassembly|handoff|parallel_sort",
        "objects": [
            {
                "label": "red 2x4 brick",
                "box_2d": [y1, x1, y2, x2],
                "arm_assignment": "left|right|both",
                "action_type": "pick|hold|pull|place",
                "coordination_required": true|false
            }
        ]
    }
    """
```

**Important: Handle edge cases:**
- What if no assembled pieces exist for disassembly?
- What if bins are out of frame?
- What if objects are unreachable by one arm?

### 2. Create `record_bbox_bimanual.py`

**Key additions to observation space:**
```python
observation.update({
    # Task identification
    'task_type': task_type,  # 'disassembly', 'handoff', 'parallel_sort'
    
    # Left arm targets (8 values per bbox)
    'left_arm_target_y1_top': norm_left_target_top[0],
    'left_arm_target_x1_top': norm_left_target_top[1],
    'left_arm_target_y2_top': norm_left_target_top[2],
    'left_arm_target_x2_top': norm_left_target_top[3],
    'left_arm_target_y1_front': norm_left_target_front[0],
    'left_arm_target_x1_front': norm_left_target_front[1],
    'left_arm_target_y2_front': norm_left_target_front[2],
    'left_arm_target_x2_front': norm_left_target_front[3],
    
    # Right arm targets (8 values per bbox)
    'right_arm_target_y1_top': norm_right_target_top[0],
    # ... (same pattern)
    
    # Coordination flags
    'coordination_required': 1.0 if coordination_required else 0.0,
    'primary_arm': 1.0 if primary_arm == 'left' else 0.0,
    
    # Action types encoded as one-hot
    'left_action_pick': 1.0 if left_action == 'pick' else 0.0,
    'left_action_hold': 1.0 if left_action == 'hold' else 0.0,
    'left_action_place': 1.0 if left_action == 'place' else 0.0,
    'right_action_pick': 1.0 if right_action == 'pick' else 0.0,
    'right_action_pull': 1.0 if right_action == 'pull' else 0.0,
    'right_action_place': 1.0 if right_action == 'place' else 0.0,
})
```

### 3. Update `datasets/utils.py`

```python
# In hw_to_dataset_features(), around line 411:
if joint_fts and prefix == "observation":
    # Bimanual bbox features (16 coords + 9 flags = 25 additional states)
    bimanual_states = [
        # Left arm bbox coordinates (8)
        'left_arm_target_y1_top', 'left_arm_target_x1_top', 
        'left_arm_target_y2_top', 'left_arm_target_x2_top',
        'left_arm_target_y1_front', 'left_arm_target_x1_front', 
        'left_arm_target_y2_front', 'left_arm_target_x2_front',
        
        # Right arm bbox coordinates (8)
        'right_arm_target_y1_top', 'right_arm_target_x1_top',
        'right_arm_target_y2_top', 'right_arm_target_x2_top',
        'right_arm_target_y1_front', 'right_arm_target_x1_front',
        'right_arm_target_y2_front', 'right_arm_target_x2_front',
        
        # Coordination and action flags (9)
        'coordination_required', 'primary_arm',
        'left_action_pick', 'left_action_hold', 'left_action_place',
        'right_action_pick', 'right_action_pull', 'right_action_place',
        'task_type'  # Will be encoded as float (0=disassembly, 1=handoff, 2=sort)
    ]
    
    joint_fts = {**joint_fts, **{name: float for name in bimanual_states}}
```

### 4. Task Detection Logic

```python
def detect_bimanual_task(objects, scene_context):
    """
    Auto-detect which bimanual task to perform based on scene.
    
    Priority order:
    1. If assembled LEGO detected → disassembly
    2. If large pieces (>4x4) present → handoff  
    3. If multiple bins visible → parallel sort
    4. Default → single arm fallback
    """
```

### 5. Coordination Requirements

**Disassembly Coordination:**
- Left arm must reach hold position BEFORE right arm pulls
- Both arms must maintain relative positions during pull
- Add "sync_checkpoint" flag when coordination critical

**Handoff Coordination:**
- Define "handoff zone" as 3D space between arms
- Left arm must reach zone first (if holding large object)
- Right arm approaches only after left is stable

**Parallel Sort Coordination:**
- No strict coordination needed
- Collision detection: if both targeting same object, reassign

### 6. Error Handling

Add these checks:
- If one arm can't reach assigned object, reassign to other arm
- If no valid targets for a task, fallback to simpler task
- If coordination fails repeatedly, switch to sequential operation

### 7. Testing Strategy

Create test scenarios:
1. Scene with only assembled pieces (forces disassembly)
2. Scene with only large pieces (forces handoff)
3. Scene with mixed objects (tests task selection)
4. Scene with occlusions (tests error handling)

## Implementation Order

1. Start with `gemini_perception_bimanual.py` - get detection working
2. Update `datasets/utils.py` - ensure features are captured
3. Create `record_bbox_bimanual.py` - integrate everything
4. Test with simple parallel sorting first
5. Add disassembly (more complex coordination)
6. Add handoff last (most complex)

## Critical Success Factors

1. **Symmetry**: Left/right assignments should be swappable
2. **Graceful Degradation**: If bimanual fails, fallback to sequential
3. **Clear Visualization**: Show which arm targets which object
4. **Deterministic**: Same scene should produce same task assignment
```

Key improvements:
1. Added specific observation space structure
2. Included error handling and edge cases  
3. Added coordination timing requirements
4. Specified testing strategy
5. Made the feature vector explicit (25 additional states)
6. Added implementation order for incremental development