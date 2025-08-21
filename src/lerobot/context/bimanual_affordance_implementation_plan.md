# Bimanual Affordance System Implementation Plan

## Executive Summary

This plan extends the existing LeRobot bimanual robot system (BiSO100Follower/Leader) with visual affordance detection for complex LEGO manipulation tasks. We will build upon the established `left_`/`right_` prefix architecture while adding intelligent task detection and coordination.

## Current System Analysis

### Existing Bimanual Infrastructure ✅
- **BiSO100Follower/Leader**: Production-ready bimanual robot system
- **Prefix Architecture**: Clean `left_`/`right_` separation for actions/observations  
- **Composition Pattern**: Two SO100 arms working in coordination
- **Shared Cameras**: Multi-view perception already supported

### Current Affordance System ✅
- **gemini_perception.py**: Single-arm bbox detection with Gemini 2.0
- **record_bbox_multi.py**: Multi-camera recording with affordances
- **datasets/utils.py**: Feature creation pipeline for ML training

## Target Tasks

1. **LEGO Disassembly**: One arm holds base piece, other pulls apart components
2. **Handoff Tasks**: Transfer large objects between arms based on size/weight
3. **Parallel Sorting**: Both arms sort different objects simultaneously into bins

## Implementation Plan

### Phase 1: Bimanual Affordance Detection Engine

#### 1.1 Create `gemini_perception_bimanual.py`

**Core Innovation**: Task-aware affordance detection that assigns objects to specific arms based on scene context.

```python
def get_2D_bbox_bimanual(img, task_type=None, arm_positions=None) -> str:
    """
    Detects objects and assigns them to arms based on bimanual task requirements.
    
    Returns structured JSON:
    {
        "task_type": "disassembly|handoff|parallel_sort|sequential_fallback", 
        "coordination_required": true|false,
        "objects": [
            {
                "label": "red 2x4 LEGO brick",
                "box_2d": [y1, x1, y2, x2],  # Normalized [0,1] coordinates
                "arm_assignment": "red|white|both|either",
                "action_type": "pick|hold|pull|place|handoff_receive|handoff_give", 
                "priority": 1|2|3,  # Execution order
                "coordination_frame": "simultaneous|sequential|leader_follower"
            }
        ],
        "workspace_constraints": {
            "red_reachable_zone": [y1, x1, y2, x2],
            "white_reachable_zone": [y1, x1, y2, x2], 
            "handoff_zone": [y1, x1, y2, x2]
        }
    }
    """
```

**Key Features**:
- **Automatic Task Detection**: Scene analysis determines optimal bimanual strategy
- **Workspace Awareness**: Considers arm reach and collision avoidance
- **Coordination Planning**: Defines temporal relationships between arm actions
- **Pure Bimanual Focus**: Designed specifically for coordinated two-arm manipulation

**Example Gemini Prompts**:
```python
def get_disassembly_prompt():
    return """Identify LEGO assemblies that can be disassembled. 
    Mark the larger/base piece as "hold_target" for the red arm.
    Mark the piece to be removed as "pull_target" for the white arm.
    Also identify any loose pieces as "pick_red" or "pick_white" based on their position."""

def get_handoff_prompt(): 
    return """Identify objects for bimanual handoff based on size.
    Mark large pieces (4x4 or bigger) as "pick_red" (red arm picks first).
    Mark the handoff zone (empty space between arms) as "handoff_zone".
    Mark the final bins as "place_target"."""

def get_parallel_sort_prompt():
    return """Identify objects for parallel sorting between red and white arms.
    Assign objects on the left side of workspace as "pick_red".
    Assign objects on the right side of workspace as "pick_white".
    Mark sorting bins clearly for each arm."""
```

#### 1.2 Task-Specific Detection Logic

```python
# Task detection hierarchy (in order of complexity)
def detect_optimal_bimanual_task(scene_objects):
    if has_assembled_lego_pieces(scene_objects):
        return setup_disassembly_task(scene_objects)
    elif has_large_objects_needing_handoff(scene_objects):  
        return setup_handoff_task(scene_objects)
    elif has_multiple_sorting_targets(scene_objects):
        return setup_parallel_sort_task(scene_objects)
    else:
        return setup_default_bimanual_sort(scene_objects)
```

**Disassembly Task Logic**:
- Identify connected LEGO assemblies using spatial relationships
- Assign larger/base piece to `red_arm` as "hold_target"
- Assign removable piece to `white_arm` as "pull_target" 
- Ensure synchronized motion coordination

**Handoff Task Logic**:
- Detect objects too large/heavy for single arm manipulation
- Plan handoff trajectory through shared workspace
- Define handoff zones equidistant from both arms

**Parallel Sort Logic**:
- Partition objects by workspace accessibility
- Prevent conflicts when both arms target same object
- Optimize for maximum parallel efficiency

### Phase 2: Recording System Integration

#### 2.1 Create `record_bbox_bimanual.py`

**Integration Strategy**: Extend existing `record_bbox_multi.py` to support the BiSO100 bimanual robot with affordance-aware recording.

**Key Observation Space Extensions**:
```python
# Bimanual affordance observations (25 additional features)
bimanual_observation = {
    # Task context (3 features)
    'task_type': task_type_encoded,  # 0=disassembly, 1=handoff, 2=parallel_sort
    'coordination_required': 1.0 if coordination_required else 0.0,
    'primary_arm': 1.0 if primary_arm == 'red' else 0.0,  # Which arm leads
    
    # Red arm target bbox (8 features) 
    'red_target_y1_top': normalized_red_top[0],
    'red_target_x1_top': normalized_red_top[1], 
    'red_target_y2_top': normalized_red_top[2],
    'red_target_x2_top': normalized_red_top[3],
    'red_target_y1_front': normalized_red_front[0],
    'red_target_x1_front': normalized_red_front[1],
    'red_target_y2_front': normalized_red_front[2], 
    'red_target_x2_front': normalized_red_front[3],
    
    # White arm target bbox (8 features)
    'white_target_y1_top': normalized_white_top[0],
    'white_target_x1_top': normalized_white_top[1],
    'white_target_y2_top': normalized_white_top[2],
    'white_target_x2_top': normalized_white_top[3],
    'white_target_y1_front': normalized_white_front[0],
    'white_target_x1_front': normalized_white_front[1],
    'white_target_y2_front': normalized_white_front[2],
    'white_target_x2_front': normalized_white_front[3],
    
    # Action type encodings (6 features - one-hot encoded)
    'red_action_pick': 1.0 if red_action == 'pick' else 0.0,
    'red_action_hold': 1.0 if red_action == 'hold' else 0.0,
    'red_action_place': 1.0 if red_action == 'place' else 0.0,
    'white_action_pick': 1.0 if white_action == 'pick' else 0.0,
    'white_action_pull': 1.0 if white_action == 'pull' else 0.0,
    'white_action_place': 1.0 if white_action == 'place' else 0.0,
}
```

**Recording Workflow**:
1. Capture multi-view images using existing camera system
2. Run bimanual affordance detection on both views
3. Reconcile multi-view detections for consistency
4. Map affordances to BiSO100 action space using existing prefixes
5. Record enhanced observations with bimanual context

#### 2.2 Multi-View Consistency

**Challenge**: Ensure red/white arm assignments are consistent across top and front camera views.

**Solution**:
```python
def reconcile_multiview_affordances(top_view_result, front_view_result):
    """
    Merge affordance detections from multiple camera views.
    Prioritizes spatial consistency and arm reachability.
    """
    # Use 3D workspace mapping to resolve conflicts
    # Validate arm assignments against reachability constraints
    # Merge bounding boxes using geometric consistency
    return reconciled_bimanual_plan
```

### Phase 3: Dataset Integration

#### 3.1 Update `datasets/utils.py`

**Location**: Modify the `hw_to_dataset_features()` function around line 411.

```python
if joint_fts and prefix == "observation":
    # Original single-arm bbox features (8 coordinates)
    single_arm_states = ['pick_y1', 'pick_x1', 'pick_y2', 'pick_x2', 
                         'place_y1', 'place_x1', 'place_y2', 'place_x2']
    
    # New bimanual affordance features (25 coordinates)  
    bimanual_states = [
        # Task context
        'task_type', 'coordination_required', 'primary_arm',
        
        # Red arm targets  
        'red_target_y1_top', 'red_target_x1_top', 'red_target_y2_top', 'red_target_x2_top',
        'red_target_y1_front', 'red_target_x1_front', 'red_target_y2_front', 'red_target_x2_front',
        
        # White arm targets
        'white_target_y1_top', 'white_target_x1_top', 'white_target_y2_top', 'white_target_x2_top', 
        'white_target_y1_front', 'white_target_x1_front', 'white_target_y2_front', 'white_target_x2_front',
        
        # Action encodings
        'red_action_pick', 'red_action_hold', 'red_action_place',
        'white_action_pick', 'white_action_pull', 'white_action_place'
    ]
    
    # Use bimanual states for bimanual robots
    additional_states = bimanual_states
        
    joint_fts = {**joint_fts, **{name: float for name in additional_states}}
```

**Pure Bimanual Design**: Focused exclusively on red/white arm coordination without single-arm fallbacks.

### Phase 4: Advanced Coordination Features

#### 4.1 Temporal Coordination

**Challenge**: Some bimanual tasks require precise timing (e.g., one arm must reach position before other moves).

**Solution**: Add coordination checkpoints to observation space:
```python
'coordination_checkpoint': 1.0 if critical_timing_required else 0.0,
'red_arm_ready': 1.0 if red_arm_at_target else 0.0,
'white_arm_ready': 1.0 if white_arm_at_target else 0.0,
'sync_required': 1.0 if both_arms_must_move_together else 0.0
```

#### 4.2 Collision Avoidance

**Workspace Partitioning**: Define safe zones for each arm to prevent collisions during parallel operations.

**Dynamic Constraints**: Real-time adjustment of target assignments if collision risk detected.

#### 4.3 Error Recovery

**Smart Re-assignment**: If one arm cannot reach its target, dynamically reassign to the other arm while maintaining bimanual coordination.

**Collision Prevention**: Real-time adjustment of arm trajectories to prevent workspace conflicts.

## Implementation Sequence

### Milestone 1: Core Affordance Engine (Week 1-2)
1. ✅ Implement `gemini_perception_bimanual.py` with basic task detection
2. ✅ Create task-specific prompts for disassembly, handoff, sorting
3. ✅ Test affordance detection on sample LEGO scenes
4. ✅ Validate multi-view consistency algorithms

### Milestone 2: Recording Integration (Week 3-4) 
1. ✅ Create `record_bbox_bimanual.py` extending existing multi-camera system
2. ✅ Integrate with BiSO100 bimanual robot using established prefix architecture
3. ✅ Update `datasets/utils.py` with bimanual feature definitions
4. ✅ Record test datasets for each bimanual task type

### Milestone 3: Advanced Coordination (Week 5-6)
1. ✅ Implement temporal coordination and synchronization features  
2. ✅ Add collision avoidance and workspace partitioning
3. ✅ Create error recovery and re-assignment mechanisms
4. ✅ Comprehensive testing with real LEGO manipulation tasks

### Milestone 4: Optimization & Validation (Week 7-8)
1. ✅ Performance optimization for real-time operation
2. ✅ Validation across diverse LEGO scenarios
3. ✅ Documentation and example usage
4. ✅ Integration testing with policy training pipelines

## Success Criteria

### Technical Validation
- [ ] **Accuracy**: >90% correct arm assignment for each task type
- [ ] **Consistency**: <5% variance between multi-view detections  
- [ ] **Performance**: <200ms latency for affordance detection
- [ ] **Safety**: Zero collisions during parallel operations

### Task Performance  
- [ ] **Disassembly**: Successfully separate connected LEGO pieces
- [ ] **Handoff**: Smooth transfer of large objects between arms
- [ ] **Parallel Sort**: Both arms working simultaneously without conflicts
- [ ] **Smart Recovery**: Automatic re-assignment and trajectory adjustment when conflicts arise

### Integration Quality
- [ ] **Backward Compatibility**: Existing single-arm systems unaffected
- [ ] **Code Quality**: Follows LeRobot conventions and patterns
- [ ] **Documentation**: Clear examples and usage instructions
- [ ] **Testing**: Comprehensive unit and integration test coverage

## Risk Mitigation

### Technical Risks
- **Multi-view Inconsistency**: Robust geometric reconciliation algorithms
- **Coordination Complexity**: Start simple, add features incrementally  
- **Performance Bottlenecks**: Profiling and optimization from day one

### System Integration Risks  
- **Integration Safety**: Ensure clean integration with existing BiSO100 bimanual system
- **Hardware Dependencies**: Extensive testing with BiSO100 hardware
- **Gemini API Limits**: Implement caching and rate limiting

## Next Steps

1. **Immediate**: Get user approval for this implementation plan
2. **Phase 1**: Begin with `gemini_perception_bimanual.py` implementation
3. **Incremental**: Build and test each component individually  
4. **Integration**: Combine components into full bimanual affordance system

This plan leverages LeRobot's existing strengths while adding the intelligent coordination needed for complex bimanual LEGO manipulation tasks.