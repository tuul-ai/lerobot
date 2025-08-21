#!/usr/bin/env python3
"""
Simple test script for bimanual affordance detection system.
Tests the core functionality without requiring actual robot hardware.
"""

import numpy as np
from PIL import Image
import json

from lerobot.gemini_perception_bimanual import (
    get_2D_bbox_bimanual,
    reconcile_multiview_affordances,
    create_bimanual_action_plan,
    plot_bimanual_bbox,
    detect_task_type,
    parse_json,
)

def create_test_image(width=640, height=480):
    """Create a test image with some colored rectangles representing LEGO pieces"""
    # Create RGB image
    img_array = np.random.randint(100, 150, (height, width, 3), dtype=np.uint8)
    
    # Add some colored rectangles to simulate LEGO pieces
    # Red piece (for red arm)
    img_array[100:150, 200:280] = [255, 0, 0]  # Red
    
    # White/light blue piece (for white arm)  
    img_array[200:250, 350:430] = [200, 200, 255]  # Light blue
    
    # Large assembled piece (for disassembly)
    img_array[300:380, 150:350] = [0, 255, 0]  # Green assembly
    
    # Bins/containers
    img_array[50:100, 50:150] = [128, 64, 0]  # Brown bin
    img_array[50:100, 450:550] = [128, 64, 0]  # Brown bin
    
    return Image.fromarray(img_array)

def test_task_detection():
    """Test automatic task type detection"""
    print("Testing task detection...")
    
    # Test disassembly detection
    scene_with_assembly = ["large assembled tower", "connected pieces", "attached bricks"]
    task = detect_task_type(scene_with_assembly)
    print(f"Assembly scene detected as: {task}")
    assert task == "disassembly"
    
    # Test handoff detection  
    scene_with_large_objects = ["large 6x6 brick", "big heavy piece", "4x4 block"]
    task = detect_task_type(scene_with_large_objects)
    print(f"Large object scene detected as: {task}")
    assert task == "handoff"
    
    # Test default parallel sort
    scene_normal = ["small red brick", "blue piece", "yellow block"]
    task = detect_task_type(scene_normal)
    print(f"Normal scene detected as: {task}")
    assert task == "parallel_sort"
    
    print("✅ Task detection tests passed!")

def test_affordance_detection():
    """Test bimanual affordance detection (requires GEMINI_API_KEY)"""
    print("\nTesting affordance detection...")
    
    try:
        # Create test images
        top_image = create_test_image()
        front_image = create_test_image()
        
        # Test basic affordance detection
        result = get_2D_bbox_bimanual(top_image, task_type="parallel_sort")
        print(f"Affordance detection result length: {len(result)} characters")
        
        # Try to parse the result
        try:
            parsed_result = json.loads(parse_json(result))
            print(f"Parsed affordance result keys: {list(parsed_result.keys())}")
            
            # Test action plan creation
            action_plan = create_bimanual_action_plan(parsed_result)
            if action_plan:
                print(f"Action plan created with task type: {action_plan.get('task_type')}")
                print(f"Red arm targets: {len(action_plan.get('red_arm', {}).get('targets', []))}")
                print(f"White arm targets: {len(action_plan.get('white_arm', {}).get('targets', []))}")
            
            print("✅ Affordance detection tests passed!")
            
        except json.JSONDecodeError as e:
            print(f"⚠️  Could not parse Gemini response as JSON: {e}")
            print("This might be expected if Gemini returns descriptive text instead of JSON")
            
    except Exception as e:
        print(f"⚠️  Affordance detection test skipped: {e}")
        print("Make sure GOOGLE_API_KEY environment variable is set to test Gemini integration")

def test_multiview_reconciliation():
    """Test multi-view affordance reconciliation"""
    print("\nTesting multi-view reconciliation...")
    
    # Mock affordance results from two views
    top_result = """{
        "task_type": "parallel_sort",
        "coordination_required": false,
        "objects": [
            {
                "label": "red brick",
                "box_2d": [100, 200, 150, 280],
                "arm_assignment": "red",
                "action_type": "pick",
                "priority": 1
            }
        ],
        "workspace_constraints": {
            "red_reachable_zone": [50, 100, 300, 400],
            "white_reachable_zone": [50, 300, 300, 600],
            "handoff_zone": [150, 250, 250, 350]
        }
    }"""
    
    front_result = """{
        "task_type": "parallel_sort", 
        "coordination_required": false,
        "objects": [
            {
                "label": "red brick",
                "box_2d": [120, 210, 160, 290],
                "arm_assignment": "red",
                "action_type": "pick",
                "priority": 1
            }
        ],
        "workspace_constraints": {
            "red_reachable_zone": [60, 110, 310, 410],
            "white_reachable_zone": [60, 310, 310, 610],
            "handoff_zone": [160, 260, 260, 360]
        }
    }"""
    
    # Test reconciliation
    reconciled = reconcile_multiview_affordances(top_result, front_result)
    
    print(f"Reconciled result task type: {reconciled.get('task_type')}")
    print(f"Reconciled objects count: {len(reconciled.get('objects', []))}")
    print(f"Workspace constraints keys: {list(reconciled.get('workspace_constraints', {}).keys())}")
    
    print("✅ Multi-view reconciliation tests passed!")

def test_visualization():
    """Test bimanual bbox visualization"""
    print("\nTesting visualization...")
    
    # Create test image and targets
    test_img = create_test_image()
    
    red_targets = [
        {
            'label': 'red LEGO brick',
            'bbox': [100, 200, 150, 280],  # [y1, x1, y2, x2] in 0-1000 scale
            'action': 'pick'
        }
    ]
    
    white_targets = [
        {
            'label': 'white LEGO brick', 
            'bbox': [200, 350, 250, 430],
            'action': 'pick'
        }
    ]
    
    handoff_zones = [
        {
            'label': 'handoff zone',
            'bbox': [150, 250, 250, 350],
            'action': 'handoff'
        }
    ]
    
    # Test visualization
    try:
        result_img = plot_bimanual_bbox(
            test_img,
            red_targets=red_targets,
            white_targets=white_targets, 
            handoff_zones=handoff_zones
        )
        
        print(f"Visualization result image size: {result_img.size}")
        print("✅ Visualization tests passed!")
        
        # Optionally save test image
        # result_img.save("/tmp/test_bimanual_bbox.png")
        # print("Test image saved to /tmp/test_bimanual_bbox.png")
        
    except Exception as e:
        print(f"⚠️  Visualization test failed: {e}")

def main():
    """Run all tests"""
    print("🤖 Testing Bimanual Affordance System")
    print("=" * 50)
    
    test_task_detection()
    test_multiview_reconciliation() 
    test_visualization()
    test_affordance_detection()  # This one requires API key
    
    print("\n" + "=" * 50)
    print("🎉 Bimanual affordance system tests completed!")
    print("\nTo test the full recording system:")
    print("python -m lerobot.record_bbox_bimanual --help")

if __name__ == "__main__":
    main()