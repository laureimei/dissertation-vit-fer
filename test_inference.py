#!/usr/bin/env python3
"""
Example script to test inference with sample images
"""

import os
import json
from inference import MultiModalFERInference
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_inference_examples():
    """Test inference with different scenarios"""
    
    # Path to your trained model
    checkpoint_path = "./experiments/multimodal_vit_fer_rgb_20250822_145743/best_model.pth"
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Please update the checkpoint_path to point to your trained model")
        return
    
    # Initialize inference
    try:
        inferencer = MultiModalFERInference(checkpoint_path)
        print(f"Model loaded successfully!")
        print(f"Mode: {inferencer.mode}")
        print(f"Device: {inferencer.device}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Example 1: Test with sample images from your dataset
    print("\n" + "="*60)
    print("EXAMPLE 1: Testing with existing dataset images")
    print("="*60)
    
    # Find some sample images from your data
    data_dir = "../../group-4-dissertation/data/vit/Data"
    
    if inferencer.mode == 'rgb':
        # Test RGB mode
        rgb_dir = os.path.join(data_dir, "RGB")
        if os.path.exists(rgb_dir):
            rgb_files = [f for f in os.listdir(rgb_dir) if f.startswith('R_') and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            if rgb_files:
                sample_rgb = os.path.join(rgb_dir, rgb_files[0])
                print(f"Testing RGB image: {sample_rgb}")
                
                try:
                    result = inferencer.predict_single(rgb_path=sample_rgb)
                    print_prediction_result(result)
                    
                    # Create visualization
                    inferencer.visualize_prediction(
                        rgb_path=sample_rgb,
                        save_path="./test_rgb_prediction.png",
                        show_plot=False
                    )
                    print("Visualization saved as 'test_rgb_prediction.png'")
                    
                except Exception as e:
                    print(f"Error during prediction: {e}")
    
    elif inferencer.mode == 'thermal':
        # Test Thermal mode
        thermal_dir = os.path.join(data_dir, "Thermal")
        if os.path.exists(thermal_dir):
            thermal_files = [f for f in os.listdir(thermal_dir) if f.startswith('T_') and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            if thermal_files:
                sample_thermal = os.path.join(thermal_dir, thermal_files[0])
                print(f"Testing Thermal image: {sample_thermal}")
                
                try:
                    result = inferencer.predict_single(thermal_path=sample_thermal)
                    print_prediction_result(result)
                    
                    # Create visualization
                    inferencer.visualize_prediction(
                        thermal_path=sample_thermal,
                        save_path="./test_thermal_prediction.png",
                        show_plot=False
                    )
                    print("Visualization saved as 'test_thermal_prediction.png'")
                    
                except Exception as e:
                    print(f"Error during prediction: {e}")
    
    elif inferencer.mode == 'combined':
        # Test Combined mode
        rgb_dir = os.path.join(data_dir, "RGB")
        thermal_dir = os.path.join(data_dir, "Thermal")
        
        if os.path.exists(rgb_dir) and os.path.exists(thermal_dir):
            rgb_files = [f for f in os.listdir(rgb_dir) if f.startswith('R_') and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            thermal_files = [f for f in os.listdir(thermal_dir) if f.startswith('T_') and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            if rgb_files and thermal_files:
                # Try to find a matching pair
                sample_rgb = os.path.join(rgb_dir, rgb_files[0])
                sample_thermal = os.path.join(thermal_dir, thermal_files[0])
                
                print(f"Testing RGB image: {sample_rgb}")
                print(f"Testing Thermal image: {sample_thermal}")
                
                try:
                    result = inferencer.predict_single(rgb_path=sample_rgb, thermal_path=sample_thermal)
                    print_prediction_result(result)
                    
                    # Create visualization
                    inferencer.visualize_prediction(
                        rgb_path=sample_rgb,
                        thermal_path=sample_thermal,
                        save_path="./test_combined_prediction.png",
                        show_plot=False
                    )
                    print("Visualization saved as 'test_combined_prediction.png'")
                    
                except Exception as e:
                    print(f"Error during prediction: {e}")
    
    # Example 2: Batch inference
    print("\n" + "="*60)
    print("EXAMPLE 2: Batch inference with multiple images")
    print("="*60)
    
    try:
        # Create a sample batch list
        batch_list = []
        
        if inferencer.mode == 'rgb':
            rgb_dir = os.path.join(data_dir, "RGB")
            if os.path.exists(rgb_dir):
                rgb_files = [f for f in os.listdir(rgb_dir) if f.startswith('R_') and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))][:3]
                for rgb_file in rgb_files:
                    batch_list.append({"rgb": os.path.join(rgb_dir, rgb_file)})
        
        elif inferencer.mode == 'thermal':
            thermal_dir = os.path.join(data_dir, "Thermal")
            if os.path.exists(thermal_dir):
                thermal_files = [f for f in os.listdir(thermal_dir) if f.startswith('T_') and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))][:3]
                for thermal_file in thermal_files:
                    batch_list.append({"thermal": os.path.join(thermal_dir, thermal_file)})
        
        elif inferencer.mode == 'combined':
            rgb_dir = os.path.join(data_dir, "RGB")
            thermal_dir = os.path.join(data_dir, "Thermal")
            if os.path.exists(rgb_dir) and os.path.exists(thermal_dir):
                rgb_files = [f for f in os.listdir(rgb_dir) if f.startswith('R_') and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))][:3]
                thermal_files = [f for f in os.listdir(thermal_dir) if f.startswith('T_') and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))][:3]
                
                for i in range(min(len(rgb_files), len(thermal_files))):
                    batch_list.append({
                        "rgb": os.path.join(rgb_dir, rgb_files[i]),
                        "thermal": os.path.join(thermal_dir, thermal_files[i])
                    })
        
        if batch_list:
            print(f"Running batch inference on {len(batch_list)} images...")
            batch_results = inferencer.predict_batch(batch_list)
            
            # Print batch results summary
            successful = [r for r in batch_results if 'error' not in r]
            print(f"Successfully processed {len(successful)}/{len(batch_results)} images")
            
            if successful:
                predictions = [r['predicted_class'] for r in successful]
                print("\nBatch Prediction Summary:")
                for emotion in inferencer.class_names:
                    count = predictions.count(emotion)
                    if count > 0:
                        print(f"  {emotion}: {count} images")
                
                # Save batch results
                with open("./test_batch_results.json", 'w') as f:
                    json.dump(batch_results, f, indent=2)
                print("Batch results saved as 'test_batch_results.json'")
        
    except Exception as e:
        print(f"Error in batch inference: {e}")
    
    print("\n" + "="*60)
    print("INFERENCE TESTING COMPLETED!")
    print("="*60)
    print("\nGenerated files:")
    print("- test_*_prediction.png: Visualization of single prediction")
    print("- test_batch_results.json: Batch prediction results")


def print_prediction_result(result):
    """Print formatted prediction result"""
    print(f"\nPredicted Emotion: {result['predicted_class']}")
    print(f"Confidence: {result['confidence_percent']:.2f}%")
    
    print("\nTop 3 Predictions:")
    for i, pred in enumerate(result['top3_predictions'], 1):
        print(f"  {i}. {pred['class']}: {pred['confidence']:.2f}%")


def create_sample_image_list():
    """Create a sample image list JSON file for batch inference"""
    data_dir = "../../group-4-dissertation/data/vit/Data"
    
    # Example for different modes
    sample_lists = {
        'rgb_only': [],
        'thermal_only': [],
        'combined': []
    }
    
    # RGB only samples
    rgb_dir = os.path.join(data_dir, "RGB")
    if os.path.exists(rgb_dir):
        rgb_files = [f for f in os.listdir(rgb_dir) if f.startswith('R_') and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))][:5]
        sample_lists['rgb_only'] = [{"rgb": os.path.join(rgb_dir, f)} for f in rgb_files]
    
    # Thermal only samples
    thermal_dir = os.path.join(data_dir, "Thermal")
    if os.path.exists(thermal_dir):
        thermal_files = [f for f in os.listdir(thermal_dir) if f.startswith('T_') and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))][:5]
        sample_lists['thermal_only'] = [{"thermal": os.path.join(thermal_dir, f)} for f in thermal_files]
    
    # Combined samples
    if sample_lists['rgb_only'] and sample_lists['thermal_only']:
        combined = []
        for i in range(min(len(sample_lists['rgb_only']), len(sample_lists['thermal_only']))):
            combined.append({
                "rgb": sample_lists['rgb_only'][i]["rgb"],
                "thermal": sample_lists['thermal_only'][i]["thermal"]
            })
        sample_lists['combined'] = combined
    
    # Save sample lists
    with open("./sample_image_lists.json", 'w') as f:
        json.dump(sample_lists, f, indent=2)
    
    print("Sample image lists created in 'sample_image_lists.json'")
    return sample_lists


if __name__ == "__main__":
    print("Creating sample image lists...")
    create_sample_image_list()
    
    print("\nStarting inference testing...")
    test_inference_examples()