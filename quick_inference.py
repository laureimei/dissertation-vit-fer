#!/usr/bin/env python3
"""
Quick inference script for single image testing
Usage examples:
    python quick_inference.py --checkpoint path/to/model.pth --rgb_image path/to/image.jpg
    python quick_inference.py --checkpoint path/to/model.pth --thermal_image path/to/image.jpg
    python quick_inference.py --checkpoint path/to/model.pth --rgb_image rgb.jpg --thermal_image thermal.jpg
"""

import argparse
import os
from inference import MultiModalFERInference


def main():
    parser = argparse.ArgumentParser(description='Quick inference for FER model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--rgb_image', type=str,
                        help='Path to RGB image')
    parser.add_argument('--thermal_image', type=str,
                        help='Path to thermal image')
    parser.add_argument('--save_viz', action='store_true',
                        help='Save visualization to file')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return
    
    if args.rgb_image and not os.path.exists(args.rgb_image):
        print(f"Error: RGB image not found: {args.rgb_image}")
        return
    
    if args.thermal_image and not os.path.exists(args.thermal_image):
        print(f"Error: Thermal image not found: {args.thermal_image}")
        return
    
    if not args.rgb_image and not args.thermal_image:
        print("Error: Please provide at least one image (--rgb_image or --thermal_image)")
        return
    
    try:
        # Load model
        print(f"Loading model from {args.checkpoint}...")
        inferencer = MultiModalFERInference(args.checkpoint)
        print(f"✓ Model loaded successfully!")
        print(f"Mode: {inferencer.mode}")
        
        # Validate inputs for model mode
        if inferencer.mode == 'rgb' and not args.rgb_image:
            print(f"Error: RGB image is required for RGB model")
            return
        elif inferencer.mode == 'thermal' and not args.thermal_image:
            print(f"Error: Thermal image is required for thermal model")
            return
        elif inferencer.mode == 'combined' and (not args.rgb_image or not args.thermal_image):
            print(f"Error: Both RGB and thermal images are required for combined model")
            return
        
        # Run inference
        print(f"\nRunning inference...")
        result = inferencer.predict_single(args.rgb_image, args.thermal_image)
        
        # Display results
        print("\n" + "="*60)
        print("🎭 EMOTION RECOGNITION RESULT")
        print("="*60)
        print(f"📊 Model Mode: {result['mode'].upper()}")
        print(f"😊 Predicted Emotion: {result['predicted_class'].upper()}")
        print(f"🎯 Confidence: {result['confidence_percent']:.1f}%")
        
        print(f"\n🏆 Top 3 Predictions:")
        for i, pred in enumerate(result['top3_predictions'], 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
            print(f"   {emoji} {pred['class']}: {pred['confidence']:.1f}%")
        
        print(f"\n📈 All Emotion Probabilities:")
        for emotion, prob in result['all_probabilities'].items():
            bar_length = int(prob * 20)  # Scale to 20 chars
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"   {emotion:>10}: {bar} {prob*100:5.1f}%")
        
        # Create visualization if requested
        if args.save_viz:
            viz_path = "prediction_result.png"
            print(f"\n📸 Creating visualization...")
            inferencer.visualize_prediction(
                args.rgb_image, 
                args.thermal_image,
                save_path=viz_path,
                show_plot=False
            )
            print(f"✓ Visualization saved as '{viz_path}'")
        
        print("\n" + "="*60)
        print("✅ INFERENCE COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error during inference: {e}")
        print("Please check your model checkpoint and image paths.")


if __name__ == "__main__":
    main()