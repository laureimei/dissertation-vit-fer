#!/usr/bin/env python3
"""
Inference script for multimodal FER model
Test trained model with new images
"""

import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import argparse
import json
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns

from model import create_multimodal_vit_model
from dataset import get_rgb_transforms, get_thermal_transforms
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiModalFERInference:
    """Inference class for multimodal FER model"""
    
    def __init__(self, checkpoint_path: str, device: str = 'auto'):
        """
        Initialize inference model
        
        Args:
            checkpoint_path: Path to the trained model checkpoint
            device: Device to run inference ('cuda', 'cpu', or 'auto')
        """
        self.checkpoint_path = checkpoint_path
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Load checkpoint and model
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.config = self.checkpoint['config']
        self.mode = self.checkpoint['mode']
        
        # Create model
        self.model = create_multimodal_vit_model(
            mode=self.config['mode'],
            fusion_strategy=self.config.get('fusion_strategy', 'early'),
            fusion_type=self.config.get('fusion_type', 'concat'),
            fusion_layer=self.config.get('fusion_layer', 'feature'),
            model_name=self.config['model_name'],
            num_classes=self.config['num_classes'],
            dropout_rate=self.config['dropout_rate'],
            freeze_backbone=False,  # Set to False for inference
            use_gradient_checkpointing=False
        )
        
        # Load model weights
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Class names
        self.class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
        
        # Setup transforms
        self.rgb_transform = get_rgb_transforms(
            image_size=self.config['image_size'],
            is_training=False
        )
        self.thermal_transform = get_thermal_transforms(
            image_size=self.config['image_size'],
            is_training=False
        )
        
        logger.info(f"Model loaded successfully from {checkpoint_path}")
        logger.info(f"Mode: {self.mode}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def preprocess_image(self, image_path: str, modality: str = 'rgb') -> torch.Tensor:
        """
        Preprocess a single image
        
        Args:
            image_path: Path to the image
            modality: 'rgb' or 'thermal'
            
        Returns:
            Preprocessed image tensor
        """
        # Load image
        image = Image.open(image_path)
        
        if modality == 'rgb':
            image = image.convert('RGB')
            tensor = self.rgb_transform(image)
        else:  # thermal
            # Convert thermal to grayscale then to 3-channel
            if image.mode != 'L':
                image = image.convert('L')
            image = image.convert('RGB')
            tensor = self.thermal_transform(image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        return tensor.to(self.device)
    
    def predict_single(self, rgb_path: str = None, thermal_path: str = None) -> Dict:
        """
        Predict emotion for a single image or image pair
        
        Args:
            rgb_path: Path to RGB image (required for rgb and combined modes)
            thermal_path: Path to thermal image (required for thermal and combined modes)
            
        Returns:
            Dictionary with predictions and probabilities
        """
        with torch.no_grad():
            if self.mode == 'rgb':
                if rgb_path is None:
                    raise ValueError("RGB image path is required for RGB mode")
                
                rgb_tensor = self.preprocess_image(rgb_path, 'rgb')
                outputs = self.model(rgb_tensor)
                
            elif self.mode == 'thermal':
                if thermal_path is None:
                    raise ValueError("Thermal image path is required for thermal mode")
                
                thermal_tensor = self.preprocess_image(thermal_path, 'thermal')
                outputs = self.model(thermal_tensor)
                
            elif self.mode == 'combined':
                if rgb_path is None or thermal_path is None:
                    raise ValueError("Both RGB and thermal image paths are required for combined mode")
                
                rgb_tensor = self.preprocess_image(rgb_path, 'rgb')
                thermal_tensor = self.preprocess_image(thermal_path, 'thermal')
                outputs = self.model(rgb_tensor, thermal_tensor)
            
            logits = outputs['logits']
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        # Get top-3 predictions
        top3_probs, top3_indices = torch.topk(probabilities[0], 3)
        top3_predictions = [
            {
                'class': self.class_names[idx.item()],
                'probability': prob.item(),
                'confidence': prob.item() * 100
            }
            for prob, idx in zip(top3_probs, top3_indices)
        ]
        
        result = {
            'predicted_class': self.class_names[predicted_class],
            'confidence': confidence,
            'confidence_percent': confidence * 100,
            'all_probabilities': {
                self.class_names[i]: prob.item() 
                for i, prob in enumerate(probabilities[0])
            },
            'top3_predictions': top3_predictions,
            'mode': self.mode
        }
        
        return result
    
    def predict_batch(self, image_paths: List[Dict[str, str]]) -> List[Dict]:
        """
        Predict emotions for a batch of images
        
        Args:
            image_paths: List of dictionaries with 'rgb' and/or 'thermal' keys
            
        Returns:
            List of prediction results
        """
        results = []
        
        for i, paths in enumerate(image_paths):
            try:
                rgb_path = paths.get('rgb')
                thermal_path = paths.get('thermal')
                
                result = self.predict_single(rgb_path, thermal_path)
                result['image_index'] = i
                result['input_paths'] = paths
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing image {i}: {e}")
                results.append({
                    'image_index': i,
                    'input_paths': paths,
                    'error': str(e)
                })
        
        return results
    
    def visualize_prediction(self, rgb_path: str = None, thermal_path: str = None, 
                           save_path: str = None, show_plot: bool = True) -> Dict:
        """
        Visualize prediction with input image(s) and probability bar chart
        
        Args:
            rgb_path: Path to RGB image
            thermal_path: Path to thermal image
            save_path: Path to save the visualization
            show_plot: Whether to display the plot
            
        Returns:
            Prediction result
        """
        # Get prediction
        result = self.predict_single(rgb_path, thermal_path)
        
        # Create visualization
        if self.mode == 'combined':
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # RGB image
            if rgb_path:
                rgb_img = Image.open(rgb_path)
                axes[0].imshow(rgb_img)
                axes[0].set_title('RGB Image')
                axes[0].axis('off')
            
            # Thermal image
            if thermal_path:
                thermal_img = Image.open(thermal_path)
                axes[1].imshow(thermal_img, cmap='gray' if thermal_img.mode == 'L' else None)
                axes[1].set_title('Thermal Image')
                axes[1].axis('off')
            
            # Probabilities
            ax_prob = axes[2]
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Single image
            if rgb_path and self.mode == 'rgb':
                img = Image.open(rgb_path)
                axes[0].imshow(img)
                axes[0].set_title('RGB Image')
            elif thermal_path and self.mode == 'thermal':
                img = Image.open(thermal_path)
                axes[0].imshow(img, cmap='gray' if img.mode == 'L' else None)
                axes[0].set_title('Thermal Image')
            
            axes[0].axis('off')
            ax_prob = axes[1]
        
        # Plot probabilities
        emotions = self.class_names
        probabilities = [result['all_probabilities'][emotion] for emotion in emotions]
        
        bars = ax_prob.bar(emotions, probabilities, 
                          color=['red' if emotion == result['predicted_class'] else 'lightblue' 
                                for emotion in emotions])
        
        ax_prob.set_title(f'Emotion Probabilities\nPredicted: {result["predicted_class"]} '
                         f'({result["confidence_percent"]:.1f}%)')
        ax_prob.set_ylabel('Probability')
        ax_prob.set_ylim(0, 1)
        ax_prob.tick_params(axis='x', rotation=45)
        
        # Add probability labels on bars
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax_prob.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{prob:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return result


def main():
    parser = argparse.ArgumentParser(description='Inference for multimodal FER model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--rgb_image', type=str,
                        help='Path to RGB image')
    parser.add_argument('--thermal_image', type=str,
                        help='Path to thermal image')
    parser.add_argument('--image_list', type=str,
                        help='Path to JSON file with list of image paths')
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                        help='Output directory for results')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualization of predictions')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize inference model
    inferencer = MultiModalFERInference(args.checkpoint, args.device)
    
    if args.image_list:
        # Batch inference
        logger.info(f"Running batch inference from {args.image_list}")
        
        with open(args.image_list, 'r') as f:
            image_paths = json.load(f)
        
        results = inferencer.predict_batch(image_paths)
        
        # Save results
        results_file = os.path.join(args.output_dir, 'batch_predictions.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Batch results saved to {results_file}")
        
        # Print summary
        successful = [r for r in results if 'error' not in r]
        logger.info(f"Successfully processed {len(successful)}/{len(results)} images")
        
        if successful:
            predictions = [r['predicted_class'] for r in successful]
            logger.info("Prediction summary:")
            for emotion in inferencer.class_names:
                count = predictions.count(emotion)
                logger.info(f"  {emotion}: {count} images")
    
    else:
        # Single inference
        logger.info("Running single image inference")
        
        result = inferencer.predict_single(args.rgb_image, args.thermal_image)
        
        # Print results
        print("\n" + "="*50)
        print("EMOTION RECOGNITION RESULT")
        print("="*50)
        print(f"Mode: {result['mode']}")
        print(f"Predicted Emotion: {result['predicted_class']}")
        print(f"Confidence: {result['confidence_percent']:.2f}%")
        print("\nTop 3 Predictions:")
        for i, pred in enumerate(result['top3_predictions'], 1):
            print(f"  {i}. {pred['class']}: {pred['confidence']:.2f}%")
        
        print("\nAll Probabilities:")
        for emotion, prob in result['all_probabilities'].items():
            print(f"  {emotion}: {prob:.4f}")
        
        # Save results
        results_file = os.path.join(args.output_dir, 'single_prediction.json')
        with open(results_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Create visualization
        if args.visualize:
            viz_path = os.path.join(args.output_dir, 'prediction_visualization.png')
            inferencer.visualize_prediction(
                args.rgb_image, 
                args.thermal_image, 
                save_path=viz_path,
                show_plot=False
            )


if __name__ == "__main__":
    main()