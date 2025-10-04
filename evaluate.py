import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import Dict, List, Tuple, Optional

from dataset import create_data_loaders, FERDataset
from model import create_vit_model
from utils import PerformanceAnalyzer, load_checkpoint


class ModelEvaluator:
    """Comprehensive model evaluation for FER ViT model"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        class_names: List[str]
    ):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.analyzer = PerformanceAnalyzer(class_names)
    
    def evaluate_on_dataloader(
        self,
        dataloader: torch.utils.data.DataLoader,
        return_predictions: bool = True
    ) -> Dict:
        """Evaluate model on a dataloader"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_logits = []
        total_loss = 0.0
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Evaluating"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(pixel_values=images, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                if return_predictions:
                    all_logits.extend(logits.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        
        # Per-class metrics
        f1_per_class = f1_score(all_labels, all_predictions, average=None)
        precision_per_class = precision_score(all_labels, all_predictions, average=None)
        recall_per_class = recall_score(all_labels, all_predictions, average=None)
        
        results = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'f1_per_class': f1_per_class.tolist(),
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'predictions': all_predictions,
            'labels': all_labels,
            'logits': all_logits if return_predictions else None
        }
        
        return results
    
    def evaluate_single_image(
        self,
        image_path: str,
        transform: Optional[torch.nn.Module] = None
    ) -> Tuple[int, np.ndarray, float]:
        """Evaluate model on a single image"""
        from PIL import Image
        from dataset import get_transforms
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        if transform is None:
            transform = get_transforms(224, is_training=False)
        
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(pixel_values=image_tensor)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(logits, dim=-1).item()
            confidence = probabilities[0, predicted_class].item()
        
        return predicted_class, probabilities[0].cpu().numpy(), confidence
    
    def compare_models(
        self,
        other_models: Dict[str, torch.nn.Module],
        test_loader: torch.utils.data.DataLoader,
        save_dir: str
    ):
        """Compare multiple models on test set"""
        results = {}
        
        # Evaluate main model
        main_results = self.evaluate_on_dataloader(test_loader, return_predictions=False)
        results['main_model'] = {
            'accuracy': main_results['accuracy'],
            'f1': main_results['f1'],
            'precision': main_results['precision'],
            'recall': main_results['recall']
        }
        
        # Evaluate other models
        for name, model in other_models.items():
            evaluator = ModelEvaluator(model, self.device, self.class_names)
            model_results = evaluator.evaluate_on_dataloader(test_loader, return_predictions=False)
            results[name] = {
                'accuracy': model_results['accuracy'],
                'f1': model_results['f1'],
                'precision': model_results['precision'],
                'recall': model_results['recall']
            }
        
        # Plot comparison
        self._plot_model_comparison(results, save_dir)
        
        return results
    
    def _plot_model_comparison(self, results: Dict, save_dir: str):
        """Plot comparison between models"""
        models = list(results.keys())
        metrics = ['accuracy', 'f1', 'precision', 'recall']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [results[model][metric] for model in models]
            axes[i].bar(models, values, alpha=0.7)
            axes[i].set_title(f'{metric.capitalize()} Comparison')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].set_ylim(0, 1)
            
            # Add value labels on bars
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_misclassifications(
        self,
        test_results: Dict,
        save_dir: str,
        top_k: int = 5
    ):
        """Analyze misclassified samples"""
        predictions = test_results['predictions']
        labels = test_results['labels']
        
        # Find misclassified samples
        misclassified_indices = []
        for i, (pred, true) in enumerate(zip(predictions, labels)):
            if pred != true:
                misclassified_indices.append(i)
        
        # Group by true class and predicted class
        confusion_analysis = {}
        for idx in misclassified_indices:
            true_class = labels[idx]
            pred_class = predictions[idx]
            
            key = f"{self.class_names[true_class]}_to_{self.class_names[pred_class]}"
            if key not in confusion_analysis:
                confusion_analysis[key] = []
            confusion_analysis[key].append(idx)
        
        # Get top confusion pairs
        top_confusions = sorted(
            confusion_analysis.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:top_k]
        
        # Save analysis
        analysis = {
            'total_misclassified': len(misclassified_indices),
            'total_samples': len(predictions),
            'misclassification_rate': len(misclassified_indices) / len(predictions),
            'top_confusion_pairs': [
                {
                    'pair': pair,
                    'count': len(indices),
                    'percentage': len(indices) / len(misclassified_indices) * 100
                }
                for pair, indices in top_confusions
            ]
        }
        
        with open(os.path.join(save_dir, 'misclassification_analysis.json'), 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return analysis
    
    def generate_attention_visualizations(
        self,
        sample_images: List[str],
        save_dir: str,
        layer_idx: int = -1
    ):
        """Generate attention visualizations for sample images"""
        from dataset import get_transforms
        from PIL import Image
        
        transform = get_transforms(224, is_training=False)
        
        for i, img_path in enumerate(sample_images):
            # Load image
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(self.device)
            
            # Get attention weights
            attention_weights = self.model.get_attention_weights(image_tensor, layer_idx)
            
            # Average attention across heads
            attention = attention_weights[0].mean(dim=0)  # [num_patches + 1, num_patches + 1]
            
            # Remove CLS token and reshape
            patch_attention = attention[0, 1:].reshape(14, 14)  # Assuming 224x224 -> 14x14 patches
            
            # Plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            
            # Original image
            ax1.imshow(image)
            ax1.set_title('Original Image')
            ax1.axis('off')
            
            # Attention heatmap
            im = ax2.imshow(patch_attention.cpu().numpy(), cmap='hot', interpolation='nearest')
            ax2.set_title('Attention Heatmap')
            ax2.axis('off')
            plt.colorbar(im, ax=ax2)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'attention_visualization_{i}.png'), dpi=300, bbox_inches='tight')
            plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate ViT model on FER dataset')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/vit', help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--model_name', type=str, default='google/vit-base-patch16-224-in21k', help='Model name')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    _, _, test_loader = create_data_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    model = create_vit_model(
        model_name=args.model_name,
        num_classes=7,
        dropout_rate=0.1,
        freeze_backbone=False
    )
    
    # Load checkpoint
    checkpoint = load_checkpoint(args.checkpoint_path, model)
    model.to(device)
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Create evaluator
    class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    evaluator = ModelEvaluator(model, device, class_names)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = evaluator.evaluate_on_dataloader(test_loader)
    
    # Generate comprehensive analysis
    print("Generating comprehensive analysis...")
    analysis = evaluator.analyzer.generate_detailed_analysis(
        test_results['labels'],
        test_results['predictions'],
        args.output_dir
    )
    
    # Analyze misclassifications
    print("Analyzing misclassifications...")
    misc_analysis = evaluator.analyze_misclassifications(test_results, args.output_dir)
    
    # Print results
    print("\n=== Evaluation Results ===")
    print(f"Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"Test F1-Score: {test_results['f1']:.4f}")
    print(f"Test Precision: {test_results['precision']:.4f}")
    print(f"Test Recall: {test_results['recall']:.4f}")
    
    print("\n=== Per-Class Results ===")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: F1={test_results['f1_per_class'][i]:.4f}, "
              f"Precision={test_results['precision_per_class'][i]:.4f}, "
              f"Recall={test_results['recall_per_class'][i]:.4f}")
    
    print(f"\n=== Misclassification Analysis ===")
    print(f"Total misclassified: {misc_analysis['total_misclassified']}/{misc_analysis['total_samples']}")
    print(f"Misclassification rate: {misc_analysis['misclassification_rate']:.4f}")
    
    print("\nTop confusion pairs:")
    for pair_info in misc_analysis['top_confusion_pairs']:
        print(f"  {pair_info['pair']}: {pair_info['count']} ({pair_info['percentage']:.1f}%)")
    
    # Save final results
    final_results = {
        'test_metrics': {k: v for k, v in test_results.items() if isinstance(v, (int, float, list))},
        'misclassification_analysis': misc_analysis,
        'checkpoint_info': {
            'epoch': checkpoint['epoch'],
            'checkpoint_path': args.checkpoint_path
        }
    }
    
    with open(os.path.join(args.output_dir, 'evaluation_summary.json'), 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print(f"\nEvaluation completed! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()