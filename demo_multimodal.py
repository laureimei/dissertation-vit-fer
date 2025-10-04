#!/usr/bin/env python3
"""
Demo script for multimodal FER pipeline
Tests all three modes: RGB-only, Thermal-only, and Combined (with fusion strategies)
"""

import os
import torch
import argparse
from dataset import create_multimodal_data_loaders, analyze_multimodal_dataset
from model import create_multimodal_vit_model, get_optimizer_and_scheduler
from train import MultiModalFERTrainer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_dataset(data_dir: str, use_augmented: bool = False):
    """Test dataset loading for all modes"""
    print("="*60)
    print("TESTING DATASET MODULE")
    print("="*60)
    
    # Analyze dataset
    analyze_multimodal_dataset(data_dir, use_augmented=use_augmented)
    
    # Test data loaders for each mode
    for mode in ['rgb', 'thermal', 'combined']:
        print(f"\n--- Testing {mode.upper()} mode data loaders ---")
        try:
            train_loader, test_loader = create_multimodal_data_loaders(
                data_dir=data_dir,
                mode=mode,
                batch_size=4,
                image_size=224,
                use_augmented=use_augmented
            )
            
            print(f"✓ Data loaders created successfully")
            print(f"  Train batches: {len(train_loader)}")
            print(f"  Test batches: {len(test_loader)}")
            
            # Test loading a batch
            if len(train_loader) > 0:
                batch = next(iter(train_loader))
                if mode == 'combined':
                    data, labels = batch
                    rgb_shape = data['rgb'].shape
                    thermal_shape = data['thermal'].shape
                    print(f"  RGB batch shape: {rgb_shape}")
                    print(f"  Thermal batch shape: {thermal_shape}")
                    print(f"  Labels shape: {labels.shape}")
                else:
                    images, labels = batch
                    print(f"  Batch shape: {images.shape}")
                    print(f"  Labels shape: {labels.shape}")
            
        except Exception as e:
            print(f"✗ Error in {mode} mode: {e}")


def test_models(device: torch.device):
    """Test model creation for all configurations"""
    print("\n" + "="*60)
    print("TESTING MODEL MODULE")
    print("="*60)
    
    # Test configurations
    configs = [
        {'mode': 'rgb', 'description': 'RGB-only ViT'},
        {'mode': 'thermal', 'description': 'Thermal-only ViT'},
        {'mode': 'combined', 'fusion_strategy': 'early', 'fusion_type': 'concat', 'description': 'Early Fusion (Concat)'},
        {'mode': 'combined', 'fusion_strategy': 'early', 'fusion_type': 'add', 'description': 'Early Fusion (Add)'},
        {'mode': 'combined', 'fusion_strategy': 'late', 'fusion_type': 'concat', 'fusion_layer': 'feature', 'description': 'Late Fusion (Feature Concat)'},
        {'mode': 'combined', 'fusion_strategy': 'late', 'fusion_type': 'attention', 'fusion_layer': 'prediction', 'description': 'Late Fusion (Prediction Attention)'},
    ]
    
    batch_size = 2
    dummy_rgb = torch.randn(batch_size, 3, 224, 224).to(device)
    dummy_thermal = torch.randn(batch_size, 3, 224, 224).to(device)
    dummy_labels = torch.randint(0, 7, (batch_size,)).to(device)
    
    for config in configs:
        print(f"\n--- Testing {config['description']} ---")
        
        try:
            # Create model
            model = create_multimodal_vit_model(
                mode=config['mode'],
                fusion_strategy=config.get('fusion_strategy', 'early'),
                fusion_type=config.get('fusion_type', 'concat'),
                fusion_layer=config.get('fusion_layer', 'feature'),
                num_classes=7,
                dropout_rate=0.1
            )
            model.to(device)
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"✓ Model created successfully")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
            
            # Test forward pass
            with torch.no_grad():
                if config['mode'] == 'combined':
                    outputs = model(dummy_rgb, dummy_thermal, dummy_labels)
                else:
                    if config['mode'] == 'rgb':
                        outputs = model(dummy_rgb, dummy_labels)
                    else:  # thermal
                        outputs = model(dummy_thermal, dummy_labels)
                
                print(f"  Output logits shape: {outputs['logits'].shape}")
                print(f"  Loss: {outputs['loss']:.4f}")
                
        except Exception as e:
            print(f"✗ Error: {e}")


def test_training_setup(data_dir: str, device: torch.device, mode: str = 'rgb'):
    """Test training setup without actual training"""
    print(f"\n" + "="*60)
    print(f"TESTING TRAINING SETUP - {mode.upper()} MODE")
    print("="*60)
    
    try:
        # Configuration
        config = {
            'data_dir': data_dir,
            'output_dir': './test_experiments',
            'mode': mode,
            'fusion_strategy': 'early',
            'fusion_type': 'concat',
            'fusion_layer': 'feature',
            'model_name': 'google/vit-base-patch16-224-in21k',
            'num_classes': 7,
            'dropout_rate': 0.1,
            'freeze_backbone': False,
            'use_gradient_checkpointing': False,
            'batch_size': 4,
            'num_epochs': 2,  # Short test
            'learning_rate': 5e-5,
            'weight_decay': 0.01,
            'warmup_steps': 10,
            'gradient_clip_norm': 1.0,
            'image_size': 224,
            'num_workers': 2,
            'val_split': 0.2,
            'use_augmented': False,
            'use_class_weights': True,
            'early_stopping_patience': 5,
            'early_stopping_min_delta': 0.001,
            'use_wandb': False,
            'wandb_project': 'test-multimodal-fer'
        }
        
        # Create data loaders
        print("Creating data loaders...")
        train_loader, test_loader = create_multimodal_data_loaders(
            data_dir=config['data_dir'],
            mode=config['mode'],
            batch_size=config['batch_size'],
            image_size=config['image_size'],
            num_workers=config['num_workers'],
            val_split=config['val_split'],
            use_augmented=config['use_augmented']
        )
        
        print(f"✓ Data loaders created")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
        # Create model
        print("Creating model...")
        model = create_multimodal_vit_model(
            mode=config['mode'],
            fusion_strategy=config['fusion_strategy'],
            fusion_type=config['fusion_type'],
            fusion_layer=config['fusion_layer'],
            model_name=config['model_name'],
            num_classes=config['num_classes'],
            dropout_rate=config['dropout_rate'],
            freeze_backbone=config['freeze_backbone'],
            use_gradient_checkpointing=config['use_gradient_checkpointing']
        )
        model.to(device)
        print("✓ Model created")
        
        # Create optimizer and scheduler
        total_steps = len(train_loader) * config['num_epochs']
        optimizer, scheduler = get_optimizer_and_scheduler(
            model=model,
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'],
            warmup_steps=config['warmup_steps'],
            num_training_steps=total_steps
        )
        print("✓ Optimizer and scheduler created")
        
        # Create trainer
        trainer = MultiModalFERTrainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=config,
            mode=config['mode'],
            experiment_name=f'test_{mode}_demo'
        )
        print("✓ Trainer created")
        
        # Test one epoch of training (without full training loop)
        print("Testing one training step...")
        model.train()
        
        # Get one batch and test forward/backward pass
        batch = next(iter(train_loader))
        
        if mode == 'combined':
            data, labels = batch
            rgb_images = data['rgb'].to(device)
            thermal_images = data['thermal'].to(device)
            labels = labels.to(device)
            outputs = model(rgb_images, thermal_images, labels)
        else:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images, labels)
        
        loss = outputs['loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        print(f"✓ Training step completed")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Learning rate: {scheduler.get_last_lr()[0]:.6f}")
        
    except Exception as e:
        print(f"✗ Error in training setup: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Demo multimodal FER pipeline')
    parser.add_argument('--data_dir', type=str, default='data/vit/Data',
                        help='Path to data directory')
    parser.add_argument('--use_augmented', action='store_true',
                        help='Include augmented data in testing')
    parser.add_argument('--test_dataset', action='store_true', default=True,
                        help='Test dataset module')
    parser.add_argument('--test_models', action='store_true', default=True,
                        help='Test model module')
    parser.add_argument('--test_training', action='store_true', default=True,
                        help='Test training setup')
    parser.add_argument('--mode', type=str, default='rgb', choices=['rgb', 'thermal', 'combined'],
                        help='Mode for training test')
    
    args = parser.parse_args()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test dataset
    if args.test_dataset:
        test_dataset(args.data_dir, args.use_augmented)
    
    # Test models
    if args.test_models:
        test_models(device)
    
    # Test training setup
    if args.test_training:
        test_training_setup(args.data_dir, device, args.mode)
    
    print("\n" + "="*60)
    print("DEMO COMPLETED")
    print("="*60)
    print("\nAll major components have been tested!")
    print("\nTo start actual training, you can:")
    print("1. For RGB-only: python train.py (default)")
    print("2. For Thermal-only: modify config['mode'] = 'thermal' in train.py")
    print("3. For Combined: modify config['mode'] = 'combined' and set fusion options")
    print("\nExample combined configurations:")
    print("- Early fusion (concat): mode='combined', fusion_strategy='early', fusion_type='concat'")
    print("- Late fusion (attention): mode='combined', fusion_strategy='late', fusion_type='attention'")


if __name__ == "__main__":
    main()