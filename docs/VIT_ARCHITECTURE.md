# ViT Architecture Documentation

### Table of Contents

- [Overview](#overview)
- [Single Modal ViT (ViTForFER)](#single-modal-vit-vitforfer)
- [Early Fusion ViT (EarlyFusionViT)](#early-fusion-vit-earlyfusionvit)
  - [Fusion Type](#fusion-types)
- [Late Fusion ViT (LateFusionViT)](#late-fusion-vit-latefusionvit)
  - [Fusion Layer](#fusion-layer-details)
- [How Fine-tuning Works](#how-fine-tuning-works-)
- [Usage / Examples](#usageexamples)
- [Training Results](#results)
___ 

### Overview
A comprehensive implementation of Vision Transformer (ViT) for Facial Expression Recognition (FER) supporting RGB, Thermal, and Combined (RGB+Thermal) modalities with multiple fusion strategies.

Full detailed documentation can be found in Vision Transformers section in the [GitBook Documentation](https://university-of-the-west-england.gitbook.io/university-of-the-west-england-docs/model-documentation/vision-transformers-vit#architecture)

### Fusion Strategies

#### Early Fusion
- **Input-level fusion**: Concatenates RGB and Thermal images
- **Architecture**: Single ViT encoder processes fused input
- **Best for**: Learning joint representations from the start

#### Late Fusion
- **Feature/Prediction-level fusion**: Separate ViT encoders for each modality
- **Architecture**: Two ViT encoders + fusion layer
- **Best for**: Learning modality-specific features first

### Fusion Types
- **Concatenation**: `torch.cat([rgb_features, thermal_features], dim=1)`
- **Addition**: `rgb_features + thermal_features`
- **Attention**: Multi-head attention mechanism for feature weighting


# Single Modal ViT (ViTForFER)
- Uses pre-trained google/vit-base-patch16-224-in21k
- Architecture: 12 layers, 768 hidden size, 12 attention heads
- Custom classifier: 2-layer MLP with LayerNorm and dropout
- Input: 224x224 RGB or thermal images

<img src="evaluation_results/RGB/single_modal_archi.jpg" style="height:500px;">

# Early Fusion ViT (EarlyFusionViT)
- Combines RGB + thermal at input level
- Concat mode: 6-channel input (RGB=3 + Thermal=3)
- Add mode: Element-wise addition of RGB and thermal
- Modifies patch embedding layer for multi-channel input

In Early Fusion, RGB and thermal images are merged before entering the ViT encoder. There are three common fusion types:

<img src="evaluation_results/Combined_EarlyFusion/early_fusion_archi.jpg" style="width:470px;">

### Fusion Types
   - **Concat**  
     Combines RGB and thermal channels by concatenating them, resulting in a 6-channel input (3 RGB + 3 thermal).  
     The patch embedding layer is modified to handle this multi-channel input, allowing the model to learn joint features from both modalities right from the start.
   

   - **Add**  
     Combines RGB and thermal images by adding their pixel values element-wise, producing a single 3-channel image.  
     The model processes this fused image as normal, blending information from both modalities at the input level.
   

   - **Attention**  
     Uses an attention mechanism to fuse RGB and thermal inputs dynamically at the input token level.  
     This lets the model learn how much weight to assign each modality for every patch, improving fusion by focusing on the most informative features.
   


# Late Fusion ViT (LateFusionViT)

- Separate encoders process RGB and thermal inputs independently.
- Fusion happens either at the feature level (combining encoded features) or prediction level (combining outputs of separate classifiers).
- Fusion strategies control how information from both modalities is merged.
  - The fusion type (e.g., concat, add, attention) defines how features or predictions are combined adaptively.

### Fusion Layer Details
- **Feature Fusion**: Combines the feature outputs from each modality’s ViT encoder before passing to a shared classifier.


- **Prediction Fusion**: Applies separate classifiers on each modality’s features, then combines the predictions (e.g., averaging or weighted sum).

<img src="evaluation_results/Combined_LateFusion/late_fusion_archi.jpg"  style="width:470px;">


##  How Fine-tuning Works 
 
Fine-tuning Strategies
 
1. Classifier-only fine-tuning (freeze backbone)
model = ViTForFER(freeze_backbone=True)
 
2. Full fine-tuning (train entire model)
model = ViTForFER(freeze_backbone=False)
 
3. Progressive unfreezing
model = ViTForFER(freeze_backbone=True)

4. Train classifier first, then:
model.unfreeze_backbone()
 
## Key Fine-tuning Features
Custom Classifier Head:

```
nn.Sequential(
nn.LayerNorm(768), # Normalize features
nn.Dropout(0.1), # Prevent overfitting
nn.Linear(768, 384), # Hidden layer
nn.GELU(), # Activation
nn.Dropout(0.05), # More dropout
nn.Linear(384, 7) # Output 7 emotions
)
```
 
Different Learning Rates:
- Backbone: learning_rate * 0.1 (lower for pre-trained weights).
- Classifier: learning_rate (higher for new weights).
 
Memory Optimisation:
- Gradient checkpointing to reduce memory usage.
- Supports freezing backbone to save compute.

## Usage/Examples
 
1. Single RGB Model
 
   Create model
   ```
   model = create_multimodal_vit_model(mode='rgb')
    ```
   For fine-tuning
   ```
   model = ViTForFER(
   model_name="google/vit-base-patch16-224-in21k",
   num_classes=7,
   freeze_backbone=False, # Full fine-tuning
   dropout_rate=0.1
   )
   ```
    
2. Multi-modal Model
 
   Early fusion (concat RGB + thermal)
   ```
   model = create_multimodal_vit_model(
   mode='combined',
   fusion_strategy='early',
   fusion_type='concat'
   )
   ```
 
   Late fusion (separate encoders)
   ```
   model = create_multimodal_vit_model(
   mode='combined', 
   fusion_strategy='late',
   fusion_type='attention',
   fusion_layer='feature'
   )
   ```
 
3. Smart Optimizer Setup
    ```
   optimizer, scheduler = get_optimizer_and_scheduler(
   model=model,
   learning_rate=5e-5,
   warmup_steps=1000,
   num_training_steps=10000
   )
   ```

##  Fine-tuning Process

1. Load pre-trained weights from HuggingFace.
2. Replace classifier head with custom 7-class head.
3. Optionally freeze backbone for initial training.
4. Use different learning rates for backbone vs classifier.
5. Apply cosine scheduler with warmup.
6. Gradually unfreeze layers if needed.


### Results
All collected results per modalities can be found in ```docs``` folder and look for [VIT_RESULTS.md](https://gitlab.uwe.ac.uk/lmr2-sanejo/group-4-dissertation/-/blob/feature4/vit_model/docs/VIT_RESULTS.md)

# Authors
- Fiorella Scarpino (21010043)
- May Sanejo (15006280)
- Soumia Kouadri (24058628)
