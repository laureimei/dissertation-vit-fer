
# ViT Results
Here is the collection of Vision Transformers results for FER.


### Table of Contents 
Modalities Results
- [RGB Modality](#rgb-modality)
  - [Training Time](#rgb-training-time) 
  - [Training history](#rgb-training-history)
  - [Classification report](#rgb-classification-report)
  - [Confusion Matrix](#rgb-confusion-matrix)
  - [Per-class metrics](#rgb-per-class-metrics)

- [Thermal Modality](#thermal-modality)
  - [Training Time](#thermal-training-time) 
  - [Training history](#thermal-training-history)
  - [Classification report](#thermal-classification-report)
  - [Confusion Matrix](#thermal-confusion-matrix)
  - [Per-class metrics](#thermal-per-class-metrics)

- [Multimodal Early Fusion](#multi-modal-early-fusion-concat)
  - [Training Time](#multi-modal-early-fusion-training-time) 
  - [Training history](#multi-modal-early-fusion-training-history)
  - [Classification report](#multi-modal-early-fusion-classification-report)
  - [Confusion Matrix](#multi-modal-early-fusion-confusion-matrix)
  - [Per-class metrics](#multi-modal-early-fusion-per-class-metrics)

- [Multimodal Late Fusion](#multi-modal-late-fusion-attention-features)
  - [Training Time](#multi-modal-late-fusion-training-time) 
  - [Training history](#multi-modal-late-fusion-training-history)
  - [Classification report](#multi-modal-late-fusion-classification-report)
  - [Confusion Matrix](#multi-modal-late-fusion-confusion-matrix)
  - [Per-class metrics](#multi-modal-late-fusion-per-class-metrics)

___ 

## RGB Modality

#### RGB Training Time
<img src="evaluation_results/RGB/30Ep_single_training_rgb/multimodal_vit_fer_rgb_20250822_145743/training_min_rgb.png">

#### RGB Training History
<img src="evaluation_results/RGB/30Ep_single_training_rgb/multimodal_vit_fer_rgb_20250822_145743/training_history.png">

#### RGB Classification Report
<img src="evaluation_results/RGB/30Ep_single_training_rgb/multimodal_vit_fer_rgb_20250822_145743/rgb_classification_table.png">

#### RGB Confusion Matrix
<img src="evaluation_results/RGB/30Ep_single_training_rgb/multimodal_vit_fer_rgb_20250822_145743/confusion_matrix_normalized.png">

#### RGB Per-class metrics
<img src="evaluation_results/RGB/30Ep_single_training_rgb/multimodal_vit_fer_rgb_20250822_145743/per_class_metrics.png">



## Thermal Modality

#### Thermal Training Time
<img src="evaluation_results/Thermal/30Ep_single_training_thermal/multimodal_vit_fer_thermal_20250822_193734/training_min_thermal.png">

#### Thermal Training History
<img src="evaluation_results/Thermal/30Ep_single_training_thermal/multimodal_vit_fer_thermal_20250822_193734/training_history.png">

#### Thermal Classification Report
<img src="evaluation_results/Thermal/30Ep_single_training_thermal/multimodal_vit_fer_thermal_20250822_193734/thermal_classification_table.png">

#### Thermal Confusion Matrix
<img src="evaluation_results/Thermal/30Ep_single_training_thermal/multimodal_vit_fer_thermal_20250822_193734/confusion_matrix_normalized.png">

#### Thermal Per-class metrics
<img src="evaluation_results/Thermal/30Ep_single_training_thermal/multimodal_vit_fer_thermal_20250822_193734/per_class_metrics.png">



## Multi-modal Early Fusion (Concat)]

#### Multi-modal Early Fusion Training Time
<img src="evaluation_results/Combined_EarlyFusion/30Ep_training_early_fusion/multimodal_vit_fer_combined_20250821_104017/training_min_earlyfusion.png">

#### Multi-modal Early Fusion Training History
<img src="evaluation_results/Combined_EarlyFusion/30Ep_training_early_fusion/multimodal_vit_fer_combined_20250821_104017/training_history.png">

#### Multi-modal Early Fusion Classification Report
<img src="evaluation_results/Combined_EarlyFusion/30Ep_training_early_fusion/multimodal_vit_fer_combined_20250821_104017/combinedEarly_classification_table.png">

#### Multi-modal Early Fusion Confusion Matrix
<img src="evaluation_results/Combined_EarlyFusion/30Ep_training_early_fusion/multimodal_vit_fer_combined_20250821_104017/confusion_matrix_normalized.png">

#### Multi-modal Early Fusion Per-class metrics
<img src="evaluation_results/Combined_EarlyFusion/30Ep_training_early_fusion/multimodal_vit_fer_combined_20250821_104017/per_class_metrics.png">


## Multi-modal Late Fusion (Attention, Features)

#### Multi-modal Late Fusion Training Time
<img src="evaluation_results/Combined_LateFusion/30Ep_training_late_fusion/multimodal_vit_fer_combined_20250824_124943/training_min_latefusion.png">

#### Multi-modal Late Fusion Training History
<img src="evaluation_results/Combined_LateFusion/30Ep_training_late_fusion/multimodal_vit_fer_combined_20250824_124943/training_history.png">

#### Multi-modal Late Fusion Classification Report
<img src="evaluation_results/Combined_LateFusion/30Ep_training_late_fusion/multimodal_vit_fer_combined_20250824_124943/combinedLate_classification_table.png">

#### Multi-modal Late Fusion Confusion Matrix
<img src="evaluation_results/Combined_LateFusion/30Ep_training_late_fusion/multimodal_vit_fer_combined_20250824_124943/confusion_matrix_normalized.png">

#### Multi-modal Late Fusion Per-class metrics
<img src="evaluation_results/Combined_LateFusion/30Ep_training_late_fusion/multimodal_vit_fer_combined_20250824_124943/per_class_metrics.png">
