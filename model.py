import torch
import torch.nn as nn
from transformers import ViTConfig, ViTModel, ViTForImageClassification
from transformers import AutoImageProcessor
from typing import Optional, Dict, Any, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
    References from Hugging Face Transformers ViT documentation:
    https://huggingface.co/docs/transformers/en/model_doc/vit
"""

class ViTForFER(nn.Module):
    """
    Vision Transformer for Facial Expression Recognition
    Fine-tuned on FER dataset with 7 emotion classes
    """
    
    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224-in21k",
        num_classes: int = 7,
        dropout_rate: float = 0.1,
        freeze_backbone: bool = False,
        use_gradient_checkpointing: bool = False
    ):
        """
        Args:
            model_name: Pre-trained ViT model name from HuggingFace
            num_classes: Number of emotion classes (7 for FER)
            dropout_rate: Dropout rate for the classifier head
            freeze_backbone: Whether to freeze the backbone during fine-tuning
            use_gradient_checkpointing: Whether to use gradient checkpointing
        """
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.freeze_backbone = freeze_backbone
        
        # Load pre-trained ViT configuration
        self.config = ViTConfig.from_pretrained(model_name)
        self.config.num_labels = num_classes
        self.config.id2label = {
            0: "angry", 1: "disgust", 2: "fear", 3: "happy",
            4: "neutral", 5: "sad", 6: "surprised"
        }
        self.config.label2id = {v: k for k, v in self.config.id2label.items()}
        
        # Initialise ViT model
        self.vit = ViTForImageClassification.from_pretrained(
            model_name,
            config=self.config,
            ignore_mismatched_sizes=True
        )
        
        # Enable gradient checkpointing if requested
        if use_gradient_checkpointing:
            self.vit.gradient_checkpointing_enable()
        
        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()
            
        # Replace classifier head with custom one
        self._replace_classifier_head()
        
        # Initialize image processor
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        
    def _freeze_backbone(self):
        """Freeze the ViT backbone parameters"""
        for param in self.vit.vit.parameters():
            param.requires_grad = False
        logger.info("ViT backbone frozen")
    
    def _replace_classifier_head(self):
        """Replace the classifier head with a custom one"""
        hidden_size = self.config.hidden_size
        
        # Custom classifier head with dropout and layer normalization
        self.vit.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate / 2),
            nn.Linear(hidden_size // 2, self.num_classes)
        )
        
        # Initialize weights
        self._init_classifier_weights()
        
    def _init_classifier_weights(self):
        """Initialize classifier weights"""
        for module in self.vit.classifier.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
    
    def forward(self, pixel_values: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """
        Forward pass
        
        Args:
            pixel_values: Input images tensor
            labels: Optional labels for computing loss
            
        Returns:
            Dictionary containing logits, loss (if labels provided), and other outputs
        """
        outputs = self.vit(pixel_values=pixel_values, labels=labels)
        return outputs
    
    def get_features(self, pixel_values: torch.Tensor):
        """
        Get features from ViT backbone (before classifier)
        
        Args:
            pixel_values: Input images tensor
            
        Returns:
            Features tensor from ViT backbone
        """
        with torch.no_grad():
            outputs = self.vit.vit(pixel_values=pixel_values)
            # Get the [CLS] token representation
            features = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)
            return features
    
    def get_attention_weights(self, pixel_values: torch.Tensor, layer_idx: int = -1):
        """
        Get attention weights for visualisation
        
        Args:
            pixel_values: Input images tensor
            layer_idx: Which layer's attention to return (-1 for last layer)
            
        Returns:
            Attention weights tensor
        """
        with torch.no_grad():
            outputs = self.vit.vit(pixel_values=pixel_values, output_attentions=True)
            attentions = outputs.attentions
            return attentions[layer_idx]
    
    def unfreeze_backbone(self):
        """Unfreeze the backbone for full fine-tuning"""
        for param in self.vit.vit.parameters():
            param.requires_grad = True
        self.freeze_backbone = False
        logger.info("ViT backbone unfrozen")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "freeze_backbone": self.freeze_backbone,
            "dropout_rate": self.dropout_rate,
            "image_size": self.config.image_size,
            "patch_size": self.config.patch_size,
            "hidden_size": self.config.hidden_size,
            "num_attention_heads": self.config.num_attention_heads,
            "num_hidden_layers": self.config.num_hidden_layers
        }


class EarlyFusionViT(nn.Module):
    """
    Early Fusion ViT: Concatenates RGB and Thermal images at input level
    """
    
    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224-in21k",
        num_classes: int = 7,
        dropout_rate: float = 0.1,
        freeze_backbone: bool = False,
        use_gradient_checkpointing: bool = False,
        fusion_type: str = "concat"
    ):
        """
        Args:
            model_name: Pre-trained ViT model name
            num_classes: Number of emotion classes
            dropout_rate: Dropout rate for classifier
            freeze_backbone: Whether to freeze backbone
            use_gradient_checkpointing: Whether to use gradient checkpointing
            fusion_type: How to fuse RGB and Thermal ("concat" or "add")
        """
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.freeze_backbone = freeze_backbone
        self.fusion_type = fusion_type
        
        # Load pre-trained ViT configuration
        self.config = ViTConfig.from_pretrained(model_name)
        self.config.num_labels = num_classes
        self.config.id2label = {
            0: "angry", 1: "disgust", 2: "fear", 3: "happy",
            4: "neutral", 5: "sad", 6: "surprised"
        }
        self.config.label2id = {v: k for k, v in self.config.id2label.items()}
        
        # Modify input channels based on fusion type
        if fusion_type == "concat":
            # RGB (3) + Thermal (3) = 6 channels
            self.input_channels = 6
            # Modify config to handle 6-channel input
            self.config.num_channels = 6
        else:  # add
            # RGB and Thermal both have 3 channels, output is 3 channels
            self.input_channels = 3
        
        # Create ViT backbone
        self.vit = ViTModel.from_pretrained(model_name, config=self.config, ignore_mismatched_sizes=True)
        
        # Modify the patch embedding layer for different input channels
        if self.input_channels != 3:
            self._modify_patch_embedding()
        
        # Enable gradient checkpointing if requested
        if use_gradient_checkpointing:
            self.vit.gradient_checkpointing_enable()
        
        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()
        
        # Create classifier head
        self._create_classifier_head()
        
    def _modify_patch_embedding(self):
        """Modify patch embedding layer for different input channels"""
        original_conv = self.vit.embeddings.patch_embeddings.projection
        
        # Check if the conv layer already has the right number of channels
        if original_conv.in_channels == self.input_channels:
            # Already has the right number of channels, initialize properly
            if self.input_channels == 6:
                with torch.no_grad():
                    # Initialize the 6-channel weights by duplicating the first 3 channels
                    # Get the original 3-channel weights from a fresh model
                    from transformers import ViTConfig, ViTModel
                    temp_config = ViTConfig.from_pretrained(self.model_name)
                    temp_vit = ViTModel.from_pretrained(self.model_name, config=temp_config)
                    original_3ch_weight = temp_vit.embeddings.patch_embeddings.projection.weight
                    
                    # Copy RGB weights to first 3 channels and thermal channels
                    original_conv.weight[:, :3, :, :] = original_3ch_weight
                    original_conv.weight[:, 3:6, :, :] = original_3ch_weight
            return
        
        # Create new conv layer with different input channels
        new_conv = nn.Conv2d(
            self.input_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        
        # Initialize weights
        with torch.no_grad():
            if self.input_channels == 6:  # concat case
                # Copy RGB weights to first 3 channels
                new_conv.weight[:, :3, :, :] = original_conv.weight
                # Copy RGB weights to thermal channels (channels 3-6)
                new_conv.weight[:, 3:6, :, :] = original_conv.weight
            
            if original_conv.bias is not None:
                new_conv.bias.copy_(original_conv.bias)
        
        # Replace the projection layer
        self.vit.embeddings.patch_embeddings.projection = new_conv
        
    def _freeze_backbone(self):
        """Freeze the ViT backbone parameters"""
        for param in self.vit.parameters():
            param.requires_grad = False
        logger.info("ViT backbone frozen")
    
    def _create_classifier_head(self):
        """Create classifier head"""
        hidden_size = self.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate / 2),
            nn.Linear(hidden_size // 2, self.num_classes)
        )
        
        # Initialize weights
        self._init_classifier_weights()
    
    def _init_classifier_weights(self):
        """Initialize classifier weights"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
    
    def forward(self, rgb_images: torch.Tensor, thermal_images: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """
        Forward pass for early fusion
        
        Args:
            rgb_images: RGB images tensor (B, 3, H, W)
            thermal_images: Thermal images tensor (B, 3, H, W)
            labels: Optional labels for computing loss
            
        Returns:
            Dictionary containing logits and loss (if labels provided)
        """
        # Fuse RGB and Thermal at input level
        if self.fusion_type == "concat":
            # Concatenate along channel dimension
            fused_input = torch.cat([rgb_images, thermal_images], dim=1)  # (B, 6, H, W)
        else:  # add
            # Element-wise addition
            fused_input = rgb_images + thermal_images  # (B, 3, H, W)
        
        # Forward through ViT backbone
        outputs = self.vit(pixel_values=fused_input)
        
        # Get [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]  # (B, hidden_size)
        
        # Forward through classifier
        logits = self.classifier(cls_output)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return {
            'logits': logits,
            'loss': loss,
            'last_hidden_state': outputs.last_hidden_state
        }
    
    def unfreeze_backbone(self):
        """Unfreeze the backbone for full fine-tuning"""
        for param in self.vit.parameters():
            param.requires_grad = True
        self.freeze_backbone = False
        logger.info("ViT backbone unfrozen")


class LateFusionViT(nn.Module):
    """
    Late Fusion ViT: Separate ViT encoders for RGB and Thermal, fuse at feature/prediction level
    """
    
    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224-in21k",
        num_classes: int = 7,
        dropout_rate: float = 0.1,
        freeze_backbone: bool = False,
        use_gradient_checkpointing: bool = False,
        fusion_type: str = "concat",
        fusion_layer: str = "feature"
    ):
        """
        Args:
            model_name: Pre-trained ViT model name
            num_classes: Number of emotion classes
            dropout_rate: Dropout rate for classifier
            freeze_backbone: Whether to freeze backbone
            use_gradient_checkpointing: Whether to use gradient checkpointing
            fusion_type: How to fuse features ("concat", "add", "attention")
            fusion_layer: Where to fuse ("feature" or "prediction")
        """
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.freeze_backbone = freeze_backbone
        self.fusion_type = fusion_type
        self.fusion_layer = fusion_layer
        
        # Load pre-trained ViT configuration
        self.config = ViTConfig.from_pretrained(model_name)
        hidden_size = self.config.hidden_size
        
        # Create separate ViT encoders for RGB and Thermal
        self.rgb_vit = ViTModel.from_pretrained(model_name)
        self.thermal_vit = ViTModel.from_pretrained(model_name)
        
        # Enable gradient checkpointing if requested
        if use_gradient_checkpointing:
            self.rgb_vit.gradient_checkpointing_enable()
            self.thermal_vit.gradient_checkpointing_enable()
        
        # Freeze backbones if requested
        if freeze_backbone:
            self._freeze_backbone()
        
        if fusion_layer == "feature":
            # Fuse at feature level, then single classifier
            if fusion_type == "concat":
                fusion_input_size = hidden_size * 2
            elif fusion_type == "attention":
                # Use attention to fuse features
                self.attention_fusion = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
                fusion_input_size = hidden_size
            else:  # add
                fusion_input_size = hidden_size
            
            # Single classifier after fusion
            self.classifier = nn.Sequential(
                nn.LayerNorm(fusion_input_size),
                nn.Dropout(dropout_rate),
                nn.Linear(fusion_input_size, hidden_size // 2),
                nn.GELU(),
                nn.Dropout(dropout_rate / 2),
                nn.Linear(hidden_size // 2, num_classes)
            )
        else:  # prediction level fusion
            # Separate classifiers for each modality
            self.rgb_classifier = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Dropout(dropout_rate / 2),
                nn.Linear(hidden_size // 2, num_classes)
            )
            
            self.thermal_classifier = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Dropout(dropout_rate / 2),
                nn.Linear(hidden_size // 2, num_classes)
            )
            
            if fusion_type == "attention":
                # Attention-based prediction fusion
                self.prediction_attention = nn.Linear(num_classes * 2, num_classes)
        
        # Initialize weights
        self._init_classifier_weights()
    
    def _freeze_backbone(self):
        """Freeze the ViT backbone parameters"""
        for param in self.rgb_vit.parameters():
            param.requires_grad = False
        for param in self.thermal_vit.parameters():
            param.requires_grad = False
        logger.info("ViT backbones frozen")
    
    def _init_classifier_weights(self):
        """Initialize classifier weights"""
        if hasattr(self, 'classifier'):
            for module in self.classifier.modules():
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
        
        if hasattr(self, 'rgb_classifier'):
            for module in self.rgb_classifier.modules():
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
        
        if hasattr(self, 'thermal_classifier'):
            for module in self.thermal_classifier.modules():
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
    
    def forward(self, rgb_images: torch.Tensor, thermal_images: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """
        Forward pass for late fusion
        
        Args:
            rgb_images: RGB images tensor (B, 3, H, W)
            thermal_images: Thermal images tensor (B, 3, H, W)
            labels: Optional labels for computing loss
            
        Returns:
            Dictionary containing logits and loss (if labels provided)
        """
        # Forward through separate ViT encoders
        rgb_outputs = self.rgb_vit(pixel_values=rgb_images)
        thermal_outputs = self.thermal_vit(pixel_values=thermal_images)
        
        # Get [CLS] token representations
        rgb_features = rgb_outputs.last_hidden_state[:, 0, :]  # (B, hidden_size)
        thermal_features = thermal_outputs.last_hidden_state[:, 0, :]  # (B, hidden_size)
        
        if self.fusion_layer == "feature":
            # Fuse at feature level
            if self.fusion_type == "concat":
                fused_features = torch.cat([rgb_features, thermal_features], dim=1)
            elif self.fusion_type == "attention":
                # Stack features for attention
                stacked_features = torch.stack([rgb_features, thermal_features], dim=1)  # (B, 2, hidden_size)
                fused_features, _ = self.attention_fusion(stacked_features, stacked_features, stacked_features)
                fused_features = fused_features.mean(dim=1)  # Average the attended features
            else:  # add
                fused_features = rgb_features + thermal_features
            
            # Forward through single classifier
            logits = self.classifier(fused_features)
            
        else:  # prediction level fusion
            # Get predictions from separate classifiers
            rgb_logits = self.rgb_classifier(rgb_features)
            thermal_logits = self.thermal_classifier(thermal_features)
            
            # Fuse predictions
            if self.fusion_type == "concat":
                logits = (rgb_logits + thermal_logits) / 2  # Simple average
            elif self.fusion_type == "attention":
                # Attention-based fusion of predictions
                concat_logits = torch.cat([rgb_logits, thermal_logits], dim=1)
                logits = self.prediction_attention(concat_logits)
            else:  # add
                logits = rgb_logits + thermal_logits
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return {
            'logits': logits,
            'loss': loss,
            'rgb_features': rgb_features,
            'thermal_features': thermal_features
        }
    
    def unfreeze_backbone(self):
        """Unfreeze the backbones for full fine-tuning"""
        for param in self.rgb_vit.parameters():
            param.requires_grad = True
        for param in self.thermal_vit.parameters():
            param.requires_grad = True
        self.freeze_backbone = False
        logger.info("ViT backbones unfrozen")


def create_multimodal_vit_model(
    mode: str = 'rgb',
    fusion_strategy: str = 'early',
    fusion_type: str = 'concat',
    fusion_layer: str = 'feature',
    model_name: str = "google/vit-base-patch16-224-in21k",
    num_classes: int = 7,
    dropout_rate: float = 0.1,
    freeze_backbone: bool = False,
    use_gradient_checkpointing: bool = False
) -> Union[ViTForFER, EarlyFusionViT, LateFusionViT]:
    """
    Create a multimodal ViT model for FER
    
    Args:
        mode: 'rgb', 'thermal', or 'combined'
        fusion_strategy: 'early' or 'late' (only for combined mode)
        fusion_type: 'concat', 'add', or 'attention' (for fusion)
        fusion_layer: 'feature' or 'prediction' (for late fusion)
        model_name: Pre-trained ViT model name
        num_classes: Number of emotion classes
        dropout_rate: Dropout rate for classifier
        freeze_backbone: Whether to freeze backbone
        use_gradient_checkpointing: Whether to use gradient checkpointing
        
    Returns:
        Appropriate ViT model based on mode
    """
    if mode in ['rgb', 'thermal']:
        # Single modality model
        model = ViTForFER(
            model_name=model_name,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            freeze_backbone=freeze_backbone,
            use_gradient_checkpointing=use_gradient_checkpointing
        )
        logger.info(f"Created single modality ViT model for {mode}")
        
    elif mode == 'combined':
        if fusion_strategy == 'early':
            # Early fusion model
            model = EarlyFusionViT(
                model_name=model_name,
                num_classes=num_classes,
                dropout_rate=dropout_rate,
                freeze_backbone=freeze_backbone,
                use_gradient_checkpointing=use_gradient_checkpointing,
                fusion_type=fusion_type
            )
            logger.info(f"Created early fusion ViT model with {fusion_type} fusion")
            
        else:  # late fusion
            # Late fusion model
            model = LateFusionViT(
                model_name=model_name,
                num_classes=num_classes,
                dropout_rate=dropout_rate,
                freeze_backbone=freeze_backbone,
                use_gradient_checkpointing=use_gradient_checkpointing,
                fusion_type=fusion_type,
                fusion_layer=fusion_layer
            )
            logger.info(f"Created late fusion ViT model with {fusion_type} fusion at {fusion_layer} level")
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'rgb', 'thermal', or 'combined'")
    
    return model


def get_optimizer_and_scheduler(
    model: Union[ViTForFER, EarlyFusionViT, LateFusionViT],
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    warmup_steps: int = 1000,
    num_training_steps: int = 10000,
    optimizer_type: str = "adamw"
):
    """
    Get optimizer and learning rate scheduler
    
    Args:
        model: ViT model
        learning_rate: Learning rate
        weight_decay: Weight decay
        warmup_steps: Number of warmup steps
        num_training_steps: Total training steps
        optimizer_type: Type of optimizer ("adamw" or "sgd")
        
    Returns:
        optimizer, scheduler
    """
    # Different learning rates for different parts
    backbone_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'classifier' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
    
    # Set different learning rates
    param_groups = [
        {'params': backbone_params, 'lr': learning_rate * 0.1},  # Lower LR for backbone
        {'params': classifier_params, 'lr': learning_rate}       # Higher LR for classifier
    ]
    
    # Create optimizer
    if optimizer_type.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    else:  # SGD
        optimizer = torch.optim.SGD(
            param_groups,
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
            nesterov=True
        )
    
    # Learning rate scheduler
    from transformers import get_cosine_schedule_with_warmup
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )
    
    return optimizer, scheduler


if __name__ == "__main__":
    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test different model configurations
    configs = [
        {'mode': 'rgb'},
        {'mode': 'thermal'},
        {'mode': 'combined', 'fusion_strategy': 'early', 'fusion_type': 'concat'},
        {'mode': 'combined', 'fusion_strategy': 'early', 'fusion_type': 'add'},
        {'mode': 'combined', 'fusion_strategy': 'late', 'fusion_type': 'concat', 'fusion_layer': 'feature'},
        {'mode': 'combined', 'fusion_strategy': 'late', 'fusion_type': 'attention', 'fusion_layer': 'prediction'},
    ]
    
    batch_size = 4
    dummy_rgb = torch.randn(batch_size, 3, 224, 224).to(device)
    dummy_thermal = torch.randn(batch_size, 3, 224, 224).to(device)
    dummy_labels = torch.randint(0, 7, (batch_size,)).to(device)
    
    for config in configs:
        print(f"\n=== Testing {config} ===")
        
        try:
            model = create_multimodal_vit_model(**config)
            model.to(device)
            
            # Test forward pass
            with torch.no_grad():
                if config['mode'] == 'combined':
                    outputs = model(dummy_rgb, dummy_thermal, dummy_labels)
                else:
                    if config['mode'] == 'rgb':
                        outputs = model(dummy_rgb, dummy_labels)
                    else:  # thermal
                        outputs = model(dummy_thermal, dummy_labels)
                
                print(f"Output logits shape: {outputs['logits'].shape}")
                print(f"Loss: {outputs['loss']}")
                
        except Exception as e:
            print(f"Error: {e}")