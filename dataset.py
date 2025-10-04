import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional
import re
from pathlib import Path

class MultiModalFERDataset(Dataset):
    """
    Multi-modal Facial Expression Recognition Dataset for RGB and Thermal images
    Supports 7 emotion classes: angry, disgust, fear, happy, neutral, sad, surprised
    Supports 3 modes: 'rgb', 'thermal', 'combined'
    """
    
    def __init__(self, 
                 data_dir: str, 
                 mode: str = 'rgb',
                 split_ratio: float = 0.8,
                 split_type: str = 'train',
                 transform_rgb=None, 
                 transform_thermal=None,
                 use_augmented: bool = False):
        """
        Args:
            data_dir: Path to the Data directory containing RGB/Thermal/RgbAug/ThermalAug folders
            mode: 'rgb', 'thermal', or 'combined' for fusion approaches
            split_ratio: Ratio for train/test split (0.8 means 80% train, 20% test)
            split_type: 'train' or 'test'
            transform_rgb: Transform for RGB images
            transform_thermal: Transform for thermal images  
            use_augmented: Whether to include augmented data
        """
        self.data_dir = data_dir
        self.mode = mode.lower()
        self.split_ratio = split_ratio
        self.split_type = split_type
        self.transform_rgb = transform_rgb
        self.transform_thermal = transform_thermal
        self.use_augmented = use_augmented
        
        # Validate mode
        if self.mode not in ['rgb', 'thermal', 'combined']:
            raise ValueError("Mode must be 'rgb', 'thermal', or 'combined'")
            
        # Define emotion classes (mapping from filename format)
        self.emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
        self.class_to_idx = {emotion: idx for idx, emotion in enumerate(self.emotion_classes)}
        self.idx_to_class = {idx: emotion for emotion, idx in self.class_to_idx.items()}
        
        # Load image paths and labels
        self.rgb_paths = []
        self.thermal_paths = []
        self.labels = []
        self._load_data()
        
    def _load_data(self):
        """Load all image paths and corresponding labels from filename-based structure"""
        # Define directories to search
        rgb_dirs = [os.path.join(self.data_dir, 'RGB')]
        thermal_dirs = [os.path.join(self.data_dir, 'Thermal')]
        
        if self.use_augmented:
            rgb_dirs.append(os.path.join(self.data_dir, 'RgbAug'))
            thermal_dirs.append(os.path.join(self.data_dir, 'ThermalAug'))
        
        # Collect all RGB and Thermal files
        rgb_files = []
        thermal_files = []
        
        # Load RGB files
        for rgb_dir in rgb_dirs:
            if os.path.exists(rgb_dir):
                for filename in os.listdir(rgb_dir):
                    if filename.startswith('R_') and filename.lower().endswith(('.jpg', '.jpeg', '.bmp', '.png')):
                        rgb_files.append((os.path.join(rgb_dir, filename), filename))
        
        # Load Thermal files  
        for thermal_dir in thermal_dirs:
            if os.path.exists(thermal_dir):
                for filename in os.listdir(thermal_dir):
                    if filename.startswith('T_') and filename.lower().endswith(('.jpg', '.jpeg', '.bmp', '.png')):
                        thermal_files.append((os.path.join(thermal_dir, filename), filename))
        
        # Create data based on mode
        if self.mode == 'combined':
            self._create_paired_data(rgb_files, thermal_files)
        elif self.mode == 'rgb':
            self._create_single_modal_data(rgb_files, 'rgb')
        elif self.mode == 'thermal':
            self._create_single_modal_data(thermal_files, 'thermal')
        
        # Apply train/test split
        self._apply_split()
    
    def _parse_filename(self, filename: str) -> Tuple[str, str]:
        """Parse filename to extract emotion class and unique ID
        Format: [R|T]_Classname_ID_Source.ext
        Returns: (emotion_class, unique_id)
        """
        # Remove extension and split by underscore
        basename = os.path.splitext(filename)[0]
        parts = basename.split('_')
        
        if len(parts) >= 4:
            modality = parts[0]  # R or T
            emotion = parts[1].lower()
            unique_id = parts[2]
            source = parts[3]
            
            # Map 'surprised' to 'surprised' (handle naming inconsistency)
            if emotion == 'surprised':
                emotion = 'surprised'
            
            return emotion, f"{unique_id}_{source}"
        else:
            raise ValueError(f"Invalid filename format: {filename}")
    
    def _create_paired_data(self, rgb_files: List, thermal_files: List):
        """Create paired RGB-Thermal data for combined mode"""
        # Create mapping from unique_id to file paths (only one per unique_id to avoid duplication)
        rgb_map = {}
        thermal_map = {}
        
        for rgb_path, rgb_filename in rgb_files:
            try:
                emotion, unique_id = self._parse_filename(rgb_filename)
                if emotion in self.class_to_idx:
                    # Only keep the first file per unique_id to avoid duplication
                    if unique_id not in rgb_map:
                        rgb_map[unique_id] = (rgb_path, emotion)
            except:
                continue
        
        for thermal_path, thermal_filename in thermal_files:
            try:
                emotion, unique_id = self._parse_filename(thermal_filename)
                if emotion in self.class_to_idx:
                    # Only keep the first file per unique_id to avoid duplication
                    if unique_id not in thermal_map:
                        thermal_map[unique_id] = (thermal_path, emotion)
            except:
                continue
        
        # Find common unique_ids that have both RGB and Thermal
        common_ids = set(rgb_map.keys()) & set(thermal_map.keys())
        
        for unique_id in common_ids:
            rgb_path, rgb_emotion = rgb_map[unique_id]
            thermal_path, thermal_emotion = thermal_map[unique_id]
            
            # Ensure emotions match
            if rgb_emotion == thermal_emotion:
                self.rgb_paths.append(rgb_path)
                self.thermal_paths.append(thermal_path)
                self.labels.append(self.class_to_idx[rgb_emotion])
    
    def _create_single_modal_data(self, files: List, modality: str):
        """Create single modal data for RGB-only or Thermal-only mode"""
        for file_path, filename in files:
            try:
                emotion, unique_id = self._parse_filename(filename)
                if emotion in self.class_to_idx:
                    if modality == 'rgb':
                        self.rgb_paths.append(file_path)
                        self.thermal_paths.append(None)
                    else:  # thermal
                        self.rgb_paths.append(None)
                        self.thermal_paths.append(file_path)
                    self.labels.append(self.class_to_idx[emotion])
            except:
                continue
    
    def _apply_split(self):
        """Apply train/test split based on split_ratio"""
        total_samples = len(self.labels)
        train_size = int(total_samples * self.split_ratio)
        
        # Create indices and shuffle
        indices = np.random.RandomState(42).permutation(total_samples)
        
        if self.split_type == 'train':
            selected_indices = indices[:train_size]
        else:  # test
            selected_indices = indices[train_size:]
        
        # Filter data based on selected indices
        self.rgb_paths = [self.rgb_paths[i] for i in selected_indices]
        self.thermal_paths = [self.thermal_paths[i] for i in selected_indices] 
        self.labels = [self.labels[i] for i in selected_indices]
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        
        if self.mode == 'rgb':
            # RGB only mode
            rgb_image = Image.open(self.rgb_paths[idx]).convert('RGB')
            if self.transform_rgb:
                rgb_image = self.transform_rgb(rgb_image)
            return rgb_image, label
            
        elif self.mode == 'thermal':
            # Thermal only mode  
            thermal_image = Image.open(self.thermal_paths[idx])
            # Convert thermal to grayscale then to 3-channel for ViT compatibility
            if thermal_image.mode != 'L':
                thermal_image = thermal_image.convert('L')
            thermal_image = thermal_image.convert('RGB')  # Convert to 3-channel
            
            if self.transform_thermal:
                thermal_image = self.transform_thermal(thermal_image)
            return thermal_image, label
            
        elif self.mode == 'combined':
            # Combined mode - return both RGB and Thermal
            rgb_image = Image.open(self.rgb_paths[idx]).convert('RGB')
            thermal_image = Image.open(self.thermal_paths[idx])
            
            # Convert thermal to grayscale then to 3-channel
            if thermal_image.mode != 'L':
                thermal_image = thermal_image.convert('L')
            thermal_image = thermal_image.convert('RGB')
            
            if self.transform_rgb:
                rgb_image = self.transform_rgb(rgb_image)
            if self.transform_thermal:
                thermal_image = self.transform_thermal(thermal_image)
                
            return {'rgb': rgb_image, 'thermal': thermal_image}, label
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get the distribution of classes in the dataset"""
        distribution = {}
        for emotion in self.emotion_classes:
            count = self.labels.count(self.class_to_idx[emotion])
            distribution[emotion] = count
        return distribution
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced dataset"""
        class_counts = np.bincount(self.labels)
        total_samples = len(self.labels)
        class_weights = total_samples / (len(self.emotion_classes) * class_counts)
        return torch.FloatTensor(class_weights)


def get_rgb_transforms(image_size: int = 224, is_training: bool = True):
    """
    Get RGB data transforms for training and validation
    
    Args:
        image_size: Target image size for ViT
        is_training: Whether this is for training (applies augmentation)
    """
    if is_training:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def get_thermal_transforms(image_size: int = 224, is_training: bool = True):
    """
    Get Thermal data transforms for training and validation
    Note: Thermal images are treated as grayscale converted to 3-channel
    
    Args:
        image_size: Target image size for ViT
        is_training: Whether this is for training (applies augmentation)
    """
    if is_training:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            # More conservative augmentation for thermal images
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            # Use ImageNet normalization for consistency with pretrained ViT
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def create_multimodal_data_loaders(
    data_dir: str,
    mode: str = 'rgb',
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    val_split: float = 0.2,
    use_augmented: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test data loaders for multimodal FER
    
    Args:
        data_dir: Path to the Data directory containing RGB/Thermal folders
        mode: 'rgb', 'thermal', or 'combined' 
        batch_size: Batch size for training
        image_size: Target image size for ViT
        num_workers: Number of workers for data loading
        val_split: Fraction of data to use for validation (applied to train split)
        use_augmented: Whether to include augmented data
    
    Returns:
        train_loader, test_loader
    """
    # Create transforms
    rgb_train_transform = get_rgb_transforms(image_size, is_training=True)
    rgb_test_transform = get_rgb_transforms(image_size, is_training=False)
    thermal_train_transform = get_thermal_transforms(image_size, is_training=True)
    thermal_test_transform = get_thermal_transforms(image_size, is_training=False)
    
    # Calculate split ratio for train vs test
    train_split_ratio = 1.0 - val_split
    
    # Create datasets
    train_dataset = MultiModalFERDataset(
        data_dir=data_dir,
        mode=mode,
        split_ratio=train_split_ratio,
        split_type='train',
        transform_rgb=rgb_train_transform,
        transform_thermal=thermal_train_transform,
        use_augmented=use_augmented
    )
    
    test_dataset = MultiModalFERDataset(
        data_dir=data_dir,
        mode=mode,
        split_ratio=train_split_ratio,
        split_type='test',
        transform_rgb=rgb_test_transform,
        transform_thermal=thermal_test_transform,
        use_augmented=use_augmented
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def analyze_multimodal_dataset(data_dir: str, use_augmented: bool = False):
    """Analyze the multimodal dataset and print statistics"""
    print("=== Multimodal Dataset Analysis ===")
    
    # Analyze different modes and splits
    for mode in ['rgb', 'thermal', 'combined']:
        print(f"\n=== {mode.upper()} Mode ===")
        
        for split in ['train', 'test']:
            print(f"\n{split.upper()} Split:")
            
            try:
                dataset = MultiModalFERDataset(
                    data_dir=data_dir,
                    mode=mode,
                    split_ratio=0.8,
                    split_type=split,
                    use_augmented=use_augmented
                )
                
                print(f"Total samples: {len(dataset)}")
                
                # Class distribution
                distribution = dataset.get_class_distribution()
                print("Class distribution:")
                for emotion, count in distribution.items():
                    if len(dataset) > 0:
                        percentage = (count / len(dataset)) * 100
                        print(f"  {emotion}: {count} ({percentage:.1f}%)")
                
                # Class weights
                if len(dataset) > 0:
                    weights = dataset.get_class_weights()
                    print("Class weights:")
                    for i, (emotion, weight) in enumerate(zip(dataset.emotion_classes, weights)):
                        print(f"  {emotion}: {weight:.3f}")
                        
            except Exception as e:
                print(f"Error loading {mode} {split} dataset: {e}")


if __name__ == "__main__":
    # Example usage
    data_dir = "../vit/vit/data/vit/Data"
    
    # Analyze dataset
    analyze_multimodal_dataset(data_dir, use_augmented=True)
    
    # Test different modes
    for mode in ['rgb', 'thermal', 'combined']:
        print(f"\n=== Testing {mode.upper()} Mode ===")
        
        try:
            # Create data loaders
            train_loader, test_loader = create_multimodal_data_loaders(
                data_dir, mode=mode, batch_size=8, image_size=224, use_augmented=True
            )
            
            print(f"Data loaders created:")
            print(f"Train batches: {len(train_loader)}")
            print(f"Test batches: {len(test_loader)}")
            
            # Test loading a batch
            if len(train_loader) > 0:
                batch = next(iter(train_loader))
                if mode == 'combined':
                    data, labels = batch
                    rgb_images = data['rgb']
                    thermal_images = data['thermal']
                    print(f"RGB batch shape: {rgb_images.shape}")
                    print(f"Thermal batch shape: {thermal_images.shape}")
                    print(f"Labels shape: {labels.shape}")
                else:
                    images, labels = batch
                    print(f"Batch shape: {images.shape}")
                    print(f"Labels shape: {labels.shape}")
            else:
                print("No data available for this mode")
                
        except Exception as e:
            print(f"Error testing {mode} mode: {e}")