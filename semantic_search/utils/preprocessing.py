"""
Data Preprocessing Utilities
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import torchvision.transforms as transforms


class DocumentPreprocessor:
    """Preprocessor for multimodal document data."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize document preprocessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Image transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((
                config.get('image_size', 224),
                config.get('image_size', 224)
            )),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Vocabulary mappings (in practice, load from file)
        self.char_to_id = self._build_char_vocab()
        
    def _build_char_vocab(self) -> Dict[str, int]:
        """Build character vocabulary."""
        vocab = {'<PAD>': 0, '<UNK>': 1}
        # ASCII characters
        for i in range(32, 127):
            vocab[chr(i)] = len(vocab)
        return vocab
    
    def preprocess_text(
        self,
        text: str,
        max_length: int = 512
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess text into token IDs.
        
        Args:
            text: Input text string
            max_length: Maximum sequence length
            
        Returns:
            Token IDs and attention mask
        """
        # Simple word-level tokenization (in practice, use proper tokenizer)
        tokens = text.lower().split()[:max_length]
        
        # Convert to IDs (simplified - use proper vocabulary)
        token_ids = [hash(token) % 50000 for token in tokens]
        
        # Pad
        padding_length = max_length - len(token_ids)
        token_ids.extend([0] * padding_length)
        mask = [1] * len(tokens) + [0] * padding_length
        
        return torch.tensor(token_ids), torch.tensor(mask)
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image.
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed image tensor
        """
        return self.image_transform(image)
    
    def preprocess_bboxes(
        self,
        bboxes: List[List[float]],
        image_width: int,
        image_height: int,
        max_panels: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess bounding boxes.
        
        Args:
            bboxes: List of bounding boxes [x, y, w, h]
            image_width: Image width for normalization
            image_height: Image height for normalization
            max_panels: Maximum number of panels
            
        Returns:
            Normalized bboxes and mask
        """
        # Normalize bboxes
        normalized_bboxes = []
        for bbox in bboxes[:max_panels]:
            x, y, w, h = bbox
            normalized_bboxes.append([
                x / image_width,
                y / image_height,
                w / image_width,
                h / image_height
            ])
        
        # Pad
        num_bboxes = len(normalized_bboxes)
        padding_length = max_panels - num_bboxes
        normalized_bboxes.extend([[0, 0, 0, 0]] * padding_length)
        mask = [1] * num_bboxes + [0] * padding_length
        
        return torch.tensor(normalized_bboxes, dtype=torch.float32), torch.tensor(mask)
    
    def preprocess_characters(
        self,
        text: str,
        max_length: int = 1024
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess text at character level.
        
        Args:
            text: Input text
            max_length: Maximum character length
            
        Returns:
            Character IDs and mask
        """
        chars = list(text)[:max_length]
        char_ids = [self.char_to_id.get(c, self.char_to_id['<UNK>']) for c in chars]
        
        # Pad
        padding_length = max_length - len(char_ids)
        char_ids.extend([self.char_to_id['<PAD>']] * padding_length)
        mask = [1] * len(chars) + [0] * padding_length
        
        return torch.tensor(char_ids), torch.tensor(mask)
    
    def preprocess_reading_order(
        self,
        elements: List[Dict[str, Any]],
        max_elements: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Preprocess reading order information.
        
        Args:
            elements: List of document elements with type, order, and hierarchy
            max_elements: Maximum number of elements
            
        Returns:
            Element types, reading orders, hierarchy levels, and mask
        """
        # Element type mapping
        type_mapping = {
            'paragraph': 0, 'heading': 1, 'caption': 2, 'list': 3,
            'table': 4, 'figure': 5, 'other': 6
        }
        
        element_types = []
        reading_orders = []
        hierarchy_levels = []
        
        for elem in elements[:max_elements]:
            element_types.append(type_mapping.get(elem.get('type', 'other'), 6))
            reading_orders.append(elem.get('order', 0))
            hierarchy_levels.append(elem.get('level', 0))
        
        # Normalize reading orders
        if reading_orders:
            max_order = max(reading_orders)
            reading_orders = [o / max(max_order, 1) for o in reading_orders]
        
        # Pad
        num_elements = len(element_types)
        padding_length = max_elements - num_elements
        element_types.extend([0] * padding_length)
        reading_orders.extend([0] * padding_length)
        hierarchy_levels.extend([0] * padding_length)
        mask = [1] * num_elements + [0] * padding_length
        
        return (
            torch.tensor(element_types),
            torch.tensor(reading_orders, dtype=torch.float32),
            torch.tensor(hierarchy_levels),
            torch.tensor(mask)
        )
    
    def preprocess_document(self, document: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Preprocess complete document.
        
        Args:
            document: Document dictionary with various modalities
            
        Returns:
            Preprocessed document data
        """
        preprocessed = {}
        
        # Text
        if 'text' in document:
            text_ids, text_mask = self.preprocess_text(document['text'])
            preprocessed['text_ids'] = text_ids
            preprocessed['text_mask'] = text_mask
        
        # Image
        if 'image' in document:
            if isinstance(document['image'], str):
                image = Image.open(document['image']).convert('RGB')
            else:
                image = document['image']
            preprocessed['images'] = self.preprocess_image(image).unsqueeze(0)
        
        # Panels
        if 'bboxes' in document:
            bboxes, bbox_mask = self.preprocess_bboxes(
                document['bboxes'],
                document.get('image_width', 1000),
                document.get('image_height', 1000)
            )
            preprocessed['bboxes'] = bboxes.unsqueeze(0)
            preprocessed['bbox_mask'] = bbox_mask
        
        # Characters
        if 'ocr_text' in document:
            char_ids, char_mask = self.preprocess_characters(document['ocr_text'])
            preprocessed['char_ids'] = char_ids
            preprocessed['char_mask'] = char_mask
        
        # Reading order
        if 'elements' in document:
            elem_types, orders, levels, reading_mask = self.preprocess_reading_order(
                document['elements']
            )
            preprocessed['element_types'] = elem_types
            preprocessed['reading_orders'] = orders
            preprocessed['hierarchy_levels'] = levels
            preprocessed['reading_mask'] = reading_mask
        
        return preprocessed
