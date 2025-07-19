"""Dataset classes for Koguma-LM training."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class KogumaDataset(Dataset):
    """Dataset for pretraining Koguma-LM."""
    
    def __init__(
        self,
        data_paths: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        block_size: int = 2048,
        cache_dir: Optional[str] = None,
        streaming: bool = False
    ):
        """
        Initialize dataset.
        
        Args:
            data_paths: Path(s) to data files (jsonl format)
            tokenizer: Tokenizer to use
            block_size: Maximum sequence length
            cache_dir: Directory for caching processed data
            streaming: Whether to stream data (for large datasets)
        """
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.streaming = streaming
        
        # Ensure data_paths is a list
        if isinstance(data_paths, str):
            data_paths = [data_paths]
        
        self.data_paths = [Path(p) for p in data_paths]
        
        # Load and process data
        if not streaming:
            self.examples = self._load_and_tokenize()
            logger.info(f"Loaded {len(self.examples)} examples")
        else:
            # For streaming, we'll load on-the-fly
            self.file_handles = []
            self.current_positions = []
            self._init_streaming()
    
    def _load_and_tokenize(self) -> List[Dict[str, torch.Tensor]]:
        """Load and tokenize all data."""
        all_examples = []
        
        for data_path in self.data_paths:
            if not data_path.exists():
                logger.warning(f"Data file not found: {data_path}")
                continue
            
            logger.info(f"Loading data from {data_path}")
            
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        text = data.get('text', data.get('content', ''))
                        
                        if not text or len(text) < 100:  # Skip short texts
                            continue
                        
                        # Tokenize
                        tokens = self.tokenizer(
                            text,
                            truncation=True,
                            max_length=self.block_size,
                            return_tensors='pt'
                        )
                        
                        # Create example
                        example = {
                            'input_ids': tokens['input_ids'].squeeze(),
                            'attention_mask': tokens['attention_mask'].squeeze()
                        }
                        
                        all_examples.append(example)
                        
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse JSON: {line[:100]}...")
                        continue
        
        return all_examples
    
    def _init_streaming(self):
        """Initialize streaming mode."""
        for data_path in self.data_paths:
            if data_path.exists():
                handle = open(data_path, 'r', encoding='utf-8')
                self.file_handles.append(handle)
                self.current_positions.append(0)
    
    def __len__(self):
        if self.streaming:
            # For streaming, we don't know the exact length
            return 1000000  # Placeholder
        return len(self.examples)
    
    def __getitem__(self, idx):
        if self.streaming:
            # For streaming, read next example
            file_idx = idx % len(self.file_handles)
            handle = self.file_handles[file_idx]
            
            line = handle.readline()
            if not line:
                # Restart from beginning
                handle.seek(0)
                line = handle.readline()
            
            try:
                data = json.loads(line.strip())
                text = data.get('text', data.get('content', ''))
                
                tokens = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.block_size,
                    padding='max_length',
                    return_tensors='pt'
                )
                
                return {
                    'input_ids': tokens['input_ids'].squeeze(),
                    'attention_mask': tokens['attention_mask'].squeeze(),
                    'labels': tokens['input_ids'].squeeze()
                }
            except:
                # Return a dummy example if parsing fails
                return self.__getitem__((idx + 1) % len(self))
        else:
            example = self.examples[idx]
            return {
                'input_ids': example['input_ids'],
                'attention_mask': example['attention_mask'],
                'labels': example['input_ids']  # For language modeling
            }
    
    def __del__(self):
        """Clean up file handles."""
        if hasattr(self, 'file_handles'):
            for handle in self.file_handles:
                handle.close()


class DistillationDataset(Dataset):
    """Dataset for knowledge distillation training."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        block_size: int = 2048,
        include_teacher_logits: bool = False
    ):
        """
        Initialize distillation dataset.
        
        Args:
            data_path: Path to distillation data (jsonl format)
            tokenizer: Tokenizer to use
            block_size: Maximum sequence length
            include_teacher_logits: Whether to include pre-computed teacher logits
        """
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.include_teacher_logits = include_teacher_logits
        
        self.examples = self._load_data(data_path)
        logger.info(f"Loaded {len(self.examples)} distillation examples")
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """Load distillation data."""
        examples = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    
                    # Extract prompt and response
                    prompt = data.get('prompt', '')
                    response = data.get('response', '')
                    full_text = prompt + response
                    
                    # Skip if too short
                    if len(full_text) < 50:
                        continue
                    
                    # Tokenize
                    tokens = self.tokenizer(
                        full_text,
                        truncation=True,
                        max_length=self.block_size,
                        padding='max_length',
                        return_tensors='pt'
                    )
                    
                    example = {
                        'input_ids': tokens['input_ids'].squeeze(),
                        'attention_mask': tokens['attention_mask'].squeeze(),
                        'labels': tokens['input_ids'].squeeze(),
                        'teacher': data.get('teacher', 'unknown'),
                        'confidence': data.get('confidence', 1.0)
                    }
                    
                    # Add teacher logits if available
                    if self.include_teacher_logits and 'logits' in data:
                        example['teacher_logits'] = torch.tensor(data['logits'])
                    
                    examples.append(example)
                    
                except Exception as e:
                    logger.warning(f"Failed to process example: {e}")
                    continue
        
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def create_data_collator(tokenizer: PreTrainedTokenizer, padding: bool = True):
    """Create a data collator for batching."""
    def collate_fn(examples):
        # Stack tensors
        input_ids = torch.stack([ex['input_ids'] for ex in examples])
        attention_mask = torch.stack([ex['attention_mask'] for ex in examples])
        labels = torch.stack([ex['labels'] for ex in examples])
        
        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        
        # Add optional fields
        if 'teacher_logits' in examples[0]:
            batch['teacher_logits'] = torch.stack([ex['teacher_logits'] for ex in examples])
        
        if 'confidence' in examples[0]:
            batch['confidence'] = torch.tensor([ex['confidence'] for ex in examples])
        
        return batch
    
    return collate_fn