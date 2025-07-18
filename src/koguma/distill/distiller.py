"""Multi-teacher knowledge distillation for Koguma-LM."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logger = logging.getLogger(__name__)


class MultiTeacherDistiller:
    """Handles knowledge distillation from multiple teacher models."""
    
    def __init__(
        self,
        student_model: nn.Module,
        teacher_configs: List[Dict],
        temperature: float = 5.0,
        alpha: float = 0.7,
        device: str = "cuda"
    ):
        """
        Initialize multi-teacher distiller.
        
        Args:
            student_model: The student model to train
            teacher_configs: List of teacher model configurations
            temperature: Distillation temperature
            alpha: Weight for distillation loss (1-alpha for task loss)
            device: Device to run models on
        """
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        self.device = device
        
        # Load teacher models
        self.teachers = {}
        self.teacher_weights = {}
        self.teacher_specializations = {}
        
        for config in teacher_configs:
            name = config["name"]
            logger.info(f"Loading teacher model: {name}")
            
            model = AutoModelForCausalLM.from_pretrained(
                name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto"
            )
            model.eval()
            
            self.teachers[name] = model
            self.teacher_weights[name] = config.get("weight", 1.0)
            self.teacher_specializations[name] = config.get("specialization", "general")
    
    def get_teacher_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task_type: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get weighted logits from all teacher models.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            task_type: Optional task type for specialized weighting
            
        Returns:
            Tuple of (weighted_logits, confidence_scores)
        """
        all_logits = []
        weights = []
        
        with torch.no_grad():
            for name, teacher in self.teachers.items():
                # Get teacher outputs
                outputs = teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                logits = outputs.logits
                
                # Adjust weight based on task specialization
                weight = self.teacher_weights[name]
                if task_type and self.teacher_specializations[name] == task_type:
                    weight *= 1.5  # Boost weight for specialized teacher
                
                all_logits.append(logits)
                weights.append(weight)
        
        # Normalize weights
        weights = torch.tensor(weights, device=self.device)
        weights = weights / weights.sum()
        
        # Compute weighted average of logits
        weighted_logits = torch.zeros_like(all_logits[0])
        for i, (logits, weight) in enumerate(zip(all_logits, weights)):
            weighted_logits += weight * logits
        
        # Compute confidence scores (based on agreement between teachers)
        logits_stack = torch.stack(all_logits)
        confidence = 1.0 - logits_stack.var(dim=0).mean()
        
        return weighted_logits, confidence
    
    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the distillation loss.
        
        Args:
            student_logits: Logits from student model
            teacher_logits: Logits from teacher model(s)
            labels: Ground truth labels (optional)
            attention_mask: Attention mask for valid positions
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Ensure logits have same shape
        if student_logits.shape != teacher_logits.shape:
            min_len = min(student_logits.shape[1], teacher_logits.shape[1])
            student_logits = student_logits[:, :min_len, :]
            teacher_logits = teacher_logits[:, :min_len, :]
            if labels is not None:
                labels = labels[:, :min_len]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :min_len]
        
        # Compute soft targets
        student_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # KL divergence loss for distillation
        distill_loss = F.kl_div(
            student_probs,
            teacher_probs,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        total_loss = distill_loss
        metrics = {"distill_loss": distill_loss.item()}
        
        # Add task loss if labels provided
        if labels is not None:
            # Shift for causal LM
            shift_logits = student_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Cross entropy loss
            task_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='mean'
            )
            
            # Combine losses
            total_loss = self.alpha * distill_loss + (1 - self.alpha) * task_loss
            metrics["task_loss"] = task_loss.item()
        
        metrics["total_loss"] = total_loss.item()
        
        return total_loss, metrics
    
    def select_best_teacher_for_input(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> str:
        """
        Select the best teacher for a given input based on perplexity.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Name of the best teacher model
        """
        best_teacher = None
        best_perplexity = float('inf')
        
        with torch.no_grad():
            for name, teacher in self.teachers.items():
                outputs = teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                    return_dict=True
                )
                
                perplexity = torch.exp(outputs.loss).item()
                
                if perplexity < best_perplexity:
                    best_perplexity = perplexity
                    best_teacher = name
        
        logger.debug(f"Best teacher for input: {best_teacher} (perplexity: {best_perplexity:.2f})")
        return best_teacher
    
    def adaptive_distillation_step(
        self,
        student_model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        task_type: Optional[str] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Perform one adaptive distillation step.
        
        Args:
            student_model: Student model
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Ground truth labels (optional)
            task_type: Task type for specialized weighting
            
        Returns:
            Tuple of (loss, metrics)
        """
        # Get student outputs
        student_outputs = student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        student_logits = student_outputs.logits
        
        # Get weighted teacher logits
        teacher_logits, confidence = self.get_teacher_logits(
            input_ids, attention_mask, task_type
        )
        
        # Compute distillation loss
        loss, metrics = self.compute_distillation_loss(
            student_logits,
            teacher_logits,
            labels,
            attention_mask
        )
        
        # Add confidence to metrics
        metrics["teacher_confidence"] = confidence.item()
        
        return loss, metrics