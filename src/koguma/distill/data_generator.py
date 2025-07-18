"""Generate high-quality distillation data from teacher models."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)


class DistillationDataGenerator:
    """Generate synthetic data using teacher models for distillation."""
    
    def __init__(
        self,
        teacher_models: Dict[str, str],
        output_dir: str,
        max_length: int = 2048,
        min_length: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
        device: str = "mps"  # Default to MPS for M3 Ultra
    ):
        """
        Initialize data generator.
        
        Args:
            teacher_models: Dict mapping model names to their paths/IDs
            output_dir: Directory to save generated data
            max_length: Maximum generation length
            min_length: Minimum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            device: Device to run generation on
        """
        self.teacher_models = teacher_models
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_length = max_length
        self.min_length = min_length
        self.temperature = temperature
        self.top_p = top_p
        self.device = device
        
        # Load tokenizers
        self.tokenizers = {}
        for name, model_id in teacher_models.items():
            logger.info(f"Loading tokenizer for {name}")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            self.tokenizers[name] = tokenizer
    
    def generate_prompts(self, prompt_templates: List[str], num_prompts: int) -> List[str]:
        """
        Generate diverse prompts for data generation.
        
        Args:
            prompt_templates: List of prompt templates
            num_prompts: Number of prompts to generate
            
        Returns:
            List of generated prompts
        """
        prompts = []
        
        # Task-specific prompt categories
        task_categories = {
            "japanese": [
                "次の文章を要約してください：",
                "以下のトピックについて説明してください：",
                "次の質問に答えてください：",
                "以下の文章を続けてください：",
            ],
            "code": [
                "Write a Python function that ",
                "Implement a solution for ",
                "Debug the following code: ",
                "Explain this code: ",
            ],
            "math": [
                "Solve the following problem: ",
                "Explain the concept of ",
                "Prove that ",
                "Calculate ",
            ],
            "general": [
                "What is ",
                "Explain how ",
                "Describe the process of ",
                "Compare and contrast ",
            ]
        }
        
        # Generate prompts from each category
        for _ in range(num_prompts):
            category = np.random.choice(list(task_categories.keys()))
            template = np.random.choice(task_categories[category])
            
            # Add variety to prompts
            if category == "japanese":
                topics = ["機械学習", "自然言語処理", "量子コンピュータ", "環境問題", "経済政策"]
                prompt = template + np.random.choice(topics)
            elif category == "code":
                tasks = ["sorts a list", "finds prime numbers", "implements binary search", 
                        "converts string to integer", "validates email addresses"]
                prompt = template + np.random.choice(tasks)
            elif category == "math":
                problems = ["the derivative of x^3", "the integral of sin(x)", 
                          "the Pythagorean theorem", "2+2", "the quadratic formula"]
                prompt = template + np.random.choice(problems)
            else:
                topics = ["artificial intelligence", "climate change", "quantum mechanics",
                         "machine learning", "natural language processing"]
                prompt = template + np.random.choice(topics)
            
            prompts.append(prompt)
        
        return prompts
    
    def generate_with_teacher(
        self,
        teacher_name: str,
        prompts: List[str],
        batch_size: int = 4
    ) -> List[Dict]:
        """
        Generate responses using a specific teacher model.
        
        Args:
            teacher_name: Name of the teacher model
            prompts: List of prompts to generate from
            batch_size: Batch size for generation
            
        Returns:
            List of generation results
        """
        model_id = self.teacher_models[teacher_name]
        tokenizer = self.tokenizers[teacher_name]
        
        logger.info(f"Loading teacher model: {teacher_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        model.eval()
        
        results = []
        
        for i in tqdm(range(0, len(prompts), batch_size), desc=f"Generating with {teacher_name}"):
            batch_prompts = prompts[i:i + batch_size]
            
            # Tokenize
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length // 2  # Leave room for generation
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_length - inputs.input_ids.shape[1],
                    min_length=self.min_length,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            # Decode and store results
            for j, (prompt, output) in enumerate(zip(batch_prompts, outputs)):
                generated_text = tokenizer.decode(output, skip_special_tokens=True)
                
                # Calculate confidence (based on perplexity)
                with torch.no_grad():
                    loss = model(
                        input_ids=output.unsqueeze(0),
                        labels=output.unsqueeze(0),
                        return_dict=True
                    ).loss
                    perplexity = torch.exp(loss).item()
                    confidence = 1.0 / (1.0 + np.log(perplexity))
                
                results.append({
                    "prompt": prompt,
                    "response": generated_text[len(prompt):].strip(),
                    "full_text": generated_text,
                    "teacher": teacher_name,
                    "confidence": confidence,
                    "perplexity": perplexity,
                    "length": len(output)
                })
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        
        return results
    
    def filter_quality(
        self,
        generations: List[Dict],
        min_confidence: float = 0.7,
        min_length: int = 50,
        max_perplexity: float = 100.0
    ) -> List[Dict]:
        """
        Filter generated data based on quality metrics.
        
        Args:
            generations: List of generated samples
            min_confidence: Minimum confidence score
            min_length: Minimum response length
            max_perplexity: Maximum allowed perplexity
            
        Returns:
            Filtered list of high-quality samples
        """
        filtered = []
        
        for gen in generations:
            # Quality checks
            if gen["confidence"] < min_confidence:
                continue
            if len(gen["response"]) < min_length:
                continue
            if gen["perplexity"] > max_perplexity:
                continue
            
            # Check for repetitions
            response_words = gen["response"].split()
            if len(response_words) > 10:
                unique_ratio = len(set(response_words)) / len(response_words)
                if unique_ratio < 0.5:  # Too repetitive
                    continue
            
            filtered.append(gen)
        
        logger.info(f"Filtered {len(filtered)} / {len(generations)} samples")
        return filtered
    
    def merge_teacher_outputs(
        self,
        all_generations: Dict[str, List[Dict]]
    ) -> List[Dict]:
        """
        Merge outputs from multiple teachers, resolving conflicts.
        
        Args:
            all_generations: Dict mapping teacher names to their generations
            
        Returns:
            Merged list of samples
        """
        # Group by prompt
        prompt_groups = {}
        
        for teacher, generations in all_generations.items():
            for gen in generations:
                prompt = gen["prompt"]
                if prompt not in prompt_groups:
                    prompt_groups[prompt] = []
                prompt_groups[prompt].append(gen)
        
        # Select best response for each prompt
        merged = []
        
        for prompt, candidates in prompt_groups.items():
            # Sort by confidence
            candidates.sort(key=lambda x: x["confidence"], reverse=True)
            
            # Take the best one
            best = candidates[0]
            
            # Add ensemble confidence
            ensemble_confidence = np.mean([c["confidence"] for c in candidates])
            best["ensemble_confidence"] = ensemble_confidence
            
            merged.append(best)
        
        return merged
    
    def generate_distillation_dataset(
        self,
        num_samples_per_teacher: int = 10000,
        batch_size: int = 4,
        save_intermediate: bool = True
    ) -> str:
        """
        Generate full distillation dataset from all teachers.
        
        Args:
            num_samples_per_teacher: Number of samples to generate per teacher
            batch_size: Batch size for generation
            save_intermediate: Whether to save intermediate results
            
        Returns:
            Path to final dataset file
        """
        all_generations = {}
        
        # Generate prompts
        prompts = self.generate_prompts([], num_samples_per_teacher)
        
        # Generate with each teacher
        for teacher_name in self.teacher_models:
            logger.info(f"Generating with {teacher_name}")
            
            generations = self.generate_with_teacher(
                teacher_name,
                prompts,
                batch_size
            )
            
            # Filter quality
            filtered = self.filter_quality(generations)
            all_generations[teacher_name] = filtered
            
            # Save intermediate results
            if save_intermediate:
                intermediate_file = self.output_dir / f"{teacher_name}_generations.jsonl"
                with open(intermediate_file, "w", encoding="utf-8") as f:
                    for gen in filtered:
                        f.write(json.dumps(gen, ensure_ascii=False) + "\n")
        
        # Merge all teacher outputs
        merged_dataset = self.merge_teacher_outputs(all_generations)
        
        # Save final dataset
        final_file = self.output_dir / "distillation_dataset.jsonl"
        with open(final_file, "w", encoding="utf-8") as f:
            for sample in merged_dataset:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        
        logger.info(f"Generated {len(merged_dataset)} samples saved to {final_file}")
        
        return str(final_file)