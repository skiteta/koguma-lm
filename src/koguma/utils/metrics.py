"""Metrics computation utilities."""

import torch
import numpy as np
from typing import Dict, List, Optional


def compute_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100
) -> Dict[str, float]:
    """
    Compute metrics for language modeling.
    
    Args:
        predictions: Model predictions (logits)
        labels: Ground truth labels
        ignore_index: Index to ignore in loss computation
        
    Returns:
        Dictionary of metrics
    """
    # Shift for causal LM
    shift_predictions = predictions[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Flatten
    shift_predictions = shift_predictions.view(-1, shift_predictions.size(-1))
    shift_labels = shift_labels.view(-1)
    
    # Compute loss
    loss = torch.nn.functional.cross_entropy(
        shift_predictions,
        shift_labels,
        ignore_index=ignore_index,
        reduction='mean'
    )
    
    # Compute perplexity
    try:
        perplexity = torch.exp(loss).item()
    except OverflowError:
        perplexity = float('inf')
    
    # Compute accuracy
    predictions_argmax = shift_predictions.argmax(dim=-1)
    mask = shift_labels != ignore_index
    correct = (predictions_argmax == shift_labels) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    
    return {
        'loss': loss.item(),
        'perplexity': perplexity,
        'accuracy': accuracy.item()
    }


def compute_bleu_score(
    predictions: List[str],
    references: List[str],
    max_n: int = 4
) -> Dict[str, float]:
    """
    Compute BLEU score for generated text.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        max_n: Maximum n-gram to consider
        
    Returns:
        Dictionary with BLEU scores
    """
    from collections import Counter
    import math
    
    def get_ngrams(text: str, n: int) -> Counter:
        """Get n-grams from text."""
        tokens = text.split()
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i+n]))
        return Counter(ngrams)
    
    def compute_precision(pred_ngrams: Counter, ref_ngrams: Counter) -> float:
        """Compute precision for n-grams."""
        if not pred_ngrams:
            return 0.0
        
        overlap = sum((pred_ngrams & ref_ngrams).values())
        total = sum(pred_ngrams.values())
        
        return overlap / total if total > 0 else 0.0
    
    # Compute BLEU scores for different n-grams
    bleu_scores = {}
    precisions = []
    
    for n in range(1, max_n + 1):
        total_precision = 0.0
        
        for pred, ref in zip(predictions, references):
            pred_ngrams = get_ngrams(pred, n)
            ref_ngrams = get_ngrams(ref, n)
            precision = compute_precision(pred_ngrams, ref_ngrams)
            total_precision += precision
        
        avg_precision = total_precision / len(predictions)
        precisions.append(avg_precision)
        bleu_scores[f'bleu_{n}'] = avg_precision
    
    # Compute geometric mean (BLEU-4)
    if all(p > 0 for p in precisions):
        log_precisions = [math.log(p) for p in precisions]
        geometric_mean = math.exp(sum(log_precisions) / len(log_precisions))
        bleu_scores['bleu'] = geometric_mean
    else:
        bleu_scores['bleu'] = 0.0
    
    return bleu_scores


def compute_rouge_scores(
    predictions: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    Compute ROUGE scores for summarization.
    
    Args:
        predictions: List of predicted summaries
        references: List of reference summaries
        
    Returns:
        Dictionary with ROUGE scores
    """
    from collections import Counter
    
    def get_tokens(text: str) -> List[str]:
        """Simple tokenization."""
        return text.lower().split()
    
    def compute_f1(pred_tokens: List[str], ref_tokens: List[str]) -> float:
        """Compute F1 score."""
        pred_counter = Counter(pred_tokens)
        ref_counter = Counter(ref_tokens)
        
        overlap = sum((pred_counter & ref_counter).values())
        
        if not overlap:
            return 0.0
        
        precision = overlap / sum(pred_counter.values())
        recall = overlap / sum(ref_counter.values())
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    # Compute ROUGE-1 and ROUGE-2
    rouge_scores = {}
    
    # ROUGE-1 (unigram)
    rouge_1_scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = get_tokens(pred)
        ref_tokens = get_tokens(ref)
        score = compute_f1(pred_tokens, ref_tokens)
        rouge_1_scores.append(score)
    
    rouge_scores['rouge_1'] = np.mean(rouge_1_scores)
    
    # ROUGE-2 (bigram)
    rouge_2_scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = get_tokens(pred)
        ref_tokens = get_tokens(ref)
        
        # Get bigrams
        pred_bigrams = [f"{pred_tokens[i]} {pred_tokens[i+1]}" 
                       for i in range(len(pred_tokens)-1)]
        ref_bigrams = [f"{ref_tokens[i]} {ref_tokens[i+1]}" 
                      for i in range(len(ref_tokens)-1)]
        
        if pred_bigrams and ref_bigrams:
            score = compute_f1(pred_bigrams, ref_bigrams)
            rouge_2_scores.append(score)
    
    if rouge_2_scores:
        rouge_scores['rouge_2'] = np.mean(rouge_2_scores)
    else:
        rouge_scores['rouge_2'] = 0.0
    
    # ROUGE-L (longest common subsequence)
    rouge_l_scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = get_tokens(pred)
        ref_tokens = get_tokens(ref)
        
        # Simple LCS calculation
        m, n = len(pred_tokens), len(ref_tokens)
        lcs_length = 0
        
        if m > 0 and n > 0:
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if pred_tokens[i-1] == ref_tokens[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            lcs_length = dp[m][n]
        
        if lcs_length > 0:
            precision = lcs_length / m
            recall = lcs_length / n
            f1 = 2 * (precision * recall) / (precision + recall)
            rouge_l_scores.append(f1)
    
    if rouge_l_scores:
        rouge_scores['rouge_l'] = np.mean(rouge_l_scores)
    else:
        rouge_scores['rouge_l'] = 0.0
    
    return rouge_scores