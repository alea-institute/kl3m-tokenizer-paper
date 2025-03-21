"""
Utility functions for KL3M analysis.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import numpy as np
import tiktoken
from tokenizers import Tokenizer

from . import constants


def ensure_output_dir() -> Path:
    """Ensure the output directory exists.
    
    Returns:
        Path to the output directory
    """
    constants.OUTPUT_DIR.mkdir(exist_ok=True)
    return constants.OUTPUT_DIR


def load_tokenizer(name: str, model_id: str = None) -> Tokenizer:
    """Load a tokenizer from Hugging Face.
    
    Args:
        name: A friendly name for the tokenizer
        model_id: The model identifier for loading (default: lookup in constants)
        
    Returns:
        The loaded tokenizer
    """
    if model_id is None:
        model_id = constants.TOKENIZER_MODEL_IDS.get(name)
        if model_id is None:
            raise ValueError(f"Model ID not found for {name} and none provided")
    
    try:
        tokenizer = Tokenizer.from_pretrained(model_id)
        print(f"Loaded tokenizer: {name}")
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer {name} ({model_id}): {e}")
        raise


def load_tiktoken_model(name: str, model_id: str = None):
    """Load a tiktoken model.
    
    Args:
        name: A friendly name for the model
        model_id: The model identifier for loading (default: use name)
        
    Returns:
        The loaded tiktoken encoder
    """
    try:
        if model_id is None:
            model_id = name
        
        encoder = tiktoken.encoding_for_model(model_id)
        print(f"Loaded tiktoken model: {name}")
        return encoder
    except Exception as e:
        print(f"Error loading tiktoken model {name}: {e}")
        raise


def get_tokenizer_vocab(tokenizer: Tokenizer) -> Dict[str, int]:
    """Get the vocabulary of a standard tokenizer.
    
    Args:
        tokenizer: The tokenizer
        
    Returns:
        The vocabulary as a dictionary mapping tokens to IDs
    """
    return tokenizer.get_vocab()


def get_tiktoken_vocab(encoder) -> Dict[str, int]:
    """Get a sample of vocabulary from a tiktoken encoder.
    
    Args:
        encoder: The tiktoken encoder
        
    Returns:
        A sample of the vocabulary as a dictionary mapping tokens to IDs
    """
    vocab = {}
    # Sample a limited set of token IDs that are likely to be valid
    for i in range(min(10000, encoder.n_vocab)):
        try:
            token = encoder.decode([i])
            if token:
                vocab[token] = i
        except Exception:
            continue
    return vocab


def save_figure(fig, name: str):
    """Save a figure to both PNG and PDF formats.
    
    Args:
        fig: The matplotlib figure
        name: The filename without extension
    """
    output_dir = ensure_output_dir()
    
    # Save as PNG
    fig.savefig(f"{output_dir / name}.png", bbox_inches='tight')
    # Save as PDF
    fig.savefig(f"{output_dir / name}.pdf", bbox_inches='tight')
    
    print(f"Saved {name} plots to {output_dir / name}.png and {output_dir / name}.pdf")


def save_text(content: str, name: str, ext: str = "txt"):
    """Save text content to a file.
    
    Args:
        content: The text content to save
        name: The filename without extension
        ext: The file extension (default: "txt")
    """
    output_dir = ensure_output_dir()
    output_path = output_dir / f"{name}.{ext}"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Saved {name} to {output_path}")


def save_json(data: Union[Dict, List], name: str):
    """Save JSON data to a file.
    
    Args:
        data: The data to save
        name: The filename without extension
    """
    output_dir = ensure_output_dir()
    output_path = output_dir / f"{name}.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {name} to {output_path}")


def analyze_tokens(tokenizer_name: str, tokenizer: Optional[Tokenizer] = None, 
                  tiktoken_encoder=None) -> Dict[str, Any]:
    """Analyze a tokenizer and return various statistics.
    
    Args:
        tokenizer_name: Name of the tokenizer
        tokenizer: The tokenizer object (if available)
        tiktoken_encoder: The tiktoken encoder (if available)
        
    Returns:
        Dictionary of tokenizer statistics
    """
    if tokenizer is None and tiktoken_encoder is None:
        raise ValueError("Either tokenizer or tiktoken_encoder must be provided")
    
    # Initialize results dictionary
    results = {"name": tokenizer_name}
    
    # Get vocabulary size
    if tokenizer is not None:
        results["vocab_size"] = tokenizer.get_vocab_size()
        vocab = get_tokenizer_vocab(tokenizer)
    else:  # tiktoken_encoder
        results["vocab_size"] = tiktoken_encoder.n_vocab
        vocab = get_tiktoken_vocab(tiktoken_encoder)
    
    # Analyze token lengths
    token_lengths = Counter(len(token) for token in vocab.keys())
    results["token_lengths"] = dict(sorted(token_lengths.items()))
    
    # Find special tokens
    if tokenizer is not None:
        special_tokens = [token for token in vocab.keys() 
                         if (token.startswith('<') and token.endswith('>')) or 
                            (token.startswith('[') and token.endswith(']')) or
                            (token.startswith('<|') and token.endswith('|>'))]
        results["special_tokens"] = special_tokens
    
    # Analyze domain-specific tokens
    if tokenizer is not None:
        compiled_patterns = {
            'legal': [re.compile(p, re.IGNORECASE) for p in constants.LEGAL_PATTERNS],
            'financial': [re.compile(p, re.IGNORECASE) for p in constants.FINANCIAL_PATTERNS],
            'html': [re.compile(p) for p in constants.HTML_PATTERNS],
            'json': [re.compile(p) for p in constants.JSON_PATTERNS]
        }
        
        tokens_by_domain = defaultdict(list)
        
        # For each token, check if it matches any pattern
        for token in vocab.keys():
            for domain, patterns in compiled_patterns.items():
                for pattern in patterns:
                    if pattern.search(token):
                        tokens_by_domain[domain].append(token)
                        break
        
        # Limit to a reasonable number of examples per domain
        for domain in tokens_by_domain:
            # Shuffle to get a random sample
            tokens = tokens_by_domain[domain]
            if len(tokens) > 10:
                np.random.shuffle(tokens)
                tokens_by_domain[domain] = tokens[:10]
        
        results["domain_tokens"] = dict(tokens_by_domain)
    
    return results