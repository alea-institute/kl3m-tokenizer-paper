"""
TokenizerAnalyzer class for loading and analyzing tokenizers.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union

import tiktoken
from tokenizers import Tokenizer
from datasets import load_dataset

from . import constants
from . import utils


class TokenizerAnalyzer:
    """Class for analyzing and comparing tokenizers."""
    
    def __init__(self, output_dir: Path = constants.OUTPUT_DIR):
        """Initialize the analyzer.
        
        Args:
            output_dir: Directory to save output files and figures
        """
        self.tokenizers = {}
        self.tiktoken_models = {}
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.stats = {}
        
    def load_tokenizer(self, name: str, model_id: str, is_tiktoken: bool = False) -> None:
        """Load a tokenizer from Hugging Face or tiktoken.
        
        Args:
            name: A friendly name for the tokenizer
            model_id: The model identifier for loading
            is_tiktoken: Whether this is a tiktoken model
        """
        try:
            if is_tiktoken:
                self.tiktoken_models[name] = tiktoken.encoding_for_model(model_id)
                print(f"Loaded tiktoken model: {name}")
            else:
                self.tokenizers[name] = Tokenizer.from_pretrained(model_id)
                print(f"Loaded tokenizer: {name}")
        except Exception as e:
            print(f"Error loading tokenizer {name} ({model_id}): {e}")
    
    def load_all_tokenizers(self) -> None:
        """Load all tokenizers for analysis."""
        # KL3M tokenizers
        self.load_tokenizer("kl3m-001-32k", "alea-institute/kl3m-001-32k")
        self.load_tokenizer("kl3m-003-64k", "alea-institute/kl3m-003-64k")
        self.load_tokenizer("kl3m-004-128k-cased", "alea-institute/kl3m-004-128k-cased")
        self.load_tokenizer("kl3m-004-128k-uncased", "alea-institute/kl3m-004-128k-uncased")
        
        # Character tokenizers
        self.load_tokenizer("kl3m-004-char-4k-cased", "alea-institute/kl3m-004-char-4k-cased")
        self.load_tokenizer("kl3m-004-char-8k-cased", "alea-institute/kl3m-004-char-8k-cased")
        self.load_tokenizer("kl3m-004-char-16k-cased", "alea-institute/kl3m-004-char-16k-cased")
        
        # Comparison tokenizers - Using a public model instead of gated Llama
        self.load_tokenizer("gpt2", "gpt2")
        self.load_tokenizer("llama3", "meta-llama/Llama-3.2-1B-Instruct")
        self.load_tokenizer("roberta-base", "roberta-base")
        
        # tiktoken models - using get_encoding which is more reliable than encoding_for_model
        try:
            import tiktoken
            self.tiktoken_models["gpt-4o"] = tiktoken.encoding_for_model("gpt-4o")
            print("Loaded tiktoken models: gpt-4o")
        except Exception as e:
            print(f"Error loading tiktoken models: {e}")
    
    def _get_tokenizer_vocab(self, name: str) -> Dict[str, int]:
        """Get the vocabulary of a tokenizer.
        
        Args:
            name: The tokenizer name
            
        Returns:
            The vocabulary as a dictionary mapping tokens to IDs
        """
        if name in self.tokenizers:
            return self.tokenizers[name].get_vocab()
        elif name in self.tiktoken_models:
            # For tiktoken, we need a different approach since not all token IDs are directly decodable
            # Return a sample of tokens that can be safely decoded
            encoder = self.tiktoken_models[name]
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
        else:
            return {}
    
    def analyze_vocab_size(self) -> Dict[str, int]:
        """Analyze vocabulary size for all tokenizers.
        
        Returns:
            Dictionary mapping tokenizer name to vocabulary size
        """
        vocab_sizes = {}
        
        # Get vocab sizes for standard tokenizers
        for name, tokenizer in self.tokenizers.items():
            vocab_sizes[name] = tokenizer.get_vocab_size()
        
        # Get vocab sizes for tiktoken models
        for name, encoder in self.tiktoken_models.items():
            vocab_sizes[name] = encoder.n_vocab
        
        return vocab_sizes
    
    def analyze_token_lengths(self) -> Dict[str, Dict[int, int]]:
        """Analyze the distribution of token lengths for each tokenizer.
        
        Returns:
            Dictionary mapping tokenizer name to a distribution of token lengths
        """
        token_lengths = {}
        
        # Analyze standard tokenizers
        for name in self.tokenizers:
            vocab = self._get_tokenizer_vocab(name)
            # Count the length distribution of tokens in the vocabulary
            length_counts = Counter(len(token) for token in vocab.keys())
            token_lengths[name] = dict(sorted(length_counts.items()))
        
        # Analyze tiktoken models
        for name in self.tiktoken_models:
            vocab = self._get_tokenizer_vocab(name)
            # Count the length distribution of tokens in the vocabulary
            length_counts = Counter(len(token) for token in vocab.keys())
            token_lengths[name] = dict(sorted(length_counts.items()))
        
        return token_lengths
    
    def analyze_token_efficiency(self, dataset_limit: int = 100) -> Dict[str, Dict[str, List[int]]]:
        """Analyze tokenization efficiency on real-world datasets.
        
        Args:
            dataset_limit: Maximum number of documents to load from each dataset
            
        Returns:
            Dictionary mapping tokenizer names to dataset categories to token counts
        """
        efficiency = {}
        
        # Load sample data from datasets
        datasets = self.load_data(limit=dataset_limit)
        
        if not datasets:
            print("Warning: No datasets loaded. Efficiency analysis will be empty.")
            return {}
            
        print(f"Analyzing token efficiency with {len(datasets)} datasets...")
        
        # Analyze standard tokenizers
        for name, tokenizer in self.tokenizers.items():
            print(f"Processing tokenizer: {name}")
            dataset_counts = {}
            for dataset_name, texts in datasets.items():
                counts = []
                for text in texts:
                    try:
                        encoding = tokenizer.encode(text)
                        counts.append(len(encoding.ids))
                    except Exception as e:
                        print(f"Warning: Error encoding text with {name}: {e}")
                dataset_counts[dataset_name] = counts
            efficiency[name] = dataset_counts
        
        # Analyze tiktoken models
        for name, encoder in self.tiktoken_models.items():
            print(f"Processing tiktoken model: {name}")
            dataset_counts = {}
            for dataset_name, texts in datasets.items():
                counts = []
                for text in texts:
                    try:
                        encoding = encoder.encode(text)
                        counts.append(len(encoding))
                    except Exception as e:
                        print(f"Warning: Error encoding text with {name}: {e}")
                dataset_counts[dataset_name] = counts
            efficiency[name] = dataset_counts
        
        return efficiency
    
    def analyze_special_tokens(self) -> Dict[str, List[str]]:
        """Identify special tokens in each tokenizer.
        
        Returns:
            Dictionary mapping tokenizer names to lists of special tokens
        """
        special_tokens = {}
        
        # Collect special tokens, they usually have specific formats like <|name|>
        for name, tokenizer in self.tokenizers.items():
            vocab = tokenizer.get_vocab()
            # Look for tokens that match special token patterns
            special = [token for token in vocab.keys() 
                      if (token.startswith('<') and token.endswith('>')) or 
                         (token.startswith('[') and token.endswith(']')) or
                         (token.startswith('<|') and token.endswith('|>'))]
            special_tokens[name] = special
        
        # tiktoken doesn't expose special tokens directly, so we'll skip those
        
        return special_tokens
    
    def compute_all_statistics(self, dataset_limit: int = 100) -> Dict[str, Any]:
        """Compute all statistics for the loaded tokenizers.
        
        Args:
            dataset_limit: Maximum number of documents to load from each dataset
            
        Returns:
            Dictionary containing all computed statistics
        """
        print("Computing tokenizer statistics...")
        
        stats = {
            "vocab_size": self.analyze_vocab_size(),
            "token_lengths": self.analyze_token_lengths(),
            "token_efficiency": self.analyze_token_efficiency(dataset_limit=dataset_limit),
            "special_tokens": self.analyze_special_tokens()
        }
        
        self.stats = stats
        print("Statistics computation complete.")
        return stats
    
    def load_data(self, limit: int = 100) -> Dict[str, List[str]]:
        """Load sample data from datasets.
        
        Args:
            limit: Maximum number of documents to load from each dataset
            
        Returns:
            Dictionary mapping dataset names to lists of document texts
        """
        print(f"Loading sample data from {len(constants.DATASET_IDS)} datasets...")
        result = {}
        
        for dataset_name, dataset_label in constants.DATASET_IDS.items():
            print(f"Loading {dataset_label} from {dataset_name}...")
            try:
                # Load the dataset with streaming enabled
                ds = load_dataset(dataset_name, split='train', streaming=True)
                documents = []
                
                # Load the first 'limit' documents
                for i, item in enumerate(ds):
                    if i >= limit:
                        break
                    
                    # Check if the item has tokens or text
                    if 'tokens' in item:
                        # Use the kl3m-004-128k-cased tokenizer to decode tokens
                        if 'kl3m-004-128k-cased' in self.tokenizers:
                            tokenizer = self.tokenizers['kl3m-004-128k-cased']
                            text = tokenizer.decode(item['tokens'])
                            documents.append(text)
                        else:
                            print(f"Warning: kl3m-004-128k-cased tokenizer not loaded, skipping token decoding for {dataset_label}")
                    elif 'text' in item:
                        # Use the text field directly
                        documents.append(item['text'])
                    else:
                        print(f"Warning: Item in {dataset_label} has neither 'tokens' nor 'text' field")
                
                result[dataset_label] = documents
                print(f"Loaded {len(documents)} documents from {dataset_label}")
            
            except Exception as e:
                print(f"Error loading dataset {dataset_name}: {e}")
        
        return result
    
    def export_statistics(self, filename: str = "tokenizer_statistics.json") -> None:
        """Export computed statistics to a JSON file.
        
        Args:
            filename: The filename for the output JSON
        """
        if not self.stats:
            print("No stats available. Please run compute_all_statistics() first.")
            return
        
        # Convert defaultdicts to regular dicts for JSON serialization
        export_stats = {}
        for category, stats in self.stats.items():
            if category == "token_lengths" or category == "vocab_size":
                # These are already serializable
                export_stats[category] = stats
            elif category == "token_efficiency":
                # Convert defaultdicts to regular dicts
                export_efficiency = {}
                for name, domains in stats.items():
                    export_efficiency[name] = dict(domains)
                export_stats[category] = export_efficiency
            elif category == "special_tokens":
                # This is already serializable
                export_stats[category] = stats
            elif category == "domain_tokens":
                # This is already serializable
                export_stats[category] = stats
        
        # Write to file
        utils.save_json(export_stats, filename)