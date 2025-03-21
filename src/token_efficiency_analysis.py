#!/usr/bin/env python
"""
Script to analyze tokenizer efficiency (tokens per character) across different datasets.
"""

import csv
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from kl3m_analysis.analyzer import TokenizerAnalyzer
from kl3m_analysis.constants import OUTPUT_DIR, KL3M_STANDARD_COLORS, OTHER_COLORS
from kl3m_analysis.visualization import setup_plot_style
from kl3m_analysis.utils import save_figure, save_text


def analyze_token_efficiency():
    """Analyze tokenizer efficiency across different datasets."""
    # Initialize analyzer
    analyzer = TokenizerAnalyzer()
    
    # Load all tokenizers
    analyzer.load_all_tokenizers()
    
    # Prepare output directories
    figures_dir = Path(OUTPUT_DIR)
    figures_dir.mkdir(exist_ok=True)
    
    # Load data from datasets
    datasets = analyzer.load_data(limit=100)  # Using a moderate sample size for analysis
    
    # Calculate token efficiency and token counts for each dataset and tokenizer
    efficiency_data, token_count_data = calculate_token_efficiency(analyzer, datasets)
    
    # Save efficiency results to CSV
    save_to_csv(efficiency_data, figures_dir / "token_efficiency.csv")
    
    # Generate LaTeX table for efficiency
    generate_latex_table(efficiency_data, figures_dir / "token_efficiency.txt")
    
    # Save token count results to CSV
    save_token_counts_to_csv(token_count_data, figures_dir / "token_count.csv")
    
    # Generate LaTeX table for token counts
    generate_token_counts_latex_table(token_count_data, figures_dir / "token_count.txt")
    
    # Create visualizations
    create_efficiency_visualizations(efficiency_data, analyzer, figures_dir)
    
    print(f"Results saved to {figures_dir}")


def calculate_token_efficiency(analyzer, datasets):
    """Calculate token efficiency (tokens per character) for each dataset and tokenizer.
    
    Args:
        analyzer: The TokenizerAnalyzer instance
        datasets: Dictionary mapping dataset names to lists of texts
        
    Returns:
        Tuple of (efficiency_data, token_counts_data)
    """
    efficiency_results = defaultdict(dict)
    token_counts_results = defaultdict(dict)
    
    print("Calculating token efficiency...")
    
    # Order tokenizers to match the domain_term_analysis.py order
    tokenizer_order = [
        "kl3m-004-128k-cased",
        "kl3m-004-128k-uncased",
        "kl3m-003-64k",
        "gpt-4o",
        "llama3",
        "gpt2"
    ]
    
    # Filter to available tokenizers and maintain order (excluding char tokenizers)
    standard_tokenizers = {name: tokenizer for name, tokenizer in analyzer.tokenizers.items() 
                          if 'char' not in name and name != 'kl3m-001-32k' and name != 'roberta-base'}
    tiktoken_models = analyzer.tiktoken_models
    
    # Process each dataset
    for dataset_name, texts in datasets.items():
        print(f"Processing dataset: {dataset_name}")
        
        dataset_efficiency = {}
        dataset_token_counts = {}
        total_chars = sum(len(text) for text in texts)
        
        # Process with standard tokenizers
        for name, tokenizer in standard_tokenizers.items():
            if 'char' in name:
                continue  # Skip character tokenizers
                
            token_counts = []
            for text in texts:
                try:
                    encoding = tokenizer.encode(text)
                    token_counts.append(len(encoding.ids))
                except Exception as e:
                    print(f"  Error with {name} on {dataset_name}: {e}")
            
            if token_counts:
                total_tokens = sum(token_counts)
                # Calculate tokens per character (higher is worse)
                efficiency = total_tokens / total_chars if total_chars > 0 else 0
                dataset_efficiency[name] = efficiency
                dataset_token_counts[name] = total_tokens
        
        # Process with tiktoken models
        for name, encoder in tiktoken_models.items():
            token_counts = []
            for text in texts:
                try:
                    encoding = encoder.encode(text)
                    token_counts.append(len(encoding))
                except Exception as e:
                    print(f"  Error with {name} on {dataset_name}: {e}")
            
            if token_counts:
                total_tokens = sum(token_counts)
                # Calculate tokens per character (higher is worse)
                efficiency = total_tokens / total_chars if total_chars > 0 else 0
                dataset_efficiency[name] = efficiency
                dataset_token_counts[name] = total_tokens
        
        efficiency_results[dataset_name] = dataset_efficiency
        token_counts_results[dataset_name] = dataset_token_counts
    
    # Calculate average efficiency across all datasets
    all_datasets_efficiency = defaultdict(list)
    all_datasets_token_counts = defaultdict(list)
    
    for dataset_name, tokenizer_results in efficiency_results.items():
        for tokenizer_name, efficiency in tokenizer_results.items():
            all_datasets_efficiency[tokenizer_name].append(efficiency)
    
    averages_efficiency = {}
    for tokenizer_name, efficiencies in all_datasets_efficiency.items():
        if efficiencies:
            averages_efficiency[tokenizer_name] = sum(efficiencies) / len(efficiencies)
    
    efficiency_results["Average"] = averages_efficiency
    
    # Calculate total token counts across all datasets
    for dataset_name, tokenizer_results in token_counts_results.items():
        for tokenizer_name, tokens in tokenizer_results.items():
            all_datasets_token_counts[tokenizer_name].append(tokens)
    
    total_token_counts = {}
    for tokenizer_name, counts in all_datasets_token_counts.items():
        if counts:
            total_token_counts[tokenizer_name] = sum(counts)
    
    token_counts_results["Total"] = total_token_counts
    
    return efficiency_results, token_counts_results


def save_to_csv(efficiency_data, output_path):
    """Save token efficiency data to a CSV file.
    
    Args:
        efficiency_data: Dictionary mapping datasets to tokenizer efficiency values
        output_path: Path to save the CSV file
    """
    # Get all tokenizer names from results
    tokenizer_names = set()
    for dataset_results in efficiency_data.values():
        tokenizer_names.update(dataset_results.keys())
    
    # Order tokenizers to match the domain_term_analysis.py order
    tokenizer_order = [
        "kl3m-004-128k-cased",
        "kl3m-004-128k-uncased",
        "kl3m-003-64k",
        "gpt-4o",
        "llama3",
        "gpt2"
    ]
    
    # Filter to available tokenizers and maintain order
    tokenizers = [t for t in tokenizer_order if t in tokenizer_names]
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['Dataset'] + tokenizers)
        
        # Write data rows (sort datasets but keep "Average" at the end)
        datasets = sorted([d for d in efficiency_data.keys() if d != "Average"])
        if "Average" in efficiency_data:
            datasets.append("Average")
        
        for dataset in datasets:
            row = [dataset]
            for tokenizer in tokenizers:
                efficiency = efficiency_data[dataset].get(tokenizer, "N/A")
                if isinstance(efficiency, (int, float)):
                    row.append(f"{efficiency:.4f}")
                else:
                    row.append(efficiency)
            writer.writerow(row)
    
    print(f"CSV data saved to {output_path}")


def generate_latex_table(efficiency_data, output_path):
    """Generate a LaTeX table for token efficiency data.
    
    Args:
        efficiency_data: Dictionary mapping datasets to tokenizer efficiency values
        output_path: Path to save the LaTeX table
    """
    # Get all tokenizer names from results
    tokenizer_names = set()
    for dataset_results in efficiency_data.values():
        tokenizer_names.update(dataset_results.keys())
    
    # Order tokenizers to match the domain_term_analysis.py order
    tokenizer_order = [
        "kl3m-004-128k-cased",
        "kl3m-004-128k-uncased",
        "kl3m-003-64k",
        "gpt-4o",
        "llama3",
        "gpt2"
    ]
    
    # Filter to available tokenizers and maintain order
    tokenizers = [t for t in tokenizer_order if t in tokenizer_names]
    
    # Format tokenizer names for LaTeX (escape special characters)
    def format_for_latex(name):
        return name.replace("_", "\\_").replace("-", "\\mbox{-}")
    
    with open(output_path, 'w') as f:
        # Start the LaTeX table
        f.write("% LaTeX table for token efficiency analysis (tokens per character)\n")
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Token efficiency (tokens per character) across datasets}\n")
        f.write("\\label{tab:token-efficiency}\n")
        f.write("\\small\n")
        
        # Define table columns - One for dataset plus one for each tokenizer
        col_spec = "l" + "r" * len(tokenizers)
        f.write("\\begin{tabular}{" + col_spec + "}\n")
        f.write("\\toprule\n")
        
        # Header row
        f.write("Dataset")
        for tokenizer in tokenizers:
            f.write(f" & {format_for_latex(tokenizer)}")
        f.write(" \\\\\n")
        f.write("\\midrule\n")
        
        # Data rows (sort datasets but keep "Average" at the end)
        datasets = sorted([d for d in efficiency_data.keys() if d != "Average"])
        
        for dataset in datasets:
            # Format dataset name for LaTeX
            latex_dataset = dataset.replace("_", "\\_").replace("&", "\\&")
            f.write(f"{latex_dataset}")
            
            # Get the most efficient value for highlighting
            valid_values = [eff for eff in efficiency_data[dataset].values() 
                           if isinstance(eff, (int, float))]
            min_value = min(valid_values) if valid_values else None
            
            for tokenizer in tokenizers:
                efficiency = efficiency_data[dataset].get(tokenizer, "N/A")
                
                if isinstance(efficiency, (int, float)):
                    # Highlight the most efficient (lowest) value
                    if min_value is not None and abs(efficiency - min_value) < 0.0001:  # Small epsilon for float comparison
                        f.write(f" & \\textbf{{{efficiency:.4f}}}")
                    else:
                        f.write(f" & {efficiency:.4f}")
                else:
                    f.write(" & N/A")
            
            f.write(" \\\\\n")
        
        # Add the average row if it exists
        if "Average" in efficiency_data:
            f.write("\\midrule\n")
            f.write("Average")
            
            # Get the most efficient average value for highlighting
            valid_values = [eff for eff in efficiency_data["Average"].values() 
                           if isinstance(eff, (int, float))]
            min_value = min(valid_values) if valid_values else None
            
            for tokenizer in tokenizers:
                efficiency = efficiency_data["Average"].get(tokenizer, "N/A")
                
                if isinstance(efficiency, (int, float)):
                    # Highlight the most efficient (lowest) value
                    if min_value is not None and abs(efficiency - min_value) < 0.0001:
                        f.write(f" & \\textbf{{{efficiency:.4f}}}")
                    else:
                        f.write(f" & {efficiency:.4f}")
                else:
                    f.write(" & N/A")
            
            f.write(" \\\\\n")
        
        # End the LaTeX table
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption*{\\small \\textit{Note: Lower values indicate more efficient tokenization (fewer tokens per character).}}\n")
        f.write("\\end{table}\n")
    
    print(f"LaTeX table saved to {output_path}")


def save_token_counts_to_csv(token_count_data, output_path):
    """Save token count data to a CSV file.
    
    Args:
        token_count_data: Dictionary mapping datasets to tokenizer token counts
        output_path: Path to save the CSV file
    """
    # Get all tokenizer names from results
    tokenizer_names = set()
    for dataset_results in token_count_data.values():
        tokenizer_names.update(dataset_results.keys())
    
    # Order tokenizers to match the domain_term_analysis.py order
    tokenizer_order = [
        "kl3m-004-128k-cased",
        "kl3m-004-128k-uncased",
        "kl3m-003-64k",
        "gpt-4o",
        "llama3",
        "gpt2"
    ]
    
    # Filter to available tokenizers and maintain order
    tokenizers = [t for t in tokenizer_order if t in tokenizer_names]
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['Dataset'] + tokenizers)
        
        # Write data rows (sort datasets but keep "Total" at the end)
        datasets = sorted([d for d in token_count_data.keys() if d != "Total"])
        if "Total" in token_count_data:
            datasets.append("Total")
        
        for dataset in datasets:
            row = [dataset]
            for tokenizer in tokenizers:
                token_count = token_count_data[dataset].get(tokenizer, "N/A")
                if isinstance(token_count, (int, float)):
                    row.append(f"{token_count}")
                else:
                    row.append(token_count)
            writer.writerow(row)
    
    print(f"Token count CSV data saved to {output_path}")


def generate_token_counts_latex_table(token_count_data, output_path):
    """Generate a LaTeX table for token count data.
    
    Args:
        token_count_data: Dictionary mapping datasets to tokenizer token counts
        output_path: Path to save the LaTeX table
    """
    # Get all tokenizer names from results
    tokenizer_names = set()
    for dataset_results in token_count_data.values():
        tokenizer_names.update(dataset_results.keys())
    
    # Order tokenizers to match the domain_term_analysis.py order
    tokenizer_order = [
        "kl3m-004-128k-cased",
        "kl3m-004-128k-uncased",
        "kl3m-003-64k",
        "gpt-4o",
        "llama3",
        "gpt2"
    ]
    
    # Filter to available tokenizers and maintain order
    tokenizers = [t for t in tokenizer_order if t in tokenizer_names]
    
    # Format tokenizer names for LaTeX (escape special characters)
    def format_for_latex(name):
        return name.replace("_", "\\_").replace("-", "\\mbox{-}")
    
    with open(output_path, 'w') as f:
        # Start the LaTeX table
        f.write("% LaTeX table for token count analysis\n")
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Total token count across datasets}\n")
        f.write("\\label{tab:token-count}\n")
        f.write("\\small\n")
        
        # Define table columns - One for dataset plus one for each tokenizer
        col_spec = "l" + "r" * len(tokenizers)
        f.write("\\begin{tabular}{" + col_spec + "}\n")
        f.write("\\toprule\n")
        
        # Header row
        f.write("Dataset")
        for tokenizer in tokenizers:
            f.write(f" & {format_for_latex(tokenizer)}")
        f.write(" \\\\\n")
        f.write("\\midrule\n")
        
        # Data rows (sort datasets but keep "Total" at the end)
        datasets = sorted([d for d in token_count_data.keys() if d != "Total"])
        
        for dataset in datasets:
            # Format dataset name for LaTeX
            latex_dataset = dataset.replace("_", "\\_").replace("&", "\\&")
            f.write(f"{latex_dataset}")
            
            # Get the most efficient (lowest) value for highlighting
            valid_values = [count for count in token_count_data[dataset].values() 
                           if isinstance(count, (int, float))]
            min_value = min(valid_values) if valid_values else None
            
            for tokenizer in tokenizers:
                token_count = token_count_data[dataset].get(tokenizer, "N/A")
                
                if isinstance(token_count, (int, float)):
                    # Format large numbers with comma separators
                    formatted_count = f"{token_count:,}"
                    
                    # Highlight the most efficient (lowest) value
                    if min_value is not None and token_count == min_value:
                        f.write(f" & \\textbf{{{formatted_count}}}")
                    else:
                        f.write(f" & {formatted_count}")
                else:
                    f.write(" & N/A")
            
            f.write(" \\\\\n")
        
        # Add the total row if it exists
        if "Total" in token_count_data:
            f.write("\\midrule\n")
            f.write("Total")
            
            # Get the most efficient (lowest) value for highlighting
            valid_values = [count for count in token_count_data["Total"].values() 
                           if isinstance(count, (int, float))]
            min_value = min(valid_values) if valid_values else None
            
            for tokenizer in tokenizers:
                token_count = token_count_data["Total"].get(tokenizer, "N/A")
                
                if isinstance(token_count, (int, float)):
                    # Format large numbers with comma separators
                    formatted_count = f"{token_count:,}"
                    
                    # Highlight the most efficient (lowest) value
                    if min_value is not None and token_count == min_value:
                        f.write(f" & \\textbf{{{formatted_count}}}")
                    else:
                        f.write(f" & {formatted_count}")
                else:
                    f.write(" & N/A")
            
            f.write(" \\\\\n")
        
        # End the LaTeX table
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption*{\\small \\textit{Note: Lower values indicate more efficient tokenization (fewer tokens to represent the same text).}}\n")
        f.write("\\end{table}\n")
    
    print(f"Token count LaTeX table saved to {output_path}")


def create_efficiency_visualizations(efficiency_data, analyzer, output_dir):
    """Create visualizations for token efficiency data.
    
    Args:
        efficiency_data: Dictionary mapping datasets to tokenizer efficiency values
        analyzer: The TokenizerAnalyzer instance for getting vocab sizes
        output_dir: Directory to save outputs
    """
    # Setup plot style
    setup_plot_style()
    
    # Define a consistent color scheme using constants
    colors = {
        'kl3m-004-128k-cased': KL3M_STANDARD_COLORS[0],    # Blue
        'kl3m-004-128k-uncased': KL3M_STANDARD_COLORS[1],  # Lighter blue
        'kl3m-003-64k': KL3M_STANDARD_COLORS[2],           # Even lighter blue
        'gpt-4o': OTHER_COLORS[0],                         # Red
        'llama3': OTHER_COLORS[2],                         # Orange
        'gpt2': OTHER_COLORS[3],                           # Lighter orange
    }
    
    # 1. Create bar chart of average efficiency, sorted by efficiency
    if "Average" in efficiency_data:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        averages = efficiency_data["Average"]
        
        # Sort tokenizers by efficiency (ascending - lower is better)
        sorted_tokenizers = sorted(averages.keys(), key=lambda t: averages[t])
        sorted_values = [averages[t] for t in sorted_tokenizers]
        
        # Create a color list in the same order as the sorted tokenizers
        bar_colors = [colors.get(t, '#333333') for t in sorted_tokenizers]
        
        # Create the bar chart
        bars = ax.bar(sorted_tokenizers, sorted_values, color=bar_colors)
        
        # Add value labels on top of each bar with increased font size
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{height:.4f}', ha='center', va='bottom', 
                   fontsize=14, fontweight='bold')
        
        ax.set_title('Token Efficiency by Tokenizer (tokens per character)')
        ax.set_xlabel('Tokenizer')
        ax.set_ylabel('Tokens per Character (lower is better)')
        ax.set_ylim(0, max(sorted_values) * 1.2)  # Add some headroom for labels
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save figure
        save_figure(fig, 'token_efficiency_bars')
        plt.close(fig)
    
    # 2. Create scatter plot of vocab size vs efficiency
    # Get vocabulary sizes
    vocab_sizes = analyzer.analyze_vocab_size()
    
    # Filter out character tokenizers and excluded tokenizers
    vocab_sizes = {name: size for name, size in vocab_sizes.items() 
                  if 'char' not in name and name != 'kl3m-001-32k' and name != 'roberta-base'}
    
    if "Average" in efficiency_data and vocab_sizes:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        averages = efficiency_data["Average"]
        
        # Create data for scatter plot
        x_data = []  # Vocab sizes
        y_data = []  # Efficiency values
        labels = []  # Tokenizer names
        point_colors = []  # Colors
        
        for tokenizer, efficiency in averages.items():
            if tokenizer in vocab_sizes:
                x_data.append(vocab_sizes[tokenizer])
                y_data.append(efficiency)
                labels.append(tokenizer)
                point_colors.append(colors.get(tokenizer, '#333333'))
        
        # Create the scatter plot
        scatter = ax.scatter(x_data, y_data, c=point_colors, s=100)
        
        # Add labels for each point with increased font size
        for i, label in enumerate(labels):
            ax.annotate(label, (x_data[i], y_data[i]), 
                       xytext=(10, 5), textcoords='offset points',
                       fontsize=14, fontweight='bold')
        
        ax.set_title('Vocabulary Size vs. Token Efficiency')
        ax.set_xlabel('Vocabulary Size')
        ax.set_ylabel('Tokens per Character (lower is better)')
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add best-fit line to show trend
        if len(x_data) > 1:
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            ax.plot(sorted(x_data), p(sorted(x_data)), "r--", alpha=0.5)
        
        # Save figure
        save_figure(fig, 'vocab_size_vs_efficiency')
        plt.close(fig)


if __name__ == "__main__":
    analyze_token_efficiency()