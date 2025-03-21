#!/usr/bin/env python
"""
Script to analyze and visualize token size distributions across different tokenizers.
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
from kl3m_analysis.constants import OUTPUT_DIR
from kl3m_analysis.visualization import setup_plot_style, group_tokenizers
from kl3m_analysis.utils import save_figure, save_text


def analyze_token_sizes():
    """Analyze and visualize token size distributions."""
    # Initialize analyzer
    analyzer = TokenizerAnalyzer()
    
    # Load all tokenizers
    analyzer.load_all_tokenizers()
    
    # Prepare output directories
    figures_dir = Path(OUTPUT_DIR)
    figures_dir.mkdir(exist_ok=True)
    
    # Analyze token lengths
    token_lengths = analyzer.analyze_token_lengths()
    
    # Filter out character tokenizers and prepare data for visualization
    filtered_token_lengths = {
        name: lengths for name, lengths in token_lengths.items() 
        if 'char' not in name
    }
    
    # Note: For tiktoken models like GPT-4o, only a subset of tokens
    # can be individually decoded, so stats are based on a partial sample
    
    # Create visualization
    create_token_size_visualization(filtered_token_lengths, figures_dir)
    
    # Save results to CSV
    save_to_csv(filtered_token_lengths, figures_dir / "token_size_distribution.csv")
    
    # Generate LaTeX table
    generate_latex_table(filtered_token_lengths, figures_dir / "token_size_distribution.txt")
    
    # Print summary statistics
    print("\n===== TOKEN SIZE DISTRIBUTION SUMMARY =====")
    
    # Order tokenizers to match the domain_term_analysis.py order
    tokenizer_order = [
        "kl3m-004-128k-cased",
        "kl3m-004-128k-uncased",
        "gpt-4o",
        "llama3",
        "roberta-base",
        "gpt2"
    ]
    
    # Filter to available tokenizers and maintain order
    tokenizers = [t for t in tokenizer_order if t in filtered_token_lengths]
    
    # Print header
    print(f"{'Tokenizer':<25} {'Short Tokens (≤5)':<15} {'Medium Tokens (6-10)':<15} {'% Short':<10}")
    print("-" * 65)
    
    # Calculate stats for each tokenizer
    for tokenizer in tokenizers:
        lengths = filtered_token_lengths[tokenizer]
        
        # Count tokens with length <= 5
        short_tokens = sum(count for length, count in lengths.items() 
                         if int(length) <= 5)
        
        # Count tokens with length 6-10
        medium_tokens = sum(count for length, count in lengths.items() 
                          if 6 <= int(length) <= 10)
        
        # Calculate percentage of short tokens
        total = short_tokens + medium_tokens
        pct_short = (short_tokens / total) * 100 if total > 0 else 0
        
        # Format numbers with thousands separator
        short_str = f"{short_tokens:,}"
        medium_str = f"{medium_tokens:,}"
        
        print(f"{tokenizer:<25} {short_str:<15} {medium_str:<15} {pct_short:.1f}%")
    
    print(f"\nResults saved to {figures_dir}")


def create_token_size_visualization(token_lengths, output_dir):
    """Create visualizations for token size distributions.
    
    Args:
        token_lengths: Dictionary mapping tokenizer names to token length distributions
        output_dir: Directory to save outputs
    """
    # Setup plot style
    setup_plot_style()
    
    # Import colors from constants
    from kl3m_analysis.constants import KL3M_STANDARD_COLORS, OTHER_COLORS, LINESTYLES, MARKERS
    
    # Define a consistent color scheme using constants
    colors = {
        'kl3m-004-128k-cased': KL3M_STANDARD_COLORS[0],    # First KL3M standard color
        'kl3m-004-128k-uncased': KL3M_STANDARD_COLORS[1],  # Second KL3M standard color
        'gpt-4o': OTHER_COLORS[0],                         # First other color (red)
        'llama3': OTHER_COLORS[2],                         # Third other color (orange)
        'roberta-base': OTHER_COLORS[1],                   # Second other color (darker red)
        'gpt2': OTHER_COLORS[3],                           # Fourth other color (lighter orange)
    }
    
    # Define markers to differentiate lines
    markers = {
        'kl3m-004-128k-cased': MARKERS['standard'][0],
        'kl3m-004-128k-uncased': MARKERS['standard'][1],
        'gpt-4o': MARKERS['other'][0],
        'llama3': MARKERS['other'][1],
        'roberta-base': MARKERS['other'][2],
        'gpt2': MARKERS['other'][3],
    }
    
    # Define line styles
    line_styles = {
        'kl3m-004-128k-cased': LINESTYLES[0],
        'kl3m-004-128k-uncased': LINESTYLES[1],
        'gpt-4o': LINESTYLES[0],
        'llama3': LINESTYLES[1],
        'roberta-base': LINESTYLES[2],
        'gpt2': LINESTYLES[3],
    }
    
    # Order tokenizers to match the domain_term_analysis.py order
    tokenizer_order = [
        "kl3m-004-128k-cased",
        "kl3m-004-128k-uncased",
        "gpt-4o",
        "llama3",
        "roberta-base",
        "gpt2"
    ]
    
    # Filtered tokenizers in the desired order
    filtered_tokenizers = [t for t in tokenizer_order if t in token_lengths]
    
    # Create figure for histogram (all token lengths combined)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for histogram
    data = []
    
    # Debug the token_lengths structure
    print("Token length structure:")
    for tokenizer, lengths in token_lengths.items():
        print(f"{tokenizer}: {list(lengths.items())[:5]} ... (total items: {len(lengths)})")
    
    for tokenizer in filtered_tokenizers:
        if tokenizer not in token_lengths:
            continue
            
        lengths = token_lengths[tokenizer]
        for length, count in lengths.items():
            try:
                length_int = int(length)
                # Filter to tokens of 10 characters or less
                if length_int <= 10:
                    data.extend([length_int] * count)
            except ValueError:
                # Skip if length can't be converted to int
                print(f"Warning: Could not convert {length} to int for {tokenizer}")
                continue
                
    # Create DataFrame for seaborn
    df = pd.DataFrame(data, columns=['Token Length'])
    
    # Create histogram
    sns.histplot(data=df, x='Token Length', bins=range(1, 12), kde=False, ax=ax)
    
    ax.set_title('Distribution of Token Sizes (≤ 10 characters)')
    ax.set_xlabel('Token Length (characters)')
    ax.set_ylabel('Count')
    ax.set_xticks(range(1, 11))
    
    # Save figure
    save_figure(fig, 'token_size_histogram')
    plt.close(fig)
    
    # Create a line plot figure comparing tokenizers using percentages
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate total vocabulary size for each tokenizer
    vocab_sizes = {}
    for tokenizer in filtered_tokenizers:
        vocab_sizes[tokenizer] = sum(token_lengths[tokenizer].values())
    
    # Prepare data for line plot - convert to percentages
    for tokenizer in filtered_tokenizers:
        lengths = token_lengths[tokenizer]
        x_values = range(1, 11)
        y_values = []
        
        for length in x_values:
            # Try both the integer and string versions of the length key
            count = lengths.get(length, 0) or lengths.get(str(length), 0)
            # Calculate as percentage of vocabulary
            percentage = (count / vocab_sizes[tokenizer]) * 100 if vocab_sizes[tokenizer] > 0 else 0
            y_values.append(percentage)
        
        # Plot line for this tokenizer
        ax.plot(x_values, y_values, label=tokenizer, 
                color=colors.get(tokenizer, '#333333'),
                marker=markers.get(tokenizer, 'o'),
                linestyle=line_styles.get(tokenizer, '-'),
                linewidth=2,
                markersize=8)
    
    # Customize the plot
    ax.set_title('Token Size Distribution by Tokenizer (≤ 10 characters)')
    ax.set_xlabel('Token Length (characters)')
    ax.set_ylabel('Percentage of Vocabulary')
    ax.set_xticks(range(1, 11))
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0, max([25, ax.get_ylim()[1]]))  # Ensure y-axis doesn't go too high
    ax.grid(True, alpha=0.3)
    
    # Add legend with better positioning and increased font size
    ax.legend(title='Tokenizer', bbox_to_anchor=(1.02, 1), loc='upper left', 
             fontsize=14, title_fontsize=16)
    
    # Save figure
    save_figure(fig, 'token_size_comparison')
    plt.close(fig)
    
    # Create a third figure with stacked bar chart for different token length categories
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define categories
    categories = ['1-5 chars', '6-10 chars', '>10 chars']
    
    # Calculate percentages for each category
    data = {tokenizer: [] for tokenizer in filtered_tokenizers}
    
    for tokenizer in filtered_tokenizers:
        lengths = token_lengths[tokenizer]
        total = vocab_sizes[tokenizer]
        
        # Count tokens in each category
        count_1_5 = sum(count for length, count in lengths.items() 
                       if int(length) <= 5)
        count_6_10 = sum(count for length, count in lengths.items() 
                        if 6 <= int(length) <= 10)
        count_over_10 = sum(count for length, count in lengths.items() 
                           if int(length) > 10)
        
        # Convert to percentages
        pct_1_5 = (count_1_5 / total) * 100 if total > 0 else 0
        pct_6_10 = (count_6_10 / total) * 100 if total > 0 else 0
        pct_over_10 = (count_over_10 / total) * 100 if total > 0 else 0
        
        data[tokenizer] = [pct_1_5, pct_6_10, pct_over_10]
    
    # Create DataFrame for plotting
    df = pd.DataFrame(data, index=categories)
    
    # Plot stacked bar chart
    df.plot(kind='bar', stacked=True, ax=ax, color=[colors.get(t, '#333333') for t in df.columns])
    
    # Add value labels on bars
    for i, tokenizer in enumerate(filtered_tokenizers):
        bottom = 0
        for j, category in enumerate(categories):
            value = df.loc[category, tokenizer]
            if value > 5:  # Only label if value is significant enough to see
                ax.text(i, bottom + value/2, f"{value:.1f}%", 
                        ha='center', va='center', color='white', fontweight='bold')
            bottom += value
    
    ax.set_title('Token Size Categories by Tokenizer')
    ax.set_xlabel('Token Length Category')
    ax.set_ylabel('Percentage of Vocabulary')
    ax.set_ylim(0, 100)  # Set y-axis from 0 to 100%
    ax.legend(title='Tokenizer', bbox_to_anchor=(1.02, 1), loc='upper left',
             fontsize=14, title_fontsize=16)
    
    # Save figure
    save_figure(fig, 'token_size_categories')
    plt.close(fig)


def save_to_csv(token_lengths, output_path):
    """Save token length distributions to a CSV file.
    
    Args:
        token_lengths: Dictionary mapping tokenizer names to token length distributions
        output_path: Path to save the CSV file
    """
    # Determine all unique token lengths
    all_lengths = set()
    for lengths in token_lengths.values():
        all_lengths.update(int(length) for length in lengths.keys())
    
    # Filter to lengths <= 10 and sort
    all_lengths = sorted([l for l in all_lengths if l <= 10])
    
    # Order tokenizers to match the domain_term_analysis.py order
    tokenizer_order = [
        "kl3m-004-128k-cased",
        "kl3m-004-128k-uncased",
        "gpt-4o",
        "llama3",
        "roberta-base",
        "gpt2"
    ]
    
    # Filter to available tokenizers and maintain order
    tokenizers = [t for t in tokenizer_order if t in token_lengths]
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['Token Length'] + tokenizers)
        
        # Write data rows
        for length in all_lengths:
            row = [length]
            for tokenizer in tokenizers:
                # Try both the integer and string versions of the length key
                count = token_lengths[tokenizer].get(length, 0) or token_lengths[tokenizer].get(str(length), 0)
                row.append(count)
            writer.writerow(row)
        
        # Add a summary row with total tokens in each size range
        row = ['Total (≤10 chars)']
        for tokenizer in tokenizers:
            total = sum(count for length, count in token_lengths[tokenizer].items() 
                        if int(length) <= 10)
            row.append(total)
        writer.writerow(row)
        
        # Add a row with percentage of vocabulary
        row = ['% of Vocabulary']
        for tokenizer in tokenizers:
            total_vocab = sum(token_lengths[tokenizer].values())
            total_small = sum(count for length, count in token_lengths[tokenizer].items() 
                             if int(length) <= 10)
            percentage = (total_small / total_vocab) * 100 if total_vocab > 0 else 0
            row.append(f"{percentage:.2f}%")
        writer.writerow(row)
    
    print(f"CSV data saved to {output_path}")


def generate_latex_table(token_lengths, output_path):
    """Generate a LaTeX table for token length distributions.
    
    Args:
        token_lengths: Dictionary mapping tokenizer names to token length distributions
        output_path: Path to save the LaTeX table
    """
    # Determine all unique token lengths
    all_lengths = set()
    for lengths in token_lengths.values():
        all_lengths.update(int(length) for length in lengths.keys())
    
    # Filter to lengths <= 10 and sort
    all_lengths = sorted([l for l in all_lengths if l <= 10])
    
    # Order tokenizers to match the domain_term_analysis.py order
    tokenizer_order = [
        "kl3m-004-128k-cased",
        "kl3m-004-128k-uncased",
        "gpt-4o",
        "llama3",
        "roberta-base",
        "gpt2"
    ]
    
    # Filter to available tokenizers and maintain order
    tokenizers = [t for t in tokenizer_order if t in token_lengths]
    
    # Format tokenizer names for LaTeX (escape special characters)
    def format_for_latex(name):
        return name.replace("_", "\\_").replace("-", "\\mbox{-}")
    
    with open(output_path, 'w') as f:
        # Start the LaTeX table
        f.write("% LaTeX table for token size distribution analysis\n")
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Token size distribution (percentage of vocabulary by character length)}\n")
        f.write("\\label{tab:token-size-distribution}\n")
        f.write("\\small\n")
        
        # Define table columns - One for length plus one for each tokenizer
        col_spec = "l" + "r" * len(tokenizers)
        f.write("\\begin{tabular}{" + col_spec + "}\n")
        f.write("\\toprule\n")
        
        # Header row
        f.write("Length")
        for tokenizer in tokenizers:
            f.write(f" & {format_for_latex(tokenizer)}")
        f.write(" \\\\\n")
        f.write("\\midrule\n")
        
        # Calculate total vocabulary size for each tokenizer
        vocab_sizes = {}
        for tokenizer in tokenizers:
            vocab_sizes[tokenizer] = sum(token_lengths[tokenizer].values())
        
        # Data rows as percentages
        for length in all_lengths:
            f.write(f"{length}")
            
            for tokenizer in tokenizers:
                # Try both the integer and string versions of the length key
                count = token_lengths[tokenizer].get(length, 0) or token_lengths[tokenizer].get(str(length), 0)
                # Calculate percentage
                percentage = (count / vocab_sizes[tokenizer]) * 100 if vocab_sizes[tokenizer] > 0 else 0
                f.write(f" & {percentage:.1f}\\%")
            
            f.write(" \\\\\n")
        
        # Add a summary row for tokens ≤ 5 characters
        f.write("\\midrule\n")
        f.write("Total $\\leq 5$")
        
        for tokenizer in tokenizers:
            total_small = sum(count for length, count in token_lengths[tokenizer].items() 
                             if int(length) <= 5)
            percentage = (total_small / vocab_sizes[tokenizer]) * 100 if vocab_sizes[tokenizer] > 0 else 0
            f.write(f" & {percentage:.1f}\\%")
        
        f.write(" \\\\\n")
        
        # Add a summary row for tokens 6-10 characters
        f.write("Total 6-10")
        
        for tokenizer in tokenizers:
            total_medium = sum(count for length, count in token_lengths[tokenizer].items() 
                             if 6 <= int(length) <= 10)
            percentage = (total_medium / vocab_sizes[tokenizer]) * 100 if vocab_sizes[tokenizer] > 0 else 0
            f.write(f" & {percentage:.1f}\\%")
        
        f.write(" \\\\\n")
        
        # Add total row for all tokens ≤ 10 characters
        f.write("Total $\\leq 10$")
        
        for tokenizer in tokenizers:
            total_small = sum(count for length, count in token_lengths[tokenizer].items() 
                             if int(length) <= 10)
            percentage = (total_small / vocab_sizes[tokenizer]) * 100 if vocab_sizes[tokenizer] > 0 else 0
            # Highlight the highest percentage in bold
            f.write(f" & {percentage:.1f}\\%")
        
        f.write(" \\\\\n")
        
        # End the LaTeX table
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption*{\\small \\textit{Note: For tiktoken models like GPT-4o, only a subset of tokens can be individually decoded, so statistics are based on a partial sample.}}\n")
        f.write("\\end{table}\n")
    
    print(f"LaTeX table saved to {output_path}")


if __name__ == "__main__":
    analyze_token_sizes()