#!/usr/bin/env python
"""
Script to analyze domain-specific term tokenization across different tokenizers.
"""

import csv
import os
from pathlib import Path

from kl3m_analysis.analyzer import TokenizerAnalyzer
from kl3m_analysis.constants import LEGAL_TERMS, FINANCIAL_TERMS, OUTPUT_DIR


def generate_term_token_counts():
    """Generate token counts for domain-specific terms."""
    # Initialize analyzer
    analyzer = TokenizerAnalyzer()
    
    # Load all tokenizers
    analyzer.load_all_tokenizers()
    
    # Prepare output directories
    figures_dir = Path(OUTPUT_DIR)
    figures_dir.mkdir(exist_ok=True)
    
    # Process domain terms
    legal_results = process_terms(analyzer, LEGAL_TERMS, "Legal")
    financial_results = process_terms(analyzer, FINANCIAL_TERMS, "Financial")
    
    # Calculate aggregate statistics
    aggregate_stats = calculate_aggregate_stats(legal_results, financial_results)
    
    # Print aggregate statistics to stdout
    print_aggregate_stats(aggregate_stats)
    
    # Save results to CSV
    save_to_csv(legal_results, financial_results, figures_dir / "domain_term_counts.csv")
    
    # Generate LaTeX tables
    generate_latex_table(legal_results, financial_results, figures_dir / "domain_term_comparison.txt")
    generate_aggregate_latex_table(aggregate_stats, figures_dir / "domain_term_aggregate.txt")
    
    print(f"Results saved to {figures_dir}")


def process_terms(analyzer, terms, domain_name):
    """Process a list of terms and calculate token counts for each tokenizer.
    
    Args:
        analyzer: The TokenizerAnalyzer instance
        terms: List of domain-specific terms
        domain_name: Name of the domain (for logging)
        
    Returns:
        Dictionary mapping terms to tokenizer name to token count
    """
    results = {}
    
    print(f"Processing {domain_name} terms...")
    for term in terms:
        term_results = {}
        print(f"  Term: {term}")
        
        # Process with standard tokenizers
        for name, tokenizer in analyzer.tokenizers.items():
            try:
                encoding = tokenizer.encode(term)
                token_count = len(encoding.ids)
                term_results[name] = token_count
                print(f"    {name}: {token_count} tokens")
            except Exception as e:
                print(f"    Error with {name}: {e}")
                term_results[name] = None
        
        # Process with tiktoken models
        for name, encoder in analyzer.tiktoken_models.items():
            try:
                encoding = encoder.encode(term)
                token_count = len(encoding)
                term_results[name] = token_count
                print(f"    {name}: {token_count} tokens")
            except Exception as e:
                print(f"    Error with {name}: {e}")
                term_results[name] = None
        
        results[term] = term_results
    
    return results


def save_to_csv(legal_results, financial_results, output_path):
    """Save results to a CSV file.
    
    Args:
        legal_results: Dictionary of legal term results
        financial_results: Dictionary of financial term results
        output_path: Path to save the CSV file
    """
    # Gather all tokenizer names from results
    tokenizer_names = set()
    for results in [legal_results, financial_results]:
        for term_results in results.values():
            tokenizer_names.update(term_results.keys())
    
    tokenizer_names = sorted(tokenizer_names)
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['Domain', 'Term'] + tokenizer_names)
        
        # Write legal terms
        for term, counts in legal_results.items():
            row = ['Legal', term]
            for tokenizer in tokenizer_names:
                row.append(counts.get(tokenizer, 'N/A'))
            writer.writerow(row)
        
        # Write financial terms
        for term, counts in financial_results.items():
            row = ['Financial', term]
            for tokenizer in tokenizer_names:
                row.append(counts.get(tokenizer, 'N/A'))
            writer.writerow(row)
    
    print(f"CSV data saved to {output_path}")


def calculate_aggregate_stats(legal_results, financial_results):
    """Calculate aggregate statistics across all tokenizers.
    
    Args:
        legal_results: Dictionary of legal term results
        financial_results: Dictionary of financial term results
        
    Returns:
        Dictionary with aggregate statistics per tokenizer and domain
    """
    # Initialize result structure
    stats = {
        "legal": {},
        "financial": {},
        "overall": {}
    }
    
    # Get all tokenizer names
    all_tokenizers = set()
    for results in [legal_results, financial_results]:
        for term_results in results.values():
            all_tokenizers.update(term_results.keys())
    
    # Initialize counters
    for tokenizer in all_tokenizers:
        stats["legal"][tokenizer] = []
        stats["financial"][tokenizer] = []
    
    # Collect all token counts
    for term, counts in legal_results.items():
        for tokenizer, count in counts.items():
            if count is not None:
                stats["legal"][tokenizer].append(count)
    
    for term, counts in financial_results.items():
        for tokenizer, count in counts.items():
            if count is not None:
                stats["financial"][tokenizer].append(count)
    
    # Calculate means for each tokenizer
    for tokenizer in all_tokenizers:
        legal_counts = stats["legal"].get(tokenizer, [])
        financial_counts = stats["financial"].get(tokenizer, [])
        
        if legal_counts:
            stats["legal"][tokenizer] = sum(legal_counts) / len(legal_counts)
        else:
            stats["legal"][tokenizer] = None
            
        if financial_counts:
            stats["financial"][tokenizer] = sum(financial_counts) / len(financial_counts)
        else:
            stats["financial"][tokenizer] = None
        
        # Calculate overall mean
        all_counts = legal_counts + financial_counts
        if all_counts:
            stats["overall"][tokenizer] = sum(all_counts) / len(all_counts)
        else:
            stats["overall"][tokenizer] = None
    
    return stats


def print_aggregate_stats(stats):
    """Print aggregate statistics to stdout.
    
    Args:
        stats: Dictionary with aggregate statistics
    """
    print("\n===== AGGREGATE TOKEN COUNT STATISTICS =====")
    
    # Select key tokenizers to display
    key_tokenizers = [
        "kl3m-004-128k-cased",
        "kl3m-004-char-16k-cased",
        "gpt-4o",
        "llama3",
        "gpt2",
        "roberta-base"
    ]
    
    # Print header
    print(f"{'Tokenizer':<25} {'Legal Mean':<12} {'Financial Mean':<15} {'Overall Mean':<12}")
    print("-" * 65)
    
    # Print data for each tokenizer
    for tokenizer in sorted(key_tokenizers):
        legal_mean = stats["legal"].get(tokenizer)
        financial_mean = stats["financial"].get(tokenizer)
        overall_mean = stats["overall"].get(tokenizer)
        
        legal_str = f"{legal_mean:.2f}" if legal_mean is not None else "N/A"
        financial_str = f"{financial_mean:.2f}" if financial_mean is not None else "N/A"
        overall_str = f"{overall_mean:.2f}" if overall_mean is not None else "N/A"
        
        print(f"{tokenizer:<25} {legal_str:<12} {financial_str:<15} {overall_str:<12}")
    
    print("\n")


def generate_aggregate_latex_table(stats, output_path):
    """Generate LaTeX table for aggregate statistics.
    
    Args:
        stats: Dictionary with aggregate statistics
        output_path: Path to save the LaTeX table
    """
    # Select key tokenizers to include in the table - order matters
    key_tokenizers = [
        "kl3m-004-128k-cased",      # First column - KL3M cased model
        "kl3m-004-128k-uncased",    # Second column - KL3M uncased model
        "gpt-4o",                   # Third column - GPT-4o
        "llama3",                   # Fourth column - Llama3
        "roberta-base",             # Fifth column - RoBERTa
        "gpt2"                      # Sixth column - GPT-2
    ]
    
    # Format tokenizer names for LaTeX (escape special characters)
    def format_for_latex(name):
        return name.replace("_", "\\_").replace("-", "\\mbox{-}")
    
    with open(output_path, 'w') as f:
        # Start the LaTeX table
        f.write("% LaTeX table for domain-specific term tokenization aggregate statistics\n")
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Average token count by domain across tokenizers}\n")
        f.write("\\label{tab:domain-term-aggregate}\n")
        
        # Define table columns
        f.write("\\begin{tabular}{lrrr}\n")
        f.write("\\toprule\n")
        
        # Header row
        f.write("Tokenizer & Legal Terms & Financial Terms & Overall \\\\\n")
        f.write("\\midrule\n")
        
        # Get the minimum values for highlighting
        legal_mins = {t: stats["legal"][t] for t in key_tokenizers if stats["legal"].get(t) is not None}
        financial_mins = {t: stats["financial"][t] for t in key_tokenizers if stats["financial"].get(t) is not None}
        overall_mins = {t: stats["overall"][t] for t in key_tokenizers if stats["overall"].get(t) is not None}
        
        min_legal = min(legal_mins.values()) if legal_mins else None
        min_financial = min(financial_mins.values()) if financial_mins else None
        min_overall = min(overall_mins.values()) if overall_mins else None
        
        # Data rows
        for tokenizer in key_tokenizers:
            latex_name = format_for_latex(tokenizer)
            f.write(f"{latex_name}")
            
            # Legal terms
            legal_mean = stats["legal"].get(tokenizer)
            if legal_mean is not None:
                if min_legal is not None and abs(legal_mean - min_legal) < 0.001:  # Compare with small epsilon for float comparison
                    f.write(f" & \\textbf{{{legal_mean:.2f}}}")
                else:
                    f.write(f" & {legal_mean:.2f}")
            else:
                f.write(" & N/A")
            
            # Financial terms
            financial_mean = stats["financial"].get(tokenizer)
            if financial_mean is not None:
                if min_financial is not None and abs(financial_mean - min_financial) < 0.001:
                    f.write(f" & \\textbf{{{financial_mean:.2f}}}")
                else:
                    f.write(f" & {financial_mean:.2f}")
            else:
                f.write(" & N/A")
            
            # Overall
            overall_mean = stats["overall"].get(tokenizer)
            if overall_mean is not None:
                if min_overall is not None and abs(overall_mean - min_overall) < 0.001:
                    f.write(f" & \\textbf{{{overall_mean:.2f}}}")
                else:
                    f.write(f" & {overall_mean:.2f}")
            else:
                f.write(" & N/A")
            
            f.write(" \\\\\n")
        
        # End the LaTeX table
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"Aggregate LaTeX table saved to {output_path}")


def generate_latex_table(legal_results, financial_results, output_path):
    """Generate LaTeX table format for the results.
    
    Args:
        legal_results: Dictionary of legal term results
        financial_results: Dictionary of financial term results
        output_path: Path to save the LaTeX table
    """
    # Select key tokenizers to include in the table (to keep it reasonably sized)
    # Order matters for the visual comparison
    key_tokenizers = [
        "kl3m-004-128k-cased",      # First column - KL3M cased model
        "kl3m-004-128k-uncased",    # Second column - KL3M uncased model
        "gpt-4o",                   # Third column - GPT-4o
        "llama3",                   # Fourth column - Llama3
        "roberta-base",             # Fifth column - RoBERTa
        "gpt2"                      # Sixth column - GPT-2
    ]
    
    # Format tokenizer names for LaTeX (escape special characters)
    def format_for_latex(name):
        return name.replace("_", "\\_").replace("-", "\\mbox{-}")
    
    with open(output_path, 'w') as f:
        # Start the LaTeX table
        f.write("% LaTeX table for domain-specific term tokenization comparison\n")
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Token count comparison for domain-specific terminology across tokenizers}\n")
        f.write("\\label{tab:domain-term-comparison}\n")
        f.write("\\small\n")
        
        # Define table columns - Using a more readable format with left alignment for text
        # One column for Domain, one for Term, and one for each tokenizer (r for right-aligned numbers)
        f.write("\\begin{tabular}{@{}llrrrrrr@{}}\n")
        f.write("\\toprule\n")
        
        # Header row with rotated column headers for tokenizers to save space
        f.write("Domain & Term")
        for tokenizer in key_tokenizers:
            latex_name = format_for_latex(tokenizer)
            f.write(f" & \\rotatebox{{90}}{{{latex_name}}}")
        f.write(" \\\\\n")
        f.write("\\midrule\n")
        
        # Legal terms
        for i, (term, counts) in enumerate(legal_results.items()):
            # Add domain name only for the first row
            if i == 0:
                f.write("Legal")
            else:
                f.write("")
            
            # Format term with special characters for LaTeX compatibility
            latex_term = term.replace("§", "\\S").replace("_", "\\_").replace("#", "\\#")
            f.write(f" & {latex_term}")
            
            for tokenizer in key_tokenizers:
                count = counts.get(tokenizer, "N/A")
                # Highlight the best (smallest) token count in each row
                valid_counts = [c for t, c in counts.items() if t in key_tokenizers and c is not None]
                if valid_counts and count == min(valid_counts):
                    f.write(f" & \\textbf{{{count}}}")
                else:
                    f.write(f" & {count}")
            
            f.write(" \\\\\n")
            
            # Add a small gap after the last legal term
            if i == len(legal_results) - 1:
                f.write("\\addlinespace[0.5em]\n")
        
        # Financial terms
        for i, (term, counts) in enumerate(financial_results.items()):
            # Add domain name only for the first row
            if i == 0:
                f.write("Financial")
            else:
                f.write("")
            
            # Format term with special characters for LaTeX compatibility
            latex_term = term.replace("§", "\\S").replace("_", "\\_").replace("#", "\\#").replace("&", "\\&")
            f.write(f" & {latex_term}")
            
            for tokenizer in key_tokenizers:
                count = counts.get(tokenizer, "N/A")
                # Highlight the best (smallest) token count in each row
                valid_counts = [c for t, c in counts.items() if t in key_tokenizers and c is not None]
                if valid_counts and count == min(valid_counts):
                    f.write(f" & \\textbf{{{count}}}")
                else:
                    f.write(f" & {count}")
                    
            f.write(" \\\\\n")
        
        # End the LaTeX table
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"LaTeX table saved to {output_path}")


if __name__ == "__main__":
    generate_term_token_counts()