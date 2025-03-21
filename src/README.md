# KL3M Tokenizer Analysis Code

This directory contains the source code for analyzing KL3M tokenizers and generating figures and statistics for the KL3M tokenizer paper.

## Setup

The environment is managed with `uv`. To set up:

```bash
# Create and activate virtual environment
uv venv
# Install dependencies
uv pip install -e .
```

## Structure

- **kl3m_analysis/**: Core package with shared functionality
  - `analyzer.py`: TokenizerAnalyzer class for loading and analyzing tokenizers
  - `constants.py`: Shared constants, sample texts, and configuration
  - `utils.py`: Utility functions for saving data and common operations
  - `visualization.py`: Plotting and visualization utilities

- **Analysis Scripts**:
  - `token_efficiency_analysis.py`: Compares token counts across different tokenizers
  - `token_size_distribution.py`: Analyzes token length distribution
  - `domain_term_analysis.py`: Evaluates tokenization of domain-specific terminology

## Running the Analysis

Each analysis script can be run independently:

```bash
# Run token efficiency analysis
uv run python token_efficiency_analysis.py

# Run token size distribution analysis
uv run python token_size_distribution.py

# Run domain-specific term analysis
uv run python domain_term_analysis.py
```

## Known Issues and Limitations

1. **Dataset Limit**: The analysis scripts use a small limit (20 samples per dataset in `token_efficiency_analysis.py`) for faster execution. For full analysis, increase the `limit` parameter.

2. **Tokenizer Availability**: The scripts need access to specific tokenizers. Some tokenizers may require authentication with the Hugging Face Hub.

3. **Default Path Assumptions**: The scripts assume outputs go to `../figures/`. Ensure this directory exists.

4. **Missing analyze_domain_specific_tokens()**: There's a reference to this function in the `compute_all_statistics()` method in analyzer.py, but it's not implemented.

5. **Error Handling**: Some tokenizer loading operations might fail if models aren't available - these are caught but might affect results.

## For More Information

See the main README.md in the project root for complete documentation on the repository structure and paper.