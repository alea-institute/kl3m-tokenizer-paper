# KL3M Tokenizers: A Family of Domain-Specific and Character-Level Tokenizers

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Authors

- Michael J. Bommarito II (ALEA Institute, Stanford CodeX) - Contact: hello@aleainstitute.ai
- Daniel Martin Katz (Illinois Tech - Chicago Kent Law, Bucerius Law School, ALEA Institute, Stanford CodeX)
- Jillian Bommarito (ALEA Institute)

## Table of Contents

- [Abstract](#abstract)
- [Paper](#paper)
  - [Building the Paper](#building-the-paper)
  - [Paper Structure](#paper-structure)
- [Code](#code)
  - [Environment Setup](#environment-setup)
  - [Analysis Scripts](#analysis-scripts)
  - [Running the Code](#running-the-code)
- [Figures and Results](#figures-and-results)
- [Replication Guide](#replication-guide)
- [Citation](#citation)
- [License](#license)

## Abstract

We present the KL3M tokenizers, a family of specialized tokenizers for legal, financial, and governmental text. Despite established work on tokenization, specialized tokenizers for professional domains remain understudied. Our paper offers two main contributions to this area.

First, we introduce domain-specific BPE tokenizers for legal, financial, and governmental text. Our kl3m-004-128k-cased tokenizer uses 9-17% fewer tokens than GPT-4o and Llama3 for domain-specific documents, despite having a smaller vocabulary. For specialized terminology, our cased tokenizer is even more efficient, using up to 83% fewer tokens for legal terms and 39% fewer tokens for financial terms.

Second, we develop character-level BPE tokenizers (4K, 8K, and 16K vocabulary sizes) for text correction tasks like OCR post-processing. These tokenizers keep consistent token boundaries between error-containing and correct text, making it easier for models to learn correction patterns.

These tokenizers help professional applications by fitting more text in context windows, reducing computational needs, and preserving the meaning of domain-specific terms. Our analysis shows these efficiency gains directly benefit the processing of long legal and financial documents. We release all tokenizers and code through GitHub and Hugging Face to support further research in specialized tokenization.

## Paper

This repository contains the LaTeX source code for the paper "KL3M Tokenizers: A Family of Domain-Specific and Character-Level Tokenizers for Legal, Financial, and Preprocessing Applications."

### Building the Paper

To build the paper, you need LaTeX installed on your system. We provide a Makefile to simplify the build process:

```bash
# Build the paper
cd kl3m-tokenizer-paper && make

# Clean and rebuild
make rebuild
```

### Paper Structure

The paper is organized into the following sections:

- Introduction: Overview of tokenization challenges in specialized domains
- Background: Review of tokenization methods and related work
- Methodology: Description of our tokenizer development approach
- Evaluation: Evaluation methodology and metrics
- Results: Comparison with other tokenizers and performance analysis
- Discussion: Implications for domain-specific applications
- Conclusion: Summary and future work
- Appendix: Additional details and analyses

## Code

The repository includes the Python code used to analyze tokenizer performance and generate figures for the paper.

### Environment Setup

The Python environment is managed with `uv`. To set up the environment:

```bash
cd kl3m-tokenizer-paper/src
uv venv
uv pip install -e .
```

### Analysis Scripts

The main analysis scripts are:

- `token_efficiency_analysis.py`: Compares token counts across different tokenizers
- `token_size_distribution.py`: Analyzes token length distribution
- `domain_term_analysis.py`: Evaluates tokenization of domain-specific terminology

### Running the Code

To run the analysis scripts:

```bash
cd kl3m-tokenizer-paper/src
uv run python token_efficiency_analysis.py
uv run python token_size_distribution.py
uv run python domain_term_analysis.py
```

## Figures and Results

The `figures/` directory contains the output from our analyses:

- Token efficiency comparisons (`token_efficiency_*.{png,pdf}`)
- Token size distributions (`token_size_*.{png,pdf}`)
- Domain-specific term analyses (`domain_term_*.{csv,txt}`)

These figures demonstrate the efficiency gains achieved by our specialized tokenizers compared to general-purpose tokenizers when processing domain-specific text.

## Replication Guide

To replicate our results:

1. Clone this repository
2. Set up the Python environment as described in [Environment Setup](#environment-setup)
3. Run the analysis scripts as described in [Running the Code](#running-the-code)
4. The scripts will generate figures and data files in the `figures/` directory
5. Build the paper using the Makefile to see how the results are incorporated

For access to the tokenizers:
- The tokenizers are available on Hugging Face: [ALEA/kl3m-tokenizers](https://huggingface.co/ALEA)
- The source code for the tokenizers is available in the [kl3m-tokenizers](https://github.com/aleainstitute/kl3m-tokenizers) repository

## Citation

If you use our tokenizers or refer to our work, please cite:

```bibtex
@inproceedings{bommarito2025kl3m,
  title={KL3M Tokenizers: A Family of Domain-Specific and Character-Level Tokenizers for Legal, Financial, and Preprocessing Applications},
  author={Bommarito II, Michael J. and Katz, Daniel Martin and Bommarito, Jillian},
  booktitle={Proceedings of...},
  year={2025}
}
```

## License

This project uses dual licensing:

- **Source Code**: All source code is licensed under the [MIT License](https://opensource.org/licenses/MIT)
- **Data and Publications**: All data, figures, and publication content are licensed under the [Creative Commons Attribution 4.0 International License (CC-BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

This means you are free to use and build upon our code (under MIT terms) and cite our research findings (under CC-BY 4.0 terms).