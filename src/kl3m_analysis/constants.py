"""
Constants used across the KL3M analysis scripts.
"""

from pathlib import Path

# Configure output directories
OUTPUT_DIR = Path("../figures")

# Create consistent color schemes
# Blue palette for KL3M standard tokenizers
KL3M_STANDARD_COLORS = ['#1f77b4', '#3a86c8', '#5599da', '#70adec']
# Green palette for KL3M character tokenizers
KL3M_CHAR_COLORS = ['#2ca02c', '#3eb53e', '#50ca50', '#62df62']
# Red/orange palette for other tokenizers
OTHER_COLORS = ['#d62728', '#e74c3c', '#ff7f0e', '#ff9f51']

# Define patterns for similar colors
PATTERNS = ['', '////', '....', 'xxxx', '\\\\\\\\', '||||', '++++', '----', '****', 'oooo']

# Define line styles for distinguishing similar colors
LINESTYLES = ['-', '--', '-.', ':']

# Define markers for each group
MARKERS = {
    'standard': ['o', 'D', '^', 's'], 
    'char': ['P', 'X', '*', 'p'], 
    'other': ['h', 'H', 'v', '8']
}

# Sample texts from different domains
SAMPLE_TEXTS = {
    "legal": [
        "The Comptroller of the Currency shall make a public rulemaking on the matter pursuant to section 553 of title 5, United States Code.",
        "This Securities Purchase Agreement (this \"Agreement\") is dated as of November 21, 2017, between Company Name, Inc., a Delaware corporation.",
        "Pursuant to 11 U.S.C. § 362(a), all entities are stayed from taking any action against the Debtor or the property of the estate.",
        "The Court finds that the petitioner's writ of habeas corpus under 28 U.S.C. § 2254 is time-barred by the Antiterrorism and Effective Death Penalty Act's one-year statute of limitations.",
        "IN THE MATTER OF THE ESTATE OF JOHN DOE, DECEASED. Case No. CV-2023-12345-PR. NOTICE OF HEARING ON PETITION FOR FORMAL PROBATE OF WILL AND APPOINTMENT OF PERSONAL REPRESENTATIVE.",
        "Plaintiff's Motion for Summary Judgment pursuant to Fed. R. Civ. P. 56(a) is hereby GRANTED, as there exists no genuine dispute as to any material fact.",
        "The doctrine of stare decisis et non quieta movere requires courts to follow precedent unless there is a special justification for departure from established jurisprudence.",
        "The undersigned counsel certifies that this brief complies with the type-volume limitation of Fed. R. App. P. 32(a)(7)(B) and contains 12,453 words excluding parts exempted by Rule 32(f).",
        "A preponderance-of-the-evidence standard applies to respondent's affirmative defense that petitioner's constructive-discharge claim is time-barred. See Tex. Dep't of Cmty. Affairs v. Burdine, 450 U.S. 248, 254–55.",
        "In Chevron U.S.A., Inc. v. Natural Resources Defense Council, Inc., 467 U.S. 837 (1984), the Supreme Court established the now-eponymous two-step framework for judicial review of agency interpretation of statutes."
    ],
    "financial": [
        "The company reported Q3 earnings of $2.45 per share, exceeding analyst expectations of $1.98.",
        "EBITDA increased by 14.3% year-over-year to $342.5 million for the fiscal year ending December 31, 2023.",
        "Form 10-K Annual Report Pursuant to Section 13 or 15(d) of the Securities Exchange Act of 1934 for the fiscal year ended December 31, 2022.",
        "The debt-to-equity ratio decreased from 1.85x to 1.37x, indicating significant deleveraging and improved balance sheet strength.",
        "Adjusted free cash flow (FCF) for Q2 reached $127.6M, representing a 37% FCF-to-adjusted-EBITDA conversion rate.",
        "The Board of Directors declared a quarterly cash dividend of $0.75 per share, payable on June 15, 2023, to shareholders of record as of the close of business on May 31, 2023.",
        "XYZ Corp. announced the completion of its previously-disclosed acquisition of ABC Technologies for $3.2B, or approximately 12.5x TTM EBITDA.",
        "The sell-side analyst consensus estimates reflected a P/E ratio of 22.4x and a price-to-book value of 3.2x, suggesting moderate overvaluation relative to historical averages.",
        "Management reiterated full-year 2023 guidance, with revenue expected to grow 9-11% YoY on a constant-currency basis and non-GAAP operating margin projected at 28-30%.",
        "According to NYSE Rule 123C(1)(e)(i), market-on-close (MOC) orders must be submitted by 3:50 p.m. ET, except in the case of regulatory circuit breakers as defined in Regulation SHO."
    ],
    "general": [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models have demonstrated remarkable capabilities in natural language processing tasks.",
        "Climate change poses significant challenges to coastal communities and biodiversity conservation efforts worldwide.",
        "The semi-centennial celebration commemorated fifty years since the establishment of the university's interdisciplinary research center.",
        "After the unprecedented rainfall, local authorities implemented anti-flood measures including the reinforcement of levees along the river's edge.",
        "The actress's breathtaking performance in the historical drama-comedy earned her widespread acclaim and multiple award nominations.",
        "Scientists discovered a previously-unknown species of deep-sea microorganisms that can survive in extreme hydrothermal vent environments.",
        "The municipality's newly-elected council members unanimously approved the infrastructure-improvement plan despite budgetary constraints.",
        "According to the World Health Organization (WHO), approximately 2.3 billion people lack access to basic handwashing facilities with soap and water.",
        "Anthropologists studying pre-Columbian civilizations have uncovered evidence of sophisticated agricultural techniques and astronomical knowledge."
    ]
}

# Legal patterns to look for
LEGAL_PATTERNS = [
    r'§', r'U\.S\.C', r'v\.', r'et al', r'plaintiff', r'defendant',
    r'court', r'pursuant', r'statute', r'regulation', r'law', r'act',
    r'amendment', r'constitution', r'motion', r'brief'
]

# Financial patterns to look for
FINANCIAL_PATTERNS = [
    r'EBITDA', r'ROI', r'IPO', r'SEC', r'Q[1-4]', r'10-[KQ]',
    r'dividend', r'earnings', r'fiscal', r'revenue', r'profit',
    r'stock', r'share', r'equity', r'asset', r'liability'
]

# HTML/Markdown patterns to look for
HTML_PATTERNS = [
    r'</?[a-z]+>', r'</[a-z]+>', r'href=', r'class=', r'style=',
    r'id=', r'div', r'span', r'script', r'table', r'tr', r'td'
]

# JSON patterns to look for
JSON_PATTERNS = [
    r'\{\"', r'\"\}', r'\[\"', r'\"\]', r'\":\"', r'\",\"', r'\":'
]

# Legal terms for domain comparison
LEGAL_TERMS = [
    "11 U.S.C. § 362(a)",
    "res judicata",
    "stare decisis",
    "habeas corpus",
    "certiorari",
    "de novo review",
    "28 C.F.R. § 14.2(a)",
    "42 U.S.C. § 1983",
    "Fed. R. Civ. P. 12(b)(6)",
    "prima facie"
]

# Financial terms for domain comparison
FINANCIAL_TERMS = [
    "EBITDA",
    "P/E ratio",
    "10-K filing",
    "SEC Form 8-K",
    "quarterly dividend",
    "year-over-year growth",
    "Basel III compliance",
    "GAAP accounting",
    "ROI analysis",
    "market capitalization"
]

# Legal phrases for tables
LEGAL_PHRASES = [
    "Fed. R. Civ. P. 12(b)(6)",
    "Pursuant to 11 U.S.C. § 362(a), all entities are stayed from taking"
]

# OCR error example (for tokenization examples)
OCR_ERROR_TEXT = "Thc Vnited S tates 5enate is nesp0nslbe for the"

# Standard list of tokenizers to analyze
DEFAULT_TOKENIZERS = [
    # KL3M standard tokenizers
    "kl3m-001-32k", 
    "kl3m-003-64k",
    "kl3m-004-128k-cased",
    "kl3m-004-128k-uncased",
    # KL3M character tokenizers
    "kl3m-004-char-4k-cased",
    "kl3m-004-char-8k-cased",
    "kl3m-004-char-16k-cased",
    # Comparison tokenizers
    "llama3",
    "roberta-base",
    "gpt2",
    "gpt-4o"
]

# Standard model IDs for tokenizers
TOKENIZER_MODEL_IDS = {
    "kl3m-001-32k": "alea-institute/kl3m-001-32k",
    "kl3m-003-64k": "alea-institute/kl3m-003-64k",
    "kl3m-004-128k-cased": "alea-institute/kl3m-004-128k-cased",
    "kl3m-004-128k-uncased": "alea-institute/kl3m-004-128k-uncased",
    "kl3m-004-char-4k-cased": "alea-institute/kl3m-004-char-4k-cased",
    "kl3m-004-char-8k-cased": "alea-institute/kl3m-004-char-8k-cased",
    "kl3m-004-char-16k-cased": "alea-institute/kl3m-004-char-16k-cased",
    "gpt2": "gpt2",
    "llama3": "meta-llama/Llama-3.2-1B-Instruct",
    "roberta-base": "roberta-base"
}

# Tiktoken models
TIKTOKEN_MODELS = ["gpt-4o"]

# Dataset IDs
DATASET_IDS = {
    'alea-institute/kl3m-data-usc': "US Code",
    'alea-institute/kl3m-data-govinfo-chrg': "Congressional Hearings",
    'alea-institute/kl3m-data-pacer-cand': "Court Documents",
    'alea-institute/kl3m-data-edgar-agreements': "SEC Filings",
    'HuggingFaceFW/fineweb-edu': 'General Content',
}