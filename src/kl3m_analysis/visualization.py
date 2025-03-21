"""
Visualization utilities for KL3M analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from typing import Dict, List, Optional, Union

from . import constants


def setup_plot_style():
    """Setup the matplotlib plot style for consistent visuals."""
    # Set style for all plots - use a clean, publication-quality style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
    # Increase font sizes for better readability
    plt.rcParams['font.size'] = 14         # Base font size
    plt.rcParams['axes.labelsize'] = 16    # Axis labels
    plt.rcParams['axes.titlesize'] = 20    # Subplot titles
    plt.rcParams['xtick.labelsize'] = 14   # X-axis tick labels
    plt.rcParams['ytick.labelsize'] = 14   # Y-axis tick labels
    plt.rcParams['legend.fontsize'] = 14   # Legend text
    plt.rcParams['figure.titlesize'] = 22  # Main figure title
    plt.rcParams['lines.linewidth'] = 2.5  # Line width
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['figure.figsize'] = (12, 8) # Larger figure size
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.3
    
    # Set Seaborn context for better looking plots
    sns.set_context("talk", rc={"lines.linewidth": 2.5})


def group_tokenizers(tokenizer_names: List[str]) -> Dict[str, List[str]]:
    """Group tokenizers into standard, character, and other categories.
    
    Args:
        tokenizer_names: List of tokenizer names to group
        
    Returns:
        Dictionary with grouped tokenizers
    """
    grouped = {
        "kl3m_standard": [],
        "kl3m_char": [],
        "others": []
    }
    
    for name in tokenizer_names:
        if name.startswith("kl3m") and "char" not in name:
            grouped["kl3m_standard"].append(name)
        elif name.startswith("kl3m") and "char" in name:
            grouped["kl3m_char"].append(name)
        else:
            grouped["others"].append(name)
    
    return grouped


def get_color_for_tokenizer(tokenizer_name: str, group_idx: int = 0) -> str:
    """Get the appropriate color for a tokenizer based on its group.
    
    Args:
        tokenizer_name: Name of the tokenizer
        group_idx: Index within the group (for selecting color from palette)
        
    Returns:
        Hex color code
    """
    if tokenizer_name.startswith("kl3m") and "char" not in tokenizer_name:
        return constants.KL3M_STANDARD_COLORS[group_idx % len(constants.KL3M_STANDARD_COLORS)]
    elif tokenizer_name.startswith("kl3m") and "char" in tokenizer_name:
        return constants.KL3M_CHAR_COLORS[group_idx % len(constants.KL3M_CHAR_COLORS)]
    else:
        return constants.OTHER_COLORS[group_idx % len(constants.OTHER_COLORS)]


def get_marker_for_tokenizer(tokenizer_name: str, group_idx: int = 0) -> str:
    """Get the appropriate marker for a tokenizer based on its group.
    
    Args:
        tokenizer_name: Name of the tokenizer
        group_idx: Index within the group (for selecting marker)
        
    Returns:
        Marker style
    """
    if tokenizer_name.startswith("kl3m") and "char" not in tokenizer_name:
        markers = constants.MARKERS['standard']
        return markers[group_idx % len(markers)]
    elif tokenizer_name.startswith("kl3m") and "char" in tokenizer_name:
        markers = constants.MARKERS['char']
        return markers[group_idx % len(markers)]
    else:
        markers = constants.MARKERS['other']
        return markers[group_idx % len(markers)]


def create_tokenizer_legend(ax, tokenizer_groups: Dict[str, List[str]], 
                           tokenizer_handles: Dict[str, any]):
    """Create a grouped legend for tokenizers.
    
    Args:
        ax: The matplotlib axes to add the legend to
        tokenizer_groups: Dictionary mapping group names to lists of tokenizer names
        tokenizer_handles: Dictionary mapping tokenizer names to their plot handles
    """
    legend_elements = []
    
    # KL3M Standard section (if any)
    if tokenizer_groups["kl3m_standard"]:
        legend_elements.append(Line2D([0], [0], color='white', lw=0, 
                                     label='KL3M Standard Tokenizers'))
        for name in tokenizer_groups["kl3m_standard"]:
            if name in tokenizer_handles:
                legend_elements.append(tokenizer_handles[name])
    
    # KL3M Character section (if any)
    if tokenizer_groups["kl3m_char"]:
        legend_elements.append(Line2D([0], [0], color='white', lw=0, 
                                     label='KL3M Character Tokenizers'))
        for name in tokenizer_groups["kl3m_char"]:
            if name in tokenizer_handles:
                legend_elements.append(tokenizer_handles[name])
    
    # Other Models section (if any)
    if tokenizer_groups["others"]:
        legend_elements.append(Line2D([0], [0], color='white', lw=0, 
                                     label='Other Models'))
        for name in tokenizer_groups["others"]:
            if name in tokenizer_handles:
                legend_elements.append(tokenizer_handles[name])
    
    # Add the legend with increased font size
    ax.legend(handles=legend_elements, 
             loc='upper left', 
             bbox_to_anchor=(1.02, 1),
             fontsize=14,  # Increased from 10 to 14
             frameon=True,
             fancybox=True,
             shadow=True,
             title="Tokenizer Groups",
             title_fontsize=16)  # Added explicit title font size


def create_line_legend_handle(tokenizer_name: str, group_idx: int = 0) -> Line2D:
    """Create a Line2D handle for legend based on tokenizer name.
    
    Args:
        tokenizer_name: Name of the tokenizer
        group_idx: Index within the group
        
    Returns:
        Line2D handle for the legend
    """
    color = get_color_for_tokenizer(tokenizer_name, group_idx)
    marker = get_marker_for_tokenizer(tokenizer_name, group_idx)
    linestyle = constants.LINESTYLES[group_idx % len(constants.LINESTYLES)]
    
    return Line2D([0], [0], color=color, marker=marker, linestyle=linestyle,
                 markersize=8, linewidth=2.5, label=tokenizer_name)


def create_patch_legend_handle(tokenizer_name: str, group_idx: int = 0) -> Patch:
    """Create a Patch handle for legend based on tokenizer name.
    
    Args:
        tokenizer_name: Name of the tokenizer
        group_idx: Index within the group
        
    Returns:
        Patch handle for the legend
    """
    color = get_color_for_tokenizer(tokenizer_name, group_idx)
    pattern = constants.PATTERNS[group_idx % len(constants.PATTERNS)]
    
    return Patch(facecolor=color, edgecolor='black', 
                hatch=pattern, label=tokenizer_name)