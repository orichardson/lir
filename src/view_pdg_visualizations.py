#!/usr/bin/env python3
"""
Simple script to view PDG visualizations.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import sys

def view_visualizations():
    """Display all generated PDG visualizations."""
    
    # List of visualization files
    viz_files = [
        "pdg_visualizations.png",
        "attention_strategy_visualizations.png", 
        "pdg_properties_analysis.png",
        "lir_experimental_results.png"
    ]
    
    # Check which files exist
    existing_files = []
    for file in viz_files:
        if Path(file).exists():
            existing_files.append(file)
        else:
            print(f"Warning: {file} not found")
    
    if not existing_files:
        print("No visualization files found. Please run visualize_pdgs.py first.")
        return
    
    # Create figure with subplots
    n_files = len(existing_files)
    if n_files == 1:
        fig, axes = plt.subplots(1, 1, figsize=(12, 8))
        axes = [axes]
    elif n_files == 2:
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    elif n_files == 3:
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
    
    # Display each visualization
    for i, file in enumerate(existing_files):
        if i < len(axes):
            img = mpimg.imread(file)
            axes[i].imshow(img)
            axes[i].set_title(file.replace('.png', '').replace('_', ' ').title(), 
                            fontsize=12, fontweight='bold')
            axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(existing_files), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.suptitle('LIR PDG Visualizations', fontsize=16, fontweight='bold', y=0.98)
    plt.show()
    
    print(f"\nDisplayed {len(existing_files)} visualization(s):")
    for file in existing_files:
        print(f"  - {file}")

def main():
    """Main function."""
    print("=== PDG Visualization Viewer ===")
    view_visualizations()

if __name__ == "__main__":
    main()
