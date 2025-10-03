#!/usr/bin/env python3
"""
View the bidirectional PDG dataset visualizations.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

def view_bidirectional_visualizations():
    """Display the bidirectional PDG visualizations."""
    
    # Check if visualization files exist
    dataset_img = Path("bidirectional_pdg_dataset.png")
    stats_img = Path("bidirectional_pdg_statistics.png")
    
    if not dataset_img.exists():
        print("❌ bidirectional_pdg_dataset.png not found!")
        return
    
    if not stats_img.exists():
        print("❌ bidirectional_pdg_statistics.png not found!")
        return
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle('Bidirectional PDG Dataset Visualizations', fontsize=16, fontweight='bold')
    
    # Load and display dataset visualization
    img1 = mpimg.imread(dataset_img)
    axes[0].imshow(img1)
    axes[0].set_title('Bidirectional PDG Dataset: 20 PDGs with A→B AND B→A edges', fontsize=12)
    axes[0].axis('off')
    
    # Load and display statistics visualization
    img2 = mpimg.imread(stats_img)
    axes[1].imshow(img2)
    axes[1].set_title('Bidirectional PDG Dataset Statistics', fontsize=12)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Bidirectional PDG visualizations displayed!")
    print(f"📊 Dataset visualization: {dataset_img}")
    print(f"📈 Statistics visualization: {stats_img}")
    print("\n🔍 Key Features:")
    print("  - Red edges indicate bidirectional pairs (A→B AND B→A)")
    print("  - Gray edges are unidirectional")
    print("  - 4 different patterns: chain, random, grid, star")
    print("  - 20 PDGs total with varying complexity")

if __name__ == "__main__":
    view_bidirectional_visualizations()
