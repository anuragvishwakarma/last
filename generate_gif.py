import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio
import os
import numpy as np

# Create output folder
os.makedirs("fsma_gif", exist_ok=True)

# Define steps in order
steps = [
    "User Query\n(Streamlit UI)",
    "Query Parser\n(Extract Equipment & Pos)",
    "FAISS Retrieval\n(PDFs + CSV)",
    "Multi-Agent Processing\n• Tech Spec Agent\n• Maintenance Log Analyst\n• Workflow Manager",
    "Response Synthesis\n(Combine all insights)",
    "Final Answer\n(Display in UI)"
]

colors = [
    "#4A90E2",  # Blue
    "#50C878",  # Emerald
    "#FFA500",  # Orange
    "#9B59B6",  # Purple
    "#3498DB",  # Light Blue
    "#2ECC71"   # Green
]

# Generate frames
filenames = []
for i in range(len(steps)):
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    # Draw all steps (dimmed)
    for j, step in enumerate(steps):
        y = 6 - j * 1.0
        alpha = 0.3 if j > i else 1.0
        color = colors[j] if j <= i else "#CCCCCC"
        rect = patches.FancyBboxPatch(
            (1, y - 0.4), 8, 0.8,
            boxstyle="round,pad=0.3",
            linewidth=2,
            edgecolor=color,
            facecolor=color if j <= i else "#F0F0F0",
            alpha=alpha
        )
        ax.add_patch(rect)
        plt.text(5, y, step, ha='center', va='center', fontsize=12, fontweight='bold' if j <= i else 'normal', color='white' if j <= i else 'black')
    
    # Add title
    plt.title("Field Support & Maintenance Assistant (FSMA)\nPowered by Amazon Bedrock + FAISS + Multi-Agent RAG", 
              fontsize=14, weight='bold', pad=20)
    
    # Add arrows between active steps
    for j in range(i):
        y1 = 6 - j * 1.0 - 0.4
        y2 = 6 - (j+1) * 1.0 + 0.4
        plt.annotate('', xy=(5, y2), xytext=(5, y1),
                     arrowprops=dict(arrowstyle='->', lw=2, color='#333333'))
    
    # Save frame
    filename = f"fsma_gif/frame_{i}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    filenames.append(filename)
    plt.close()

# Create GIF
images = []
for filename in filenames:
    images.append(imageio.v2.imread(filename))

# Save final GIF
output_path = "fsma_process_flow.gif"
imageio.mimsave(output_path, images, fps=1, loop=0)

# Cleanup frames
for filename in filenames:
    os.remove(filename)
os.rmdir("fsma_gif")

print(f"✅ GIF saved as: {output_path}")