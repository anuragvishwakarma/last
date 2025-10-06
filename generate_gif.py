import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import imageio
import os
import numpy as np

# Create output dir
os.makedirs("fsma_gif", exist_ok=True)

# Define stages (left to right)
stages = [
    "User\nQuery",
    "Streamlit\nUI",
    "FAISS\nRetrieval",
    "Multi-Agent\nProcessing",
    "Amazon Titan\n(LLM)",
    "Final\nAnswer"
]

# Sub-steps for Multi-Agent
sub_agents = ["• Tech Spec Agent", "• Maintenance Log Analyst", "• Workflow Manager"]

# Colors (AWS-inspired: Bedrock purple, FAISS orange, etc.)
colors = ["#232F3E", "#FF9900", "#FF4500", "#8A2BE2", "#28A745", "#1E88E5"]

# Animation settings
n_frames = 60  # 2 sec at 30 fps
current_step = 0

fig, ax = plt.subplots(figsize=(12, 5))
ax.set_xlim(-1, len(stages))
ax.set_ylim(-2, 2)
ax.axis('off')
fig.patch.set_facecolor('#FFFFFF')

def animate(frame):
    global current_step
    ax.clear()
    ax.set_xlim(-1, len(stages))
    ax.set_ylim(-2, 2)
    ax.axis('off')
    
    progress = frame / n_frames
    current_step = min(int(progress * len(stages)), len(stages))

    # Draw main pipeline
    for i, stage in enumerate(stages):
        x = i
        y = 0
        alpha = 0.3
        color = "#CCCCCC"
        text_color = "black"
        
        if i < current_step:
            alpha = 1.0
            color = colors[i]
            text_color = "white"
        elif i == current_step:
            # Pulsing effect
            pulse = 0.5 + 0.5 * np.sin(frame * 0.3)
            alpha = 0.4 + 0.6 * pulse
            color = colors[i]
            text_color = "white"

        # Circle node
        circle = plt.Circle((x, y), 0.4, color=color, alpha=alpha, ec='black', lw=1)
        ax.add_patch(circle)
        ax.text(x, y, stage, ha='center', va='center', fontsize=10, fontweight='bold', color=text_color)

        # Arrows
        if i < len(stages) - 1:
            ax.annotate('', xy=(x+0.6, y), xytext=(x+0.4, y),
                        arrowprops=dict(arrowstyle='->', lw=2, color='#333333'))

    # Draw sub-agents below Multi-Agent stage
    if current_step > 3:
        ma_x = 3
        for j, agent in enumerate(sub_agents):
            ax.text(ma_x, -0.8 - j*0.3, agent, ha='center', va='center', fontsize=9, color="#555555")

    # Title
    ax.text(2.5, 1.5, "Field Support & Maintenance Assistant (FSMA)", 
            fontsize=14, fontweight='bold', ha='center')

# Create animation
anim = FuncAnimation(fig, animate, frames=n_frames, interval=100, repeat=True)

# Save as GIF
anim.save("fsma_workflow.gif", writer=animation.PillowWriter(fps=60))
print("✅ GIF saved as: fsma_workflow.gif")