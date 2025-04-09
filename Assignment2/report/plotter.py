import matplotlib.pyplot as plt
import numpy as np

# Data from your simulations
depths = [2, 3, 4, 5, 6, 7, 8]
scores = [9, 9, 10, 10, 10, 10, -1]  # Final scores (positive indicates AI wins)
total_times = [29.47, 33.92, 31.14, 277.60, 343.17, 1351.68, 381.67]  # in seconds
avg_times = [t/10 if i != 6 else t for i, t in enumerate(total_times)]  # depth 8 only had 1 game

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 4))
#fig.suptitle('AI Performance in Fenix Game at Different Search Depths', fontsize=14)

# Plot 1: Scores vs Depth
ax1.bar(depths, scores, color='skyblue')
ax1.set_title('Final Scores')
ax1.set_xlabel('Search Depth')
ax1.set_ylabel('Score (AI wins)')
ax1.set_ylim(-2, 12)  # Adjust y-axis limits to fit the scores
ax1.set_xticks(depths)
ax1.axhline(y=0, color='gray', linestyle='--')
for i, v in enumerate(scores):
    ax1.text(depths[i], v + 0.2 if v >=0 else v - 0.5, str(v), ha='center')

# Plot 2: Computation Time vs Depth
ax2.plot(depths, avg_times, 'r-o', label='Average Time per Game')
ax2.set_title('Computation Time')
ax2.set_xlabel('Search Depth')
ax2.set_ylabel('Time (seconds)')
ax2.set_xticks(depths)
ax2.set_yscale('log')  # Logarithmic scale due to large time differences
ax2.grid(True, which="both", ls="-")
ax2.legend()

# Annotate the depth 8 special case
#ax1.text(0.95, 0.95, "*Depth 8: Only 1 game completed", 
#         transform=ax1.transAxes, ha='right', va='top', fontsize=9)
ax2.text(0.8, 0.95, "*Depth 8: Only 1 game completed", 
         transform=ax2.transAxes, ha='right', va='top', fontsize=9)

plt.tight_layout()
plt.show()