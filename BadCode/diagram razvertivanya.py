import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create a figure and a subplot
fig, ax = plt.subplots(figsize=(12, 8))

# Add nodes (servers)
web_server = patches.FancyBboxPatch((0.1, 0.7), 0.3, 0.15, boxstyle="round,pad=0.1", 
                                    edgecolor='black', facecolor='lightblue', linewidth=1.5)
app_server = patches.FancyBboxPatch((0.5, 0.7), 0.3, 0.15, boxstyle="round,pad=0.1", 
                                    edgecolor='black', facecolor='lightgreen', linewidth=1.5)
db_server = patches.FancyBboxPatch((0.1, 0.4), 0.3, 0.15, boxstyle="round,pad=0.1", 
                                   edgecolor='black', facecolor='lightcoral', linewidth=1.5)
nn_server = patches.FancyBboxPatch((0.5, 0.4), 0.3, 0.15, boxstyle="round,pad=0.1", 
                                   edgecolor='black', facecolor='lightsalmon', linewidth=1.5)
flask_server = patches.FancyBboxPatch((0.3, 0.1), 0.3, 0.15, boxstyle="round,pad=0.1", 
                                      edgecolor='black', facecolor='lightyellow', linewidth=1.5)
camera = patches.FancyBboxPatch((0.1, 0.1), 0.15, 0.1, boxstyle="round,pad=0.1", 
                                edgecolor='black', facecolor='lightgray', linewidth=1.5)

ax.add_patch(web_server)
ax.add_patch(app_server)
ax.add_patch(db_server)
ax.add_patch(nn_server)
ax.add_patch(flask_server)
ax.add_patch(camera)

# Add text labels for nodes
plt.text(0.25, 0.8, 'Web Server\n(Frontend)', ha='center', va='center', fontsize=12, weight='bold')
plt.text(0.65, 0.8, 'App Server\n(Backend)', ha='center', va='center', fontsize=12, weight='bold')
plt.text(0.25, 0.5, 'DB Server\n(Database)', ha='center', va='center', fontsize=12, weight='bold')
plt.text(0.65, 0.5, 'NN Server\n(Neural Networks)', ha='center', va='center', fontsize=12, weight='bold')
plt.text(0.45, 0.2, 'Flask Server', ha='center', va='center', fontsize=12, weight='bold')
plt.text(0.175, 0.15, 'Cameras', ha='center', va='center', fontsize=12, weight='bold')

# Add arrows (communication paths)
plt.arrow(0.4, 0.75, 0.1, 0, head_width=0.02, head_length=0.02, fc='k', ec='k')
plt.arrow(0.4, 0.45, 0.1, 0, head_width=0.02, head_length=0.02, fc='k', ec='k')
plt.arrow(0.25, 0.4, 0.0, -0.15, head_width=0.02, head_length=0.02, fc='k', ec='k')
plt.arrow(0.65, 0.4, 0.0, -0.15, head_width=0.02, head_length=0.02, fc='k', ec='k')
plt.arrow(0.175, 0.2, 0.125, 0, head_width=0.02, head_length=0.02, fc='k', ec='k')
plt.arrow(0.425, 0.25, 0.125, 0.15, head_width=0.02, head_length=0.02, fc='k', ec='k')

# Add text labels for communication paths
plt.text(0.45, 0.78, 'HTTP Request/Response', ha='center', va='center', fontsize=10)
plt.text(0.45, 0.48, 'SQL Query/Response', ha='center', va='center', fontsize=10)
plt.text(0.175, 0.28, 'Video Data', ha='center', va='center', fontsize=10)
plt.text(0.65, 0.28, 'NN Processing', ha='center', va='center', fontsize=10)

# Set limits and hide axes
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# Display the diagram
plt.show()
