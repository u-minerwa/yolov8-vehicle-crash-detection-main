import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create a figure and a subplot
fig, ax = plt.subplots(figsize=(10, 6))

# Add nodes (servers)
web_server = patches.FancyBboxPatch((0.1, 0.6), 0.3, 0.2, boxstyle="round,pad=0.1", 
                                    edgecolor='black', facecolor='lightblue', linewidth=1.5)
app_server = patches.FancyBboxPatch((0.5, 0.6), 0.3, 0.2, boxstyle="round,pad=0.1", 
                                    edgecolor='black', facecolor='lightgreen', linewidth=1.5)
db_server = patches.FancyBboxPatch((0.3, 0.2), 0.3, 0.2, boxstyle="round,pad=0.1", 
                                   edgecolor='black', facecolor='lightcoral', linewidth=1.5)

ax.add_patch(web_server)
ax.add_patch(app_server)
ax.add_patch(db_server)

# Add text labels for nodes
plt.text(0.25, 0.75, 'Web Server\n(Frontend)', ha='center', va='center', fontsize=12, weight='bold')
plt.text(0.65, 0.75, 'App Server\n(Backend)', ha='center', va='center', fontsize=12, weight='bold')
plt.text(0.45, 0.35, 'DB Server\n(Database)', ha='center', va='center', fontsize=12, weight='bold')

# Add arrows (communication paths)
plt.arrow(0.4, 0.7, 0.1, 0, head_width=0.02, head_length=0.02, fc='k', ec='k')
plt.arrow(0.5, 0.6, -0.1, 0, head_width=0.02, head_length=0.02, fc='k', ec='k')
plt.arrow(0.5, 0.4, -0.1, -0.1, head_width=0.02, head_length=0.02, fc='k', ec='k')
plt.arrow(0.4, 0.2, 0.1, 0.1, head_width=0.02, head_length=0.02, fc='k', ec='k')

# Add text labels for communication paths
plt.text(0.45, 0.73, 'HTTP Request/Response', ha='center', va='center', fontsize=10)
plt.text(0.45, 0.47, 'SQL Query/Response', ha='center', va='center', fontsize=10)

# Set limits and hide axes
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# Display the diagram
plt.show()
