import matplotlib.pyplot as plt
import numpy as np

# Confusion matrix values
cm = np.array([
    [1050, 93],   # TN, FP
    [70, 1073]    # FN, TP
])

plt.figure(figsize=(6,5))
plt.imshow(cm, cmap='viridis')
plt.title('Confusion Matrix – SLA-FS++ Model', fontsize=14)
plt.xlabel('Predicted label')
plt.ylabel('True label')

# Annotate values with better visibility
for i in range(2):
    for j in range(2):
        plt.text(
            j, i, cm[i, j],
            ha='center',
            va='center',
            color='white',          # better contrast
            fontsize=14,            # larger font
            fontweight='bold',      # bold text
            bbox=dict(facecolor='black', alpha=0.3, boxstyle='round')
        )

plt.colorbar()

# Save image for report
plt.savefig("confusion_matrix_sla_fspp.png", dpi=300, bbox_inches='tight')
plt.show()
