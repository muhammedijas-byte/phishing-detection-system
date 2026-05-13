import matplotlib.pyplot as plt

# Model names and accuracies
models = ['Baseline RF', 'SLA-FS++ RF']
accuracies = [0.93, 0.96]  # replace with your exact values

# Create bar chart with narrow bars
plt.figure(figsize=(6,4))
bars = plt.bar(models, accuracies, width=0.4)

# Add accuracy values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.01,
        f'{height:.2f}',
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold'
    )

# Labels and title
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.ylim(0, 1.0)

plt.show()
