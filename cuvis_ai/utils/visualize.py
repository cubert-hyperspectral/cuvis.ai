
import numpy as np

import matplotlib.pyplot as plt

def visualize_image(preds: np.ndarray, title='Empty'):
    plt.imshow(preds, cmap='viridis')  # You can choose any colormap you like
    plt.colorbar()
    plt.title(title)
    plt.axis('off')
    plt.show()