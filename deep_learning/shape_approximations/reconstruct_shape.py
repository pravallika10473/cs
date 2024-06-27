# reconstruct_image.py

from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# Load the saved model
saved_model = load_model('image_approximator.h5')

# Generate an empty input tensor (all zeros) of the same shape as the original input
input_image = np.zeros((1, 128, 128, 1))  # Example: empty input tensor of shape (1, 128, 128, 1)

# Reconstruct the image using the saved model
reconstructed_image = saved_model.predict(input_image)[0]

# Display the reconstructed image
plt.imshow(reconstructed_image.squeeze(), cmap='gray')
plt.title("Reconstructed Image")
plt.show()
