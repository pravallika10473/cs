import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from PIL import Image
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Image Reconstruction with Neural Network')
parser.add_argument('image_path', type=str, help='Path to the input image file')
args = parser.parse_args()

# Load the image
image_path = args.image_path
image = Image.open(image_path).convert('L')  # Convert to grayscale
image = image.resize((128, 128))  # Resize for simplicity
image = np.array(image) / 255.0  # Normalize pixel values
image = np.expand_dims(image, axis=-1)  # Add channel dimension

# Display the original image
plt.imshow(image.squeeze(), cmap='gray')
plt.title("Original Image")
plt.show()

# Build the model
image_shape = image.shape
flattened_shape = image_shape[0] * image_shape[1]

model = Sequential([
    Flatten(input_shape=image_shape),
    Dense(256, activation='relu'),
    Dense(flattened_shape, activation='sigmoid'),
    Reshape(image_shape)
])

model.compile(optimizer=Adam(), loss=MeanSquaredError())

# Train the model and visualize the process
epochs = 20
display_interval = 1

for epoch in range(epochs):
    model.fit(np.array([image]), np.array([image]), epochs=1, verbose=0)
    
    if epoch % display_interval == 0 or epoch == epochs - 1:
        generated_image = model.predict(np.array([image]))[0]
        
        plt.figure(figsize=(6, 3))
        
        plt.subplot(1, 2, 1)
        plt.title(f"Original Image")
        plt.imshow(image.squeeze(), cmap='gray')
        
        plt.subplot(1, 2, 2)
        plt.title(f"Generated Image - Epoch {epoch + 1}")
        plt.imshow(generated_image.squeeze(), cmap='gray')
        
        plt.show()

# Save the trained model
model.save('output_model/image_approximator.h5')  # Save model
