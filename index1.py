import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize pixel values to [0, 1]
X_train, X_test = X_train / 255.0, X_test / 255.0

# Expand dimensions for CNN input: (batch_size, height, width, channels)
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Build CNN model architecture
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 classes for digits 0-9
])

# Compile model with optimizer, loss function, and metrics
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model for 5 epochs (adjust as needed)
model.fit(X_train, y_train, epochs=5, validation_split=0.1)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'\nTest accuracy: {test_acc:.4f}')

# Ensure test accuracy is above 95%
assert test_acc > 0.95, "Test accuracy is below 95%, consider training longer or tuning model."

# Predict on 5 sample test images
sample_images = X_test[:5]
predictions = model.predict(sample_images)
predicted_labels = np.argmax(predictions, axis=1)

# Visualize the predictions
plt.figure(figsize=(10, 3))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(sample_images[i].reshape(28, 28), cmap=plt.cm.binary)
    plt.xlabel(f'Predicted: {predicted_labels[i]}')
plt.show()
