import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image settings
img_size = 224
batch_size = 32

# Data preprocessing
train_gen = ImageDataGenerator(rescale=1./255)

# Load training data
train_data = train_gen.flow_from_directory(
    'data/chest_xray/train',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary'
)

# Load validation data
val_data = train_gen.flow_from_directory(
    'data/chest_xray/val',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary'
)

# Build CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(train_data, validation_data=val_data, epochs=5)

# Save model
model.save('models/model.keras')

# Load test data
test_data = train_gen.flow_from_directory(
    'data/chest_xray/test',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary'
)

# Evaluate model
test_loss, test_acc = model.evaluate(test_data)

# Print results
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)

# Save results to file
with open('results.txt', 'w') as f:
    f.write(f"Test Loss: {test_loss}\n")
    f.write(f"Test Accuracy: {test_acc}\n")

print("Results saved to results.txt")
