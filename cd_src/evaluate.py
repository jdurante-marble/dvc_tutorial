import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the image size and batch size
image_size = (64, 64)
batch_size = 32

# Data generator for loading images
datagen = ImageDataGenerator(rescale=1./255)

val_generator = datagen.flow_from_directory(
    'data/processed/val',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='input'
)

# Load the trained autoencoder model
autoencoder = load_model('models/autoencoder.h5')

# Evaluate the autoencoder
loss = autoencoder.evaluate(val_generator)
with open('metrics/eval.txt', 'w') as f:
    f.write(f'Loss: {loss}\n')
