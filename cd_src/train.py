import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the image size and batch size
image_size = (64, 64)
batch_size = 32

# Data generator for loading images
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    'data/processed/train',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='input'
)

val_generator = datagen.flow_from_directory(
    'data/processed/val',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='input'
)

# Define the autoencoder architecture
input_img = Input(shape=(image_size[0], image_size[1], 3))
x = Flatten()(input_img)
encoded = Dense(128, activation='relu')(x)
decoded = Dense(image_size[0] * image_size[1] * 3, activation='sigmoid')(encoded)
decoded = Reshape((image_size[0], image_size[1], 3))(decoded)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder
autoencoder.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# Save the model
autoencoder.save('models/autoencoder.h5')
