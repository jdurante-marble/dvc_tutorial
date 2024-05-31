import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

def preprocess_data(src_dir, dest_dir, image_size):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(
        src_dir,
        target_size=image_size,
        batch_size=1,
        class_mode=None,
        shuffle=False
    )

    for i in range(len(generator)):
        img = generator[i][0]
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        img_path = os.path.join(dest_dir, f'image_{i}.png')
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        img.save(img_path)

src_train_dir = 'data/raw/train'
src_val_dir = 'data/raw/val'
dest_train_dir = 'data/processed/train'
dest_val_dir = 'data/processed/val'
image_size = (64, 64)

preprocess_data(src_train_dir, dest_train_dir, image_size)
preprocess_data(src_val_dir, dest_val_dir, image_size)
