import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50

def create_model(input_shape, num_classes):
    base_model = MobileNet(input_shape=input_shape, include_top=False, weights='imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=x)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(data_dir):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    
    num_classes = len(train_generator.class_indices)
    model = create_model((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), num_classes)
    
    model.fit(train_generator,
              steps_per_epoch=train_generator.samples // BATCH_SIZE,
              validation_data=validation_generator,
              validation_steps=validation_generator.samples // BATCH_SIZE,
              epochs=EPOCHS,
              verbose=1)

    
    model.save('potato_disease_mobilenet_model.h5')

    return model, train_generator.class_indices

if __name__ == '__main__':
    data_dir = 'dataset'  # Path to the image dataset


    train_model(data_dir)
