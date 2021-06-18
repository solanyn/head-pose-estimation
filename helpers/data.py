import tensorflow as tf
import numpy as np
import pandas as pd

DATA_DIR = './modified_data/'

def make_tilt_train_generator(train_data_df):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, data_format='channels_last')
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_data_df,
        directory=DATA_DIR,
        x_col="filename",
        y_col='tilt_str',
        target_size=(64, 64),
        batch_size=1,
        color_mode='rgb',
        shuffle=False,
        class_mode='categorical'
    )
    return train_generator

def make_pan_train_generator(train_data_df):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, data_format='channels_last')
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_data_df,
        directory=DATA_DIR,
        x_col="filename",
        y_col='pan_str',
        target_size=(64, 64),
        batch_size=1,
        color_mode='rgb',
        shuffle=False,
        class_mode='categorical'
    )
    return train_generator

def make_tilt_val_generator(val_data_df):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, data_format='channels_last')
    generator = datagen.flow_from_dataframe(
        dataframe=val_data_df,
        directory=DATA_DIR,
        x_col="filename",
        y_col='tilt_str',
        target_size=(64, 64),
        batch_size=1,
        color_mode="rgb",
        shuffle=False,
        class_mode='categorical'
    )
    return generator

def make_pan_val_generator(val_data_df):
   datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, data_format='channels_last')
   generator = datagen.flow_from_dataframe(
       dataframe=val_data_df,
       directory=DATA_DIR,
       x_col="filename",
       y_col='pan_str',
       target_size=(64, 64),
       batch_size=1,
       color_mode="rgb",
       shuffle=False,
       class_mode='categorical'
   )
   return generator

def make_test_generator(test_data):
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, data_format='channels_last')
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_data,
        directory=DATA_DIR,
        x_col='filename',
        target_size=(64, 64),
        batch_size=1,
        color_mode='rgb',
        shuffle=False,
        class_mode=None
    )
    return test_generator

def convert(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32) # Cast and normalize the image to [0,1]
    image = image[0,:]
    label = label[0,:]
    return image, label

def augment(image, label):
    image,label = convert(image, label)
    image = tf.image.resize_with_crop_or_pad(image, 69, 69) # Add 5 pixels of padding
    image = tf.image.random_crop(image, size=[64, 64, 3]) # Random crop back to 28x28
    image = tf.image.random_brightness(image, max_delta=0.5) # Random brightness
    return image,label



def make_tilt_train_generator(train_data_df):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, data_format='channels_last')
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_data_df,
        directory=DATA_DIR,
        x_col="filename",
        y_col='tilt_str',
        target_size=(64, 64),
        batch_size=1,
        color_mode='rgb',
        shuffle=False,
        class_mode='categorical'
    )
    return train_generator

def make_pan_train_generator(train_data_df):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, data_format='channels_last')
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_data_df,
        directory=DATA_DIR,
        x_col="filename",
        y_col='pan_str',
        target_size=(64, 64),
        batch_size=1,
        color_mode='rgb',
        shuffle=False,
        class_mode='categorical'
    )
    return train_generator

def make_tilt_val_generator(val_data_df):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, data_format='channels_last')
    generator = datagen.flow_from_dataframe(
        dataframe=val_data_df,
        directory=DATA_DIR,
        x_col="filename",
        y_col='tilt_str',
        target_size=(64, 64),
        batch_size=1,
        color_mode="rgb",
        shuffle=False,
        class_mode='categorical'
    )
    return generator

def make_pan_val_generator(val_data_df):
   datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, data_format='channels_last')
   generator = datagen.flow_from_dataframe(
       dataframe=val_data_df,
       directory=DATA_DIR,
       x_col="filename",
       y_col='pan_str',
       target_size=(64, 64),
       batch_size=1,
       color_mode="rgb",
       shuffle=False,
       class_mode='categorical'
   )
   return generator

def make_test_generator(test_data):
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, data_format='channels_last')
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_data,
        directory=DATA_DIR,
        x_col='filename',
        target_size=(64, 64),
        batch_size=1,
        color_mode='rgb',
        shuffle=False,
        class_mode=None
    )
    return test_generator

def convert(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32) # Cast and normalize the image to [0,1]
    image = image[0,:]
    label = label[0,:]
    return image, label

def augment(image, label):
    image,label = convert(image, label)
    image = tf.image.resize_with_crop_or_pad(image, 69, 69) # Add 5 pixels of padding
    image = tf.image.random_crop(image, size=[64, 64, 3]) # Random crop back to 28x28
    image = tf.image.random_brightness(image, max_delta=0.5) # Random brightness
    return image,label
