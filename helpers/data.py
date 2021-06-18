import tensorflow as tf
import numpy as np
import pandas as pd


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

