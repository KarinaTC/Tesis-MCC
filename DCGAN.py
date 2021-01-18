#  Inspired in  Tensorflow tutorial
#  https://www.tensorflow.org/tutorials/generative/dcgan
#  Tensorflow version 2.1
#  Deep Convolutional Generative Adversarial Network

import os
import PIL
import time
import glob
import pickle
import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from IPython import display

# GENERATOR MODEL
def build_generator(z_dim):
    '''
    Build the architecture of the Generator
    param z_dim: Dimension of the latent space
    return: Generator model
    '''
    model = tf.keras.Sequential()
    model.add(layers.Dense(4*4*1024, input_dim=z_dim))
    model.add(layers.Reshape((4, 4, 1024)))
    assert model.output_shape == (None, 4, 4, 1024)

    model.add(layers.Conv2DTranspose(512, (5,5), strides=(2, 2), padding='same',activation='relu'))
    model.add(layers.BatchNormalization())
    assert model.output_shape == (None, 8, 8, 512)

    model.add(layers.Conv2DTranspose(256, (5,5), strides=(2, 2), padding='same',activation='relu'))
    model.add(layers.BatchNormalization())
    assert model.output_shape == (None, 16, 16, 256)

    model.add(layers.Conv2DTranspose(128, (5,5), strides=(2, 2), padding='same',activation='relu'))
    model.add(layers.BatchNormalization())
    assert model.output_shape == (None, 32, 32, 128)

    model.add(layers.Conv2DTranspose(3, (5,5), strides=(2, 2), padding='same', activation='tanh'))
    assert model.output_shape == (None, 64, 64, 3)
    return model

# DISCRIMINATOR MODEL
def build_discriminator(img_shape):
    '''
    Build the architecture of the Discriminator
    img_shape: Dimensions of the images
    return: Discriminator model
    '''
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=img_shape))
    model.add(layers.LeakyReLU(alpha=0.2))
    #model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    #model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    #model.add(layers.Dropout(0.2))

    model.add(layers.Flatten())
    #model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1))
    return model


# LOSS FUNCTION
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# DISCRIMINATOR FUNCTION LOSS
def discriminator_loss(real_output, fake_output, smoothing_factor = 0.9):
    '''
    Compares the discriminator's predictions on real images to an array of 1s
    Compares the discriminator's predictions on fake (generated) images to an array of 0s
    real_output: Mini Batch of real images
    fake_output: Mini Batch of fake images
    smoothing_factor: Factor that set the real label
    return: Discriminator loss
    '''
    real_loss = cross_entropy(tf.ones_like(real_output)* smoothing_factor, real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


# GENERATOR FUNCTION LOSS
def generator_loss(fake_output):
    '''
    Generator tries to fool discriminator using real labels
    fake_output: Mini Batch of fake images
    return: Generator loss
    '''
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# OPTIMIZERS
# The generator and discriminator optimizers are different since we will train two networks separately
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)


# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, latent_dim]) # noise generated from normal distribution

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True) # generator produces an image

      real_output = discriminator(images, training=True) # discriminator classify real images
      fake_output = discriminator(generated_images, training=True) # discriminator classify fake images

      # Calculate losses for each of the models
      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output, smoothing_factor=0.9)

    # Adjusting Gradient of Discriminator
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    # Adjusting Gradient of Generator
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    return (disc_loss, gen_loss)


def summarize_performance(iteration, generator):
    '''
    Save generator model
    iteration: Number of epoch/iteration
    generator: Generator model
    '''
    filename = 'Generator_model_DCGAN_%03d.h5' % (iteration)
    generator.save(path + '/checkpoints/models/' +  filename)


def convert_array_to_image(array):
    '''
    Converts a numpy array to a PIL Image and undoes any rescaling.
    array: Array
    img: Image
    '''
    img = PIL.Image.fromarray(np.uint8((array + 1.0) / 2.0 * 255), mode='RGB')
    return img


def generate_and_save_images(model, epoch, test_input):
    '''
    Generate images and save models to check performance
    model: Generator model
    epoch: Number of epoch
    test_input: Vector in latent space
    '''
    predictions = model(test_input, training=False)   # `training` is set to False in order that all layers run in inference mode
    fig = plt.figure(figsize=(6,6))

    for i in range(predictions.shape[0]):
        plt.subplot(5, 5, i+1)
        predi = convert_array_to_image(predictions[i])
        plt.imshow(predi)
        plt.axis('off')

    plt.savefig(path + '/checkpoints/images/image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


losses = []
def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            d_loss, g_loss = train_step(image_batch)
        losses.append((d_loss.numpy(), g_loss.numpy()))
        # Produce images for the GIF as we go


        # Save the model every 15 epochs
        if (epoch + 1) % 10 == 0:
            display.clear_output(wait=True)
            generate_and_save_images(generator,epoch + 1,seed)
            #checkpoint.save(file_prefix = checkpoint_prefix)
            summarize_performance(epoch+1, generator)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        print("%d [D loss: %4f] [G loss: %4f]" % (epoch + 1, d_loss.numpy(), g_loss.numpy()))

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator,epochs,seed)


path = "D:/DCGAN/"

# READ DATASET
dbfile = open('FaceDataTrain_images.pkl', 'rb')
data = pickle.load(dbfile)
dbfile.close()


# PREPROCESSING
print('Shape: ', data['Images'].shape)
print('Max: ',data['Images'].max())
print('Min: ',data['Images'].min())


# RESCALE IMAGES [0,1] TO [-1,1]
X = (data['Images'] - 0.5) /0.5
print('Max value: ' + str(X.max()))
print('Min value: ' + str(X.min()))


EPOCHS = 300
latent_dim = 100
num_examples_to_generate = 25
seed = tf.random.normal([num_examples_to_generate, latent_dim]) # Use seed to visualize progress in the animated GIF)

# IMAGE DIMENSIONS
img_rows = 64
img_cols = 64
channels = 3
img_shape = (img_rows, img_cols, channels) # Input image dimensions

BUFFER_SIZE = 50039  # Number of images in the dataset
BATCH_SIZE = 128

# Load elements of size=buffer size and shuffle them at each iteration in n number of batches
train_dataset = tf.data.Dataset.from_tensor_slices(X).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


# DEFINE MODELS
generator = build_generator(latent_dim)
discriminator = build_discriminator(img_shape)

train(train_dataset, EPOCHS)