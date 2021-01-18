#  Inspired in  Tensorflow tutorial
#  https://www.tensorflow.org/tutorials/generative/dcgan
#  Tensorflow version 2.1
#  Wasserstein Generative Adversarial Network with Gradient Penalty

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
    assert model.output_shape == (None, 4, 4, 1024) # Note: None is the batch size

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
def build_critic(img_shape):
    '''
    Build the architecture of the Critic
    img_shape: Dimensions of the images
    return: Critic model
    '''
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=img_shape))
    model.add(layers.LeakyReLU(alpha=0.2))
    #model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    #model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    #model.add(layers.Dropout(0.2))

    model.add(layers.Flatten())
    #model.add(layers.Dropout(0.4))
    model.add(layers.Dense(1))

    return model


def gradient_penalty(images, generated_images):
    alpha = tf.random.normal([images.shape[0], 1, 1, 1], 0.0, 1.0)
    #alpha = tf.random.uniform([BATCH_SIZE, 1, 1, 1], 0.0, 1.0)
    diff = generated_images - images
    interpolated = images + alpha * diff
    with tf.GradientTape() as t:
        t.watch(interpolated)
        pred = critic(interpolated, training = True)
    gradients = t.gradient(pred, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(gradients),axis = [1,2,3]))
    c_regularizer = tf.reduce_mean((norm - 1.0)**2)
    return c_regularizer

# CRITIC FUNCTION LOSS
def discriminator_loss(real_output, fake_output,c_regularizer,gradient_penalty_weight=10.0):
    critic_loss = (tf.reduce_mean(real_output) - tf.reduce_mean(fake_output) + c_regularizer * gradient_penalty_weight)
    return critic_loss


# GENERATOR FUNCTION LOSS
def generator_loss(fake_output):
    # Labels are true here because generator thinks he produces real images.
    gen_loss =  tf.reduce_mean(fake_output)
    return gen_loss


# OPTIMIZERS
# The generator and discriminator optimizers are different since we will train two networks separately
generator_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001,beta_1=0,beta_2=0.9)
critic_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001,beta_1=0,beta_2=0.9)


# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    # noise = tf.random.normal([BATCH_SIZE, latent_dim]) # noise generated from normal distribution
    for i in range(2): #Train 2 times more the critic than the generator
        # Get the latent vector
        noise = tf.random.normal([images.shape[0], latent_dim])  # noise generated from normal distribution

        with tf.GradientTape() as gen_tape, tf.GradientTape() as critic_tape:
            generated_images = generator(noise, training=True)  # generator produces an image
            real_output = critic(images, training=True)  # critic classify real images
            fake_output = critic(generated_images, training=True)  # critic classify fake images
            c_regularizer = gradient_penalty(images, generated_images)

            critic_loss = critic_loss(real_output, fake_output, c_regularizer)

        # Adjusting Gradient of Discriminator
        gradients_of_critic = critic_tape.gradient(critic_loss, critic.trainable_variables)
        critic_optimizer.apply_gradients(zip(gradients_of_critic, critic.trainable_variables))

    noise = tf.random.normal([BATCH_SIZE, latent_dim])  # noise generated from normal distribution
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)  # generator produces an image
        fake_output = critic(generated_images, training=True)  # discriminator classify fake images

        gen_loss = generator_loss(fake_output)

    # Adjusting Gradient of Generator
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    return (critic_loss, gen_loss)


def summarize_performance(iteration, generator):
    '''
    Save generator model
    iteration: Number of epoch/iteration
    generator: Generator model
    '''
    filename = 'Generator_model_WGAN-GP_%03d.h5' % (iteration)
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
            c_loss, g_loss = train_step(image_batch)
        losses.append((c_loss.numpy(), g_loss.numpy()))
        # Produce images for the GIF as we go

        # Save the model every 10 epochs
        if (epoch + 1) % 10 == 0:
            display.clear_output(wait=True)
            generate_and_save_images(generator,epoch + 1,seed)
            #checkpoint.save(file_prefix = checkpoint_prefix)
            summarize_performance(epoch+1, generator)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        print("%d [D loss: %4f] [G loss: %4f]" % (epoch + 1, c_loss.numpy(), g_loss.numpy()))

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator,epochs,seed)


path = "D:/WGAN-GP/"

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


EPOCHS = 150
latent_dim = 100
num_examples_to_generate = 25
seed = tf.random.normal([num_examples_to_generate, latent_dim]) # Use seed to visualize progress in the animated GIF)

# IMAGE DIMENSIONS
img_rows = 64
img_cols = 64
channels = 3
img_shape = (img_rows, img_cols, channels) # Input image dimensions

BUFFER_SIZE = 50039  # Number of images in the dataset
BATCH_SIZE = 64

# Load elements of size=buffer size and shuffle them at each iteration in n number of batches
train_dataset = tf.data.Dataset.from_tensor_slices(X).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


# DEFINE MODELS
generator = build_generator(latent_dim)
critic = build_critic(img_shape)

train(train_dataset, EPOCHS)