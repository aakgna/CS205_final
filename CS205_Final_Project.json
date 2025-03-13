import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import time

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
IMAGE_CHANNELS = 3
BATCH_SIZE = 64
EPOCHS = 100
DATA_DIRECTORIES = ['/kaggle/input/cubicasa5k/cubicasa5k/cubicasa5k/colorful']

def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB').resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    img_array = np.array(img).astype(np.float32)
    return (img_array - 127.5) / 127.5

def gather_image_paths(data_dirs):
    all_image_paths = []
    for directory in data_dirs:
        for root, _, files in os.walk(directory):
            all_image_paths.extend(os.path.join(root, f) for f in files if f.lower().endswith('.png'))
    return all_image_paths

#Creation of dataset
def create_dataset(image_paths):
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(lambda x: tf.numpy_function(load_and_preprocess_image, [x], Tout=tf.float32),
                          num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

#models
def create_discriminator_model():
    model = tf.keras.Sequential(name="discriminator")
    model.add(tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=[IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    for _ in range(4):
        model.add(tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model

def create_generator_model():
    model = tf.keras.Sequential(name="generator")
    model.add(tf.keras.layers.Dense(256 * 4 * 4, input_dim=100))
    model.add(tf.keras.layers.Reshape((4, 4, 256)))
    for _ in range(5):
        model.add(tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(3, (3, 3), activation='tanh', padding='same'))
    return model

def compute_discriminator_loss(real_output, fake_output):
    return tf.keras.losses.BinaryCrossentropy(from_logits=False)(tf.ones_like(real_output), real_output) + \
           tf.keras.losses.BinaryCrossentropy(from_logits=False)(tf.zeros_like(fake_output), fake_output)

def compute_generator_loss(fake_output):
    return tf.keras.losses.BinaryCrossentropy(from_logits=False)(tf.ones_like(fake_output), fake_output)

#Training
@tf.function
def execute_training_step(images, generator, discriminator, gen_optimizer, disc_optimizer):
    noise = tf.random.normal([BATCH_SIZE, 100])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = compute_generator_loss(fake_output)
        disc_loss = compute_discriminator_loss(real_output, fake_output)
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    gen_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss, disc_loss

def execute_training(dataset, generator, discriminator, gen_optimizer, disc_optimizer):
    seed = tf.random.normal([16, 100])
    for epoch in range(EPOCHS):
        for image_batch in dataset:
            g_loss, d_loss = execute_training_step(image_batch, generator, discriminator, gen_optimizer, disc_optimizer)
        # Generate and save images
        save_generated_images(generator, epoch + 1, seed)

def save_generated_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    predictions = (predictions + 1) / 2.0
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i])
        plt.axis('off')
    plt.savefig(f"generated_epoch_{epoch}.png")
    plt.close()

#Main
if __name__ == "__main__":
    all_image_paths = gather_image_paths(DATA_DIRECTORIES)
    dataset = create_dataset(all_image_paths)
    generator = create_generator_model()
    discriminator = create_discriminator_model()
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    execute_training(dataset, generator, discriminator, generator_optimizer, discriminator_optimizer)
