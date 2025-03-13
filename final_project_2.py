import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import time

class FloorplanGAN:
    def __init__(self):
        self.image_height = 128
        self.image_width = 128
        self.image_channels = 3
        self.batch_size = 64
        self.epochs = 100
        self.latent_dim = 100
        self.data_directories = ['/kaggle/input/cubicasa5k/cubicasa5k/cubicasa5k/colorful']
        
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        self.gen_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.disc_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

        self.seed = tf.random.normal([16, self.latent_dim])

    def _build_discriminator(self):
        inputs = tf.keras.layers.Input(shape=[self.image_height, self.image_width, self.image_channels])

        x = tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')(inputs)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        
        for _ in range(4):
            x = tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        
        x = tf.keras.layers.Flatten()(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs, name="discriminator")

    def _build_generator(self):
        inputs = tf.keras.layers.Input(shape=(self.latent_dim,))

        x = tf.keras.layers.Dense(256 * 4 * 4)(inputs)
        x = tf.keras.layers.Reshape((4, 4, 256))(x)

        for _ in range(5):
            x = tf.keras.layers.Conv2DTranspose(
                128, (4, 4), strides=(2, 2), padding='same', activation='relu'
            )(x)

        outputs = tf.keras.layers.Conv2D(
            self.image_channels, (3, 3), activation='tanh', padding='same'
        )(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs, name="generator")

    def _discriminator_loss(self, real_output, fake_output):
        real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(
            tf.ones_like(real_output), real_output
        )
        fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(
            tf.zeros_like(fake_output), fake_output
        )
        return real_loss + fake_loss

    def _generator_loss(self, fake_output):
        return tf.keras.losses.BinaryCrossentropy(from_logits=False)(
            tf.ones_like(fake_output), fake_output
        )

    @staticmethod
    def load_and_preprocess_image(image_path):
        img = Image.open(image_path).convert('RGB').resize((128, 128))
        img_array = np.array(img).astype(np.float32)
        return (img_array - 127.5) / 127.5

    def create_dataset(self):
        all_image_paths = []
        for directory in self.data_directories:
            for root, _, files in os.walk(directory):
                image_paths = [os.path.join(root, f) for f in files if f.lower().endswith('.png')]
                all_image_paths.extend(image_paths)

        dataset = tf.data.Dataset.from_tensor_slices(all_image_paths)
        dataset = dataset.map(
            lambda x: tf.numpy_function(
                self.load_and_preprocess_image, [x], Tout=tf.float32
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        return dataset.shuffle(buffer_size=1000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([self.batch_size, self.latent_dim])
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            
            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            
            gen_loss = self._generator_loss(fake_output)
            disc_loss = self._discriminator_loss(real_output, fake_output)

        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))
        
        return gen_loss, disc_loss

    def save_generated_images(self, epoch):
        """Generates and saves sample images"""
        predictions = self.generator(self.seed, training=False)
        predictions = (predictions + 1) / 2.0
        
        fig = plt.figure(figsize=(4, 4))
        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i])
            plt.axis('off')
        
        plt.savefig(f"generated_epoch_{epoch}.png")
        plt.close()

    def train(self):
        dataset = self.create_dataset()
        
        for epoch in range(self.epochs):
            start_time = time.time()

            for image_batch in dataset:
                g_loss, d_loss = self.train_step(image_batch)

            self.save_generated_images(epoch + 1)

            print(f"Epoch {epoch+1}/{self.epochs}, "
                  f"Generator Loss: {g_loss:.4f}, "
                  f"Discriminator Loss: {d_loss:.4f}, "
                  f"Time: {time.time() - start_time:.2f} sec")

def main():
    gan = FloorplanGAN()
    gan.train()

if __name__ == "__main__":
    main()
