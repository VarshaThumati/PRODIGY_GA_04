import os
import tensorflow as tf

from model import Generator, Discriminator
from losses import generator_loss, discriminator_loss


# Configuration
LEARNING_RATE = 0.0002
BETA_1 = 0.5


# Create models
generator = Generator()
discriminator = Discriminator()


# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(
    LEARNING_RATE,
    beta_1=BETA_1
)

discriminator_optimizer = tf.keras.optimizers.Adam(
    LEARNING_RATE,
    beta_1=BETA_1
)


def train_step(input_image, target_image):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        # Generate image
        gen_output = generator(
            input_image,
            training=True
        )

        # Discriminator on real pair
        disc_real_output = discriminator(
            [input_image, target_image],
            training=True
        )

        # Discriminator on generated pair
        disc_generated_output = discriminator(
            [input_image, gen_output],
            training=True
        )

        # Calculate losses
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
            disc_generated_output,
            gen_output,
            target_image
        )

        disc_loss = discriminator_loss(
            disc_real_output,
            disc_generated_output
        )

    # Generator gradients
    generator_gradients = gen_tape.gradient(
        gen_total_loss,
        generator.trainable_variables
    )

    # Discriminator gradients
    discriminator_gradients = disc_tape.gradient(
        disc_loss,
        discriminator.trainable_variables
    )

    # Update Generator
    generator_optimizer.apply_gradients(
        zip(
            generator_gradients,
            generator.trainable_variables
        )
    )

    # Update Discriminator
    discriminator_optimizer.apply_gradients(
        zip(
            discriminator_gradients,
            discriminator.trainable_variables
        )
    )

    return (
        gen_total_loss,
        gen_gan_loss,
        gen_l1_loss,
        disc_loss
    )