import tensorflow as tf


# Binary Cross-Entropy loss
loss_object = tf.keras.losses.BinaryCrossentropy(
    from_logits=True
)

# Weight for L1 reconstruction loss
LAMBDA = 100


def discriminator_loss(disc_real_output, disc_generated_output):

    real_loss = loss_object(
        tf.ones_like(disc_real_output),
        disc_real_output
    )

    generated_loss = loss_object(
        tf.zeros_like(disc_generated_output),
        disc_generated_output
    )

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


def generator_loss(
    disc_generated_output,
    gen_output,
    target
):

    # Adversarial loss
    gan_loss = loss_object(
        tf.ones_like(disc_generated_output),
        disc_generated_output
    )

    # Reconstruction loss
    l1_loss = tf.reduce_mean(
        tf.abs(target - gen_output)
    )

    # Total generator loss
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss