import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()

        self.patch_size=patch_size
    
    def call(self, images):
        batch_size = tf.shape(images)[0]

        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )
        patch_dims = patches.shape[-1]

        patches = tf.reshape(patches, [batch_size, -1, patch_dims])

        return patches


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()

        self.num_patches = num_patches
        # projection encoding layer
        self.projection = layers.Dense(units=projection_dim)
        # position encoding layer
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches)

        return self.projection(patches) + self.position_embedding(positions)
