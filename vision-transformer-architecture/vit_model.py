import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from vit_patch_layers import Patches, PatchEncoder


class VisionTransformer():

    def __init__(self, input_shape, image_size, patch_size, projection_dim, num_transformer_layers, num_attention_heads, num_mlp_heads, output_classes) -> None:
        self.input_shape = input_shape
        self.image_size = image_size
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.num_transformer_layers = num_transformer_layers
        self.num_attention_heads = num_attention_heads
        self.num_mlp_heads = num_mlp_heads
        self.output_classes = output_classes

        self.transformer_units = [projection_dim*2, projection_dim]
        self.num_patches = (image_size // patch_size) ** 2

        self.model = self._build_architecture()

    def _build_architecture(self):
        # adding input layer
        inputs = layers.Input(self.input_shape)

        resized = layers.Resizing(self.image_size, self.image_size)(inputs)

        # creating patches of image
        patches = Patches(self.patch_size)(resized)

        # encoding patches
        patches_encoded = PatchEncoder(self.num_patches, self.projection_dim)(patches)

        # transformer blocks
        patches_encoded = self._add_transformer_layers(patches_encoded)

        # creating [batch size, projection_dim] tensor
        representation = self._add_representation_block(patches_encoded)

        # add multi-layered-perceptron
        features = self._add_mlp_block(representation, hidden_units=self.num_mlp_heads, dropout=0.5)

        # classify outputs
        outputs = layers.Dense(self.output_classes)(features)

        return keras.Model(inputs=inputs, outputs=outputs)

    def _add_mlp_block(self, x, hidden_units, dropout):
        for units in hidden_units:
            x = layers.Dense(units)(x)
            x = tfa.layers.GELU()(x)
            x = layers.Dropout(dropout)(x)
        return x

    def _add_transformer_layers(self, patches_encoded):

        for i in range(self.num_transformer_layers):
            # layer normalization
            x1 = layers.LayerNormalization(epsilon=1e-6)(patches_encoded)
            # multi head attention
            attention_output = layers.MultiHeadAttention(
                num_heads=self.num_attention_heads, 
                key_dim=self.projection_dim, 
                dropout=0.1
            )(x1, x1)

            # skip connection
            x2 = layers.Add()([attention_output, patches_encoded])

            # layer normalization
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)

            # mlp
            x3 = self._add_mlp_block(x3, hidden_units=self.transformer_units, dropout=0.1)

            # skip connection
            patches_encoded=layers.Add()([x3, x2])
        
        return patches_encoded

    def _add_representation_block(self, patches_encoded):
        representation = layers.LayerNormalization(epsilon=1e-06)(patches_encoded)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.5)(representation)

        return representation

    def compile(self, learning_rate=0.001, weight_decay=0.0001):
        """
        Compiles model
        """
        optimizer=tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
        self.model.compile(
            optimizer=optimizer, 
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy")
            ]
        )
    
    def fit(self, X, y, batch_size=32, epochs=100):
        self.model.fit(x=X, y=y, batch_size=batch_size, epochs=epochs)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)
    
    def summary(self):
        self.model.summary()

    def save(self, save_dir="."):
        self.model.save(filepath=save_dir)







