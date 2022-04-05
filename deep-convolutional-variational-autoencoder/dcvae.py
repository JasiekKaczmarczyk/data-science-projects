from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Dense, Flatten, Reshape, Conv2DTranspose, Activation, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
import os
import pickle
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

class DCVAE:

    def __init__(self, input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim):
        """
        Deep Convolutional Variantional Autoencoder

        Parameters:
        input_shape (tuple): shape of the input image
        conv_filters (tuple): convolutional filters in each layer
        conv_kernels (tuple): size of kernel in each layer
        conv_strides (tuple): strides in each layer
        latent_space_dim [int]: size of the bottleneck layer
        """
        # input shape
        self.input_shape=input_shape
        # list of conv layers
        self.conv_filters=conv_filters
        # list of kernels in each layer
        self.conv_kernels=conv_kernels
        # list of strides in each convolution layer
        self.conv_strides=conv_strides
        # bottleneck width
        self.latent_space_dim=latent_space_dim

        # reconstruction loss weight
        self.reconstruction_loss_weight=1000000

        # keras architectures
        self.encoder = None
        self.decoder = None
        self.model = None

        self._num_conv_layers=len(conv_filters)
        self._shape_before_bottleneck=None
        self._model_input=None

        # build DCVAE
        self._build()


    def summary(self):
        """
        Prints summary of model
        """
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def _build(self):
        """
        Builds DCVAE
        """
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    # BUILDING ENCODER
    def _build_encoder(self):
        """
        Method that builds encoder part
        """
        # creating input layer
        encoder_input=self._add_encoder_input()

        # building convolutional layers
        conv_layers=self._add_conv_layers(encoder_input)

        # build bottleneck
        bottleneck=self._add_bottleneck(conv_layers)

        # build encoder
        self.encoder=Model(encoder_input, bottleneck, name="encoder")

        # assign input for encoder
        self._model_input=encoder_input
    
    def _add_encoder_input(self):
        """
        Creates input layer
        """
        return Input(shape=self.input_shape, name="encoder_input")

    def _add_conv_layers(self, encoder_input):
        """
        Creates all convolutional blocks in encoder
        """
        x=encoder_input

        for layer_index in range(self._num_conv_layers):
            x=self._add_conv_layer(layer_index, x)
        
        return x

    def _add_conv_layer(self, layer_index, x):
        """
        Adds conv block to graph of layers consisting of conv2d + ReLU + batch normalization layer
        """
        # creating convolutional layer
        conv_layer = Conv2D(
            filters=self.conv_filters[layer_index], 
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"encoder_conv_layer_{layer_index}"
        )


        # applying layer to graph
        x = conv_layer(x)
        # applying ReLU
        x = ReLU(name=f"encoder_relu_layer_{layer_index}")(x)
        # applying batch normalization
        x = BatchNormalization(name=f"encoder_bn_layer_{layer_index}")(x)

        return x
    
    def _add_bottleneck(self, x):
        """
        Flattens data, adds bottleneck with Gaussian sampling (Dense)
        """
        # shape of data before flattening
        self._shape_before_bottleneck = K.int_shape(x)[1:] # returns [batch_size, width, height, nr_channels] we don't need batch_size

        # adding layers to graph

        # applying flattening
        x = Flatten()(x)

        # applying mu and variance layers
        self.mu = Dense(self.latent_space_dim, name="mu")(x)
        self.log_variance = Dense(self.latent_space_dim, name="log_variance")(x)

        def sample_point_from_normal_distribution(args):
            mu, log_variance = args
            epsilon = K.random_normal(shape=K.shape(self.mu), mean=0, stddev=1)

            return mu + K.exp(log_variance / 2) * epsilon

        # applying custom layer
        x = Lambda(sample_point_from_normal_distribution, name="encoder_output")([self.mu, self.log_variance])

        return x

    
    # BUILDING DECODER
    def _build_decoder(self):
        """
        Method that builds decoder part
        """
        # adding decoder input
        decoder_input=self._add_decoder_input()

        # adding dense layer
        dense_layer=self._add_dense_layer(decoder_input)

        # reshaping layer to 3d
        reshape_layer=self._add_reshape_layer(dense_layer)

        # applying convolutional layers
        conv_transpose_layers=self._add_conv_transpose_layers(reshape_layer)

        decoder_output=self._add_decoder_output(conv_transpose_layers)

        # building decoder
        self.decoder=Model(decoder_input, decoder_output, name="decoder")
    
    def _add_decoder_input(self):
        """
        Adding input for decoder
        """
        return Input(shape=self.latent_space_dim, name="decoder_input")
    
    def _add_dense_layer(self, decoder_input):
        """
        Adds dense layer
        """
        # np.prod multiplies all elements in array
        num_neurons=np.prod(self._shape_before_bottleneck)
        dense_layer=Dense(num_neurons, name="decoder_dense")(decoder_input)

        return dense_layer
    
    def _add_reshape_layer(self, dense_layer):
        """
        Adds reshape layer
        """
        return Reshape(self._shape_before_bottleneck, name="reshape")(dense_layer)

    def _add_conv_transpose_layers(self, x):
        """
        Adds convolutional transpose blocks
        """
        # looping through layers in reverse without last layer
        for layer_index in reversed(range(1, self._num_conv_layers)):
            x = self._add_conv_transpose_layer(layer_index, x)

        return x
    
    def _add_conv_transpose_layer(self, layer_index, x):
        """
        Creates convoltional transpose layer
        """

        layer_num=self._num_conv_layers - layer_index

        conv_transpose_layer = Conv2DTranspose(
            filters=self.conv_filters[layer_index], 
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"decoder_conv_transpose_layer_{layer_num}"
        )

        # applying convolutional transpose
        x=conv_transpose_layer(x)
        # applying ReLU
        x=ReLU(name=f"decoder_relu_layer_{layer_num}")(x)
        # applying batch normalization
        x=BatchNormalization(name=f"decoder_bn_layer_{layer_num}")(x)

        return x

    def _add_decoder_output(self, x):
        """
        Creates decoder output layer (similar to _add_conv_transpose_layer but without ReLU and BatchNormalization and with sigmoid activation)
        """

        conv_transpose_layer = Conv2DTranspose(
            filters=1, 
            kernel_size=self.conv_kernels[0],
            strides=self.conv_strides[0],
            padding="same",
            name=f"decoder_conv_transpose_layer_{self._num_conv_layers}"
        )

        x=conv_transpose_layer(x)
        output_layer=Activation("sigmoid", name="sigmoid_layer")(x)

        return output_layer

    # BUILDING AUTOENCODER
    def _build_autoencoder(self):
        """
        Builds autoencoder
        """

        model_input=self._model_input
        model_output=self.decoder(self.encoder(model_input))
    
        # merging encoder and decoder
        self.model = Model(model_input, model_output, name="autoencoder")

    def _calculate_combined_loss(self, y_true, y_pred):
        """
        Building custom loss function
        """
        reconstruction_loss=self._calculate_reconstruction_loss(y_true, y_pred)
        kl_loss=self._calculate_kl_loss(y_true, y_pred)

        return self.reconstruction_loss_weight*reconstruction_loss + kl_loss

    def _calculate_reconstruction_loss(self, y_true, y_pred):
        """
        MSE Loss
        """
        error = y_true - y_pred
        reconstruction_loss = K.mean(K.square(error), axis=[1,2,3])
        return reconstruction_loss
    
    def _calculate_kl_loss(self, y_true, y_pred):
        """
        KL Divergence
        """
        return -0.5*K.sum(1 + self.log_variance - K.square(self.mu) - K.exp(self.log_variance), axis=1)

    # COMPILING MODEL
    def compile(self, learning_rate=0.0001):
        """
        Compiles model
        """
        optimizer=Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer, 
            loss=self._calculate_combined_loss
        )

    # TRAINING MODEL
    def train(self, X_train, batch_size, epochs):
        """
        Trains model
        """
        self.model.fit(
            x=X_train, 
            y=X_train, 
            batch_size=batch_size, 
            epochs=epochs,
            shuffle=True
            )
    
    # SAVING MODEL
    def save(self, save_folder="."):
        """
        Saves model's parameters and weights
        """
        self._create_folder_if_doesnt_exist(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)

    def _create_folder_if_doesnt_exist(self, save_folder):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

    def _save_parameters(self, save_folder):
        parameters=[
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim
        ]

        save_path=os.path.join(save_folder, "parameters.pkl")

        with open(save_path, "wb") as file:
            pickle.dump(parameters, file)

    def _save_weights(self, save_folder):
        save_path=os.path.join(save_folder, "weights.h5")
        self.model.save_weights(save_path)


    # LOADING MODEL
    @classmethod
    def load(cls, save_folder="."):
        """
        Loads trained model
        """

        parameters_path=os.path.join(save_folder, "parameters.pkl")
        weights_path=os.path.join(save_folder, "weights.h5")

        # load parameters
        with open(parameters_path, "rb") as f:
            parameters=pickle.load(f)

        # create autoencoder
        autoencoder=DCVAE(*parameters)
        
        # load weights
        autoencoder._load_weights(weights_path)

        return autoencoder

    def _load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    
    # RECONSTRUCT IMAGE
    def reconstruct(self, images):
        """
        Reconstructs image
        """
        
        latent_representations=self.encoder.predict(images)
        recostructed_images=self.decoder.predict(latent_representations)

        return recostructed_images, latent_representations