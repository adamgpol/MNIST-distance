from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Input, Flatten,Reshape, Conv2D, Conv2DTranspose,Lambda
from tensorflow.keras.models import Model, Sequential
import tensorflow as tf
import numpy as np

LATENT_SIZE=10

def vae_sampling(encoder_output):
    mean, log_sigma = encoder_output
    epsilon = tf.random.normal(shape=(tf.shape(mean)[0], LATENT_SIZE),
                              mean=0., stddev=0.1)
    return mean + tf.math.exp(log_sigma) * epsilon

def train_autoencoder(x_train, y_train,save_file=None,load_file=None,epochs=5,batch_size=128):
    #build model:
    input_layer=Input((28,28),name='encoder_input')
    encoder_reshape_layer=Reshape((28,28,1),name='encoder_reshape')(input_layer)
    conv_1=Conv2D(32, 3, activation='relu', padding='same', strides=(2,2),name='encoder_conv_1')(encoder_reshape_layer)
    conv_2=Conv2D(64, 3, activation='relu', padding='same', strides=(2,2),name='encoder_conv_2')(conv_1)
    encoder_flatten_layer=Flatten(name='encoder_flatten')(conv_2)
    z_mean = Dense(LATENT_SIZE,name='encoder_z_mean')(encoder_flatten_layer)
    z_log_sigma = Dense(LATENT_SIZE,name='encoder_z_log_sigma')(encoder_flatten_layer)

    z = Lambda(vae_sampling,name='encoder_z')([z_mean, z_log_sigma])

    # decoder_input=Input((LATENT_SIZE,),name='decoder_input')
    dense_1=Dense(7*7*32,name='decoder_dense')(z)
    decoder_reshape_layer=Reshape((7,7,32),name='decoder_reshape')(dense_1)
    convt_1=Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same',
                activation='relu',name='decoder_convt_1')(decoder_reshape_layer)
    convt_2=Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same',
                activation='relu',name='decoder_convt_2')(convt_1)
    convt_3=Conv2DTranspose(
                filters=1, kernel_size=3, strides=1, padding='same',name='decoder_output')(convt_2)
    decoder_output=Reshape((28,28))(convt_3)


    # decoder = Model(inputs=decoder_input,outputs=decoder_output)
    encoder = Model(inputs=input_layer,outputs=z_mean)
    model = Model(inputs = input_layer, outputs = decoder_output)



    # VAE loss
    reconstruction_loss = tf.keras.losses.MeanSquaredError()(input_layer, decoder_output)
    reconstruction_loss = tf.math.multiply(reconstruction_loss,28*28)
    kl_loss = tf.math.add(1.,z_log_sigma)
    kl_loss = tf.math.subtract(kl_loss,tf.math.square(z_mean))
    kl_loss = tf.math.subtract(kl_loss,tf.math.exp(z_log_sigma))
    kl_loss = tf.reduce_sum(kl_loss, axis=-1)
    kl_loss = tf.math.multiply(kl_loss,-0.5)
    vae_loss = tf.reduce_mean(tf.math.add(reconstruction_loss,kl_loss))
    model.add_loss(vae_loss)

    model.compile(optimizer="adam")


    if load_file is not None:
        model.load_weights(load_file+'.h5')
    model.fit(x_train, x_train,epochs=epochs,batch_size=batch_size)

    if save_file is not None:
        model.save_weights(save_file+'.h5')

    trained_encoder = encoder
    # trained_decoder = decoder

    return trained_encoder,model
