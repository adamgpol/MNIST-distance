from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Input, Flatten,\
                                    Reshape, LeakyReLU,\
                                    Activation, Dropout
from tensorflow.keras.models import Model, Sequential
import tensorflow as tf
import numpy as np



def train_autoencoder(x_train, y_train,LATENT_SIZE,save_file,load_file=None,epochs=5):
    #build model:
    input_layer=Input((28,28))
    flatten_layer=Flatten()(input_layer)
    dense_1=Dense(512,name='encoder_dense_1')(flatten_layer)
    dense_1=LeakyReLU()(dense_1)
    dense_1=Dropout(0.5)(dense_1)
    dense_2=Dense(256,name='encoder_dense_2')(dense_1)
    dense_2=LeakyReLU()(dense_2)
    dense_2=Dropout(0.5)(dense_2)
    dense_3=Dense(256,name='encoder_dense_3')(dense_2)
    dense_3=LeakyReLU()(dense_3)
    dense_3=Dropout(0.5)(dense_3)
    dense_4=Dense(256,name='encoder_dense_4')(dense_3)
    dense_4=LeakyReLU()(dense_4)
    dense_4=Dropout(0.5)(dense_4)
    encoder_output=Dense(LATENT_SIZE,name='encoder_output')(dense_4)

    decoder_input=Input((LATENT_SIZE,),name='decoder_input')
    dense_1=Dense(64,name='decoder_dense_1')(decoder_input)
    dense_1=LeakyReLU()(decoder_input)
    dense_1=Dropout(0.5)(dense_1)
    dense_2=Dense(128,name='decoder_dense_2')(dense_1)
    dense_2=LeakyReLU()(dense_2)
    dense_2=Dropout(0.5)(dense_2)
    dense_3=Dense(256,name='decoder_dense_3')(dense_2)
    dense_3=LeakyReLU()(dense_3)
    dense_3=Dropout(0.5)(dense_3)
    dense_4=Dense(512,name='decoder_dense_4')(dense_3)
    dense_4=LeakyReLU()(dense_4)
    dense_4=Dropout(0.5)(dense_4)
    dense_5=Dense(28*28,activation='sigmoid',name='decoder_dense_5')(dense_4)
    decoder_output=Reshape((28,28),name='decoder_output')(dense_5)

    decoder=Model(inputs=decoder_input,outputs=decoder_output)
    model = Model(inputs = input_layer, outputs = decoder(encoder_output))

    model.compile(optimizer="nadam", loss = "binary_crossentropy")

    if load_file is not None:
        model = tf.keras.models.load_model(load_file)
    model.fit(x_train, x_train,epochs=epochs,batch_size=128)

    model.save(save_file)
    trained_encoder = Model(input_layer,encoder_output)
    trained_decoder = decoder

    return trained_encoder,trained_decoder
