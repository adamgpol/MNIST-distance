from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Input, Flatten,\
                                    Reshape, LeakyReLU as LR,\
                                    Activation, Dropout
from tensorflow.keras.models import Model, Sequential
from matplotlib import pyplot as plt
from IPython import display # If using IPython, Colab or Jupyter
import numpy as np



def train_autoencoder(x_train, y_train,LATENT_SIZE,save_file,epochs=5):
    encoder = Sequential([
        Flatten(input_shape = (28, 28)),
        Dense(512),
        LR(),
        Dropout(0.5),
        Dense(256),
        LR(),
        Dropout(0.5),
        Dense(128),
        LR(),
        Dropout(0.5),
        Dense(64),
        LR(),
        Dropout(0.5),
        Dense(LATENT_SIZE),
        LR(name='encoder_output')
    ])
    decoder = Sequential([
        Dense(64, input_shape = (LATENT_SIZE,),name='decoder_input'),
        LR(),
        Dropout(0.5),
        Dense(128),
        LR(),
        Dropout(0.5),
        Dense(256),
        LR(),
        Dropout(0.5),
        Dense(512),
        LR(),
        Dropout(0.5),
        Dense(784),
        Activation("sigmoid"),
        Reshape((28, 28),name='decoder_output')
    ])
    img = Input(shape = (28, 28))
    latent_vector = encoder(img)
    output = decoder(latent_vector)
    model = Model(inputs = img, outputs = output)
    model.compile("nadam", loss = "binary_crossentropy")
    EPOCHS = epochs
    #Only do plotting if you have IPython, Jupyter, or using Colab
    # for epoch in range(EPOCHS):
    #     fig, axs = plt.subplots(4, 4)
    #     rand = x_test[np.random.randint(0, 10000, 16)].reshape((4, 4, 1, 28, 28))
    #
    #     display.clear_output() # If you imported display from IPython
    #
    #     for i in range(4):
    #         for j in range(4):
    #             axs[i, j].imshow(model.predict(rand[i, j])[0], cmap = "gray")
    #             axs[i, j].axis("off")
    #
    #     plt.subplots_adjust(wspace = 0, hspace = 0)
    #     plt.show()
    #     print("-----------", "EPOCH", epoch, "-----------")
    model.fit(x_train, x_train,epochs=epochs,batch_size=128)

    model.save(save_file)
    trained_encoder = Model(model.input, model.get_layer('encoder_output').output)
    trained_decoder = Model(model.get_layer('decoder_input').input, model.get_layer('decoder_output').output)

    return trained_encoder,trained_decoder
