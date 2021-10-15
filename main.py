from tensorflow.keras.datasets import mnist
from src.distances import train_autoencoder,class_distances
from src.classifier import *
import numpy as np
from matplotlib import pyplot as plt
from IPython import display # If using IPython, Colab or Jupyter

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train/255.0
x_test = x_test/255.0
# Plot image data from x_train
LATENT_SIZE = 10
save_file='./autoencoder_models/latent_size_10'

encoder, encoder_decoder= train_autoencoder(x_train,y_train,
                                save_file=None,
                                load_file=save_file,
                                epochs=0,
                                batch_size=128)

np.set_printoptions(precision=2, linewidth=100, suppress=True)

distances=class_distances(10,x_train,y_train,encoder)
print('y_train DEBUG:',y_train[5])
print('distance matrix:\n',distances)

model = build_model()
train_model(model,x_train,get_one_hot(y_train,10),(x_test,get_one_hot(y_test,10)))

fig, axs = plt.subplots(4, 8)
rand = x_test[np.random.randint(0, 10000, 16)].reshape((4, 4, 1, 28, 28))

display.clear_output() # If you imported display from IPython

for i in range(4):
    for j in range(4):
        axs[i, j].imshow(encoder_decoder(rand[i, j])[0], cmap = "gray")
        axs[i, j].axis("off")
        axs[i, j+4].imshow((rand[i, j])[0], cmap = "gray")
        axs[i, j+4].axis("off")
        print(np.argmax(model(rand[i,j])),end=' ')
    print('')


plt.subplots_adjust(wspace = 0, hspace = 0)
plt.show()
