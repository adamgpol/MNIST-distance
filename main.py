from tensorflow.keras.datasets import mnist
from src.distances import train_autoencoder
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train/255.0
x_test = x_test/255.0
# Plot image data from x_train
LATENT_SIZE = 2
save_file='./autoencoder_models/latent_size_2'

encoder,decoder = train_autoencoder(x_train,y_train,LATENT_SIZE,save_file)

fig, axs = plt.subplots(4, 4)
rand = x_test[np.random.randint(0, 10000, 16)].reshape((4, 4, 1, 28, 28))

# display.clear_output() # If you imported display from IPython

for i in range(4):
    for j in range(4):
        axs[i, j].imshow(decoder(encoder((rand[i, j])))[0], cmap = "gray")
        axs[i, j].axis("off")

plt.subplots_adjust(wspace = 0, hspace = 0)
plt.show()
