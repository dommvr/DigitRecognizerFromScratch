import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def get_data():
    current_directory = os.getcwd()
    file = os.path.join(current_directory, 'data', 'test.csv')
    test_df = pd.read_csv(file)
    images = test_df.to_numpy()
    images = images / 255 #change pixels value from 0-255 to 0-1
    return images

images = get_data()

current_directory = os.getcwd()
weights_folder = os.path.join(current_directory, 'trained_neural_network')

b_i_h = np.load(os.path.join(weights_folder, 'b_i_h.npy'))
w_i_h = np.load(os.path.join(weights_folder, 'w_i_h.npy'))
b_h_o = np.load(os.path.join(weights_folder, 'b_h_o.npy'))
w_h_o = np.load(os.path.join(weights_folder, 'w_h_o.npy'))

while True:
    index = int(input('index: '))
    if index == 666: #type 666 to break loop and exit program without closing console
        exit()
    image = images[index]
    plt.imshow(image.reshape(28, 28), cmap="Greys") #np.reshape() == 1D array to 2D array with given shape
    image.shape += (1,)
    #Forward propagation input --> hidden
    h_pre = b_i_h + w_i_h @ image.reshape(784, 1)
    h = 1 / (1 + np.exp(-h_pre))
    #Forward propagation hidden --> output
    o_pre = b_h_o + w_h_o @ h
    o = 1 / (1 + np.exp(-o_pre))

    plt.title(f"This is a {o.argmax()}")
    plt.show()