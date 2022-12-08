import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

"""
w = weights, b = bias, i = input, h = hidden, o = output, l = label
e.g. w_i_h = weights from input layer to hidden layer
"""

def get_data():
    current_directory = os.getcwd()
    file = os.path.join(current_directory, 'data', 'train.csv')
    train_df = pd.read_csv(file)
    labels = train_df['label'].copy()
    labels = labels.to_numpy()
    labels = np.eye(10)[labels]
    images = train_df.drop('label', axis=1)
    images = images.to_numpy()
    images = images / 255 #change pixels value from 0-255 to 0-1
    return labels, images

#labels.shape = (number of pictures, 10) 10 because when there is more then 2 possible outcomes we need to present it in binary form
#images.shape = (number of pictures, number of pixels in picture)
labels, images = get_data()

#np.random.uniform(low=-0.5, high=0.5, size=(10, 20))
#Create matrix of given sizes with values between low and high
w_i_h = np.random.uniform(-0.5, 0.5, (20, 784))
w_h_o = np.random.uniform(-0.5, 0.5, (10, 20))

#np.zeros(shape)
#Create matrix of given size with 0
b_i_h = np.zeros((20, 1))
b_h_o = np.zeros((10, 1))

learn_rate = 0.01
nr_correct = 0
epochs = 10 #1-83.5, 2-91.5, 10-95, 50-97.5, 100-98.5
epochs_accuracy = []

for epoch in range(epochs):
    for image, label in zip(images, labels):
        image.shape += (1,) #image.shape = 784 Vector --> image.shape = (784, 1) Matrix
        label.shape += (1,) #label.shape = 10 Vector --> label.shape = (10, 1) Matrix
        #Forward propagation input --> hidden
        h_pre = b_i_h + w_i_h @ image # @ is used to multiply matrix
        h = 1 / (1 + np.exp(-h_pre)) #Sigmoid function 1/(1+e^-x), np.exp(x) == e^x
        #Forward propagation hidden --> output
        o_pre = b_h_o + w_h_o @ h
        o = 1 / (1 + np.exp(-o_pre))
        #Error calculation
        e = 1 / len(o) * np.sum((o - label) ** 2, axis=0) #np.sum() == sum of two matrix 
        nr_correct += int(np.argmax(o) == np.argmax(label)) #np.argmax() == returns position of max value in matrix
        #Backpropagation output --> hidden (cost function derivative)
        delta_o = o - label
        w_h_o += -learn_rate * delta_o @ np.transpose(h) #np.transform() == gives matrix transpose
        b_h_o += -learn_rate * delta_o
        #Backpropagation hidden --> input (activation function derivative)
        delta_h = np.transpose(w_h_o) @ delta_o * (h * (1 - h))
        w_i_h += -learn_rate * delta_h @ np.transpose(image)
        b_i_h += -learn_rate * delta_h
    
    #Show accuracy for this epoch
    accuracy = round((nr_correct / images.shape[0]) * 100, 2)
    epochs_accuracy.append(accuracy)
    print(f"Epoch {epoch + 1} accuracy: {accuracy}%")
    nr_correct = 0

#Save trained neural network (save neural network weights)
weights = {
    'b_i_h': b_i_h,
    'w_i_h': w_i_h,
    'b_h_o': b_h_o,
    'w_h_o': w_h_o
}

current_directory = os.getcwd()
weights_folder = os.path.join(current_directory, 'trained_neural_network')
os.makedirs(weights_folder, exist_ok=True)

for i in weights:
    np.save(os.path.join(weights_folder, f"{i}.npy"), weights[i])

#Epochs accuracy diagram
plt.plot(epochs_accuracy)
plt.xticks(range(0,len(epochs_accuracy)+1, 5))
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.title("Epochs accuracy")
plt.show()