# baseline model with multi-layer perceptrons


import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.utils import np_utils

train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

'''
The training dataset is structured as a 3-dimensional array of instance, image width and image height. 
For a multi-layer perceptron model we must reduce the images down into a vector of pixels. 
In this case the 28×28 sized images will be 784 pixel input values.
'''
# flatten 28*28 images to a 784 vector for each image
num_pixels= train_images.shape[1] * train_images.shape[1]
train_images= train_images.reshape((train_images.shape[0], num_pixels)).astype('float32')
test_images= test_images.reshape((test_images.shape[0], num_pixels)).astype('float32')

# normalising the inputs from 0-255 to 0-1
train_images= train_images/255
test_images= test_images/255

'''
Finally, the output variable is an integer from 0 to 9. This is a multi-class classification problem.
As such, it is good practice to use a one hot encoding of the class values, transforming the vector of class integers into a binary matrix.
'''

# one hot encode outputs
test_labels= np_utils.to_categorical(test_labels)
train_labels= np_utils.to_categorical(train_labels)
num_classes= test_labels.shape[1] #????

#define baseline model

def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

'''
The model is a simple neural network with one hidden layer with the same number of neurons as there are inputs (784).
A rectifier activation function is used for the neurons in the hidden layer.
A softmax activation function is used on the output layer to turn the outputs into 
probability-like values and allow one class of the 10 to be selected as the model’s output prediction. 
Logarithmic loss is used as the loss function (called categorical_crossentropy in Keras) and the efficient ADAM gradient descent algorithm is used to learn the weights.
The model is fit over 10 epochs with updates every 200 images.
 A verbose value of 2 is used to reduce the output to one line for each training epoch.
'''

# idk what is batch size, verbose
#i guess scores[1] is accuracy
#build the model
model = baseline_model()
# Fit the model
model.fit(train_images, train_labels, validation_data=(test_images,test_labels), epochs=10, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(test_images, test_labels, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

# 2.07%
# error- 1.96%


