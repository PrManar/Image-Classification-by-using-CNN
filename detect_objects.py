# -*- coding: utf-8 -*-


from abc import abstractclassmethod
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data() #load the dataset from keras
training_images, testing_images = training_images / 255, testing_images / 255 # for normalization the minimum = 0, the maximum = 255

#objects to be classified:
class_names = ['plane', 'car','bird', 'cat','deer', 'dog','frog', 'horse', 'ship', 'truck'] #should be in the same order

for i in range(16):
  plt.subplot(4,4,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(training_images[i], cmap=plt.cm.binary) #to show the training images
  plt.xlabel(class_names[training_labels[i][0]]) #the labeld of particular images
plt.show()

#Start training the model:
#reduce the amount of images that been feeding the model to save time and resources
training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

#buliding the model: conv layer then maxpooling layer and so on
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Flatten()) # make the image flatt like in 1D
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

#evaluate the model
loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

#save the model
#model.save('image_classifier.model')

#I can delete the whole training code and use only the following with images from the internet to test it:
#model = models.load_model('image_classifier.model')
