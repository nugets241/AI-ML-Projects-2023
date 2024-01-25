import os
import sys
from matplotlib import pyplot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

# Define paths
train_dir = 'images/train/'
test_dir = 'images/test/'

# Define CNN model to be trained 
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))

# Three output values (one for each class: dogs, cats, horses)
model.add(Dense(3, activation='softmax'))  # Change here

# Compile the model
opt = SGD(lr=0.001, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])  # Change here

# Create data generator for reading the image data from directories
datagen = ImageDataGenerator(rescale=1.0/255.0)

# Prepare the iterators both for test and train
train_iterator = datagen.flow_from_directory(train_dir, class_mode='categorical', batch_size=64, target_size=(200, 200))  # Change here
test_iterator = datagen.flow_from_directory(test_dir, class_mode='categorical', batch_size=64, target_size=(200, 200))  # Change here

# Train the model and save the training history 
epochs=10
history = model.fit(train_iterator, steps_per_epoch=len(train_iterator), validation_data=test_iterator, validation_steps=len(test_iterator), epochs=epochs)

# Save the trained model to a file
filename = 'basemodel'
model.save('models/' + filename + '.h5')

# Estimate the model by calculating accuracy with test images
_, accuracy = model.evaluate_generator(test_iterator, steps=len(test_iterator))
print('> Accuracy: %.2f' % (accuracy * 100.0))

# Draw the training curves
# Draw curve for loss 
pyplot.subplot(211)
pyplot.title('Cross Entropy Loss')
pyplot.plot(history.history['loss'], color='blue', label='train')
pyplot.plot(history.history['val_loss'], color='orange', label='test')

# Draw curves for accuracy
pyplot.subplot(212)
pyplot.title('Classification Accuracy')
pyplot.plot(history.history['accuracy'], color='blue', label='train')
pyplot.plot(history.history['val_accuracy'], color='orange', label='test')

# Save plots to one file
pyplot.savefig('plots/' + filename + '_plot.png')
pyplot.close()
