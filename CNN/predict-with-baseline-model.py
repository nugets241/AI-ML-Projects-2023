import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np
from pathlib import Path

# Define paths
unidentified_images = 'unidentified_images/'
identified_images = 'identified_images/'

# Create directory for the identified/recognised images
os.makedirs(name=identified_images, exist_ok=True)

# Load the trained model
model = load_model('models/basemodel.h5')

# Make operations for all images with '.jpg' extension
for file in os.listdir(unidentified_images):
   print('File: ' + file)
   # Make predictions for image if it is '*.jpg'
   if Path(unidentified_images + file).suffix == '.jpg':
       # Load the image for prediction
       new_image = plt.imread(unidentified_images + file)
       # Resize the image, it is also scaled as 1/255.0 just like with training
       resized_image = resize(new_image, (200, 200, 3))
       # Predict image class as array of propabilities
       predictions = model.predict(np.array([resized_image]))[0]
       # Print the predicted class
       print('Prediction: ')
       predicted_class = np.argmax(predictions)
       class_names = ['Cat', 'Dog', 'Horse']  # Define class names
       print(class_names[predicted_class])  # Print the name of the predicted class
       print(predictions)
       # Save the original (size unchanged) as image to be saved
       img = plt.imshow(new_image)
       # Add the predicted class as the title with prediction
       plt.title('Prediction: ' + class_names[predicted_class])  # Convert numpy.int64 to str here
       # Clean the plot and save it
       plt.axis('off')
       plt.savefig(identified_images + file.split('.')[0] + '-identified.jpg', bbox_inches='tight', pad_inches=0)
       # Close the plot before next image
       plt.close()
