from sklearn.datasets import load_files 
from keras.utils import np_utils
import numpy as np
from glob import glob
import cv2                
import matplotlib.pyplot as plt                        

#%matplotlib inline   
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image                  
from tqdm import tqdm
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.callbacks import ModelCheckpoint
from extract_bottleneck_features import *

import json
import plotly
import pandas as pd
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import plotly.graph_objs as go
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
import sqlite3
app = Flask(__name__)

# define function to load train, test, and validation datasets
#def load_dataset(path):
 #   data = load_files(path)
  #  dog_files = np.array(data['filenames'])
  #  dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
   # return dog_files, dog_targets

# load train, test, and validation datasets
#train_files, train_targets = load_dataset('../../../data/dog_images/train')
#valid_files, valid_targets = load_dataset('../../../data/dog_images/valid')
#test_files, test_targets = load_dataset('../../../data/dog_images/test')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("../../../data/dog_images/train/*/"))]

#Detect humans
                           
# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()

def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')

#preprocess data
def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)
def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)
#prediction with resnet50
def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))
### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))

### TODO: Obtain bottleneck features from another pre-trained CNN.
#bottleneck_features = np.load('bottleneck_features/DogResnet50Data.npz')
#train_Resnet50 = bottleneck_features['train']
#valid_Resnet50 = bottleneck_features['valid']
#test_Resnet50=  bottleneck_features['test']
### TODO: Define your architecture.
Resnet50_model= Sequential()
Resnet50_model.add(GlobalAveragePooling2D(input_shape=train_Resnet50.shape[1:]))
Resnet50_model.add(Dense(133, activation='softmax'))
### TODO: Compile the model.
#Resnet50_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

### TODO: Train the model.
#checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.Resnet50.hdf5', verbose=1, save_best_only=True)
#Resnet50_model.fit(train_Resnet50, train_targets,validation_data=(valid_Resnet50,valid_targets), epochs= 20, batch_size=20, callbacks=#[checkpointer],verbose=1)
### TODO: Load the model weights with the best validation loss
Resnet50_model.load_weights('saved_models/weights.best.Resnet50.hdf5')

### TODO: Write a function that takes a path to an image as input
### and returns the dog breed that is predicted by the model.
def breed_detection(path):
    bottleneck_features = extract_Resnet50(path_to_tensor(path))
    predicted_vector=Resnet50_model.predict(bottleneck_features)
    breed= dog_names[np.argmax(predicted_vector)]   
    return breed.split('.')[1]            
            
def human_vs_dog(path):
    dog=dog_detector(path)
    human=face_detector(path)
    if dog==True:
        breed=breed_detection(path)
        return ('The breed is', breed)
        
    else:
        if human== True:
            most_similar_breed= breed_detection(path)
            return ('This person looks like a {}'.format(most_similar_breed))
        else:
            return ("It's not  an image of a dog nor a human")                                  
          
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    
    classification_results = human_vs_dog(query)

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results)

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()      
            
            
            
            
            