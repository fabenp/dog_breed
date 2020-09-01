# dog_breed
Detect the breed of the dog in the image provided and if it's a human image, returns the resembling  breed.

### Summary
This repository contains the web_app file with the detailed step by step development of a CNN from scratch and a CNN with pretrained model.
This aims of the developed function is to detect if an image is of a human or a dog and returns the dog breed corresponding to the image path submitted by the user. If it's a human image, the app returns to which breed the person has more similarity.

To detect the dog breed a convolutional neural network was adopted: It consists mainly of conv2D and MaxPooling2D layers.  
The CNN was trained on a database  from sklearn containing 113 dog breeds but the accuracy was very low.
  
A pretrained model with Resnet50 was then used and the best model weights were saved to a 'weights.best.Resnet50.hdf5' file. The accuracy obtained was higher than 80%, and 
the function developed showed satsfying results in predicting dogs breeds, however, when testing it on human faces, it appears that it relies on the hair coloring to make the prediction. As an improvement, it would be interesting to compare human faces based on the distance between the face features. 

### Libraries used
numpy, pandas, nltk, sklearn, flask, json, c2, sqlite3, tqdm, glob
## Files in this repository
-dog_app.ipynb:  the main file with the code for the CNN model and all the functions needed for this project
- extract_bottlneck_features.py : python file containing the code to extract the bottleneck features for 4 models(VGG-19, ResNet-50, Inception, Xception)
- haarcascade_frontalface_alt: contains the pre-trained face detector data
- weights.best.Resnet50.hdf5: Once the model trained with resnet50 the best model weights are saved to this file

### Link: 
https://github.com/fabenp/dog_breed


