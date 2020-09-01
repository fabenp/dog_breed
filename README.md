
# dog_breed

## Motivation of the project

Dog breeds are often  difficult to identify and are too many to learn all of them. So what better for image classification than deep learning? it is an efficient way to   label the dog breed in a matter of a second without going through the painfull process of googling random breed names each time we look for the information.

 ### Summary
 
 In order to have a function that takes as inpput an image and output the dog breed, two approchaes were first tried for developping the CNN classifying the breeds.
In the dog_app file the step by step development of the two approaches is detailed. In the first part a  CNN from scratch was studied . It consists mainly of conv2D and MaxPooling2D layers.   The CNN was trained on a database  from sklearn containing 113 dog breeds which led to an accuracy aroud 7% . In the second part a CNN with pretrained Resnet50 model and transfer learning was investigated, the best model weights were saved to a 'weights.best.Resnet50.hdf5' file and the model had an  accuracy of 81%. Based in this CNN, a dog breed detection function was defined. 
Other than this function two more were defined, one to detect a human face and one to detect dogs.
Based on the 3 functions, the final function  detect if an image is of a human or a dog and in both cases returns the most similar dog breed. In the case of dog the accuracy was satisfying, however, when testing it on human faces, it appears that it relies on the hair coloring to make the prediction. As an improvement, it would be interesting to compare human faces based on the distance between the face features and the hair style, length and color.

### Libraries used
numpy, pandas, nltk, sklearn, flask, json, c2, sqlite3, tqdm, glob

## Files in this repository
- dog_app.ipynb:  the main file with the code for the CNN model and all the functions needed for this project
- extract_bottlneck_features.py : python file containing the code to extract the bottleneck features for 4 models(VGG-19, ResNet-50, Inception, Xception)
- haarcascade_frontalface_alt: contains the pre-trained face detector data
- weights.best.Resnet50.hdf5: Once the model trained with resnet50 the best model weights are saved to this file so that it can be directly used for prediction without training the model each time

### Links: 
https://github.com/fabenp/dog_breed
 Medium : https://medium.com/@fatma.ben.dhieb/how-to-use-python-to-identify-a-dog-breed-2438cf6ad458


