# dog_breed
Detect the breed of the dog in the image provided and if it's a human image,  the app returns the resembling dog breed.

### Summary
This app detects if an image is of a human or a dog and returns the dog breed corresponding to the image path submitted by the user. If it's a human image, the app returns to which breed the person has more similarity.

### Libraries used
numpy, pandas, nltk

## Files in this repository
extract_bottlneck_features.py
haarcascade_frontalface_alt
Procfile: This file is needed for heroku as it will indicates what to run first.
requirements.txt: 
Webapp.py: 
weights.best.Resnet50.hdf5: Once the model trained with resnet50 the best model weights are saved to this file
### HTML files:
go.html : HTML code needed to return the breed classification of the image entrered by the user.
master.html: HTML code for the design of the web app.
### Run the web app:
run.py: file to run in order to access the web app. It contains the data needed for the plots and the code to plot the barplots.

### Run the python script:
run in the command line: 

to run the app,acces the folder containing the run.py file and type in the command line:
python run.py
In another terminal window, run:
env|grep WORK
This will show the credentials needed for the web app address

### Link: 
https://github.com/fabenp/dog_breed


