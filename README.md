# Fraud Detection Case Study

## Overview
Built machine learning model to predict fraud events based on event details submitted. Compared the performance of Logistic Regression, Random Forest and AdaBoost Classifier Algorithms by fine tuning the model to maximize profit using cost benefit matrix. Developed a web app using Flask, Jinja and CSS to predict the risk level of live data with a recall score of 96% and precision of 85%.

Due to confidentiality, the data is not made public. The python code is made available in the src file. The trained model is also available as a pickle file.

## Dataset
The dataset was provided by Galvanize. Dataset had 55 columns, a mix of numerical, categorical and text data. 

## EDA and Feature Engineering
One of the biggest problems we faced was data leakage. Although we felt some features could've been strong indicators, we realized that was due to data leakage. We were able to engineer certain features based on the type of text or the format of text. This later proved to be one of the biggest indicators of whether the model performed well or not. We also dummified and binarized whenever necessary.  

## Metrics and Loss Function
Since there was a cost associated with wrong predictions, we wanted our recall rate to be as high as possible. At the same time we wanted our profits to be high as well. Hence we built a custom loss function which when passed through the grid search, chooses the model with the highest profit. 

## Model Development
We initially built a logistic regression model with 3 features (chose based on our domain knowledge). Using this as our baseline model, following the CRISP-DM methodology, we developed multiple Random Forest, Logistic Regression, AdaBoost classifier, KNN models and compared with each other. We iterated over by adding new features to the model and chose the model which performed best. At the end, the random forest model was performing better than the rest.

## Web App Deployment
We created a live web app using Flask, Jinja and CSS which we then deployed on an Amazon EC2 instance. We connected with Galvanize's Heroku app which provided live data. We then stored the data on a Mongo Database and used a pickled model to predict the risk level on the air. The model was able to predict within a few seconds. A static version of the website is hosted [here]('frauddetection.praveenraman.com'). 

## Result and Inference
The final random forest model had a recall score of 96%, and precision of 85%. Through this project we understood how to use Flask and AWS effectively. We also had a dilemma while choosing the model, whether to prefer interpretability over performance. 

## Files in src and it's use

* collection_app.py - Collects live data from a Heroku App and stores it into a MongoDB
* model.py - Compares the performance of models and stores the best model in pickle format
* my_app.py - Loads pickled model and performs the predictions on live data and pushes it to the Web app using Flask
* predict.py - Predicts the fraud risk level based on the probability

## Rough timeline

* Day 1: Project scoping, Model building, and an intro to Web apps
* Day 2: Web app and deployment


## Credits
This project would not be possible without the efforts of my fellow teammates Rosina Norton, Yuwei Kelly Peng, Kenny Durell.

