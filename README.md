# Fraud Detection Case Study

Built machine learning model to predict fraud events based on event details submitted. Compared the performance of Logistic Regression, Random Forest and AdaBoost Classifier Algorithms by fine tuning the model to maximize profit using cost benefit matrix. Developed a web app using Flask, Jinja and CSS to predict the risk level of live data with a recall score of 96% and precision of 85%.

Due to confidentiality, the data is not made public. The python code is made available in the src file. The trained model is also available as a pickle file.


## Files in src and it's use

* collection_app.py - Collects live data from a Heroku App and stores it into a MongoDB
* model.py - Compares the performance of models and stores the best model in pickle format
* my_app.py - Loads pickled model and performs the predictions on live data and pushes it to the Web app using Flask
* predict.py - Predicts the fraud risk level based on the probability

## Rough timeline 

* Day 1: Project scoping, Model building, and an intro to Web apps
* Day 2: Web app and deployment


## Credits
This project would not be possible without the efforts of my fellow teammaates Rosina Norton, Yuwei Kelly Peng, Kenny Durell.

