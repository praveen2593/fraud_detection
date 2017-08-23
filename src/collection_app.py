import os
from flask import Flask, render_template
import requests
import json
import predict as pr
import pandas as pd
import cPickle as pickle
import threading
from pymongo import MongoClient
import time
# initialization
#app = Flask(__name__)
#app.config.update(
#    DEBUG = True,
#)


# collecting page
#@app.route('/')
#def homepage():
   # threading.Timer(60.0,homepage).start()
   # predict_event_fraud()
   # return 'record collected'


def get_data():
    req = requests.get('http://galvanize-case-study-on-fraud.herokuapp.com/data_point').content
    data = json.loads(req)
    cols = ['org_twitter','body_length','user_age','sale_duration2','delivery_method',
            'org_facebook','previous_payouts','has_analytics','venue_state',
            'org_desc', 'name']
    df_data = []
    for c in cols:
       df_data.append(data[c])
    df = pd.DataFrame([df_data],columns=cols)
    return df, data

def predict_event_fraud():
    df, data = get_data()
    X = pr.feature_engineering(df)
    predict_proba = model.predict_proba(X)
    pr.send_to_db(data, predict_proba[:,1][0], collection)


if __name__ == "__main__":

    with open('best_model_0816_2.pkl') as f:
        model = pickle.load(f)

    client = MongoClient()
    db = client.new_db
    collection = db.new_test
    #app.run(host='0.0.0.0',port=8000, debug=True)
    #threading.Timer(5.0,predict_event_fraud).start()
    while True:
	time.sleep(10)
	predict_event_fraud()
    # port = int(os.environ.get("PORT", 5002))
    # app.run(host='0.0.0.0', port=port)
