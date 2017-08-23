import numpy as np
import os
import pandas as pd
from pymongo import MongoClient
import cPickle as pickle
import json

def get_data(test_script_examples):
    """Load raw data from a file and return training data and responses.

    Parameters
    ----------
    filename: The path to a json/csv file containing the raw text data and response.

    Returns
    -------
    X:
    y:
    """
    ext = os.path.splitext(test_script_examples)[-1]
    if ext == '.csv':
        df = pd.read_csv(test_script_examples)
    elif ext == '.json':
        df = pd.read_json(test_script_examples)

    with open(test_script_examples) as f:
        d = json.load(f)

    return df, d

def feature_engineering(df):
    # original features needed for feature engineering
    df = df[['org_twitter','body_length','user_age','sale_duration2','delivery_method',
            'org_facebook','previous_payouts','has_analytics','venue_state',
            'org_desc', 'name']]
    # cols: columns in final design matrix, the included features are used as id
    cols = ['body_length','user_age','sale_duration2','has_analytics']

    df['facebook_presence'] = df.org_facebook.apply(lambda x:1 if x>5 else 0)
    df['twitter_presence'] = df.org_twitter.apply(lambda x:1 if x>5 else 0)
    df['have_previous_payouts'] = df['previous_payouts'].apply(lambda x: 1 if len(x) != 0 else 0)
    df['highly_suspect_state'] = df['venue_state'].apply(lambda x: 1 if x in ['MT', 'Mt', 'AK', 'FL', 'NEW SOUTH WALES', 'Florida'] else 0)
    df['cap_name'] = df['name'].apply(lambda x: 1 if x.isupper() == True else 0)

    ls = []
    for i in df.org_desc:
        ls.append(len(i))
    df['has_org_desc'] = np.array(ls)

    cols.extend(['facebook_presence', 'twitter_presence', 'have_previous_payouts', \
    'highly_suspect_state', 'has_org_desc', 'cap_name'])
    # cols.extend(['facebook_presence', 'twitter_presence', 'have_previous_payouts', \
    # 'highly_suspect_state', 'has_org_desc'])

    df['delivery_3.0'] = df['delivery_method'].apply(lambda x: 1 if x == 3.0 else 0)
    df['delivery_1.0'] = df['delivery_method'].apply(lambda x: 1 if x == 1.0 else 0)
    df['delivery_0.0'] = df['delivery_method'].apply(lambda x: 1 if x == 0.0 else 0)
    cols.extend(['delivery_3.0', 'delivery_1.0', 'delivery_0.0'])

    return df[cols]


def predict(df_feature_engineered, model):
    # with open('best_model.pkl') as f:
    #     model = pickle.load(f)
    predict_proba = model.predict_proba(df_feature_engineered)
    print "The probability of this event being fraud is ", str(predict_proba[:,1][0])
    return predict_proba[:,1][0]

def send_to_db(data, probability, collection):
    data['probability'] = probability
    # client = MongoClient()
    # db = client.fraud_detection_db
    # collection = db.fraud_detection_probabilities
    # json_data = json.dumps(data_probability)
    # print json_data
    # print data
    collection.insert(data)


if __name__ == '__main__':
    data, raw_file = get_data('files/example.json')
    df_feature_engineered = feature_engineering(data)
    probability = predict(df_feature_engineered)
    send_to_db(raw_file,probability)
