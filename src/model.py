import numpy as np
import pandas as pd
from pymongo import MongoClient
import cPickle as pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools.tools import add_constant
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, recall_score, make_scorer, precision_score, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier



def get_data():
    client = MongoClient()
    db = client.fraud_detection
    collection = db.fraud
    df = pd.DataFrame(list(collection.find()))
    return df


def feature_engineering(df):
    """Extracting the features from the original dataframe created from the
    json data. EDA analysis and cross validation have informed which feature
    are required.

    Parameters
    ----------
    df: Pandas dataframe from json data

    Returns
    -------
    training and testing data:
    - X_train: Pandas dataframe
    - X_test: Pandas dataframe
    - y_train: Pandas dataframe
    - y_train: Pandas dataframe
    """
    # original features needed for feature engineering
    df = df[['org_twitter','body_length','user_age','sale_duration2','delivery_method',
             'org_facebook','acct_type','previous_payouts','has_analytics','venue_state',
             'org_desc', 'name']]
    # cols: columns in final design matrix, the included features are used as id
    cols = ['body_length','user_age','sale_duration2','has_analytics']

    # creating the new features
    df['facebook_presence'] = df.org_facebook.apply(lambda x:1 if x>5 else 0)
    df['twitter_presence'] = df.org_twitter.apply(lambda x:1 if x>5 else 0)
    df['have_previous_payouts'] = df['previous_payouts'].apply(lambda x: 1 if len(x) != 0 else 0)
    df['highly_suspect_state'] = df['venue_state'].apply(lambda x: 1 if x in ['MT', 'Mt', 'AK', 'FL', 'NEW SOUTH WALES', 'Florida'] else 0)
    df['cap_name'] = df['name'].apply(lambda x: 1 if x.isupper() == True else 0)

    ls = []
    for i in df.org_desc:
        ls.append(len(i))
    df['has_org_desc'] = np.array(ls)

    # adding the new feature names to the final column list
    cols.append('facebook_presence')
    cols.append('twitter_presence')
    cols.append('have_previous_payouts')
    cols.append('highly_suspect_state')
    cols.append('has_org_desc')
    cols.append('cap_name')

    # delivery methods is categorical so dummifying these variables
    delivery_methods = df['delivery_method'].unique()
    for d in delivery_methods[:-1]:
        col_name = 'delivery_'+str(d)
        cols.append(col_name)
        df[col_name] = df['delivery_method'].apply(lambda x: 1 if x == d else 0)

    print 'columns included: {}'.format(cols)

    # creating the target feature column
    df['fraud'] = df['acct_type'].apply(lambda x: True  if 'fraud' in str(x) else False)

    # creating train and test sets
    X_train, X_test, y_train, y_test = train_test_split(df[cols], df['fraud'],random_state=1)

    return X_train, X_test, y_train, y_test

def cost_function(labels, predicted_probs, threshold=.5):
    """Calculate the average profit given the confusion matrix resutls
    Parameters
    ----------
    labels: ndarray - 1D
    predicted_probs : ndarray - 1D
    threshold: float
    Returns
    -------
    profit: float
    """
    # define cost_benefit matrix
    cost_benefit = np.array([[2000,-200],[0,0]])
    predicted_probs = np.array(predicted_probs[:,-1])
    # initialize empty numpy array
    predicted_labels = np.array([0] * len(predicted_probs))
    # set predicted label
    predicted_labels[predicted_probs >= threshold] = 1
    # get confusion matrix
    cm = standard_confusion_matrix(labels, predicted_labels)
    # calculate profit
    profit = (cm * cost_benefit).sum() * 1. / len(labels)
    return profit

def standard_confusion_matrix(y_true, y_pred):
    """Make confusion matrix with format:
                  -----------
                  | TP | FP |
                  -----------
                  | FN | TN |
                  -----------
    Parameters
    ----------
    y_true : ndarray - 1D
    y_pred : ndarray - 1D
    Returns
    -------
    ndarray - 2D
    """
    [[tn, fp], [fn, tp]] = confusion_matrix(y_true, y_pred)
    return np.array([[tp, fp], [fn, tn]])


def get_best_estimator_threshold(models, X_train, y_train):
    """
    Given a dictionary of models and their parameter grid,
    from the best estimator.
    Parameters
    ----------
    models : dictionary where key = string for model name,
             value = list(model instance,param_grid)
    X_train : Pandas dataframe
    y_train : Pandas dataframe
    Returns
    ----------
    best_estimator : fitted sklearn model
    best_score : profit score for that model
    clfs : list of the best model for each model type
    """
    thresholds = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
    # thresholds = [.1, .2]
    clfs_with_t = []
    for t in thresholds:
        print 'grid search for threshold {}'.format(t)
        clfs = grid_search(models, X_train, y_train, t)
        result = [t,clfs]
        clfs_with_t.append(result)

    max_score = 0
    final_threshold = None
    best_estimator = None

    for combo in clfs_with_t:
        threshold = combo[0]
        for clf in combo[1]:
            if clf.best_score_ > max_score:
                max_score = clf.best_score_
                final_threshold = threshold
                best_estimator = clf.best_estimator_
    return best_estimator, max_score, final_threshold, clfs

def grid_search(models, X_train, y_train, threshold):
    """
    Given a dictionary of models and their parameter grid,
    from the best estimator.
    Parameters
    ----------
    models : dictionary where key = string for model name,
             value = list(model instance,param_grid)
    X_train : Pandas dataframe
    y_train : Pandas dataframe
    Returns
    ----------
    clfs : list of the best model for each model type
    """
    clfs =[]
    scorer = make_scorer(cost_function, greater_is_better=True, needs_proba=True, threshold=threshold)

    for key, value in models.items():
        print 'grid searching for {}'.format(key)
        model = GridSearchCV(value[0], cv=5,param_grid=value[1], scoring = scorer)
        clf = model.fit(X_train, y_train)
        clfs.append(clf)
    return clfs

def write_pickle(filename, model):
    """
    Write the final model to a pickle file
    Parameters
    ----------
    filename : String
    model : sklearn model instance
    Returns
    ----------
    Nothing
    """
    with open(filename, 'w') as f:
        pickle.dump(model, f)
    pass

def our_prediction(predicted_proba, threshold):
    pred_labels = [True if x >= threshold else False for x in predicted_proba]
    return np.asarray(pred_labels)

if __name__ == '__main__':
    print 'getting data...'
    df = get_data()

    print 'creating features...'
    X_train, X_test, y_train, y_test = feature_engineering(df)

    DTC = DecisionTreeClassifier(random_state = 11)

    models = {'lr' : [LogisticRegression(),{'n_jobs':[1,-1]}],
            'rfc' : [RandomForestClassifier(n_jobs=-1,random_state=3),{'n_estimators':[30,40,60,80,120],'max_depth':[20,30,40,60], 'max_features' : [2,4,'sqrt','log2']}],
            'knn' : [KNeighborsClassifier(n_jobs=-1),{'n_neighbors': [10,20,30,40],'weights':['uniform','distance']}],
            'ada': [AdaBoostClassifier(base_estimator = DTC),{'base_estimator__max_depth': [20,30,40],'base_estimator__max_features':['sqrt','log2',2,4],'n_estimators' : [20,50,70,100,120], 'learning_rate': [0.00001,0.0001,0.001]}]}

    print 'finding best estimator...'
    best_estimator, best_score, final_threshold, clfs = get_best_estimator_threshold(models,X_train,y_train)

    print 'best estimator: {} \nwith best score: {} \nand threshold {}'.format(best_estimator, best_score, final_threshold)
    print '\n'

    pred_proba = best_estimator.predict_proba(X_test)[:,-1]
    y_pred = our_prediction(pred_proba, final_threshold)

    print 'accuracy {}'.format(accuracy_score(y_test, y_pred))
    print 'f1_score {}'.format(f1_score(y_test, y_pred))
    print 'precision {}'.format(precision_score(y_test, y_pred))
    print 'recall {}'.format(recall_score(y_test, y_pred))

    print 'pickle writing...'
    write_pickle('best_model_0816_2.pkl', best_estimator)

    # This may not work but the idea is to print the threshold
    with open('threshold.txt', 'w') as f:
        to_print = "{threshold: {}}".format(final_threshold)
        f.write(to_print)
