'''
3_train.py

@version: 1.0

Created on December, 13, 2014

@author: Glenn Kroegel
@contact: glenn.kroegel@gmail.com
@summary: Train model for use in websocket

'''

# Third party imports
import numpy as np
import pandas as pd
from sklearn import linear_model, cross_validation, grid_search
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.neural_network import BernoulliRBM
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import Binarizer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn import tree
from inspect import getmembers
import cPickle as pickle
import copy
import scipy
from Functions import *


def main():

  # MODEL INITIALIZATION

  clf = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', min_samples_leaf = 250, n_jobs = -1, random_state = 62, class_weight = 'subsample') # Declare random forest object with hyperparams

  # MODEL CALCULATION
  X_train = joblib.load('dataset_x_train.joblib')
  Y_train = joblib.load('dataset_y_train.joblib')

  X_train = X_train.astype(np.float64)
  Y_train = Y_train.astype(np.float64)

  assert not np.any(np.isnan(X_train) | np.isinf(X_train))

  clf.fit(X_train, Y_train) # Train model based on train split determined from 2_preprocess.py

  X_test = joblib.load('dataset_x_test.joblib')
  Y_test = joblib.load('dataset_y_test.joblib')

  clf_px = clf.predict_proba(X_test)

  # STATISTICS

  print "R2 Score: {0}\n".format(clf.score(X_test, Y_test))

  i = 0
  ls_x = joblib.load('feature_vector.joblib')

  for feature in ls_x.columns:
    print "{0}: {1}".format(feature, clf.feature_importances_[i])
    i = i + 1

  #print clf.feature_importances_

  # EXPORT MODEL

  print("Exporting Model...")
  na_symbol = joblib.load('sym.joblib')
  with open(na_symbol+'.pkl', 'wb') as model:
    pickle.dump(clf, model)

  # EXPORT TEST SET RESULTS
  print("Exporting Results...")
  df_export = pd.DataFrame(zip(Y_test, clf_px[:,1]))
  exportData(df_export,'Test Set.csv')


if __name__ == "__main__":

  print("Studying Data...\n")

  try:

    main()

  except KeyboardInterrupt:

    print("Interupted...Exiting\n")





