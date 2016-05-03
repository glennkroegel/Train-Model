'''
2_preprocess.py

@version: 1.0

Created on May, 13, 2015

@author: Glenn Kroegel
@contact: glenn.kroegel@gmail.com
@summary: Randomly split data from 1_features.py for model training

'''

# Third party imports
import numpy as np
import pandas as pd
import sys
from sklearn import linear_model, cross_validation, grid_search
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.neural_network import BernoulliRBM
import copy
import scipy
from Functions import *

def getFeatureVector(df_data, export_type):

  df_data = copy.deepcopy(df_data)

  del df_data['OPEN']
  del df_data['HIGH']
  del df_data['LOW']
  del df_data['CLOSE']
  del df_data['VOLUME']
  del df_data['DATETIME']

  if export_type == 'DataFrame':
    _X = df_data
    return _X
  else:
    _X = df_data.as_matrix()
    return _X


def main():
  
	# INPUT
  na_symbol = sys.argv[1]

	# IMPORT DATA
  df_data = getFeatureData(na_symbol)
  print df_data.shape

  # DATA SORTING
  df_x = getFeatureVector(df_data, export_type = 'DataFrame')
  ls_x = getFeatureVector(df_data, export_type = 'Array')
  ls_y = loadBinaryClassification()
  print ls_x.shape
  print ls_y.shape

  X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(ls_x, ls_y, train_size = 0.6, random_state = 42) # Split feature vector randomly into train/test segments

  # EXPORT DATA SPLITS FOR TRAINING

  joblib.dump(na_symbol, 'sym.joblib')
  joblib.dump(df_x, 'feature_vector.joblib')
  joblib.dump(X_train, 'dataset_x_train.joblib')
  joblib.dump(Y_train, 'dataset_y_train.joblib')
  joblib.dump(X_test, 'dataset_x_test.joblib')
  joblib.dump(Y_test, 'dataset_y_test.joblib')


if __name__ == "__main__":

  print("Preparing data...\n")

  try:

    main()

    print("Data succesfully formatted. Train Model")

  except KeyboardInterrupt:

    print("Interupted...Exiting\n")





