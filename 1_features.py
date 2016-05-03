'''
1_features.py

@version: 1.0

@author: Glenn Kroegel
@contact: glenn.kroegel@gmail.com
@summary: Calculates features on historical data for model training. Features are in various subcategories: candle/volume/volatility/momentum

'''

# Third party imports
import datetime as dt
import numpy as np
import pandas as pd
import sys
import math
import openpyxl
import xlrd
import xlwt
import time
import urllib
import copy
from Technical import *
from Functions import *
import talib as ta
from talib import abstract
from sklearn.externals import joblib


def candleDeciders(df_x):

  df_candles = copy.deepcopy(df_x)

  # CALCULATION REMOVED

  ls_inputs = ['OPEN','HIGH','LOW','CLOSE','VOLUME']
  cols = [col for col in df_candles.columns if col not in ls_inputs]
  df_candles = df_candles[cols]

  return df_candles

def volumeDeciders(df_x):

  df_volume = copy.deepcopy(df_x)

  # CALCULATION REMOVED
  
  ls_inputs = ['OPEN','HIGH','LOW','CLOSE','VOLUME']
  cols = [col for col in df_volume.columns if col not in ls_inputs]
  df_volume = df_volume[cols]

  return df_volume

def volatilityDeciders(df_x):

  df_volatility = copy.deepcopy(df_x)

  # CALCULATION REMOVED

  ls_inputs = ['OPEN','HIGH','LOW','CLOSE','VOLUME']
  cols = [col for col in df_volatility.columns if col not in ls_inputs]
  df_volatility = df_volatility[cols]

  return df_volatility

def trendDeciders(df_x):

  df_trend = copy.deepcopy(df_x)

  # CALCULATION REMOVED

  ls_inputs = ['OPEN','HIGH','LOW','CLOSE','VOLUME']
  cols = [col for col in df_trend.columns if col not in ls_inputs]
  df_trend = df_trend[cols]

  return df_trend

def indicatorDeciders(df_x):

  df_indicators = copy.deepcopy(df_x)

  # CALCULATION REMOVED

  ls_inputs = ['OPEN','HIGH','LOW','CLOSE','VOLUME']
  cols = [col for col in df_indicators.columns if col not in ls_inputs]
  df_indicators = df_indicators[cols]

  return df_indicators

def getTreeFeatures(df_x):

  df_x = copy.deepcopy(df_x)
  df_pricedata = df_x

  df_candles = candleDeciders(df_x)
  df_volume = volumeDeciders(df_x)
  df_volatility = volatilityDeciders(df_x)
  df_trend = trendDeciders(df_x)
  df_indicators = indicatorDeciders(df_x)

  df_x = pd.concat([df_x, df_candles, df_volume, df_volatility, df_trend, df_indicators], axis = 1) # Combine features into one dataframe

  
  # FILTER (Data Bias)

  df_y = getBinaryClassification(df_pricedata) # Function to classify data
  df_y = df_y[df_y != 2]
  df_x = df_x.ix[df_y.index]
  df_x = df_x.replace([np.inf, -np.inf], np.nan)
  df_x = df_x.dropna()
  df_y = df_y.ix[df_x.index]

  # FILTER (Trading Hours)

  try:
    df_x.index = pd.to_datetime(df_x.index, format = "%d/%m/%Y %H:%M")
  except:
    df_x.index = pd.to_datetime(df_x.index, format = "%Y-%m-%d %H:%M:%S")
  
  df_y.index = df_x.index
  df_x = df_x.between_time(dt.time(9,00), dt.time(18,00))
  df_y = df_y.ix[df_x.index]

  # FILTER (Weekends)

  df_x = df_x.ix[df_x.index.weekday < 5]
  df_y = df_y.ix[df_x.index]

  print df_x.shape
  print df_y.shape
  df_x.index.name = 'DATETIME'
  df_y.to_csv('y.csv', index_label = 'TRUE')

  df_x = np.round(df_x, decimals = 8)

  return df_x

########################################################################################################################

#################################################################################################################


def main():
  
	# Read user input
  na_symbol = sys.argv[1]

	# Get data
  df_data = getData(na_symbol)
  df_features = getTreeFeatures(df_data)

  # Export Feature Data
  print("Exporting...")

  outfile = na_symbol+' Features'+'.csv'
  exportData(df_features, outfile)

if __name__ == "__main__":

  print("Calculating Data Features...")

  try:

    main()

  except KeyboardInterrupt:

    print('Interupted...Exiting...')





