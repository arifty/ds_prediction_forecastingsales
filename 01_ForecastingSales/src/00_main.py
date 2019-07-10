# ---------------------------------------------------------------------------------------
# Author              : Arif Thayal
# Project name        : 01_ForecastingSales
# Purpose             : Main python code to execute the Data Science flow
# Last modified by    : Arif Thayal
# Last modified date  : 01/01/2019
# ---------------------------------------------------------------------------------------

# import the libraries
import os
import pandas as pd
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta, date
import pickle
from itertools import product
from pandas import ExcelWriter

# define the directory variables
project = '01_ForecastingSales'
input_dir = os.path.join(project,'input_files')
temp_dir = os.path.join(project,'temp_files')
output_dir = os.path.join(project,'output_files')
code_dir = os.path.join(project,'src')

# define few common lists to be used
groupby_sum_cols = ['Style_display_code','SesnYrCd','week_of_season','New_Territory']
groupby_booking_cols = ['Style_display_code','SesnYrCd','New_Territory']
variable_cols = ['NetSlsUnts_WTD', 'OHInvUnts_WTD', 'bookings', 'low_obsv_flag']

# define features in a list
numerical_feat = ['bookings','year_id', 'week_of_season']
categorical_feat = ['New_Territory','season']

# input the outlier score thresholds
outlier_threshold_upper = 0.1
outlier_threshold_lower = -0.01

# define the ML algorithm to be used
model_name = 'XGBRegression'
# model_name = 'RandomForest' or 'LinearRegression' or 'SVMRegression' or 'XGBRegression'


# load the commonly used function file
exec(open(os.path.join(code_dir, "fn_01_common_functions.py")).read())

# execute step-by-step modules of this project
exec(open(os.path.join(code_dir, "mod_01_data_discovery.py")).read())
exec(open(os.path.join(code_dir, "mod_02_data_enrich.py")).read())
exec(open(os.path.join(code_dir, "mod_03_feature_selection.py")).read())
exec(open(os.path.join(code_dir, "mod_04_model_traintest.py")).read())
exec(open(os.path.join(code_dir, "mod_05_model_predict.py")).read())

# ---------------------------------------------------------------------------------------


