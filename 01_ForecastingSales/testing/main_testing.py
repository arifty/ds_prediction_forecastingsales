import os
import pandas as pd
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta, date
import pickle
from itertools import product
from pandas import ExcelWriter


project = '01_ForecastingSales'
input_dir = os.path.join(project,'input_files')
staging_dir = os.path.join(project,'staging_files')
code_dir = os.path.join(project,'src')
temp_dir = os.path.join(project,'temp_files')
output_dir = os.path.join(project,'output_files')




temp_dir+"/"+model_name+"/"+str(style_code)+ "_model_pipeline.p




rfmodel_train_status = open(output_dir+"/rf_v4_model_train_status.txt", 'w')
for file in os.listdir(temp_dir+"/RandomForest"):
    if file.endswith("_model_pipeline.p"):
        print(file)
        rfmodel_train_status.write("%s\n" % file)
        
rfmodel_train_status.close()


test_results = pickle.load(open(output_dir+"/testresults_RandomForest_v4.p", "rb" ))
# write to Excel 
writer = pd.ExcelWriter(output_dir+"/modelscore_RandomForest_v4.xlsx")
test_results.sort_values(['Style_display_code']).to_excel(writer,'scores', index = False)
writer.save()

        


groupby_sum_cols = ['Style_display_code','SesnYrCd','week_of_season','New_Territory']
groupby_booking_cols = ['Style_display_code','SesnYrCd','New_Territory']
variable_cols = ['NetSlsUnts_WTD', 'OHInvUnts_WTD', 'bookings', 'low_obsv_flag']

outlier_threshold_upper = 0.1
outlier_threshold_lower = -0.01
trainset_threshold = 3

# import the function file
exec(open(os.path.join(code_dir, "00_functions.py")).read())


# scoring list
scoring_list = pd.read_csv(input_dir+"/cs_scoring_list.csv")
scoring_list.head(6)

# pos data
pos_data = pd.read_csv(input_dir+"/cs_pos_data.csv")
pos_data.head(6)

# product data
prod_info = pd.read_csv(input_dir+"/cs_prod_info.csv")
prod_info.head(6)

# bookings data
bookings = pd.read_csv(input_dir+"/cs_bookings.csv")
bookings.head(6)


# get variables list separately
ls_scoring = scoring_list['Style_display_code'].tolist()
ls_pos = pos_data['Style_display_code'].unique().tolist()
ls_prod_info = prod_info['Style_display_code'].unique().tolist()
ls_booking = bookings['Style_display_code'].unique().tolist()

ls_terrirory = bookings['New_Territory'].unique().tolist()
ls_season = bookings['SesnYrCd'].unique().tolist()

weeknr_list = list(range(1,14))
base_df = cartesian_lists_toDF (ls_scoring, ls_season, weeknr_list, ls_terrirory, 
                                groupby_sum_cols)


# Check the missing style_codes
booking_missing_style = list(set(ls_scoring) - set(ls_booking))

# 839 codes missing
pos_missing = list(set(ls_scoring) - set(ls_pos))
len(pos_missing)

# 648 codes missing
booking_missing = list(set(ls_scoring) - set(ls_booking))
len(booking_missing)

# booking vs prod_info
prod_booking_missing = list(set(booking_missing) - set(ls_prod_info))
len(prod_booking_missing)


# TESTING
bookings_total = bookings.groupby(['Style_display_code', 'SesnYrCd']).sum().reset_index()
bookings[bookings['Style_display_code']=='AAA0599'].sum().reset_index()




# ------------------
# Find Correlations
# ------------------
prod_info_filt = prod_info[prod_info['Style_display_code'] \
                           .isin(ls_pos+ls_scoring+booking_missing_style)]
prod_info_dedup = prod_info_filt[['Style_display_code','age_desc', 'gender_desc',
                                  'product_family','category']].drop_duplicates()

cat_features = ['age_desc', 'gender_desc', 'product_family','category' ]


# duration: runs for 13 mins because of corr() function
corr_data = calculate_correlations(prod_info_dedup, 'Style_display_code', cat_features, 
                                   pos_missing+booking_missing)



## TESTING
corr_data[corr_data['order']=='0'].isnull().sum()
prod_info_dedup.count()

# ----- POS_DATA ENRICH -------
# format the data before enriching correlated data
pos_missing_style = corr_data[(corr_data['order']=='0') & 
                                    (corr_data['Style_display_code'].isin(pos_missing))]

corr_pos_enriched = pos_missing_style.merge(pos_data \
                          .rename(columns={'Style_display_code': 'correlated_Style_display_code'}), 
                          how="inner", on='correlated_Style_display_code') \
                          .drop(['correlated_Style_display_code','order'], axis=1) \
                          .set_index('Style_display_code')

# enrich the pos_data with missing scoring style codes
pos_enriched = pos_data[pos_data['Style_display_code'].isin(ls_scoring)].set_index('Style_display_code') \
                .append(corr_pos_enriched)


# TESTING
style_code = ls_scoring[10]  #'ZQY1001'
df = pos_enriched_agg[pos_enriched_agg['Style_display_code']==style_code]
df.head()


# convert the date field to datetime type
pos_enriched['Activity_Date'] = pd.to_datetime(pos_enriched['Activity_Date'], 
                                               errors ='coerce')
# fetch the seasona nd corresponding dates
season_weeks = pos_enriched.reset_index()[['SesnYrCd','Activity_Date']].drop_duplicates() \
                .sort_values(by='Activity_Date')

# calculate the week of season column by ranking 13 weeks
season_weeks['week_of_season'] = season_weeks.groupby('SesnYrCd')['Activity_Date'] \
                                  .rank(ascending=True)
# merge the new column with the pos_enrich
pos_enriched = pos_enriched.reset_index().merge(season_weeks, on=['SesnYrCd','Activity_Date'], 
                                 how='inner')

# Aggregate the data  
pos_enriched_agg = pos_enriched.fillna(0).groupby(groupby_sum_cols).sum().reset_index()

  

## TESTING
ls_pos_enriched = corr_pos_enriched.index.unique().tolist()
len(list(set(pos_missing) - set(ls_pos_enriched)))
pos_enriched_agg['Style_display_code'].nunique()   #2588
pos_enriched_agg[pos_enriched_agg['Style_display_code'] == 'EKV7612']
pos_enriched_agg[(pos_enriched_agg['Style_display_code']=='AAA0599') &
                 (pos_enriched_agg['New_Territory']=='UNITED STATES')]
test_df = pos_enriched[(pos_enriched['New_Territory']=='UNITED STATES')][['SesnYrCd','Activity_Date']]
(test_df.groupby('SesnYrCd').min(),test_df.groupby('SesnYrCd').max())
test_df['SesnYrCd'].unique()
pos_enriched['SesnYrCd'].unique()


# ----- BOOKING ENRICH -------
# format the data before enriching correlated data
booking_missing_style = corr_data[(corr_data['order']=='0') & 
                                    (corr_data['Style_display_code'].isin(booking_missing))]

corr_booking_enriched = booking_missing_style.merge(bookings \
                          .rename(columns={'Style_display_code': 'correlated_Style_display_code'}), 
                          how="inner", on='correlated_Style_display_code') \
                          .drop(['correlated_Style_display_code','order'], axis=1) \
                          .set_index('Style_display_code')

# enrich the bookings data with missing scoring style codes
bookings_enriched = bookings[bookings['Style_display_code'].isin(ls_scoring)].set_index('Style_display_code') \
                      .append(corr_booking_enriched)


# find the not enriched booking data
ls_bookings_enriched = corr_booking_enriched.index.unique().tolist()
booking_not_enriched = list(set(booking_missing) - set(ls_bookings_enriched))

## TESTING
len(booking_not_enriched)

# ------- PAUSE -----------
# save the data
pickle.dump(bookings_enriched, open(staging_dir+ '/bookings_enriched_py3.p', 'wb'))
pickle.dump(pos_enriched, open(staging_dir+ '/pos_enriched_py3.p', 'wb'))

# read again from pickle
bookings_enriched = pickle.load(open(staging_dir+ '/bookings_enriched_py3.p', "rb" ) )
pos_enriched = pickle.load(open(staging_dir+ '/pos_enriched_py3.p', "rb" ) )
# ------- PAUSE -----------


# Re-enrich the still missing bookings data
bookings_agg_1 = bookings_enriched.reset_index() \
                  .fillna(0).groupby(groupby_sum_cols).sum().reset_index()
bookings_agg_2 = bookings_agg_1.groupby(groupby_mean_cols).mean().reset_index()

bookings_re_enriched = pd.concat([bookings_agg_2] * len(booking_not_enriched),
                                 keys=booking_not_enriched) \
                        .reset_index(level=1, drop=True) \
                        .rename_axis('Style_display_code').reset_index()


# TESTING
bookings[(bookings['Style_display_code']=='AAA0599') & (bookings['SesnYrCd']== 'SP2016')].head()


# aggregate bookings at Style code & Season level
bookings_enrich_agg = bookings_agg_1.append(bookings_re_enriched)


# TESTING
bookings_enrich_agg['Style_display_code'].nunique()  #2588
bookings['Style_display_code'].nunique()             #4827



# ------- PAUSE -----------
# save the data
pickle.dump(bookings_enrich_agg, open(staging_dir+ '/bookings_enriched_agg.p', 'wb'))
pickle.dump(pos_enriched_agg, open(staging_dir+ '/pos_enriched_agg.p', 'wb'))
pickle.dump(pos_enriched_agg, open(staging_dir+ '/pos_enriched_agg_v2.p', 'wb'))

# read again from pickle
bookings_enrich_agg = pickle.load(open(staging_dir+ '/bookings_enriched_agg.p', "rb" ) )
pos_enriched_agg = pickle.load(open(staging_dir+ '/pos_enriched_agg_v2.p', "rb" ) )
# ------- PAUSE -----------



# Prepare the features
# ---------------------
# merge bookings to pos data
pos_booking_data = pos_enriched_agg.merge(bookings_enrich_agg, how="left", on=groupby_booking_cols) \
                    .fillna(0)

  #.drop('OHInvUnts_WTD', axis=1)


# data quality fixes
# ------------------

# update the sales data, whenever it is more than the bookings value
# Cap it to the bookings value for those cases
pos_booking_data.loc[((pos_booking_data['bookings']>0) & 
                      (pos_booking_data['NetSlsUnts_WTD']>pos_booking_data['bookings'])), 'NetSlsUnts_WTD'] \
            =  pos_booking_data.loc[((pos_booking_data['bookings']>0) & 
                                     (pos_booking_data['NetSlsUnts_WTD']>pos_booking_data['bookings'])), 'bookings']


# flag too few samples as difficult to predict, n_samples should be 3 or more
low_observations = pos_booking_data.groupby('Style_display_code')['Style_display_code'].filter(lambda x: x.count() < 3)
pos_booking_data['low_obsv'] = low_observations

# transform the low observation column to 0 = if no low observation and 1 = if low observation
pos_booking_data['low_obsv_flag'] = np.where((pos_booking_data['low_obsv'].isnull()), 0, 1)

# Fill all missing data with mean for style codes with missing lines from base_df
pos_booking_totalenriched = base_df.merge(pos_booking_data,on=groupby_sum_cols, how='left').fillna(0)

# filter out SUMMER 2016 entries, which needs to be predicted later
pos_booking_enrich = pos_booking_totalenriched[pos_booking_totalenriched.SesnYrCd != 'SU2016']



#for key in ls_scoring:
#  pos_booking_enrich.loc[(pos_booking_enrich['Style_display_code']==key), 'tot_sales'] = \
#                pos_booking_enrich[pos_booking_enrich['Style_display_code']==key]['NetSlsUnts_WTD'].sum()


# ---------------------------------------------------------------------------------------
## COMMENTING-OUT this, because missing values are filled as 0 with above code
## Calculate average sales per style codes 
##mean_data = pd.DataFrame(pos_booking_data.groupby(['Style_display_code'])['NetSlsUnts_WTD'].mean()).to_dict()
#
## empty dataframe to store the enriched pos_booking data
#pos_booking_enrich = pd.DataFrame()
#
## Fill all missing data with mean for style codes with missing lines from base_df
#for key in ls_scoring:
#  # enrich the base_df with mean values
#  df_enriched = base_df[base_df['Style_display_code']==key].merge(
#                          pos_booking_data[pos_booking_data['Style_display_code']==key], 
#                          on=groupby_sum_cols, how='left')
#  
#  df_enriched[variable_cols] = df_enriched[variable_cols].fillna(df_enriched[variable_cols].mean())
#  #print("enriched for key:{}_Timestamp:{}".format(str(key), format(str(datetime.now()))))
#    
#  # get the appended pos data
#  pos_booking_enrich = pos_booking_enrich.append(df_enriched)
# ---------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------
## COMMENTING-OUT this, because missing rows are enriched with above step now
## ignore all low_obsv data, to avoid the models to fail
#pos_booking_data = pos_booking_data[pos_booking_data['low_obsv_flag']==0]
#
## get the final style code list to be forecasted
#final_style_list = pos_booking_data['Style_display_code'].drop_duplicates().tolist()
# ---------------------------------------------------------------------------------------



#for style_code in ls_scoring:
#  
#  # filter the data per style_codes
#  style_filtered = pos_booking_data[pos_booking_data['Style_display_code']==style_code]
#  # get the countries list per style_code
#  countries_list = style_filtered['New_Territory'].drop_duplicates().tolist()
#  
#  # loop thru each countries to find the outlier
#  for country in countries_list:
#    country_filtered = style_filtered[style_filtered['New_Territory']==country]
#    X_train_outlier = country_filtered['NetSlsUnts_WTD'].values.reshape(-1, 1)
#    
#    # train and predict outliers using isof
#    isof = IsolationForest(n_jobs=-1,n_estimators=500, max_samples=256, random_state=23)
#    isof.fit(X_train_outlier)
#    y_pred_outlier = isof.predict(X_train_outlier)
#
#    #concatenate outlier value with the original data
#    predicted_data = np.column_stack([country_filtered.values,y_pred_outlier])
#
#    # convert to a dataframe
#    with_outlier = pd.DataFrame(predicted_data, columns=column_list)
#
#    pos_with_outlier.append(with_outlier)
# 
#pos_with_outlier[pos_with_outlier['outlier_flag']==-1]
  
  
  
  
# split the season and year fields
pos_booking_enrich['season'] = pos_booking_enrich['SesnYrCd'].str[:2]
pos_booking_enrich['year_id'] = pos_booking_enrich['SesnYrCd'].str[2:]
pos_booking_enrich.drop(['SesnYrCd','low_obsv'], axis=1, inplace=True)

numerical_feat = ['OHInvUnts_WTD', 'bookings','year_id', 'week_of_season']
categorical_feat = ['New_Territory','season']


# ---------------
# Start Modeling
# ---------------
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, cross_validate
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# define the column names
outlier_colnames = list(pos_booking_enrich.columns) + ['outlier_flag', 'outlier_score']
pred_colnames = ['Style_display_code', 'actual_testdata', 'predicted_testdata', 'RMSE', 
                 'r2', 'feature_importances', 'model_name']

# empty dataframe
pos_with_outlier = pd.DataFrame(columns=outlier_colnames)
predict_results_appended = pd.DataFrame()

for style_code in ls_scoring:
  
  # filter the data per style_codes and encode categorical features
  model_data = pos_booking_enrich[pos_booking_enrich['Style_display_code']==style_code]    
  
  X_train_outlier = pd.get_dummies(model_data,columns=categorical_feat) \
                        .drop(['Style_display_code','OHInvUnts_WTD'], axis=1)

  # train and predict outliers using isof
  isof = IsolationForest(contamination=0.1, n_jobs=-1, n_estimators=50,
                         max_features=1.0,max_samples=10, random_state=None, behaviour="new")
  isof.fit(X_train_outlier)
  outlier_pred_value = isof.predict(X_train_outlier)
  
  # The Anomaly scores are calclated for each observation and stored in 'scores_pred'
  outlier_pred_score = isof.decision_function(X_train_outlier)

  #concatenate outlier value with the original data
  model_data['outlier_flag'] = outlier_pred_value
  model_data['outlier_score']= outlier_pred_score
  
  # append the outlier data to a dataframe
  pos_with_outlier = pos_with_outlier.append(model_data)
    
  # Get X, y without outliers
  y = pos_with_outlier[(pos_with_outlier['outlier_score']
              .between(outlier_threshold_lower, outlier_threshold_upper, inclusive=True))]['NetSlsUnts_WTD']
  X = pos_with_outlier[(pos_with_outlier['outlier_score']
              .between(outlier_threshold_lower, outlier_threshold_upper, inclusive=True))] \
      .drop(['Style_display_code','NetSlsUnts_WTD', 'low_obsv_flag','outlier_flag', 'outlier_score'], 1)

  feature_list = list(X.columns)
  
  # define the preprocessor for OneHotEncoding
  preprocessor = ColumnTransformer([("numerical", "passthrough", numerical_feat), 
                                    ("categorical", OneHotEncoder(sparse=False, handle_unknown="ignore"),
                                     categorical_feat)])
  
  
  # create ML pipeline
  model_name = 'RandomForest'
  model_pipe = Pipeline([("preprocessor", preprocessor),
             (model_name,RandomForestRegressor(
                 max_depth=10,
                 min_samples_split=.3,
                 n_estimators=15,
                 max_features="auto",
                 bootstrap=True,  
                 n_jobs=-1))])
  
#  model_pipe = Pipeline([("preprocessor", preprocessor),
#                       ("model",LinearRegression())])
    
      
  # ---------------------------------------------
  # LINEAR REGRESSION Model
  
  # get the train and test data splits
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
  
  # fit the model and predict
  model_pipe.fit(X_train,y_train)  # model fit step
  predictions = model_pipe.predict(X_test)
  

  # check the coefficient of the model
  #coeff_model = pd.DataFrame(model_pipe.named_steps["model"].coef_,X.columns,columns=['Coeff'])
  
  # Get numerical feature importances
  importances = list(model_pipe.named_steps[model_name].feature_importances_)

  # List of tuples with variable and importance
  feature_importances = [(feature, np.round(importance*100, 5)) for feature, importance in zip(feature_list, importances)]
  feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
  
  # check the RMSE
  RMSE = np.sqrt(metrics.mean_squared_error(y_test,predictions))
  print("RMSE:"+str(np.round(RMSE,2)))

  # check r2 score
  r2 = metrics.r2_score(y_test, predictions, multioutput='raw_values')[0]
  print("r2:"+str(np.round(r2, 2)))
  
  # create the dataframe of predicted results on test data
  predicted_results = pd.DataFrame([style_code, np.sum(y_test), np.sum(predictions),
                                RMSE, r2, feature_importances, model_name], pred_colnames)
  predict_results_appended = predict_results_appended.append(predicted_results.T)


  # save the model and predicted results
  pickle.dump(model_pipe, open(staging_dir+"/"+str(style_code)+ "_model_pipeline.p", 'wb'))
  

# save the outliers data for reference
pickle.dump(predict_results_appended, open(output_dir+"/predicted_results_RF_v2.p", 'wb'))
pickle.dump(pos_with_outlier, open(output_dir+ '/pos_data_withOutlier_v2.p', 'wb'))

output_dir = os.path.join(project,'output_files')

#  # ------- PAUSE -----------
#  # save the data
#  pickle.dump(pos_with_outlier, open(staging_dir+ '/pos_with_outlier.p', 'wb'))
#  pickle.dump(pos_with_outlier, open(staging_dir+ '/pos_with_outlier_2.p', 'wb'))

#  # read again from pickle
#  pos_with_outlier = pickle.load(open(staging_dir+ '/pos_with_outlier_2.p', "rb" ) )
#  # ------- PAUSE -----------
  
  
  
  # RESULTS
  style_code   RMSE
  'NCN8564'   94242.58
  
#  TO CHECK:
#    6	SP	2016	192168	445139.0	187461.0
  
  'YHO1335'    1074.13   1114.8
  
# TO CHECK:
#2	HO	2014	520	3706.0	154.0


predict_results_appended = pd.DataFrame()

for style_code in ls_scoring:
  data = pickle.load(open(staging_dir+"/"+str(style_code)+ "_predicted_results.p", "rb" ))
  predict_results_appended = predict_results_appended.append(data.T)
  
pickle.dump(predict_results_appended, open(staging_dir+ '/predicted_results_lm.p', 'wb'))

predict_results_appended
predict_results_appended.describe()
predict_results_appended['r2'].mean()
predict_results_appended.min()
predict_results_appended.max()

  
  
#----------------------------------------------------------------------------------------
# Experimentation
#----------------------------------------------------------------------------------------


# scoring list
scoring_list = pd.read_csv("./01_ForecastingSales/input_files/cs_scoring_list.csv")
scoring_list.head(6)

scoring_list.info()

scoring_list['Style_display_code'].nunique()  --2588
#scoring_list['Style_display_code'].value_counts()

ls_scoring = scoring_list['Style_display_code'].tolist()


# pos data
pos_data = pd.read_csv("./01_ForecastingSales/input_files/cs_pos_data.csv")
pos_data.head(6)

pos_data.info()


pos_data['Retailer'].nunique()
pos_data['New_Territory'].nunique()
pos_data['New_Territory'].value_counts()


# 839 codes missing
ls_pos = pos_data['Style_display_code'].unique().tolist()
pos_missing = list(set(ls_scoring) - set(ls_pos))
len(pos_missing)

pos_missing[0]
# 'XEH8012'



# product data
prod_info = pd.read_csv("./01_ForecastingSales/input_files/cs_prod_info.csv")
prod_info.head(6)

prod_info['Style_display_code'].count()  --62917
prod_info['Style_display_code'].nunique() --5573
prod_info['design_code'].nunique()  -- 34961

prod_info[prod_info['Style_display_code'].isnull()]


# checking if same desing_code or technology are assigned to multiple style_codes
# answer is YES
prod_info[['Style_display_code','technology']].drop_duplicates()['technology'].value_counts().head(6)

newdf = prod_info[['Style_display_code','technology']].drop_duplicates()
newdf[newdf['technology'] == 'PATENT 456']



dfmerge = scoring_list.merge(prod_info, on='Style_display_code', how='left')

dfmerge[dfmerge.iloc[:,-1].isnull()][scoring_list.columns.tolist()]


ls_prod = prod_info['Style_display_code'].unique().tolist()


# difference is # ['EKV7612']
product_missing = list(set(ls_scoring) - set(ls_prod))
product_missing




# FIND THE Correlated style codes

corr_finaldata = pd.DataFrame(columns=['Style_display_code','matched_display_code'])
related_cols = list(prod_info.drop(['Style_display_code','design_code','product_family',
                                        'technology'],axis=1))

for style_code in pos_missing:
  
  Missing_style = prod_info[prod_info['Style_display_code']==style_code]
  Other_styles = prod_info[prod_info['Style_display_code']!=style_code]

  corr_styles = Missing_style.merge(Other_styles, on=related_cols, how="inner")
  corr_styles.rename(columns={'Style_display_code_x': 'Style_display_code', 
                              'Style_display_code_y': 'matched_display_code'}, inplace=True)
  corr_finaldata = corr_finaldata.append(corr_styles[['Style_display_code', 'matched_display_code']])


corr_sorted = corr_finaldata.groupby('Style_display_code')['matched_display_code'] \
  .count().reset_index().sort_values(by=['matched_display_code'], ascending=False)

# check if there is null
corr_sorted[corr_sorted['matched_display_code'].isnull()]

corr_sorted['Style_display_code'].nunique()

# ALL 838 style codes have more than 1 matching
corr_sorted[corr_sorted['matched_display_code']>1]['Style_display_code'].count()

# bookings data

bookings = pd.read_csv("./01_Usecase_ForecastingSales/input_files/cs_bookings.csv")
bookings.head(6)

# 648 codes missing
ls_booking = bookings['Style_display_code'].unique().tolist()
booking_missing = list(set(ls_scoring) - set(ls_booking))
len(booking_missing)


calculate_correlations(prod_info, 'Style_display_code')


# FINDING CORRELATION

prod_info_indexed = prod_info.set_index('Style_display_code')
prod_info_indexed.drop(['design_code','technology'], axis=1, inplace=True)	

# they have a third class "unknown" we'll process them as non binary categorical
num_features = []
cat_features = ['age_desc','gender_desc','color_family','category','product_family']

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder



preprocessor = ColumnTransformer([("numerical", "passthrough", num_features), 
                                  ("categorical", OneHotEncoder(sparse=False, handle_unknown="ignore"),
                                   cat_features)])

transformed_data = preprocessor.fit(prod_info_indexed)



final_data = pd.get_dummies(prod_info_indexed,columns=cat_features,drop_first=True)

# checking corr within features => THIS IS NOT I WANTED
final_data.corr().values
sns.heatmap(final_data.corr(),annot=True)


final_data.corrwith(final_data.iloc[0], axis=1)
sns.heatmap(final_data.corrwith(final_data, axis=1),annot=True)


#pd.Series(corr2_coeff_rowwise2(final_data.values,final_data.values))


pd.DataFrame(final_data.transpose()).corr()


for style_code in ls_prod:
  
  style_code = 'SZC1066'
  
  df1 = final_data[final_data.index==style_code]
  df2 = final_data[final_data.index!=style_code]
  
  corr_results = df1.corrwith(df2)


corrs = df.pivot('Category_A','Category_B').T.corr().stack()

#Category_A  Category_A
#Alan        Alan          1.000000
#            Bob          -0.986552
#Bob         Alan         -0.986552
#            Bob           1.000000
corrs.index.names = 'A','B'
corrs.reset_index()
#      A     B         0
#0  Alan  Alan  1.000000
#1  Alan   Bob -0.986552
#2   Bob  Alan -0.986552
#3   Bob   Bob  1.000000


## Finding the correlation in product data

# FIND THE Correlated style codes
# -------------------------------

#corr_finaldata = pd.DataFrame(columns=['Style_display_code','matched_display_code'])
#related_cols = list(prod_info.drop(['Style_display_code','design_code','product_family',
#                                        'technology'],axis=1))
#
#for style_code in pos_missing:
#  
#  Missing_style = prod_info[prod_info['Style_display_code']==style_code]
#  Other_styles = prod_info[prod_info['Style_display_code']!=style_code]
#
#  corr_styles = Missing_style.merge(Other_styles, on=related_cols, how="inner")
#  corr_styles.rename(columns={'Style_display_code_x': 'Style_display_code', 
#                              'Style_display_code_y': 'matched_display_code'}, inplace=True)
#  corr_finaldata = corr_finaldata.append(corr_styles[['Style_display_code', 'matched_display_code']])
#
#
#corr_sorted = corr_finaldata.groupby('Style_display_code')['matched_display_code'] \
#  .count().reset_index().sort_values(by=['matched_display_code'], ascending=False)
#
## check if there is null
#corr_sorted[corr_sorted['matched_display_code'].isnull()]
#
#corr_sorted['Style_display_code'].nunique()
#
#
## get the count of POS data available for each style_codes
#pos_style_datacount = pos_data.groupby('Style_display_code')['Activity_Date'].count().reset_index()






def corr2_coeff_rowwise2(A,B):
    A_mA = A - A.mean(1)[:,None]
    B_mB = B - B.mean(1)[:,None]
    ssA = np.einsum('ij,ij->i',A_mA,A_mA)
    ssB = np.einsum('ij,ij->i',B_mB,B_mB)
    return np.einsum('ij,ij->i',A_mA,B_mB)/np.sqrt(ssA*ssB)



