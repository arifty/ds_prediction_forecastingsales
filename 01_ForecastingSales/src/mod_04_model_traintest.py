# load the functions for Outlier detection and Regression model
exec(open(os.path.join(code_dir, "fn_02_outlier_detection.py")).read())
exec(open(os.path.join(code_dir, "fn_03_regression_algo.py")).read())
exec(open(os.path.join(code_dir, "fn_04_gridsearch_validate.py")).read())



# Train & Test the model
# ----------------------

# filter out SUMMER 2016 entries, which needs to be predicted later
pos_booking_enrich = pos_booking_totalenriched.query("season+year_id != 'SU2016'")


# get the list of style codes to predict, which is only available in ls_scoring 
pos_stylecodes = pos_booking_enrich['Style_display_code'].unique().tolist()
ls_stylecodes = [elem for elem in pos_stylecodes if elem in ls_scoring]

# call the regression model
outlier_flag = False
predict_results_appended, pos_with_outlier = model_regression_train(pos_booking_enrich, 
                          ls_stylecodes,numerical_feat, categorical_feat, model_name, outlier_flag)



# save the outliers data for reference
pickle.dump(predict_results_appended, open(output_dir+"/testresults_"+model_name+".p", 'wb'))
pickle.dump(pos_with_outlier, open(output_dir+ "/posdata_"+model_name+".p", 'wb'))



# Validate the results
# --------------------
#data_lm = pickle.load(open(output_dir+ "/testresults_"+model_name+".p", "rb" ))
#
#data_lm
#data_lm.describe()
#data_lm[['RMSE', 'r2']].mean()
#data_lm.min()
#data_lm.max()
#
#df = data_lm[data_lm['r2']>0]
#df[['RMSE', 'r2']].mean()
#df.min()
#df.max()
##
### LM
## r2 mean = 0.75
##data_lm['Style_display_code'].nunique() #2588
##df['Style_display_code'].nunique()      #2225
##
##
### RF
## r2 mean = 0.75
##data_lm['Style_display_code'].nunique() #2588
##df['Style_display_code'].nunique()      #2312
##
##
### SVR
## r2 mean = 0.40
##data_lm['Style_display_code'].nunique() #2588
##df['Style_display_code'].nunique()      #1511
#
### RF v4
## r2 mean = 0.63
##data_lm['Style_display_code'].nunique() #2505
##df['Style_display_code'].nunique()      #2148

### XGBoost v3
## r2 mean = 0.57  RMSE = 1753
##data_lm['Style_display_code'].nunique() #2505
##df['Style_display_code'].nunique()      #1946

#RMSE mean = 1258
#RMSE max = 58465
#RMSE min = 0.04
#r2 mean = 0.71
#r2 min = 0.01
#r2 max = 0.99