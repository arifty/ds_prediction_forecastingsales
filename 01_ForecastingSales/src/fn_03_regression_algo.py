from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, cross_validate


# Regression model - Train and Test
#----------------------------------------------------------------------------------------
def model_regression_train (pos_booking_enrich, ls_stylecodes, numerical_feat, 
                                 categorical_feat, model_name, outlier_removal = False):
  """
  Run the Regression algorithm (Linear Regression or Random Forest)  This function can be used to 
  train & test together.

  pos_booking_enrich (dataframe): input pos_booking data which will be trained and tested 
                                  with the model
  ls_stylecodes (list): list of style codes to be modeled separately
  numerical_feat (list): numerical features input to the model
  categorical_feat (list): categorical features input to the model
  """
  # logging
  print("Prediction model started_Timestamp:{}".format(format(str(datetime.now()))))
  

  # define the column names
  outlier_colnames = list(pos_booking_enrich.columns) + ['outlier_flag', 'outlier_score']
  pred_colnames = ['Style_display_code', 'actual_testdata', 'predicted_testdata', 'RMSE', 
                   'r2', 'feature_importances', 'time_taken', 'start_time','model_name']

  cols_todrop = ['Style_display_code','OHInvUnts_WTD']
  X_cols_todrop = ['NetSlsUnts_WTD','Style_display_code','OHInvUnts_WTD', 'low_obsv_flag',
                   'outlier_flag','outlier_score']
  
  # empty dataframe
  predict_results_appended = pd.DataFrame()  
  pos_with_outlier = pd.DataFrame(columns=outlier_colnames)
    
 

  # loop thru each style_codes, to run the modeling 
  for style_code in ls_stylecodes:
    
    # capture the timestamp
    start_time = datetime.now()
    
    # logging
    print("Style code:{}_Timestamp:{}".format(style_code, format(str(start_time))))

    # filter the data per style_codes and encode categorical features
    model_data = pos_booking_enrich[pos_booking_enrich['Style_display_code']==style_code]
    
    # if the model is for training, perform outlier detection
    model_data = outlier_detection(model_data, categorical_feat, cols_todrop)
                
    if outlier_removal:
        # Get X, y WITHOUT outliers
        y = model_data[(model_data['outlier_score']
                    .between(outlier_threshold_lower, outlier_threshold_upper, inclusive=True))]['NetSlsUnts_WTD']
        X = model_data[(model_data['outlier_score']
                    .between(outlier_threshold_lower, outlier_threshold_upper, inclusive=True))] \
                    .drop(X_cols_todrop, 1)
        
    else:
        # Get X, y WITH outliers
        y = model_data['NetSlsUnts_WTD']
        X = model_data.drop(X_cols_todrop, 1)
        
    feature_list = list(X.columns)

    # define the preprocessor for OneHotEncoding
    preprocessor = ColumnTransformer([("numerical", "passthrough", numerical_feat), 
                                      ("categorical", OneHotEncoder(sparse=False, handle_unknown="ignore"),
                                       categorical_feat)])
    
    # create ML pipeline
    if model_name == 'RandomForest':
      model_pipe = Pipeline([("preprocessor", preprocessor),
                 (model_name,RandomForestRegressor(
                     max_depth=10,
                     min_samples_split=.3,
                     n_estimators=15,
                     max_features="auto",
                     bootstrap=True,  
                     n_jobs=-1))])
      
    elif model_name == 'LinearRegression':  
      model_pipe = Pipeline([("preprocessor", preprocessor),
                             (model_name,LinearRegression())])
      
    elif model_name == 'SVMRegression':
      model_pipe = Pipeline([("preprocessor", preprocessor),
                             (model_name,SVR(
                               gamma='scale', 
                               C=1.0, 
                               epsilon=0.2))])
      
    elif model_name == 'XGBRegression':
      model_pipe = Pipeline([("preprocessor", preprocessor),
                            (model_name, XGBRegressor(
                              colsample_bytree=0.8,
                              gamma=0,
                              learning_rate=0.1,
                              max_depth=5,
                              min_child_weight=1.5,
                              n_estimators=30,
                              reg_alpha=0.75,
                              reg_lambda=0.45,
                              nthread=6, 
                              scale_pos_weight=1,
                              subsample=0.9,
                              seed=42))])
    
    # get the train and test data splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
    
    # set the best params from the GridSearch output
#    if model_name == 'XGBRegression': 
#      gridsearch_best_params = gridsearch_validate(X_train,y_train)
#      model_pipe.set_params(**gridsearch_best_params)
#      print(gridsearch_best_params)
    
    
    # fit the model and predict
    model_pipe.fit(X_train,y_train)      
    predictions = model_pipe.predict(X_test)


    # check the coefficient of the model
    #coeff_model = pd.DataFrame(model_pipe.named_steps["model"].coef_,X.columns,columns=['Coeff'])
    
    # Get numerical feature importances, if model is RANDOM-FOREST
    if model_name == 'RandomForest':      
      importances = list(model_pipe.named_steps[model_name].feature_importances_)

      # List of tuples with variable and importance
      feature_importances = [(feature, np.round(importance*100, 5)) for feature, importance in zip(feature_list, importances)]
      feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    
    elif model_name == 'XGBRegression':
      feature_importances= sorted(model_pipe.named_steps[model_name].get_booster() \
                      .get_score(importance_type='gain').items(), key=lambda x: x[1], reverse=True)
      #list( dict((k, v) for k, v in model.booster().get_fscore().items() if v >= 10).keys())
  
    else:
      feature_importances = ""

      
    # check the RMSE
    RMSE = np.sqrt(metrics.mean_squared_error(y_test,predictions))
    print("RMSE:"+str(np.round(RMSE,2)))

    # check r2 score
    r2 = metrics.r2_score(y_test, predictions, multioutput='raw_values')[0]
    print("r2:"+str(np.round(r2, 2)))
    
    # capture the timestamp
    end_time = datetime.now()
    time_taken = (end_time - start_time)
    print("end_time:{}".format(format(str(end_time))))
    print("time_taken:"+str(time_taken))

    # create the dataframe of predicted results on test data
    predicted_results = pd.DataFrame([style_code, np.sum(y_test), np.sum(predictions),
                          RMSE, r2, feature_importances, time_taken, start_time, model_name], 
                                     pred_colnames)
    predict_results_appended = predict_results_appended.append(predicted_results.T)


    # save the model and predicted results
    pickle.dump(model_pipe, open(temp_dir+"/"+model_name+"/"+str(style_code)+ "_model_pipeline.p", 'wb'))
    
    
    # append the outlier data to a dataframe, to use for future testing or prediction
    pos_with_outlier = pos_with_outlier.append(model_data)
      
    
  # logging
  print("Prediction model finished_Timestamp:{}".format(format(str(datetime.now()))))

  # return the predict results and outlier data
  return predict_results_appended, pos_with_outlier
  
# Regression model - Predict only
#----------------------------------------------------------------------------------------
def model_regression_predict (pos_booking_enrich, ls_stylecodes, numerical_feat, 
                                   categorical_feat, model_name, predict_flag = True):
  """
  Run the Regression algorithm (Linear Regression or Random Forest) for the input data. 
  This function can be used only to predict the unseen data, provided the model is trained before.

  pos_booking_enrich (dataframe): input pos_booking data which will be used
                                   to predict the y_variable
  ls_stylecodes (list): list of style codes to be predicted separately
  numerical_feat (list): numerical features input to the model
  categorical_feat (list): categorical features input to the model
  """
    
  # logging
  print("Prediction model started_Timestamp:{}".format(format(str(datetime.now()))))
  
  X_cols_todrop = ['NetSlsUnts_WTD','Style_display_code','OHInvUnts_WTD', 'low_obsv_flag',
                   'outlier_flag','outlier_score']
  
  # define the column names
  if predict_flag:
      pred_colnames = ['Style_display_code', 'predicted_sales', 'time_taken','start_time', 
                       'model_name']
      
      pos_booking_enrich[['OHInvUnts_WTD', 'low_obsv_flag','outlier_flag',
              'outlier_score']] = pd.DataFrame([[0, 0, -1, 0]],index=pos_booking_enrich.index)
  else:
      pred_colnames = ['Style_display_code', 'actual_testdata', 'predicted_testdata', 'RMSE', 
                       'r2', 'feature_importances', 'time_taken', 'start_time','model_name']
      
      pos_booking_enrich = pickle.load(open(staging_dir+"/pos_data_withOutlier_lm_v2.p", "rb" ))
  
  # empty dataframe
  predict_results_appended = pd.DataFrame()
  
  # loop thru each style_codes, to run the modeling 
  for style_code in ls_stylecodes:
        
    # capture the timestamp
    start_time = datetime.now()
    
    # logging
    print("Style code:{}_Timestamp:{}".format(style_code, format(str(start_time))))
        
    # filter the data per style_codes and encode categorical features
    model_data = pos_booking_enrich[pos_booking_enrich['Style_display_code']==style_code]
    
    # Get X, y without outliers
    y = model_data[(model_data['outlier_score']
                .between(outlier_threshold_lower, outlier_threshold_upper, inclusive=True))]['NetSlsUnts_WTD']
    X = model_data[(model_data['outlier_score']
                .between(outlier_threshold_lower, outlier_threshold_upper, inclusive=True))] \
                .drop(X_cols_todrop, 1)

    feature_list = list(X.columns)

    # define the preprocessor for OneHotEncoding
    preprocessor = ColumnTransformer([("numerical", "passthrough", numerical_feat), 
                                      ("categorical", OneHotEncoder(sparse=False, handle_unknown="ignore"),
                                       categorical_feat)])

    model_pipe = pickle.load(open(temp_dir+"/"+model_name+"/"+str(style_code)+ "_model_pipeline.p", "rb" ))
    
    if predict_flag:
        # fetch all X data to predict as X_test to predict
        X_test = X
    
    else:
        # get the train and test data splits
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)


    # fit the model and predict     
    predictions = model_pipe.predict(X_test)
    
    # If its the future prediction, results can't be validated
    if not(predict_flag):
      #  # Get numerical feature importances, incase of RandomForest
      if model_name == 'RandomForest':
        importances = list(model_pipe.named_steps[model_name].feature_importances_)

        # List of tuples with variable and importance
        feature_importances = [(feature, np.round(importance*100, 5)) for feature, importance in zip(feature_list, importances)]
        feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
      
      elif model_name == 'XGBRegression':
        feature_importances= sorted(model_pipe.named_steps[model_name].get_booster() \
                      .get_score(importance_type='gain').items(), key=lambda x: x[1], reverse=True)
      
      else:
        feature_importances = ""
        
      # check the RMSE
      RMSE = np.sqrt(metrics.mean_squared_error(y_test,predictions))
      print("RMSE:"+str(np.round(RMSE,2)))

      # check r2 score
      r2 = metrics.r2_score(y_test, predictions, multioutput='raw_values')[0]
      print("r2:"+str(np.round(r2, 2)))
    
    
    # capture the timestamp
    end_time = datetime.now()
    time_taken = (end_time - start_time)
    
    if predict_flag:
        results_list = [style_code, int(round(np.sum(predictions),0)),time_taken,start_time, model_name]
    else:
        results_list = [style_code, np.sum(y_test), int(round(np.sum(predictions),0)),RMSE, r2, 
                        feature_importances, time_taken, start_time, model_name]
  
    # create the dataframe of predicted results on test data
    predicted_results = pd.DataFrame(results_list, pred_colnames)
    predict_results_appended = predict_results_appended.append(predicted_results.T)

    
  # return the predict results and outlier data
  print("Prediction model finished_Timestamp:{}".format(format(str(datetime.now()))))
  return predict_results_appended


#RF => staging_dir+"/"+str(style_code)+ "_model_pipeline.p"
# LM => staging_dir+"/"+str(style_code)+ "_model_pipeline_lm.p", "rb" ))
