#----------------------------------------------------------------------------------------
def gridsearch_validate (X_train,y_train):
  """
  Run the GridSearchCV validation to find the best parameters for XGBoost Regressor.

  X_train (dataframe): X dataset 
  y_train (dataframe): y dataset
  """

  # -------------------------------------------------------------------------
  print("GridSearch started_Timestamp:{}".format(format(str(datetime.now()))))


  # GridSearch for tuning parameters
  parameters_for_testing = {
                          model_name+'__colsample_bytree':[0.4,0.8],
                          model_name+'__gamma':[0,0.03,0.3],
                          model_name+'__min_child_weight':[1.5,10],
                          model_name+'__learning_rate':[0.1,0.07],
                          model_name+'__max_depth':[3,5],
                          model_name+'__n_estimators':[10, 30],
                          model_name+'__reg_alpha':[0.75],
                          model_name+'__reg_lambda':[0.45],
                          model_name+'__subsample':[0.6,0.90]  
                          }

  gridCV = GridSearchCV(estimator = model_pipe,
                        param_grid = parameters_for_testing,
                        cv=5, n_jobs=-1,
                        iid=False, verbose=10,
                        scoring='neg_mean_squared_error')
  gridCV.fit(X_train,y_train)
  #print (gridCV.grid_scores_)
  print('best params:'+str(gridCV.best_params_))
  print('best score:'+str(gridCV.best_score_))

  print("GridSearch ended_Timestamp:{}".format(format(str(datetime.now()))))
  
  return gridCV.best_params_
  # -------------------------------------------------------------------------
