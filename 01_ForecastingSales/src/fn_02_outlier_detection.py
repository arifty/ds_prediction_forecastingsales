# Outlier detection model
#----------------------------------------------------------------------------------------
def outlier_detection (model_data, categorical_feat, cols_todrop):
  """
  Calculate the cartesian join of 3 lists and return a dataframe

  list1: first list
  list2: second list
  list3: third list
  list4: fourth list
  col_names: column names when converting to dataframe
  """
  from sklearn.ensemble import IsolationForest

   
  # encode the categorical features for X_train
  X_train_outlier = pd.get_dummies(model_data,columns=categorical_feat) \
                        .drop(cols_todrop, axis=1)

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

  return model_data

