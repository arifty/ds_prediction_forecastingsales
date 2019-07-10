from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, SCORERS
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, cross_validate
  
  
# apply K-fold cross validation sets
kfold = KFold(10)
final_predictions = []

rfm = RandomForestRegressor(
               max_depth=7,
               min_samples_split=.3,
               n_estimators=15,
               max_features="auto",
               bootstrap=True,  
               n_jobs=-1)
# try 1
scores =  [rfm.fit(X.values[train], y.values[train]) \
           .score(X.values[test], y.values[test])
           for train, test in kfold.split(X.values)]

# try 2
scores_mse = cross_val_score(rfm, X.values, y.values ,cv=10)
print(np.mean(scores_mse))

# try 3

scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'explained_variance']
scores = cross_validate(rfm, X.values, y.values, scoring=scoring,
                        cv=kfold.split(X.values), return_train_score=False)
sorted(scores.keys())

#sorted(SCORERS.keys())

scores['test_explained_variance']

rmse = np.round(np.sqrt(np.abs(scores['test_neg_mean_squared_error'])),2)

np.sum(rmse)