#!/usr/bin/python3 -B

# load Python modules
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn import metrics
from sklearn.model_selection import GridSearchCV


### load data
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print('train size:', train_df.shape)
print('test size:', test_df.shape)
print('have the same columns?', all(train_df.drop('target', axis=1).columns == test_df.columns))
train_df_org = train_df
test_df_org = test_df


###data cleansing
# remove duplicatives if exists
# wrt rows
train_df = train_df.drop_duplicates()
# wrt columns (get recursion error)
#train_df = train_df.T.drop_duplicates().T

rows = train_df.shape[0]
columns = train_df.shape[1]
print("The train dataset contains {0} rows and {1} columns".format(rows, columns))
print(test_df.head())


# remove constant values
train_df = train_df.loc[:, (train_df != train_df.iloc[0]).any()]
test_df = test_df.loc[:, train_df.drop('target', axis=1).columns]


# fill nan by median
train_df = train_df.replace(-1, np.NaN)
test_df = test_df.replace(-1, np.NaN)
print('nan exists in train?:', train_df.isnull().any().any())
print('nan exists in test?:', test_df.isnull().any().any())
train_median = train_df.drop('target', axis=1).median()
train_df = train_df.fillna(train_median)
test_df = test_df.fillna(train_median)


# separate data
train_y = train_df.loc[:, 'target']
train_id = train_df.loc[:, 'id']
train_df = train_df.drop(['target', 'id'], axis=1)
train_df_float = train_df.select_dtypes(include=['float64'])
train_df_int = train_df.select_dtypes(include=['int64'])
test_id = test_df.loc[:, 'id']
test_df = test_df.drop('id', axis=1)
test_df_float = test_df.select_dtypes(include=['float64'])
test_df_int = test_df.select_dtypes(include=['int64'])
print('train float:', len(train_df_float.columns))
print('train int:', len(train_df_int.columns))
print('test float:', len(test_df_float.columns))
print('test int:', len(test_df_int.columns))


# normalize data
train_df_float_mean = train_df_float.mean()
train_df_float_std = train_df_float.std()
train_df_float_norm = (train_df_float - train_df_float_mean) / (train_df_float_std + 1.e-9)
test_df_float_norm = (test_df_float - train_df_float_mean) / (train_df_float_std + 1.e-9)

train_df_norm = pd.concat((train_df_float_norm, train_df_int), axis=1)
test_df_norm = pd.concat((test_df_float_norm, test_df_int), axis=1)
print(train_df_norm.shape)
print(test_df_norm.shape)


# gini score
def gini(solution, submission):
    df = zip(solution, submission, range(len(solution)))
    df = sorted(df, key=lambda x: (x[1],-x[2]), reverse=True)
    rand = [float(i+1)/float(len(df)) for i in range(len(df))]
    totalPos = float(sum([x[0] for x in df]))
    cumPosFound = [df[0][0]]
    for i in range(1,len(df)):
        cumPosFound.append(cumPosFound[len(cumPosFound)-1] + df[i][0])
    Lorentz = [float(x)/totalPos for x in cumPosFound]
    Gini = [Lorentz[i]-rand[i] for i in range(len(df))]
    return sum(Gini)

def normalized_gini(solution, submission):
    normalized_gini = gini(solution, submission)/gini(solution, solution)
    return normalized_gini

# Normalized Gini Scorer
gini_scorer = metrics.make_scorer(normalized_gini, greater_is_better = True)

# lightgbm
param_test2 = [
    "max_bin":range(20, 10000, 100),
    "learning_rate":range(0.001, 0.3, 0.005),
    "num_leaves":range(100, 4095, 100),
    "scale_pos_weight":[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1., 5., 10., 50., 100., 500],
    "n_estimators":[10, 50, 100, 500, 1000, 3000, 5000],
    "min_child_weight": [1, 10, 100, 500, 1000, 2000],
    "subsample": range(0.4, 1, 0.1),
    "bagging_fraction": range(0.3, 1, 0.1),
    "max_depth":range(2, 40, 4),
]

# param_test2 = {
#     'max_depth':range(3,10,1),
#     'min_child_weight':range(1,6,1),
#     'subsample':[i/100.0 for i in range(6,100,5)],
#     'colsample_bytree':[i/100.0 for i in range(20,100)],
#     'gamma':[i/100.0 for i in range(0,100,5)],
#     'reg_alpha':[1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100]
# }

gsearch2 = GridSearchCV(estimator=lgb.LGBMRegressor(
                            objective='regression',
                            num_leaves=31,
                            learning_rate=0.05,
                            n_estimators=20),
                        param_grid=param_test2,
                        verbose=2,
                        # scoring=gini_scorer,
                        # n_jobs=2,
                        iid=False,
                        cv=5)
gsearch2.fit(train_df_norm, train_y)

print(gsearch2.id_scores_, gsearch2.best_params_, gsearch2.best_score_)

predict_y_lgb = gsearch2.predict(test_df_norm)
predict_y_lgb[predict_y_lgb> 1.] = 1.
predict_y_lgb[predict_y_lgb< 0.] = 0.
predict_submit = pd.concat((test_id, pd.DataFrame(data=predict_y_lgb, columns=['target'])), axis=1)
predict_submit.to_csv('./gsearch2_lgb_submission.csv', index=False)   #LB

### parameter tuning of xgboost

# Grid seach on subsample and max_features
param_test1 = {
    'max_depth':range(3,10,1),
    'min_child_weight':range(1,6,1),
    'subsample':[i/100.0 for i in range(6,100,5)],
    'colsample_bytree':[i/100.0 for i in range(20,100)],
    'gamma':[i/100.0 for i in range(0,100,5)],
    'reg_alpha':[1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100]
}

gsearch1 = GridSearchCV(estimator=xgb.XGBRegressor(max_depth=3,
                                                   learning_rate=0.001,
                                                   n_estimators=1000,
                                                   # n_jobs=2,
                                                   gamma=0,
                                                   objective='reg:logistic',
                                                   min_child_weight=1,
                                                   subsample=1,
                                                   colsample_bytree=1,
                                                   scale_pos_weight=1,
                                                   seed=27),
                        param_grid=param_test1,
                        verbose=2,
                        # scoring=gini_scorer,
                        # n_jobs=2,
                        iid=False,
                        cv=5)
gsearch1.fit(train_df_norm, train_y)

print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)

predict_y_xgb = gsearch1.predict(test_df_norm)
predict_y_xgb[predict_y_xgb> 1.] = 1.
predict_y_xgb[predict_y_xgb< 0.] = 0.
predict_submit = pd.concat((test_id, pd.DataFrame(data=predict_y_xgb, columns=['target'])), axis=1)
predict_submit.to_csv('./gsearch1_once_submission.csv', index=False)   #LB

# ensemble
predict_y_ens = predict_y_xgb + predict_y_gbm
predict_y_ens /= 2.
predict_y_ens[predict_y_ens > 1.] = 1.
predict_y_ens[predict_y_ens < 0.] = 0.
predict_submit_ens = pd.concat((test_id, pd.DataFrame(data=predict_y_ens, columns=['target'])), axis=1)
predict_submit_ens.to_csv('./ens0_submission.csv', index=False)   #LB0.170
