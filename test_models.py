import sklearn
import time
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
import xgboost
import csv as csv
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.model_selection import cross_val_score,KFold
from sklearn.metrics import mean_absolute_error
from scipy.stats import skew
from collections import OrderedDict

data = pd.read_csv('/content/gdrive/My Drive/eatconcernmodel - eatconcernmodel.csv')

train, test = train_test_split(data, test_size=0.3)

df = StandardScaler().fit_transform(train)

train = pd.DataFrame(df)

df1 = StandardScaler().fit_transform(test)

test = pd.DataFrame(df1)

trainY = train.iloc[:,-1]
trainX= train.iloc[:,:-1]
valY = test.iloc[:,-1]
valX= test.iloc[:,:-1]

X = imgs.iloc[:,:-2]
y = imgs.iloc[:,-1]

X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())

reg = LinearRegression().fit(trainX, trainY)

print("\n Linear Regression: \n")

print('MAE: \n', mean_absolute_error(valY, reg.predict(valX), multioutput='raw_values'))
# The mean squared error
print('MSE: %.5f'
      % mean_squared_error(valY, reg.predict(valX)))
# The coefficient of determination: 1 is perfect prediction
print('R-Sqrd: %.2f'
      % r2_score(valY, reg.predict(valX)))

svr = sklearn.svm.NuSVR()

kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1),
                  param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                              "gamma": np.logspace(-2, 2, 5)})

t0 = time.time()
svr.fit(trainX, trainY)
svr_fit = time.time() - t0


t0 = time.time()
kr.fit(trainX, trainY)
kr_fit = time.time() - t0

print("\n Support Vector Regression: \n")

print('MAE: \n', mean_absolute_error(valY, svr.predict(valX), multioutput='raw_values'))
# The mean squared error
print('MSE: %.5f'
      % mean_squared_error(valY, svr.predict(valX)))
# The coefficient of determination: 1 is perfect prediction
print('R-Sqrd: %.2f'
      % r2_score(valY, svr.predict(valX)))


print("\n Kernel Regression: \n")

print('MAE: \n', mean_absolute_error(valY, kr.predict(valX), multioutput='raw_values'))
# The mean squared error
print('MSE: %.5f'
      % mean_squared_error(valY, kr.predict(valX)))
# The coefficient of determination: 1 is perfect prediction
print('R-Sqrd: %.2f'
      % r2_score(valY, kr.predict(valX)))


print("\n LARS: \n")

reg = linear_model.LassoLars(alpha=.1)
reg.fit(trainX, trainY)

print('MAE: \n', mean_absolute_error(valY, reg.predict(valX), multioutput='raw_values'))
# The mean squared error
print('MSE: %.5f'
      % mean_squared_error(valY, reg.predict(valX)))
# The coefficient of determination: 1 is perfect prediction
print('R-Sqrd: %.2f'
      % r2_score(valY, reg.predict(valX)))

br = linear_model.BayesianRidge()
br.fit(trainX, trainY)

print("\n Bayesian Ridge: \n")

print('MAE: \n', mean_absolute_error(valY, br.predict(valX), multioutput='raw_values'))
# The mean squared error
print('MSE: %.5f'
      % mean_squared_error(valY, br.predict(valX)))
# The coefficient of determination: 1 is perfect prediction
print('R-Sqrd: %.2f'
      % r2_score(valY, br.predict(valX)))

# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(trainX, trainY)
regr_2.fit(trainX, trainY)

print("\n Decision Tree D3: \n")
print('MAE: \n', mean_absolute_error(valY, regr_1.predict(valX), multioutput='raw_values'))
# The mean squared error
print('MSE: %.5f'
      % mean_squared_error(valY, regr_1.predict(valX)))
# The coefficient of determination: 1 is perfect prediction
print('R-Sqrd: %.2f'
      % r2_score(valY, regr_1.predict(valX)))

print("\n Decision Tree D5: \n")
print('MAE: \n', mean_absolute_error(valY, regr_2.predict(valX), multioutput='raw_values'))
# The mean squared error
print('MSE: %.5f'
      % mean_squared_error(valY, regr_2.predict(valX)))
# The coefficient of determination: 1 is perfect prediction
print('R-Sqrd: %.2f'
      % r2_score(valY, regr_2.predict(valX)))

MLP = MLPRegressor(random_state=1, max_iter=500).fit(trainX, trainY)

print("\n MLP: \n")
print('MAE: \n', mean_absolute_error(valY, MLP.predict(valX), multioutput='raw_values'))
# The mean squared error
print('MSE: %.5f'
      % mean_squared_error(valY, MLP.predict(valX)))
# The coefficient of determination: 1 is perfect prediction
print('R-Sqrd: %.2f'
      % r2_score(valY, MLP.predict(valX)))

best_xgb_model = xgboost.XGBRegressor(colsample_bytree=0.4,
                 gamma=0,                 
                 learning_rate=0.07,
                 max_depth=3,
                 min_child_weight=1.5,
                 n_estimators=10000,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42)

best_xgb_model.fit(trainX, trainY)

print("\n xgboost: \n")
print('MAE: \n', mean_absolute_error(valY, best_xgb_model.predict(valX), multioutput='raw_values'))
# The mean squared error
print('MSE: %.5f'
      % mean_squared_error(valY, best_xgb_model.predict(valX)))
# The coefficient of determination: 1 is perfect prediction
print('R-Sqrd: %.2f'
      % r2_score(valY, best_xgb_model.predict(valX)))

import shap
shap.initjs()
explainer = shap.KernelExplainer(br.predict, trainX)
shap_values = explainer.shap_values(valX, nsamples=5)
shap.summary_plot(shap_values, valX, plot_type="bar")

