import numpy as np
from time import time
from preprocess import preprocess
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

#get the preprocessed data
input_train, input_test, output_train, output_test = preprocess(model="other")

#Linear Regression
regr_linear = linear_model.LinearRegression()
regr_linear.fit(input_train, output_train[0])
#print('Coefficients: \n', regr_linear.coef_)
print("Mean squared error: %.5f"
      % np.mean((regr_linear.predict(input_test) - output_test[0]) ** 2))
#error:0.03912

#Ridge Regression
regr_ridge = linear_model.Ridge (alpha = .5)
regr_ridge.fit(input_train, output_train[0])
#print('Coefficients: \n', regr_ridge.coef_)
print("Mean squared error: %.5f"
      % np.mean((regr_ridge.predict(input_test) - output_test[0]) ** 2))
#error:0.06832

#Ridge regression with built-in cross-validation of the alpha parameter
regr_ridgeCV = linear_model.RidgeCV(alphas=[0.0000001, 0.01, 0.1, 0.5, 1.0, 10.0])
regr_ridgeCV.fit(input_train, output_train[0])
print regr_ridgeCV.alpha_  
#print('Coefficients: \n', regr_ridgeCV.coef_)
print("Mean squared error: %.5f"
      % np.mean((regr_ridgeCV.predict(input_test) - output_test[0]) ** 2))
#alpha=1e-07
#error:0.03988

regr_lassoCV = linear_model.LassoCV(cv=10)
regr_lassoCV.fit(input_train, output_train[0])
print('alpha: \n', regr_lassoCV.alpha_)
print("Mean squared error: %.5f"
      % np.mean((regr_lassoCV.predict(input_test) - output_test[0]) ** 2))
#error:0.06847

regr_elasticnetCV = linear_model.ElasticNetCV(cv=10)
regr_elasticnetCV.fit(input_train, output_train[0])
#print('Coefficients: \n', regr_elasticnetCV.coef_)
print("Mean squared error: %.5f"
      % np.mean((regr_elasticnetCV.predict(input_test) - output_test[0]) ** 2))
#error:0.06847

