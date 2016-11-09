import numpy as np
from time import time
from preprocess import preprocess
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

#get the preprocessed data
input_train, input_test, output_train, output_test = preprocess(model="other")

param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']},
]
#grid search with cross-validation
clf = GridSearchCV(SVR(), param_grid, cv=10)
# training
print("Start training...")
t0 = time()
clf.fit(input_train, output_train[0])
print "training time:", round(time()-t0, 3), "s"
# predict
print("Start predicting...")
t0 = time()
predict_test = clf.predict(input_test)
print "predicting time:", round(time()-t0, 3), "s"
# best parameters
print("Best parametres: ", clf.best_params_)
print("Mean squared test error: ", mean_squared_error(output_test[0], predict_test))
# scores for different parameters
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))


"""
Volume of training set: 45000
Volume of test set: 5000
Start training...
training time: 129177.478 s
Start predicting...
predicting time: 5.475 s
('Best parametres: ', {'kernel': 'rbf', 'C': 1000, 'gamma': 0.1})
('Mean squared test error: ', 0.028527340638862371)
0.434 (+/-0.042) for {'kernel': 'linear', 'C': 1}
0.445 (+/-0.041) for {'kernel': 'linear', 'C': 10}
0.526 (+/-0.031) for {'kernel': 'linear', 'C': 100}
0.674 (+/-0.020) for {'kernel': 'linear', 'C': 1000}
0.690 (+/-0.016) for {'kernel': 'rbf', 'C': 1, 'gamma': 0.1}
0.623 (+/-0.023) for {'kernel': 'rbf', 'C': 1, 'gamma': 0.01}
0.452 (+/-0.038) for {'kernel': 'rbf', 'C': 1, 'gamma': 0.001}
0.459 (+/-0.024) for {'kernel': 'rbf', 'C': 1, 'gamma': 0.0001}
0.706 (+/-0.015) for {'kernel': 'rbf', 'C': 10, 'gamma': 0.1}
0.660 (+/-0.020) for {'kernel': 'rbf', 'C': 10, 'gamma': 0.01}
0.491 (+/-0.036) for {'kernel': 'rbf', 'C': 10, 'gamma': 0.001}
0.446 (+/-0.038) for {'kernel': 'rbf', 'C': 10, 'gamma': 0.0001}
0.732 (+/-0.014) for {'kernel': 'rbf', 'C': 100, 'gamma': 0.1}
0.671 (+/-0.018) for {'kernel': 'rbf', 'C': 100, 'gamma': 0.01}
0.623 (+/-0.023) for {'kernel': 'rbf', 'C': 100, 'gamma': 0.001}
0.441 (+/-0.041) for {'kernel': 'rbf', 'C': 100, 'gamma': 0.0001}
0.773 (+/-0.011) for {'kernel': 'rbf', 'C': 1000, 'gamma': 0.1}
0.701 (+/-0.015) for {'kernel': 'rbf', 'C': 1000, 'gamma': 0.01}
0.660 (+/-0.020) for {'kernel': 'rbf', 'C': 1000, 'gamma': 0.001}
0.490 (+/-0.036) for {'kernel': 'rbf', 'C': 1000, 'gamma': 0.0001}

"""

