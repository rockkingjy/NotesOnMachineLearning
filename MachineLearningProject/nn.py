import numpy as np
import matplotlib.pyplot as plt
from time import time

from preprocess import preprocess
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import validation_curve

#get the preprocessed data
input_train, input_test, output_train, output_test = preprocess()
"""
param_grid = [
  {'alpha': [0.0001], 'solver': ['sgd'], 'learning_rate': ['constant'], 'momentum': [0.9], 'early_stopping': [True]},
]

mlp = GridSearchCV(MLPRegressor(), param_grid, cv=10)
"""
param_grid = [
  {'learning_rate_init': [0.1, 0.01,0.001,0.0001,0.00001]},
]

mlp = GridSearchCV(MLPRegressor(hidden_layer_sizes=(18,),batch_size=5, max_iter=1000, solver='sgd', learning_rate='constant', early_stopping=True), param_grid, cv=10)

#mlp = MLPRegressor(hidden_layer_sizes=(18,), batch_size=5, max_iter=1000, solver='sgd', learning_rate='constant', early_stopping=True)
print("Start training...")
t0 = time()
mlp.fit(input_train, output_train[0])
print "training time:", round(time()-t0, 3), "s"

#print("Start predicting...")
predict_train = mlp.predict(input_train)
print("Mean squared train error: ", mean_squared_error(output_train[0], predict_train))
predict_test = mlp.predict(input_test)
print("Mean squared test error: ", mean_squared_error(output_test[0], predict_test)) 

# scores
print("Best parametres: ", mlp.best_params_)
means = mlp.cv_results_['mean_test_score']
stds = mlp.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, mlp.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))





"""
param_grid = [
  {'learning_rate_init': [0.1, 0.01,0.001,0.0001,0.00001]},
]
mlp = GridSearchCV(MLPRegressor(hidden_layer_sizes=(18,),batch_size=5, max_iter=1000, solver='sgd', learning_rate='constant', early_stopping=True), param_grid, cv=10)

training time: 3355.893 s
('Mean squared train error: ', 0.03448391121277037)
('Mean squared test error: ', 0.033564280842786683)
('Best parametres: ', {'learning_rate_init': 0.01})
0.701 (+/-0.022) for {'learning_rate_init': 0.1}
0.723 (+/-0.035) for {'learning_rate_init': 0.01}
0.719 (+/-0.012) for {'learning_rate_init': 0.001}
0.700 (+/-0.030) for {'learning_rate_init': 0.0001}
0.683 (+/-0.020) for {'learning_rate_init': 1e-05}

param_grid = [
  {'learning_rate_init': [0.01,0.001,0.0001,0.00001]},
]
mlp = GridSearchCV(MLPRegressor(hidden_layer_sizes=(18,),batch_size=200, max_iter=1000, solver='sgd', learning_rate='constant', early_stopping=True), param_grid, cv=10)
/usr/local/lib/python2.7/dist-packages/sklearn/neural_network/multilayer_perceptron.py:563: ConvergenceWarning: Stochastic Optimizer: Maximum iterations reached and the optimization hasn't converged yet.
  % (), ConvergenceWarning)
training time: 1028.76 s
('Mean squared train error: ', 0.037759381201235821)
('Mean squared test error: ', 0.036762305210820551)
('Best parametres: ', {'learning_rate_init': 0.01})
0.711 (+/-0.023) for {'learning_rate_init': 0.01}
0.689 (+/-0.022) for {'learning_rate_init': 0.001}
0.654 (+/-0.031) for {'learning_rate_init': 0.0001}
0.532 (+/-0.042) for {'learning_rate_init': 1e-05}

param_grid = [
  {'batch_size': [5,10,50,100,200]},
]
mlp = GridSearchCV(MLPRegressor(hidden_layer_sizes=(18,), max_iter=1000, solver='sgd', learning_rate='constant', early_stopping=True), param_grid, cv=10)
training time: 982.679 s
('Mean squared train error: ', 0.03639791911900081)
('Mean squared test error: ', 0.035726194787371483)
('Best parametres: ', {'batch_size': 5})
0.719 (+/-0.020) for {'batch_size': 5}
0.715 (+/-0.025) for {'batch_size': 10}
0.706 (+/-0.017) for {'batch_size': 50}
0.697 (+/-0.023) for {'batch_size': 100}
0.692 (+/-0.020) for {'batch_size': 200}
mlp = MLPRegressor(hidden_layer_sizes=(5,), max_iter=1000, solver='sgd', learning_rate='constant', early_stopping=True)
training time: 9.047 s
('Mean squared train error: ', 0.042390931701517372)
predicting time: 0.115 s
('Mean squared test error: ', 0.041315409900277732)
('R2 score: ', 0.6852885573739208)
mlp = MLPRegressor(hidden_layer_sizes=(10,), max_iter=1000, solver='sgd', learning_rate='constant', early_stopping=True)
training time: 8.569 s
('Mean squared train error: ', 0.04063970785161497)
predicting time: 0.122 s
('Mean squared test error: ', 0.039661652595370959)
('R2 score: ', 0.69788570571244257)
mlp = MLPRegressor(hidden_layer_sizes=(50,), max_iter=1000, solver='sgd', learning_rate='constant', early_stopping=True)
Start training...
training time: 16.748 s
('Mean squared train error: ', 0.039935493257718675)
predicting time: 0.147 s
('Mean squared test error: ', 0.038794283274906846)
('R2 score: ', 0.70449270902650207)
mlp = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, solver='sgd', learning_rate='constant', early_stopping=True)
('Mean squared train error: ', 0.039230967832868779)
predicting time: 0.178 s
('Mean squared test error: ', 0.038340241382658784)
('R2 score: ', 0.70795127761548526)
mlp = MLPRegressor(hidden_layer_sizes=(300,), max_iter=1000, solver='sgd', learning_rate='constant', early_stopping=True)
training time: 55.157 s
('Mean squared train error: ', 0.039209359452649702)
predicting time: 0.326 s
('Mean squared test error: ', 0.038091453962005931)
('R2 score: ', 0.70984636345028296)
mlp = MLPRegressor(hidden_layer_sizes=(400,), max_iter=1000, solver='sgd', learning_rate='constant', early_stopping=True)
training time: 61.549 s
('Mean squared train error: ', 0.039464712342118693)
predicting time: 0.415 s
('Mean squared test error: ', 0.03851302295775217)
('R2 score: ', 0.70663515031847668)
mlp = MLPRegressor(hidden_layer_sizes=(500,), max_iter=1000, solver='sgd', learning_rate='constant', early_stopping=True)
training time: 83.57 s
('Mean squared train error: ', 0.039599699802225392)
predicting time: 0.464 s
('Mean squared test error: ', 0.038765309996974279)
('R2 score: ', 0.70471340687550743)
mlp = MLPRegressor(hidden_layer_sizes=(1000), max_iter=1000, solver='sgd', learning_rate='constant', early_stopping=True)
training time: 158.676 s
('Mean squared train error: ', 0.039670442373727598)
predicting time: 1.344 s
('Mean squared test error: ', 0.038792967530254532)
('R2 score: ', 0.70450273143457343)
mlp = MLPRegressor(hidden_layer_sizes=(5000,), max_iter=1000, solver='sgd', learning_rate='constant', early_stopping=True)
training time: 917.801 s
('Mean squared train error: ', 0.039459126070390406)
predicting time: 4.628 s
('Mean squared test error: ', 0.038558914448981338)
('R2 score: ', 0.70628558153908294)
mlp = MLPRegressor(hidden_layer_sizes=(300,300), max_iter=1000, solver='sgd', learning_rate='constant', early_stopping=True)
training time: 378.234 s
('Mean squared train error: ', 0.037603526246434857)
predicting time: 1.154 s
('Mean squared test error: ', 0.03668465387887123)
('R2 score: ', 0.72056236710892829)

mlp = MLPRegressor(hidden_layer_sizes=(1000,1000), max_iter=1000, solver='sgd', learning_rate='constant', early_stopping=True)
training time: 2813.474 s
('Mean squared train error: ', 0.037915039110020483)
predicting time: 9.067 s
('Mean squared test error: ', 0.037024818185845333)
('R2 score: ', 0.71797123706722932)


mlp = MLPRegressor(hidden_layer_sizes=(100,100,100,100,100,100,100,100,100), solver='sgd', learning_rate='constant', early_stopping=True)
training time: 135.433 s
('Mean squared train error: ', 0.03798036496769229)
predicting time: 1.582 s
('Mean squared test error: ', 0.037042643306597484)
('R2 score: ', 0.71783545796010118)
mlp = MLPRegressor(hidden_layer_sizes=(100,100,100,100,100,100,100,100), solver='sgd', learning_rate='constant', early_stopping=True)
training time: 143.235 s
('Mean squared train error: ', 0.037079453807950385)
predicting time: 1.48 s
('Mean squared test error: ', 0.036281840144948764)
('R2 score: ', 0.72363071597968431)
mlp = MLPRegressor(hidden_layer_sizes=(100,100,100,100,100,100,100), solver='sgd', learning_rate='constant', early_stopping=True)
training time: 84.871 s
('Mean squared train error: ', 0.038390239039548323)
predicting time: 0.98 s
('Mean squared test error: ', 0.037568664086870499)
('R2 score: ', 0.71382860533512194)
mlp = MLPRegressor(hidden_layer_sizes=(100,100,100,100,100,100), solver='sgd', learning_rate='constant', early_stopping=True)
/usr/local/lib/python2.7/dist-packages/sklearn/neural_network/multilayer_perceptron.py:563: ConvergenceWarning: Stochastic Optimizer: Maximum iterations reached and the optimization hasn't converged yet.
  % (), ConvergenceWarning)
training time: 374.833 s
('Mean squared train error: ', 0.033484215789898575)
predicting time: 0.763 s
('Mean squared test error: ', 0.032938190218301272)
('R2 score: ', 0.74910026583025324)
mlp = MLPRegressor(hidden_layer_sizes=(100,100,100,100,100), solver='sgd', learning_rate='constant', early_stopping=True)
/usr/local/lib/python2.7/dist-packages/sklearn/neural_network/multilayer_perceptron.py:563: ConvergenceWarning: Stochastic Optimizer: Maximum iterations reached and the optimization hasn't converged yet.
  % (), ConvergenceWarning)
training time: 389.466 s
('Mean squared train error: ', 0.034115672651010509)
predicting time: 0.756 s
('Mean squared test error: ', 0.033472144553669002)
('R2 score: ', 0.74503298101846938)
mlp = MLPRegressor(hidden_layer_sizes=(100,100,100,100), solver='sgd', learning_rate='constant', early_stopping=True)
training time: 114.586 s
Start predicting...
('Mean squared train error: ', 0.037921733149992205)
predicting time: 0.674 s
('Mean squared test error: ', 0.037045368366069989)
('R2 score: ', 0.71781470039289064)
mlp = MLPRegressor(hidden_layer_sizes=(100,100,100), solver='sgd', learning_rate='constant', early_stopping=True)
training time: 58.95 s
Start predicting...
('Mean squared train error: ', 0.037877152848159928)
predicting time: 0.447 s
('Mean squared test error: ', 0.036900666175621141)
('R2 score: ', 0.71891693888495034)
mlp = MLPRegressor(hidden_layer_sizes=(100,100), solver='sgd', learning_rate='constant', early_stopping=True)
training time: 44.456 s
('Mean squared train error: ', 0.03813608822672189)
predicting time: 0.298 s
('Mean squared test error: ', 0.037366509437063419)
"""
"""
param_grid = [
  {'alpha': [0.0001], 'solver': ['sgd'], 'learning_rate': ['constant','invscaling','adaptive'], 'momentum': [0.9], 'early_stopping': [True]},
]
mlp = GridSearchCV(MLPRegressor(), param_grid, cv=10)
Start training...
/usr/local/lib/python2.7/dist-packages/sklearn/neural_network/multilayer_perceptron.py:563: ConvergenceWarning: Stochastic Optimizer: Maximum iterations reached and the optimization hasn't converged yet.
  % (), ConvergenceWarning)
training time: 607.91 s
Start predicting...
('Mean squared train error: ', 0.040024874089335583)
predicting time: 0.265 s
('Mean squared test error: ', 0.038980884312705515)
('R2 score: ', 0.70307131487979291)
0.698 (+/-0.021) for {'alpha': 0.0001, 'learning_rate': 'constant', 'momentum': 0.9, 'early_stopping': True, 'solver': 'sgd'}
0.365 (+/-0.081) for {'alpha': 0.0001, 'learning_rate': 'invscaling', 'momentum': 0.9, 'early_stopping': True, 'solver': 'sgd'}
0.695 (+/-0.021) for {'alpha': 0.0001, 'learning_rate': 'adaptive', 'momentum': 0.9, 'early_stopping': True, 'solver': 'sgd'}
"""
"""
param_grid = [
  {'alpha': [0.0001], 'solver': ['sgd'], 'learning_rate': ['constant'], 'momentum': [0.9], 'early_stopping': [True]},
]

Start training...
training time: 228.578 s
Start predicting...
predicting time: 0.021 s
('Mean squared error: ', 0.038394157256354075)
('R2 score: ', 0.70754058479089399)
0.695 (+/-0.021) for {'alpha': 0.0001, 'learning_rate': 'constant', 'momentum': 0.9, 'early_stopping': True, 'solver': 'sgd'}
"""
"""
param_grid = [
  {'alpha': [0.0001], 'solver': ['sgd'], 'learning_rate': ['constant'], 'momentum': [0.9]},
]

Start training...
training time: 59.889 s
Start predicting...
predicting time: 0.02 s
('Mean squared error: ', 0.042312632457146171)
('R2 score: ', 0.67769242435118571)
0.671 (+/-0.018) for {'alpha': 0.0001, 'learning_rate': 'constant', 'momentum': 0.9, 'solver': 'sgd'}
"""

"""
mlp = MLPRegressor()

Start training...
training time: 4.563 s
Start predicting...
predicting time: 0.023 s
('Mean squared error: ', 0.030877302583211405)
('R2 score: ', 0.7647986438033838)
"""
"""
param_range = np.logspace(-6, -1, 5)
train_scores, valid_scores = validation_curve(MLPRegressor(), 
			input_train, output_train[0], "alpha",param_range)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(valid_scores, axis=1)
test_scores_std = np.std(valid_scores, axis=1)

plt.title("Validation Curve with MLP")
plt.xlabel("$alpha$")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()
"""

