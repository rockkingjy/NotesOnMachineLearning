import numpy as np
from time import time
from preprocess import preprocess
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import validation_curve

#get the preprocessed data
input_train, input_validation, input_test, output_train, output_validation, output_test = preprocess(model="validation")

learning_rate_init=[0.1]
batch_size=[50]
"""
learning_rate_init=[0.1, 0.01,0.001,0.0001]
batch_size=[5,50,200]
"""
for l in learning_rate_init:
	for b in batch_size:
		print("learning_rate:",l,";batch_size:",b)
		mlp = MLPRegressor(hidden_layer_sizes=(100,50), learning_rate_init=l,batch_size=b, max_iter=1000, solver='sgd',alpha=0.0001, learning_rate='constant', early_stopping=True)
		#print("Start training...")
		t0 = time()
		mlp.fit(input_train, output_train[0])
		print "Training time:", round(time()-t0, 3), "s"
		#print("Start predicting...")
		predict_train = mlp.predict(input_train)
		print("Mean squared train error: ", mean_squared_error(output_train[0], predict_train))
		predict_validation = mlp.predict(input_validation)
		print("Mean squared validation error: ", mean_squared_error(output_validation[0], predict_validation)) 
		predict_test = mlp.predict(input_test)
		print("Mean squared test error: ", mean_squared_error(output_test[0], predict_test)) 

"""
mlp = MLPRegressor(hidden_layer_sizes=(10), learning_rate_init=0.001,batch_size=200, max_iter=1000, solver='sgd',alpha=0.0001, learning_rate='constant', early_stopping=True)
training time: 6.871 s
('Mean squared train error: ', 0.041514599316041019)
('Mean squared validation error: ', 0.040357197815880153)

hidden_layer_sizes=[(8,),(18,),(100,),(300,),(500,),(600,),(800,),(1000,)]
Training time: 9.498 s
('Mean squared train error: ', 0.043521421908057953)
('Mean squared validation error: ', 0.042649514540149286)
Training time: 8.473 s
('Mean squared train error: ', 0.041207934635343141)
('Mean squared validation error: ', 0.03995040198310542)
Training time: 21.971 s
('Mean squared train error: ', 0.039290429947297388)
('Mean squared validation error: ', 0.03818179716937925)
Training time: 45.923 s
('Mean squared train error: ', 0.03948692766049567)
('Mean squared validation error: ', 0.038325004425151875)
Training time: 52.238 s
('Mean squared train error: ', 0.040377086541921306)
('Mean squared validation error: ', 0.03948797402698407)
Training time: 114.687 s
('Mean squared train error: ', 0.039243154871851892)
('Mean squared validation error: ', 0.038204127207584308)
Training time: 101.39 s
('Mean squared train error: ', 0.040137393508918776)
('Mean squared validation error: ', 0.039154801189593268)
Training time: 168.982 s
('Mean squared train error: ', 0.03948923350519027)
('Mean squared validation error: ', 0.038425345712251993)

hidden_layer_sizes=[(100,),(100,8),(100,30),(100,50),(100,100)]
Training time: 14.318 s
('Mean squared train error: ', 0.040224377833207779)
('Mean squared validation error: ', 0.03921083407107176)
Training time: 14.886 s
('Mean squared train error: ', 0.038751345800618148)
('Mean squared validation error: ', 0.037487310288533297)
Training time: 30.323 s
('Mean squared train error: ', 0.038157273784590093)
('Mean squared validation error: ', 0.037005813220749251)
Training time: 38.576 s
('Mean squared train error: ', 0.037859849356471492)
('Mean squared validation error: ', 0.036743163908343313)
Training time: 32.02 s
('Mean squared train error: ', 0.039226781259606359)
('Mean squared validation error: ', 0.038490311443930697)

hidden_layer_sizes=[(100,50,8),(100,50,18),(100,50,30),(100,50,50)]
Training time: 18.787 s
('Mean squared train error: ', 0.03896993469858781)
('Mean squared validation error: ', 0.037770061668906917)
Training time: 22.637 s
('Mean squared train error: ', 0.038405254473041155)
('Mean squared validation error: ', 0.037237082875896291)
Training time: 48.908 s
('Mean squared train error: ', 0.037193062518818688)
('Mean squared validation error: ', 0.036215014356336415)
Training time: 82.408 s
('Mean squared train error: ', 0.035624651988229603)
('Mean squared validation error: ', 0.034587290059668402)

hidden_layer_sizes=[(100,50,50,8),(100,50,50,18),(100,50,50,30),(100,50,50)]
Training time: 34.441 s
('Mean squared train error: ', 0.037962452635549038)
('Mean squared validation error: ', 0.036838215887159388)
Training time: 54.718 s
('Mean squared train error: ', 0.037123290529314748)
('Mean squared validation error: ', 0.036192997372392711)
Training time: 43.396 s
('Mean squared train error: ', 0.037609240117987844)
('Mean squared validation error: ', 0.036442162753795422)
Training time: 50.013 s
('Mean squared train error: ', 0.037808087249852976)
('Mean squared validation error: ', 0.036541657259016244)

hidden_layer_sizes=[(100,50),(100,50,8),(100,50,18),(100,50,30),(100,50,50)]
Training time: 53.514 s
('Mean squared train error: ', 0.037668101806698842)
('Mean squared validation error: ', 0.036442866877629099)
Training time: 19.932 s
('Mean squared train error: ', 0.039039691432368337)
('Mean squared validation error: ', 0.037611934108554589)
Training time: 31.501 s
('Mean squared train error: ', 0.038124012157138967)
('Mean squared validation error: ', 0.037040938802213728)
Training time: 38.856 s
('Mean squared train error: ', 0.037845549080501452)
('Mean squared validation error: ', 0.0367207383808923)
Training time: 45.359 s
('Mean squared train error: ', 0.037829734947651385)
('Mean squared validation error: ', 0.036558040262425963)

learning_rate_init=[0.1, 0.01,0.001,0.0001]
batch_size=[5,50,200]
('learning_rate:', 0.1, ';batch_size:', 5)
Training time: 17.917 s
('Mean squared train error: ', 0.033952081604797256)
('Mean squared validation error: ', 0.033091138671051595)
('learning_rate:', 0.1, ';batch_size:', 50)
Training time: 6.024 s
('Mean squared train error: ', 0.031416204542312504)
('Mean squared validation error: ', 0.030943414216340871)
('learning_rate:', 0.1, ';batch_size:', 200)
Training time: 3.711 s
('Mean squared train error: ', 0.033362673709819782)
('Mean squared validation error: ', 0.032364611900165451)
('learning_rate:', 0.01, ';batch_size:', 5)
Training time: 26.058 s
('Mean squared train error: ', 0.031557035420465805)
('Mean squared validation error: ', 0.030742032541581851)
('learning_rate:', 0.01, ';batch_size:', 50)
Training time: 13.189 s
('Mean squared train error: ', 0.032657965075990697)
('Mean squared validation error: ', 0.031881298069200224)
('learning_rate:', 0.01, ';batch_size:', 200)
Training time: 29.388 s
('Mean squared train error: ', 0.032677787945050919)
('Mean squared validation error: ', 0.031948062558699176)
('learning_rate:', 0.001, ';batch_size:', 5)
Training time: 70.319 s
('Mean squared train error: ', 0.032862067667770369)
('Mean squared validation error: ', 0.03240046981636812)
('learning_rate:', 0.001, ';batch_size:', 50)
Training time: 74.403 s
('Mean squared train error: ', 0.033836133906444341)
('Mean squared validation error: ', 0.032995072301808891)
('learning_rate:', 0.001, ';batch_size:', 200)
Training time: 32.443 s
('Mean squared train error: ', 0.038627966478969503)
('Mean squared validation error: ', 0.037448690633770126)
('learning_rate:', 0.0001, ';batch_size:', 5)
Training time: 244.996 s
('Mean squared train error: ', 0.035539589427328031)
('Mean squared validation error: ', 0.034580324956864437)
('learning_rate:', 0.0001, ';batch_size:', 50)
Training time: 75.77 s
('Mean squared train error: ', 0.039340145958378744)
('Mean squared validation error: ', 0.038250995373604872)
('learning_rate:', 0.0001, ';batch_size:', 200)
Training time: 81.426 s
('Mean squared train error: ', 0.041863942618064837)
('Mean squared validation error: ', 0.041368620673970652)

('learning_rate:', 0.1, ';batch_size:', 50)
Training time: 7.493 s
('Mean squared train error: ', 0.030647795904989252)
('Mean squared validation error: ', 0.029661068715503076)
('Mean squared test error: ', 0.030479616037540517)
"""

