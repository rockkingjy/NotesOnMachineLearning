#!/usr/bin/python
# rows:50000; col:40; 
#input:1-6,22-27,(-1)-(-6), 18 cols;
#output:(-7)-(-12), 6 cols;

from sklearn import cross_validation

# model="validation":to separate the dataset to training, validation and test set
# otherwise, just separate to trainng and test set
def preprocess(model):
	# read from the original file
	file_handler = open('./RESPLAS.txt', "r")

	input_number = []
	output_number = []

	lines = file_handler.readlines()
	for i in range(len(lines)):
		input_number.append([])
		output_number.append([])
	
	point = 0
	for line in lines:
		items = line.split()
		for i in range(0,6):
			input_number[point].append(float(items[i]))
		for i in range(21,27):
			input_number[point].append(float(items[i]))
		for i in range(-6,0):
			input_number[point].append(float(items[i]))
		for i in range(-12,-6):
			output_number[point].append(float(items[i]))
		point = point + 1
		
	file_handler.close()
	# set 10% of the dataset as the test set, the remaining as the training set
	input_train, input_test, output_train, output_test = cross_validation.train_test_split(input_number, output_number, test_size=0.1, random_state=42)
	if(model=="validation"):
		# set 10% of the training set as the validation set, the remaing as the training set
		input_train, input_validation, output_train, output_validation = cross_validation.train_test_split(input_train, output_train, test_size=0.1, random_state=42)

	print "Volume of training set:", len(input_train) 
	if(model=="validation"):
		print "Volume of validation set:", len(input_validation) 
	print "Volume of test set:", len(input_test) 

	#tranvers the output list for convenience of training
	output_train_trans = []
	for i in range(6):
		output_train_trans.append([])	
		for j in range(len(output_train)):
			output_train_trans[i].append(output_train[j][i])
	if(model=="validation"):
		output_validation_trans = []
		for i in range(6):
			output_validation_trans.append([])	
			for j in range(len(output_validation)):
				output_validation_trans[i].append(output_validation[j][i])
	output_test_trans = []
	for i in range(6):
		output_test_trans.append([])	
		for j in range(len(output_test)):
			output_test_trans[i].append(output_test[j][i])

	if(model=="validation"):
		return input_train, input_validation, input_test, output_train_trans, output_validation_trans, output_test_trans
	else:
		return input_train, input_test, output_train_trans, output_test_trans


