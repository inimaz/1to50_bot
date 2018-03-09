# Load the libraries
import os
import cv2
import pyautogui
import mss
import numpy as np
import mss.tools
import matplotlib.pyplot as plt
from PIL import Image
from time import sleep
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.models import load_model


# TRAINING
#-------------

def load_images_from_dir(folder):
	"""
	Read and store every image in the provided directory
	"""
	# dict to hold the individual files
	img_data = {}
	# for each file in the directory,
	for img_file in os.listdir(folder):
		# read the file
		img = cv2.imread(folder+img_file)
		# use the file name with extension removed as the key
		img_data[img_file[0:img_file.index(".")]] = img
	# return the collection of keys:image
	return img_data

def preprocessing_image_files(dict_data):
	"""
	Performs following steps,
		1. Extract the individula files, 
		2. Perform preprocessing for neural network training
		3. Store them as numpy arrays for further processing
	"""
	# variable to hold image data
	data = np.array([])
	# variable to hold labels 
	label = np.array([])
	# for all images add them into array
	for key in dict_data.keys():
		# load data
		img = dict_data[key]
		# convert it into gray scale from BGR format (cv2 loads as BGV, right!)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		# convert it into RGB
		img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		# apply threshold to simple out training process
		# anything more than 150 is 1 rest 0
		ret_, img = cv2.threshold(img, 150, 1, cv2.THRESH_BINARY)
		# update the variables
		data = np.append(data, img)
		label = np.append(label, key)
	# reshape the image data
	data = data.reshape(len(dict_data), 71, 71).astype("float32")
	# normalize the input to 0 or 1
	data =  data / 255
	# processing done, return the data and label
	return data, label

def train_network(image_directory):
	"""
	Perform the following actions,
		1. Reads data, do all of the preprocessing
		2. Train the neural network on th e data and return it
	"""
	# status
	print("Loading images...")
	# load all the images
	train_imgs = load_images_from_dir(image_directory)
	# status
	print("Performing preprocessing...")
	# prepare the file for network training, 
	training_img, training_label = preprocessing_image_files(train_imgs)

	# now comes the hacking, creating duplicates
	duplicate_count = 400 
	training_img = np.tile(training_img, (duplicate_count,1,1))
	training_label = np.tile(training_label, duplicate_count)

	# reshape the images;make them 71*71 vector for input to network
	num_pixels = training_img.shape[1] * training_img.shape[2]
	training_img = training_img.reshape(training_img.shape[0], num_pixels).astype('float32')

	# one hot encode outputs
	training_label = np_utils.to_categorical(training_label)
	num_classes = training_label.shape[1]
	# status
	print("Creating neural network...")
	# create neural network
	model = create_simple_neural_network(num_pixels, num_pixels, num_classes)

	# create a tuple of training data
	training_data = (training_img, training_label)
	# status
	print("Training on the images...")
	# now start fitting of neural network 
	model = fit_neural_network(model, training_data, training_data)

	# return the trained model
	return model


def create_simple_neural_network(input_layer_len, hidden_layer_len, output_layer_len):
	# create model
	model = Sequential()
	# add input and hidden layer 
	model.add(Dense(hidden_layer_len, input_dim=input_layer_len, kernel_initializer='normal', activation='relu'))
	# add output layer
	model.add(Dense(output_layer_len, kernel_initializer='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# return the network
	return model

def fit_neural_network(model, train_data, validation_data):
	"""
	"""
	# Fit the model
	model.fit(train_data[0], train_data[1], validation_data=validation_data, epochs=5, batch_size=750, verbose=2, shuffle = False)
	# Final evaluation of the model
	scores = model.evaluate(validation_data[0], validation_data[1], verbose=0)
	print("Error: %.2f%%" % (100 - (scores[1] * 100))) 
	# return the trained model
	return model


def save_model(model, file_path):
	"""
	Save the model as file
	"""
	model.save(file_path)

# PLAYING
#-------------

# now calculate the position of each number based on the image
def get_number_boxs_position(image_location):
	"""
	Take screenshot of the complete 1to50 board, 
	calculate the x-y co-ordinates of all the numbers present in it.
	"""
	# Take the screen shot of the specified 1to50 board
	sct = mss.mss()
	im = np.array(sct.grab(image_location))
	# status
	print("Screenshot taken, performing preprocessing...")

	# now perform preprocessing before using model to predict the identify the numbers
	im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

	# Convert to grayscale and apply Gaussian filtering
	im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

	# thresholding
	ret, im = cv2.threshold(im, 150, 255, cv2.THRESH_BINARY_INV)

	# Find contours in the image
	_, ctrs, hier = cv2.findContours(im.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	#  status
	print("Trying to identify individual numbers..")
	# variable to hold all the numbers and their positions
	position_dict = {}
	# for each bounding rect, 
	# 	1. extract individual image with number
	#	2. identify the number
	# 	3. store the co-ordinates of the number
	for contour in ctrs:
		# get box's bunding co-ordinates
		x, y, w, h = cv2.boundingRect(contour)
		# extract the image
		number_img = im[y:y+h, x:x+w]
		ret, number_img = cv2.threshold(number_img ,150, 255, cv2.THRESH_BINARY_INV)
		# resize the image
		number_img = cv2.resize(number_img, (71, 71)).reshape(1, 5041)
		number_img = number_img/255
		# identify the image
		number_val = model.predict_classes(number_img)
		# update the position_dict; calculations done to hit the center of the number box
		position_dict[number_val[0]] = ( x+(w/2)+image_location['left'] , y+(h/2)+image_location['top'])
	#status
	print("Whoos..enough learning for today...")
	# return the position dict
	return position_dict


# function to perform click event on the provided position_dict
def perform_mouse_click_event(position_dict):
	"""
	Provided position dict, perform the click event.
	"""
	# status
	print("I'm ready...let the game begin...")
	# parse through the position_dict
	for num in sorted(position_dict.keys()):
		# extract the co-ordinates
		x, y = position_dict[num]
		# status
		print("Going for number ", num, " at x:", x, " y: ", y)
		# move the curser and click
		pyautogui.moveTo(x, y)
		pyautogui.click(x,y)

def play_game(board_location):
	"""
	The main function, perform the following steps,
		1. Take screenshot and calculate the number's positions
		2. Init the mouse move and click event
		3. Init the whole process twice, as 26-50 numbers are shown after 1-25 are complete
	"""
	# RUN 1: For 1 to 25
	#-----------------------
	# first, take screenshot and perform number identification using neural network aloong with number position calculation
	position_dict = get_number_boxs_position(board_location)
	# second, init mouse drag and click event for all the numbers
	perform_mouse_click_event(position_dict)

	# move the mouse out of the way
	pyautogui.moveTo(100, 100)

	# RUN 2: For 25 to 50
	#-----------------------
	# first, take screenshot and perform number identification using neural network aloong with number position calculation
	position_dict = get_number_boxs_position(board_location)
	# second, init mouse drag and click event for all the numbers
	perform_mouse_click_event(position_dict)

# if __name__ == '__main__':
	# start
	# model = train_network(image_directory = "data/numbers separated/")
	# save_model(model, "new.h5")
	# model = load_model("new.h5")
	# # Define variables
	# image_location = {'top': 239, 'left': 486, 'width': 380, 'height': 378}
	# play_game(image_location)

