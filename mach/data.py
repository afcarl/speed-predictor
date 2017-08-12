#!/bin/python

import glob
import cv2
import numpy as np
import pandas as pd
from keras.applications.mobilenet import preprocess_input as mobilenet_preprocess_input

from mach.util import full_path


def raw_train_data():
	image_path = full_path("data/orig/train/*.jpg")
	image_files = glob.glob(image_path)
	label_path = "video_data/train.txt"
	train_labels = read_txt_file(label_path)[0:len(image_files)]

	return (image_files, train_labels)

def read_txt_file(file_path):
	with open(full_path(file_path), "r") as f:
		return list(map(float, f))

def img_from_file(file):
  img = cv2.imread(file)
  return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

### Augmentation functions
CROP_SIZE = (70,390,0,640)
IMAGE_SIZE = (320, 160) # divisable by 32 for MobileNet

def preprocess_valid_images(images):
	fn = lambda i: preprocess_valid_image(i, CROP_SIZE, IMAGE_SIZE)
	return list(map(fn, images))

def preprocess_valid_image(image, crop_size, image_size):	
	image = crop_image(image,crop_size)
	return cv2.resize(image, image_size)

# augment a group of images with the same values
def augment_images(images):
	brightness_min = 0.7
	brightness_max = 1.25
	brightness = np.random.uniform(brightness_min, brightness_max)
	translation_x_min = -30
	translation_x_max = 30
	translation_y_min = -30
	translation_y_max = 30
	translation_x = np.random.uniform(translation_x_min, translation_x_max)
	translation_y = np.random.uniform(translation_y_min, translation_y_max)
	scale_x1_min = 0
	scale_x2_min = 0
	scale_y1_min = 0
	scale_y2_min = 0
	scale_x1_max = 20
	scale_x2_max = 20
	scale_y1_max = 20
	scale_y2_max = 20
	scale_x1 = np.random.uniform(scale_x1_min, scale_x1_max)
	scale_x2 = np.random.uniform(scale_x2_min, scale_x2_max)
	scale_y1 = np.random.uniform(scale_y1_min, scale_y1_max)
	scale_y2 = np.random.uniform(scale_y2_min, scale_y2_max)

	fn = lambda image: augment_image(image, CROP_SIZE, IMAGE_SIZE, brightness, translation_x, translation_y, scale_x1, scale_x2, scale_y1, scale_y2)

	return list(map(fn, images))

def augment_image(image, crop_size, image_size, brightness, translation_x, translation_y, scale_x1, scale_x2, scale_y1, scale_y2):
	image = crop_image(image,crop_size)
	image = augment_image_brightness(image, brightness)
	image = translate_image(image, translation_x, translation_y)
	image = stretch_image(image, scale_x1, scale_x2, scale_y1, scale_y2)
	return cv2.resize(image, image_size)

def crop_image(image, crop_size):
	x1, x2, y1, y2 = crop_size
	return image[x1:x2, y1:y2]

def augment_image_brightness(image, brightness):
    ### Augment brightness
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1[:,:,2] = image1[:,:,2]*brightness
    return cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)

def translate_image(image, translation_x, translation_y):
    # Translation augmentation
    Trans_M = np.float32([[1,0,translation_x],[0,1,translation_y]])
    rows,cols,channels = image.shape
    return cv2.warpAffine(image,Trans_M,(cols,rows))

def stretch_image(image,scale_x1,scale_x2,scale_y1,scale_y2):
    # Stretching augmentation
    p1 = (scale_x1,scale_y1)
    p2 = (image.shape[1]-scale_x2,scale_y1)
    p3 = (image.shape[1]-scale_x2,image.shape[0]-scale_y2)
    p4 = (scale_x1,image.shape[0]-scale_y2)

    pts1 = np.float32([[p1[0],p1[1]],
                   [p2[0],p2[1]],
                   [p3[0],p3[1]],
                   [p4[0],p4[1]]])
    pts2 = np.float32([[0,0],
                   [image.shape[1],0],
                   [image.shape[1],image.shape[0]],
                   [0,image.shape[0]]])

    M = cv2.getPerspectiveTransform(pts1,pts2)
    image = cv2.warpPerspective(image,M,(image.shape[1],image.shape[0]))
    return np.array(image,dtype=np.uint8)


def average_optical_flow_dense(images):
	"""
	input: images (RGB images) assumed to be in time sequence order
	* calculates optical flow magnitude and angle and places it into HSV image
	* Set the saturation to the saturation value of image_next
	* Set the hue to the angles returned from computing the flow params
	* set the value to the magnitude returned from computing the flow params
	* Convert from HSV to RGB and return RGB image with same size as original image
	"""
	assert(len(images) > 1)

	to_gray = lambda image: cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	gray_images = list(map(to_gray, images))
	result = []

	for i in range(len(gray_images) - 1):
		old = gray_images[i]
		new = gray_images[i+1]
		result.append(single_optical_flow_dense(old, new))

	return np.mean(result, axis=0)

def single_optical_flow_dense(old_image_gray, current_image_gray):
	"""
	input: old_image_gray, current_image_gray (gray images)
	* calculates optical flow magnitude and angle and places it into HSV image
	* Set the hue to the angles returned from computing the flow params
	* set the value to the magnitude returned from computing the flow params
	* Convert from HSV to RGB and return RGB image with same size as original image
	"""
  # set saturation
	out_shape = list(old_image_gray.shape) + [3]
	hsv_flow = np.zeros(out_shape)
	hsv_flow[...,1] = 255

	# Flow Parameters
	# flow = None
	pyr_scale = 0.5
	levels = 1
	winsize = 12
	iterations = 2
	poly_n = 5
	poly_sigma = 1.2
	extra = 0
	# obtain dense optical flow paramters
	# https://github.com/npinto/opencv/blob/master/samples/python2/opt_flow.py
	# http://docs.opencv.org/trunk/d7/d8b/tutorial_py_lucas_kanade.html
	flow = cv2.calcOpticalFlowFarneback(old_image_gray, current_image_gray,  
																			None, #flow_mat 
																			pyr_scale, 
																			levels, 
																			winsize, 
																			iterations, 
																			poly_n, 
																			poly_sigma, 
																			extra)

	# convert from cartesian to polar
	mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
	# hue corresponds to direction
	hsv_flow[...,0] = ang*180/np.pi/2
	# value corresponds to magnitude
	hsv_flow[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
	# convert HSV to int32's
	hsv_flow = np.asarray(hsv_flow, dtype= np.float32)
	# convert to RGB
	rgb = cv2.cvtColor(hsv_flow,cv2.COLOR_HSV2RGB)
	
	return rgb


def create_optical_flow_data(num_images, num_augmentations, file_path, valid_pct=0.4):
	train_files, train_labels = raw_train_data()
	end = ((len(train_labels) // num_images - 1) * num_images)
	train_results = []
	valid_results = []
	print("Processing a total of {} images.".format(len(train_labels)))

	for i in range(0, end, num_images):
		speed = train_labels[i]
		if i % 100 == 0: print("Finished with {} original images.".format(i))

		if np.random.uniform() < valid_pct:
			image = average_optical_flow_dense(list(map(img_from_file, train_files[i:i+num_images])))
			img_file_path = "{}/valid/frame_{}_{}.jpg".format(file_path, i, i+num_images)
			cv2.imwrite(img_file_path, image)
			valid_results.append((img_file_path, speed))

		else:
			for j in range(num_augmentations):
				image = average_optical_flow_dense(list(augment_images(map(img_from_file, train_files[i:i+num_images]))))
				img_file_path = "{}/train/frame_{}_{}_aug_{}.jpg".format(file_path, i, i+num_images, j)
				cv2.imwrite(img_file_path, image)
				train_results.append((img_file_path, speed))

	pd.DataFrame(train_results, columns=["file_path", "speed"]).to_csv("{}/train/files_labels.csv".format(file_path))
	pd.DataFrame(valid_results, columns=["file_path", "speed"]).to_csv("{}/valid/files_labels.csv".format(file_path))


def create_mobilenet_generators(folder_path, batch_size, is_debug):
	train = create_mobilenet_generator("{}/train".format(folder_path), batch_size, is_debug)
	valid = create_mobilenet_generator("{}/valid".format(folder_path), batch_size, is_debug)

	return (train, valid, image_shape(folder_path, mobilenet_preprocessor))

def image_shape(folder_path, preprocessor):
	image_file = pd.read_csv("{}/train/files_labels.csv".format(folder_path))['file_path'][0]
	image = img_from_file(image_file)
	return list(list(preprocessor([image], [0.]))[0])[0].shape

def create_mobilenet_generator(folder_path, batch_size, is_debug):
	df = pd.read_csv("{}/files_labels.csv".format(folder_path))
	files = df['file_path']
	labels = df['speed']
	if is_debug:
		end = min(len(files), 200)
		files = files[0:end]
		labels = labels[0:end]
	g = ImageFileGenerator(batch_size, files, labels, mobilenet_preprocessor)
	return (len(files) // batch_size , g)

def mobilenet_preprocessor(images, labels):
	map_images = map(mobilenet_preprocess_input, map(as_(np.float32), images))
	print('********mobilenet_preprocessor**********')
	return (list(map_images), labels)

def as_(dtype):
	return lambda image: np.array(image, dtype=dtype)
	

class ImageFileGenerator():
	def __init__(self, batch_size, image_files, labels, preprocessor):
		self.batch_size = batch_size
		self.image_files = image_files
		self.labels = labels
		self.index = 0
		self.preprocessor = preprocessor

	def _increment_index(self, end):
		if end == len(self.labels):
			self.index = 0
		else:
			self.index += self.batch_size

	def __next__(self):
		return self.next()

	def next(self):
		start = self.index
		end = min(self.index+self.batch_size, len(self.labels))
		batch_images = map(img_from_file, self.image_files[start:end])
		batch_labels = self.labels[start:end]

		self._increment_index(end)

		return self.preprocessor(batch_images, batch_labels) 
