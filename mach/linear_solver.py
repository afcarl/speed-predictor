#!/bin/python

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from data import img_from_file
import cv2

def test_train_data(folder_path):
	df = pd.read_csv("{}/files_labels.csv".format(folder_path))

	mean_stds = list(map(mean_std, df['flow_path'].values))

	train_mean_std, valid_mean_std, train_labels, valid_labels = train_test_split(mean_stds, df['speed'].values, test_size=0.30, random_state=42)
	scaler = MinMaxScaler()
	scaler.fit(train_mean_std)
	train_mean_std = scaler.transform(train_mean_std)
	valid_mean_std = scaler.transform(valid_mean_std)

	return train_mean_std, train_labels, valid_mean_std, valid_labels, scaler


def mean_std(image_file):
	to_hsv = lambda img: cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
	hsv_image = np.array(to_hsv(img_from_file(image_file)))
	image_mean = np.mean(hsv_image, (0,1))
	image_std = np.std(hsv_image, (0,1))
	return np.concatenate((image_mean, image_std))

def fit(folder_path="data/optical_flow_recurrent"):
	train_X, train_labels, valid_X, valid_labels, scaler = test_train_data(folder_path)

	lr = LinearRegression()
	lr.fit(train_X, train_labels)

	return np.mean((lr.predict(valid_X) - valid_labels)**2)

print(fit())
