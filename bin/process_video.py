#!/bin/python 

import os
import sys

project_path, x = os.path.split(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(project_path)

import tensorflow as tf
import skvideo.io
import cv2 
from mach.util import full_path

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('video_file', 'train', 'The name of the video file in video_data to turn into images.')
flags.DEFINE_integer('frame_count', None, 'The number of frames to convert')

def main(_):
	full_video_file = full_path("video_data/{}.mp4".format(FLAGS.video_file))
	vid = skvideo.io.vreader(full_video_file)

	folder_path = full_path("data/orig")
	image_folder_path = "{}/{}".format(folder_path, FLAGS.video_file)

	if os.path.exists(folder_path) == False:
		os.makedirs(folder_path)

	if os.path.exists(image_folder_path) == False:
		os.makedirs(image_folder_path)

	print("Preparing to process", full_video_file)
	for i, image in enumerate(vid):
	  file_out_path = "{}/frame_{}.jpg".format(image_folder_path, i)
	  cv2.imwrite(file_out_path, image) # save frame as JPEG file
	  
	  if i % 100 == 0:
	  	print("Done saving image", file_out_path)

	  if FLAGS.frame_count is not None and i >= FLAGS.frame_count:
	  	break

	print("Done processing images.")

if __name__ == '__main__':
	tf.app.run()
