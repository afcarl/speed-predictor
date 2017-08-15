### Predicting vehicle odometry from monocular video

In this project I attempt to predict a vehicles current speed based on a video taken from a a single camera on a car's front dashboard. The training sequence is 20400 frames shot at 20 fps. Each frame has a speed (m/s) associated with it. The speeds are in a txt file. 

### Images
In addition to the video images, I generated optical flow images from pairs of images using openCVs Farneback dense optical flow function. Farneback optical flow calculates the angle and magnitude of change for each pixel in an image. Conceptually, I thought the angles and magnitudes would be a good predictor of speed given consistent image time steps. If the car is moving quickly, the angles and/or the magnitudes should be larger than in situations where the car is moving slowly. There are inherently some problems with this assumption, namely it assumes everything else in the field of view isn't moving. For instance, if a car is driving slightly slower than the test vehicle, the optical flow magnitude in the pixels near the car will be smaller than pixels in parts of the image that are static. Similarly if a car is driving past in the opposite direction, the optical flow magnitude will be larger for the pixels near the car compared to pixels of static objects. To me this means that although optical flow is an extremely helpful metric for predicting vehicle speed, there may be situations where the flow actually throws the car off. To alleviate some noise between frames, I computed the average optical flow over three frames (two optical flows). Too much averaging causes the optical flow to lose granularity but none results in a fair amount of difference between frames. I employed augmentation on the images before computing optical flow although in practice I'm not sure this accomplishes much because optical flow essentially looks at the changes in the images so augmenting pairs of images won't translate to relative changes between images. 

### Model
I tried a few different models for this project but in the end a simple model worked best. The simplest model, a linear classifier based on the average and standard deviation of pixel values for each channel in the optical flow image results in a validation mse of ~30. At a minimum by NN should beat this value. Of the NNs I tried, a small one worked best. I think this is because the training set is very small so intricate models were very prone to over-fitting. The model I ended up using is created by `create_simple_optical_flow_model`. It consists of 3 CNN layers followed by 3 FCN layers. I found using dropout difficult in this project probably due to the small network and regression task. I ended up relying on some kernel regularization to help with over fitting. Because the model was so small I also found it helpful to employ a small learning rate (0.0001). 

### Results
Looking at the training results `optical_flow_3_augmented_1_200_256_10_30_75_non_recurrent.csv`, you can see that the model's training error varies wildly even during the later stages of training. This is concerning and its possible that the "best" model is actually just a model that is over-fit to the validation set. The model produces a MSE of 3.1 on the entire training dataset. TBD on how it scores on the test dataset. In `labels.ipynb` I explored the error values. It seems like a few large errors make up a majority of the overall error and most values are near 0. Its possible that averaging optical flow over more images would help with this. Its also possible that the model could be further trained to remove these larger values. I looked at whether the mse is correlated to the speeds or the change in speeds and it does not appear that it is correlated with either. Looking at the frequency plot, it seems like disproportionate amount of error occur at speeds above ~26 m/s where we have very few samples. Additionally, by clipping the max error to 20, the mse is 2.84. I think it is reasonable to expect most large errors to go away as we add more data to the model and train longer. 

### Run
This project uses a few python command line scripts to run processes. This allows me to test locally and then run the full process in AWS. To reproduce these results, from the project root, run:
1. `python ./bin/process --video_file train`
2. `python ./bin/process --video_file test`
3. `python ./bin/create_optical_flow_images.py`
4. `python ./bin/create_optical_flow_images.py --is_train_full True`
5. `python ./bin/create_optical_flow_images.py --is_test True`
6. `python ./bin/train_model.py --max_epochs 200 --batch_size 256 --folder optical_flow_3_augmented_1 --min_delta 0.01 --patience 30`
7. `python ./bin/predict/py --video_file train`
8. `python ./bin/predict/py --video_file test`

### Thoughts
In the process of working on this project I've read a few papers on predicting vehicle odometry and most seem to focus on predicting camera pose changes between two frames. Although I haven't attempted this yet, predicting camera pose changes seems like a good approach because speed can be broken down into f(change in camera pose)/change in time.

### Further areas of exploration
- I'm either failing to fit a good model, or significantly over-fitting, is this because so much of the data is from the highway? Maybe more data would help.
- Should I reformat the problem as a classifier? The speeds are between 0 and ~25 so 100 classes would get down to very decent accuracy. This might allow for more dropout.
- This is a video, does a stateful recurrent neural network make sense?
- Normally adding dropout helps with over fitting but in this regression problem, it felt like dropout caused the model to never train past a certain point and even caused the test error it swing wildly at times. Is there a better way to prevent over-fitting for regression problems?

