# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/orig_image1.jpg "Original Image 1"
[image2]: ./writeup_images/aug_image1.jpg "Augmented Image 1"
[image3]: ./writeup_images/orig_image2.jpg "Original Image 2"
[image4]: ./writeup_images/aug_image2.jpg "Augmented Image 2"
[image5]: ./writeup_images/orig_image3.jpg "Original Image 3"
[image6]: ./writeup_images/aug_image3.jpg "Augmented Image 3"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a Nvidia model convolution neural network. (model.py lines 66-91) 

The model includes RELU layers to introduce nonlinearity (code line 94), and the data is normalized in the model using a Keras lambda layer (code line 74). 

The next three are fully connected layers with 100, 50, 10 units and finally the output with one unit because we are in a regression problem.

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 82 and 88). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The training-validation data was spilt as 80-20%.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 100).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the data provided by Udacity. I used some correction for steering angles and flipped images to create more versatile data for training

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use nVidia model for CNN.

It is a simple yet effective model. In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I tried introducing lambda layer and cropping, but there were errors while creating model. I decided to try simplest approach without extra layers.

I tried with epoch of 3 but when the simulator ran autonomously, it worked well for most of the lap but at one point it went off track when the dusty road came on the right. I then trained with epoch = 7 and this time it worked well. Validation loss kept on decreasing with the epochs, which was a good sign.

The final step was to run the simulator to see how well the car was driving around track one. It successfully completed whole lap, I let it run little more just to ensure it didn't behave differently further down the line.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 66-98) consisted of a convolution neural network with the following layers and layer sizes

1. Convolution2D with kernel (5,5) , activation: relu, strides(2,2), filters:24
2. Convolution2D with kernel (5,5) , activation: relu, strides(2,2), filters:36
3. Convolution2D with kernel (5,5) , activation: relu, strides(2,2), filters:48
4. Dropout with 0.3
5. Convolution2D with kernel (3,3) , activation: relu, strides(1,1), filters:64
6. Convolution2D with kernel (3,3) , activation: relu, strides(1,1), filters:64
7. Dropout with 0.2
8. Flatten
9. Dense with output:100
10. Dropout with 0.3
11. Dense with output:50
12. Dense with output:10
13. Dense with output:1

#### 3. Creation of the Training Set & Training Process

I used the data provided by Udacity, which really helped me.

To augment the data set, I also flipped images and corrected steering angles

#### original
![alt text][image1]

#### Augmented
![alt text][image2]

#### original
![alt text][image3]

#### Augmented
![alt text][image4]

#### original
![alt text][image5]

#### Augmented
![alt text][image6]


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7 I used an adam optimizer so that manually training the learning rate wasn't necessary.
