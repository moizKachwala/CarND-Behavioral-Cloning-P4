import csv
import cv2
import numpy as np
import sklearn

def process_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    return image

samples = []
with open('data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  next(reader, None)
  for row in reader:
    steering_center = float(row[3])
    
    # create adjusted steering measurements for the side camera images
    correction = 0.2 # this is a parameter to tune
    steering_left = steering_center + correction
    steering_right = steering_center - correction
    
    # read in images from center, left and right cameras
    samples.append((('./data/'+row[0].strip()),steering_center))
    samples.append((('./data/'+row[1].strip()),steering_left))
    samples.append((('./data/'+row[2].strip()),steering_right))    

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for imgpath, measurement in batch_samples:
                center_image = process_image(imgpath.strip())
                center_angle = float(measurement)
                
                # Add the images and angles
                images.append(center_image)
                angles.append(center_angle)
                
                # Flip the images and angles
                images.append(cv2.flip(center_image, 1))
                angles.append(center_angle*-1.0)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D


model = Sequential()

# Cropping:
# remove top 70, bottom 25px, leave left / right as is
model.add(Cropping2D(cropping=((70, 25), (0, 0)), dim_ordering='tf', input_shape=(160, 320, 3)))

# Normalisation:
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(65, 320, 3), output_shape=(65, 320, 3)))

# Convolutional layers:
model.add(Convolution2D(24,5,5, subsample=(2, 2),input_shape=(160,320,3),  border_mode='valid',activation="relu"))
model.add(Convolution2D(36,5,5, subsample=(2, 2), border_mode='valid',activation="relu"))
model.add(Convolution2D(48,5,5, subsample=(2, 2), border_mode='valid',activation="relu"))

# Dropout layer to prevent overfitting:
model.add(Dropout(0.3))

model.add(Convolution2D(64,3,3, border_mode='valid',activation="relu"))
model.add(Convolution2D(64,3,3, border_mode='valid',activation="relu"))

#Dropout
model.add(Dropout(0.2))

# Flattening:
model.add(Flatten())

# RELU activation layers:
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='tanh'))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= \
            len(train_samples), validation_data=validation_generator, \
            nb_val_samples=len(validation_samples), nb_epoch=2)

model.save('model.h5')
