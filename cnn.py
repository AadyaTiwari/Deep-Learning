#!/usr/bin/env python
# coding: utf-8

# # Convolutional Neural Network

# ## Importing the Libraries

# In[1]:


import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


# In[2]:


tf.__version__


# ## Part 1 - Data Preprocessing

# Preprocessing the Training Set

# In[5]:


train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range=0.2,
    horizontal_flip = True)
training_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size = (64,64),
    batch_size = 32,
    class_mode = 'binary')


# Preprocessing the Test Set
# 

# In[7]:


test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(64,64),
    batch_size = 32,
    class_mode = 'binary')


# ## Part 2 - Building the CNN

# ###Initialisiing the CNN

# In[8]:


cnn = tf.keras.models.Sequential()


# ### Step 1 - Convolution

# In[10]:


cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu', input_shape = (64,64,3)))


# ###Step 2 - Pooling

# In[11]:


cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))


# Adding a Second Covolutional Layer

# In[12]:


cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))


# ### Step 3 - Flattening

# In[13]:


cnn.add(tf.keras.layers.Flatten())


# ### Step 4 - Full Connection

# In[14]:


cnn.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))


# ### Step 5 - Output Layer

# In[15]:


cnn.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))


# ## Part 3 - Training the CNN
# 

# Compiling the CNN

# In[16]:


cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Training the CNN on the Training Set and Evaluating it on the Test Set

# In[17]:


cnn.fit(x = training_set, validation_data = test_set, epochs = 25)


# ## Part 4 - Making a Single Prediction

# In[19]:


import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg',
                            target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
  prediction = 'dog'
else:
  prediction = 'cat'

print(prediction)

