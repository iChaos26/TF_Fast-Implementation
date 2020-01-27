#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
tf.test.gpu_device_name()


# In[15]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))


# In[16]:


"Setup for testing a simple regression model evaluation and performance"
X = np.random.rand(100).astype(np.float32)
Y = np.vectorize(lambda y: y + np.random.normal(loc=0.0, scale=0.05))(Y)


# In[17]:


class Model(object):
    """
    A model class to train a simple regression and evaluate his performance
    :param loss: Loss function to minimize the squared error
    :param train: Train method 
    """
    def __init__(self, x, y):
        self.a = tf.Variable(1.0)
        self.b = tf.Variable(1.0)
    def __call__(self, X):
        return self.a*X + self.b
    


# In[18]:


def loss(pred_y, real_y):
    return tf.reduce_mean(tf.square(pred_y - real_y))   


# In[19]:


"Gradient descent to minimize too"
opt = tf.keras.optimizers.SGD(learning_rate=0.5)


# In[24]:


def train(model,inputs, outputs):
    with tf.GradientTape() as t:
        current_loss = loss(model(inputs), outputs)
    grads = t.gradient(current_loss, [model.a, model.b])
    optimizer.apply_gradients(zip(grads,[model.a, model.b]))
    print(current_loss)


# In[25]:


model = Model(X,Y)


# In[26]:


"1000 iterations for the train method"
for i in range(1000):
    train(model,X,Y)


# In[27]:


print(model.b.numpy())

