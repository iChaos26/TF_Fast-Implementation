#!/usr/bin/env python
# coding: utf-8

# In[4]:


import tensorflow as tf
tf.test.gpu_device_name()


# In[7]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
tf.__version__


# In[17]:


X = np.random.rand(100).astype(np.float32)
a = 50
b = 40
Y = a * X + b
Y = np.vectorize(lambda y: y + np.random.normal(loc=0.0, scale=0.05))(Y)
a_var = tf.Variable(1.0)
b_var = tf.Variable(1.0)
y_var = a_var * X + b_var


# In[27]:


def tf_loss(y_var, Y):
    loss = tf.reduce_mean(tf.square(y_var - Y))
    return loss


# In[35]:


def tf_optimizer():
    optimizer = tf.optimizers.SGD ()
    return optimizer


# In[41]:


train = tf_optimizer().minimize(tf_loss(y_var, Y), var_list=None)


# In[ ]:




