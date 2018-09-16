
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import pickle


# In[2]:


def load_file(filename):
    with open (filename , 'rb') as pkl:
        file = pickle.load(pkl)

    return file


# In[3]:


#reshape corpus into an array of size batch_size * batch_len.
def make_array(file , batch_size):
"""
Description of function:
...
input:
...
output:
...
"""
    file_array = np.array(file)

    print(len(file))
    print(file_array.shape)

    batch_len = len(file)//batch_size

    reshaped_array = (file_array[:batch_len*batch_size]).reshape(batch_size , batch_len)

    return reshaped_array , batch_len


# In[5]:


def get_batch(filename , batch_size , num_steps , epochs , vocab_size):
"""
Description of function:
...
input:
...
output:
...
"""
# extra prints
    print (vocab_size)
    file = load_file(filename)

    arr , batch_len = make_array(file , batch_size)
    print (arr.shape)

    #for each batch, we use arr[:,window_size].

    for epoch in range (epochs):
        for time_step in range (batch_len-num_steps+1):
            Xs = np.copy(arr[:,time_step:time_step+num_steps])
            Xs[:,num_steps//2] = vocab_size
            yield Xs , arr[:,time_step+num_steps//2]
