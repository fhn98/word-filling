
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import pickle


# In[3]:


#reshape corpus into an array of size batch_size * batch_len.
def make_array(file , batch_size): 
    file_array = np.array(file)
    
    batch_len = len(file)//batch_size
    
    reshaped_array = (file_array[:batch_len*batch_size]).reshape(batch_size , batch_len)
    
    return reshaped_array , batch_len


# In[5]:


def get_batch(file , batch_size , num_steps , epochs , vocab_size): 
    """
    description: generating training batches for word filling
    
    inputs: file: training file
            num_steps: size of context window
            vocab_size: size of GloVe
            
    outputs: Xs: inputs [batch_size X num_steps]
             Ys: targets [batch_size X 1]
             ratio: a [batch_size X 1] array with each cell showing how much we want our target have impact on gradient decent.
                    ',' , '.' have value of 0.6. 'the' , 'and' , 'of' , 'to' have 0.8.
    """
    arr , batch_len = make_array(file , batch_size)
    
    #for each batch, we use arr[:,window_size].
    
    for epoch in range (epochs):
        for time_step in range (batch_len-num_steps+1):
            
            Xs = np.copy(arr[:,time_step:time_step+num_steps])
            Xs[:,num_steps//2] = vocab_size
            Ys = arr[:,time_step+num_steps//2]
            
            ratio = np.ones([batch_size,1] , dtype=np.float32)
            for i in range (batch_size):
                if (Ys[i]==1 or Ys[i]==2):
                    ratio[i,0]=0.6
                elif (Ys[i]==0 or Ys[i]==3 or Ys[i]==4 or Ys[i]==5):
                    ratio[i,0]=0.8
    
            yield Xs , Ys.reshape(-1,1) , ratio

