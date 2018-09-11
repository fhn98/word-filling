
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import nltk
from nltk.corpus import gutenberg
from nltk.tokenize import RegexpTokenizer


# In[3]:


class model (object):
    def __init__ (self , batch_size  , hidden_len , num_steps , num_layers , vocab_size , embd_len):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.hidden_len = hidden_len
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.embd_len = embd_len
        
        self.construct_model()
        
        
    def construct_model(self):#making needed variables and rnn models
        
        self.inputs = tf.placeholder(tf.int32 , [self.batch_size , self.num_steps])#inputs (each cell is the index of a word)
        self.targets = tf.placeholder(tf.int32 , [self.batch_size , ])#outputs (each cell is the index of the target word)
        
        
#         here we use a placeholder to load the embedding vectors in it with feed_dict
#         then we assign values of the placeholder into a variable and use that variable for embedding_lookup
#         blank_vector is a learnable vector used for representing blanks. we attach it to the end of our embedding tensor 

        embedding = tf.Variable (tf.constant (0.0 ,shape=[self.vocab_size , self.embd_len]) , trainable=False , name='embedding')
        self.weights = tf.placeholder(tf.float32 , [self.vocab_size , self.embd_len])
        self.blank_vector = tf.Variable(tf.random_uniform([1,self.embd_len] , -1.0 , 1.0))
        self.embedding_init = embedding.assign(self.weights)
        new_embedding = tf.concat([embedding , self.blank_vector] , axis = 0)
   
        embedded_weights = tf.nn.embedding_lookup(new_embedding , self.inputs)
        
       
    #basic cell followed by dropout. used for making the rnn
        def basic_cell():
            cell = tf.contrib.rnn.LSTMCell(self.hidden_len , forget_bias=1.0)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.8)
            return cell
    
    #making forward and backward layers used in rnn
        forward_cell = tf.contrib.rnn.MultiRNNCell([basic_cell() for _ in range (self.num_layers)])
        backward_cell = tf.contrib.rnn.MultiRNNCell([basic_cell() for _ in range (self.num_layers)])
            
            
    #making bidirectional rnn        
        output , _ = tf.nn.bidirectional_dynamic_rnn(forward_cell , backward_cell , inputs=embedded_weights , dtype=tf.float32)
        
    #getting hidden state of previous forward cell and following backward cell of our target index   
        fw_out = output[0][:,(self.num_steps//2),:]
        bw_out = output[1][:,(self.num_steps//2),:]
        
    #concatenating hidden states followed by a fully connected layer to compute score of each word as a candidate of blank   
        final_out = tf.concat([fw_out , bw_out] , axis=1)
        
        logits = tf.layers.dense(final_out , self.vocab_size)
    #computing softmax cross entropy   
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits , labels=tf.one_hot(indices=self.targets,depth=self.vocab_size , axis=-1))
        
        self.batch_loss = tf.reduce_mean(loss)


# In[4]:


if __name__=='__main__':
    model(100,100,50,3,400000,50)

