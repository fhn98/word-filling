
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np


# In[3]:


class model (object):
    def __init__ (self , batch_size  , hidden_len , num_steps , num_layers , vocab_size , embd_len , train_mode , keep_prob):
        """
        description: initializing class fields and then calling construct_model for constructing the model_graph

        inputs:

        outputs:
        """
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.hidden_len = hidden_len
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.embd_len = embd_len
        self.train_mode=train_mode
        self.keep_prob=keep_prob
        
        self.construct_model()
        
        
    def construct_model(self):
        """
        description: constructing our word filling model. context is passed through a bidirectional rnn. then output of 
                     blank index is passed through a fully connected layer. after that nce loss is computed and finally loss is multiplied by ratio.
        inputs:
        outputs:
        """
        
        self.inputs = tf.placeholder(tf.int32 , [None , None])#inputs (each cell is the index of a word)
        self.targets = tf.placeholder(tf.int32 , [None ,1 ])#outputs (each cell is the index of the target word)
        self.loss_ratio = tf.placeholder(tf.float32 , [None,1])
        
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
            if (self.train_mode==True):
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
            return cell
    
    #making forward and backward layers used in rnn
        forward_cell = [basic_cell() for _ in range (self.num_layers)]
        backward_cell = [basic_cell() for _ in range (self.num_layers)]
            
            
    #making bidirectional rnn        
        outputs , _  , _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=forward_cell ,cells_bw= backward_cell , inputs=embedded_weights , dtype=tf.float32)
        
    #getting hidden state of the target index  
      
        final_out = outputs[:,(self.num_steps//2),:]
        
        logits = tf.layers.dense(final_out , self.vocab_size)
        self.answers = tf.argmax(logits , axis=1) 
        
    #multiplying loss and ratio   
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits , labels=tf.one_hot(indices=self.targets,depth=self.vocab_size , axis=-1))
        
        self.batch_loss = tf.reduce_mean(loss)
