
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import pickle
import import_ipynb

import word_filling_batch_generator1
from word_filling_batch_generator1 import get_batch
from word_filling_attempt1 import model
from word_filling_test import run_test


# In[2]:


def train (file , batch_size , num_steps , epochs , embedding_array , itos , stoi,
           learning_rate , lr_decay , hidden_len , num_layers , savepath , num_samples ):
    """
    description: training out word filling model.
    
    inputs: file: training file
            vectors: an array with each row as the vector representation of its corresponding index
            itos: a list with index of each word as its index in GloVe
            stoi: a dictionary from each word to its index in GloVe
            lr_decay: after each 10000 steps, learning_rate is reassigned to learning_rate*learning_decay
            save_path: the path in which trained model is going to be saved
            num_samples: number of samples in nce loss
            
    outputs: savepath: address of the saved trained model
    
    """
    
    #loading vectors of the pretrained embedding
    with open (embedding_array , 'rb') as pkl:
        embd = pickle.load(pkl)
        
    with open (itos , 'rb') as pkl:
        itos_list = pickle.load(pkl)
        
    
    #making an instance of our word_filling model
    with tf.device('/cpu:0'):
        my_model = model (batch_size=batch_size , embd_len=embd.shape[1] , hidden_len=hidden_len ,num_layers=num_layers , num_steps=num_steps , vocab_size=embd.shape[0] , train_mode=True ,keep_prob= 0.8 , num_samples = num_samples)
    
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess :
        #used for updating learning_rate
        tmp=learning_rate
        lr = tf.Variable(0.0, trainable=False)
        lr_new_value = tf.placeholder(tf.float32 ,[])
        lr_update=tf.assign(lr,lr_new_value)
        
        #clipping gradients by norm 5
        global_step = tf.Variable(0 , trainable=False)
        params = tf.trainable_variables()
        clipped = [tf.clip_by_norm(g,5) for g in tf.gradients(my_model.batch_loss , params)]
        
        #declaring the optimizer
        optimizer = tf.train.AdamOptimizer(lr)
        train_opt = optimizer.apply_gradients(zip(clipped , params) , global_step=global_step)

        #train_opt = tf.train.AdamOptimizer(lr).minimize(my_model.batch_loss)
        
        #initializing global variables
        sess.run(tf.global_variables_initializer())
        #initializing embedding tensor in our model with loaded pretrained vectors
        sess.run(my_model.embedding_init , feed_dict={my_model.weights: embd})
        
        
        #using batch generator defined in word_filling_batch_generator1
        generated_batch=get_batch(file , batch_size , num_steps , epochs , embd.shape[0])
        
        step=0
      
        for input_batch ,target_batch , ratio in generated_batch:
            _,batch_loss = sess.run([train_opt , my_model.batch_loss] , 
                                    feed_dict={my_model.inputs:input_batch , my_model.targets:target_batch , my_model.loss_ratio:ratio})
            
            if step%10==0:
                print (str(step)+ " train loss: " +':'+str(batch_loss))

            if step%4000==0:
                tmp=tmp*lr_decay
                sess.run(lr_update , feed_dict={lr_new_value:tmp})
                
            step+=1
        
        saver.save(sess, './'+savepath+'.ckpt')
        print("Model saved in file: %s" % savepath) 
        
        return './'+savepath+'.ckpt'
            


# In[3]:


if __name__=='__main__':
    with open ('train_by_index.pkl' , 'rb') as pkl:
        file = pickle.load(pkl)
    
    savepath=train(file=file,batch_size=64,num_steps=101,epochs=2,embedding_array='vectors.pkl',itos='itos.pkl',stoi='stoi.pkl'
          , learning_rate =0.02  , lr_decay=0.8 , hidden_len = 200 ,num_layers=1 , savepath = 'model10' , num_samples=40)

