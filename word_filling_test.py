
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import pickle
import import_ipynb
from word_filling_attempt1 import model
from word_filling_test_batch_generator import get_data


# In[2]:


def run_test(itos , realfile , blankedfile , blanked_indexfile  , num_steps , blank_index , modelpath , batch_size 
             , vocab_size , embd_len , hidden_len , num_layers):
       
       with open (itos , 'rb') as pkl:
           itos_list = pickle.load(pkl)
           
       my_model = model (batch_size=batch_size , embd_len=embd_len , hidden_len=hidden_len , 
                         num_layers=num_layers , num_steps=num_steps , vocab_size=vocab_size , train_mode= False , keep_prob=1 , num_samples=50)
       
           
       saver = tf.train.Saver()
           
       itos_list.append('<blank>')
          
       test_data = get_data(realfile , blankedfile , blanked_indexfile , num_steps , blank_index , batch_size)
       
       zero = np.ones ([batch_size , 1 ], dtype=np.float32)
       f = open('test10.txt' , 'w')
       with tf.Session() as sess:
           saver.restore(sess, modelpath)
           total_loss=0
           count=0
           for context , target in test_data:
               Loss,answers = sess.run([my_model.batch_loss,my_model.answers] , feed_dict={my_model.inputs:context , my_model.targets:target , my_model.loss_ratio:zero})
               total_loss+=Loss
               count+=1
               for i in range (batch_size):
                   input_list=[itos_list[word] for word in context[i]]
                   print (input_list[:(num_steps//2)+1])
                   f.write(str(input_list[:(num_steps//2)+1])+'\n')
                   print (input_list[(num_steps//2)+1:])
                   f.write(str(input_list[(num_steps//2)+1:])+'\n')
                   print ('right answer:'+itos_list[target[i,0]])
                   f.write('right answer:'+itos_list[target[i,0]]+'\n')
                   print ('computer\'s answer:'+itos_list[answers[i]])
                   f.write('computer\'s answer:'+itos_list[answers[i]]+'\n')
               
           print ('test perplexity:'+str(np.exp((total_loss/count))))
           f.write('test perplexity:'+str(np.exp((total_loss/count)))+'\n')
       f.close()
           


# In[3]:


if __name__=='__main__':
    run_test('itos.pkl','real_test.pkl' , 'blanked_test.pkl' , 'blanked_index.pkl', num_steps=101 , blank_index=407371 ,modelpath= './model10.ckpt' , batch_size=64 , vocab_size=407371, embd_len=50 , hidden_len=200 , num_layers=1)

