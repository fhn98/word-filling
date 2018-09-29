
# coding: utf-8

# In[38]:


import numpy as np
import tensorflow as tf
import random
import pickle


# In[39]:


def convert (book , stoi , itos , vectors):
    book_by_index=[]
    i = len(stoi)
    
    for word in book:
        if word.lower() in stoi:#if the word already exists in out vocabulary
            book_by_index.append(stoi[word.lower()])
        else:
            stoi[word.lower()]=i
            itos.append(word.lower())
            new = np.random.rand(1,vectors.shape[1])
            vectors = np.concatenate((vectors , new) , axis = 0)#adding a random as the embedding of the new word to vectors
            book_by_index.append(i)
            i+=1
            
    return book_by_index , stoi , itos , vectors


# In[40]:


def make_blanks(book , blank_index):
    blanked=[]
    blanked_index=[]
    for i , word in enumerate(book):
        rand = random.random()
        
        if rand<=0.05:
            blanked.append (blank_index)
            blanked_index.append(i)
        else:
            blanked.append (word)
            
    return blanked , blanked_index
            
    


# In[41]:


def make_corpus(book_id , stoi , itos , vectors , blank_index):
    book = gutenberg.words(book_id)
    
    book_by_index , stoi , itos , vectors = convert(book , stoi , itos , vectors)
    
    blanked  , blanked_index= make_blanks(book_by_index , blank_index)
    
    return book_by_index , blanked , blanked_index


# In[42]:


def get_one(realfile , blankedfile , blanked_indexfile , num_steps , blank_index):
    with open (realfile , 'rb') as pkl:#stoi : a dictionary from word to its index in vectors
        real = pickle.load(pkl)

    with open (blankedfile , 'rb') as pkl:#index of each word in the list represents its index in vectors
        blanked = pickle.load(pkl)

    with open (blanked_indexfile , 'rb') as pkl:#each row is the word embedding of the word with corresponding index
        blanked_index = pickle.load(pkl)
        
    for index in blanked_index:
        if (index >(num_steps//2) and index<len(real)-(num_steps//2)-1):
            yield np.array (blanked[index-(num_steps//2):index+(num_steps//2)+1]) , np.array([real[index]])


# In[43]:


def get_data(realfile , blankedfile , blanked_indexfile , num_steps , blank_index , batch_size):
    x = get_one(realfile , blankedfile , blanked_indexfile , num_steps , blank_index )
    while True:
        c = np.zeros((batch_size , num_steps) , dtype=np.int32)
        t = np.zeros((batch_size,1) ,dtype=np.int32)
        for i in range (batch_size):
            c[i] , t[i,0]=next(x)
        yield c,t


# In[44]:


if __name__=='__main__':
    x=get_data('real_test.pkl' , 'blanked_test.pkl' , 'blanked_index.pkl' , 41 , 402252 , 5)
    for c , y in x:
        print (c)
        print(y)

