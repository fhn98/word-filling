
# coding: utf-8

# In[1]:

# you do not need tensorflow
# import tensorflow as tf
import numpy as np
import pickle


# In[2]:
# be awore of extra comments caused by ipynb

import nltk
from nltk.corpus import gutenberg
from nltk.tokenize import RegexpTokenizer


# In[3]:


#converting words of the book to indexes
#if there was an unknown word, we add it to the end of our dicitionary and use a random vector for its embedding
def convert (book , stoi , itos , vectors):
"""
Description of function:
...
input:
...
output:
...
"""
    book_by_index=[]
    i = len(stoi)
    for word in book:
        if word.lower() in stoi:#if the word already exists in out vocabulary
            book_by_index.append(stoi[word.lower()])
        else:
            stoi[word.lower()]=i
            itos.append(word.lower())
            new = np.random.rand(1,vectors.shape[1]) # save shape in variable to avoid extra calculation
            vectors = np.concatenate((vectors , new) , axis = 0)#adding a random as the embedding of the new word to vectors
            book_by_index.append(i)
            i+=1

    return book_by_index , stoi , itos , vectors


# In[4]:

# the module has constant paths (for writing) that is not standard
def make_corpus(book_id , stoifile , itosfile , vectorsfile):
"""
Description of function:
...
input:
...
output:
...
"""
    book = gutenberg.words(book_id)

    with open (stoifile , 'rb') as pkl:#stoi : a dictionary from word to its index in vectors
        stoi = pickle.load(pkl)

    with open (itosfile , 'rb') as pkl:#index of each word in the list represents its index in vectors
        itos = pickle.load(pkl)

    with open (vectorsfile , 'rb') as pkl:#each row is the word embedding of the word with corresponding index
        vectors = pickle.load(pkl)



    book_by_index , stoi , itos , vectors = convert(book , stoi , itos , vectors)


    print(len(book_by_index))
    print(book_by_index)

    #writing back data
    with open (stoifile,'wb') as pkl:
        pickle.dump(stoi , pkl)

    with open (itosfile,'wb') as pkl:
        pickle.dump(itos , pkl)

    with open (vectorsfile,'wb') as pkl:
        pickle.dump(vectors , pkl)

    with open ('test_by_index.pkl','wb') as pkl:
        pickle.dump(book_by_index[:5000] , pkl) # pre-defined index 5000


# In[5]:


if __name__=='__main__':
    make_corpus('shakespeare-macbeth.txt' , 'stoi.pkl' , 'itos.pkl' , 'vectors.pkl')
