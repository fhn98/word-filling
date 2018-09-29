
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import pickle
from urllib import request
from nltk import word_tokenize


# In[2]:


import nltk
from nltk.corpus import gutenberg
from nltk.tokenize import RegexpTokenizer


# In[3]:


#converting words of the book to indexes
#if there was an unknown word, we add it to the end of our dicitionary and use a random vector for its embedding
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


# In[4]:


def make_corpus(corpus , stoi , itos , vectors):     
    """
    description: converting words to their indices in GloVe and adding new words to Glove if they do not exist in stoi.
    
    inputs: stoi : a dictionary from word to its index in vectors
            itos: index of each word in the list represents its index in vectors
            vectors: each row is the word embedding of the word with corresponding index
            
    outputs: nth.
    """
    book_by_index , stoi , itos , vectors = convert(corpus , stoi , itos , vectors)
    
    #writing back data
    with open (stoifile,'wb') as pkl:
        pickle.dump(stoi , pkl)      

    with open (itosfile,'wb') as pkl:
        pickle.dump(itos , pkl)      

    with open (vectorsfile,'wb') as pkl:
        pickle.dump(vectors , pkl)

    with open ('train_by_index.pkl','wb') as pkl:
        pickle.dump(book_by_index , pkl)      


# In[ ]:


def download_books(urls):
    books=[]
    
    for book_url in urls:
        response = request.urlopen(book_url)
        raw = response.read().decode('utf8')
        books=books+list(word_tokenize(raw))
    
    return books
    


# In[5]:


if __name__=='__main__':
    books = download_books(['http://www.gutenberg.org/cache/epub/1514/pg1514.txt'] )
    with open (stoifile , 'rb') as pkl:
        stoi = pickle.load(pkl)

    with open (itosfile , 'rb') as pkl:
        itos = pickle.load(pkl)

    with open (vectorsfile , 'rb') as pkl:
        vectors = pickle.load(pkl)
        
    make_corpus(books , stoi, itos , vectors)

