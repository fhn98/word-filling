
# coding: utf-8

# In[6]:
# be aware of extra comments caused by ipynb

import numpy as np
import pickle


# what is stoi, itos??
# define function for each models
# do not use pre-identified index for number of words for this module when it is not neccessary


def get_wordembedding(inflile):
"""
construting word embedding matrix from a soure

input:
infile: file of wordembedding, a text file containing the word in first index and representation in other indexes

output:
vectors: ??
stoi: ??
itos: ??
"""
    vectors = []
    stoi = dict()
    itos = []
    i=0
    for line in infile.readlines():
        row = line.strip().split(' ')
        itos.append(row[0])
        listt = [float(j) for j in row[1:]]
        stoi[row[0]]=i
        vectors.append(listt)
        i+=1
    vectors = np.array(vectors)
    return vectors, itos, stoi



if __name__ == "__main__":
    vectors, itos, stoi = get_wordembedding(open("D:\Contest\glove.6B\glove.6B.50d.txt" , 'r' , encoding= 'UTF-8'))
    with open ('vectors.pkl' , 'wb') as pkl:
        pickle.dump(vectors , pkl)
    with open ('stoi.pkl' , 'wb') as pkl:
        pickle.dump(stoi , pkl)
    with open ('itos.pkl' , 'wb') as pkl:
        pickle.dump(itos , pkl)
