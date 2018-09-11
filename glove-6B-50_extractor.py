
# coding: utf-8

# In[6]:


import numpy as np
import pickle


# In[7]:


vectors = np.zeros ((400000, 50) , dtype=np.float32)
stoi = dict()
itos = []
i=0

with open ("D:\Contest\glove.6B\glove.6B.50d.txt" , 'r' , encoding= 'UTF-8') as infile:
    for line in infile.readlines():
        row = line.strip().split(' ')
        itos.append(row[0])
        listt = [float(j) for j in row[1:]]
        stoi[row[0]]=i
        vectors[i]=np.array(listt)
        i+=1

with open ('vectors.pkl' , 'wb') as pkl:
    pickle.dump(vectors , pkl)
with open ('stoi.pkl' , 'wb') as pkl:
    pickle.dump(stoi , pkl)
with open ('itos.pkl' , 'wb') as pkl:
    pickle.dump(itos , pkl)

