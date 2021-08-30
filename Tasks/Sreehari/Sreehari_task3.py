#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf

#import tensorflow_text
import tensorflow_text as tf_text

directory_url = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
file_names = ['cowper.txt', 'derby.txt', 'butler.txt']

file_paths = [
    tf.keras.utils.get_file(file_name, directory_url + file_name)
    for file_name in file_names
]

dataset = tf.data.TextLineDataset(file_paths)


# In[2]:


#convert to utf8
dataset_utf8 = map(lambda str:tf_text.normalize_utf8(str), dataset)

for data in dataset_utf8:
  print(data.numpy())


# In[ ]:




