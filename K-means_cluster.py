
# coding: utf-8

# # import data

# In[3]:


from __future__ import division
from sklearn.cluster import KMeans
from numbers import Number
from pandas import DataFrame
import sys, codecs, numpy


# # 預處理所需函式

# In[4]:


class autovivify_list(dict):
  '''A pickleable version of collections.defaultdict'''
  def __missing__(self, key):
    '''Given a missing key, set initial value to an empty list'''
    value = self[key] = []
    return value

  def __add__(self, x):
    '''Override addition for numeric types when self is empty'''
    if not self and isinstance(x, Number):
      return x
    raise ValueError

  def __sub__(self, x):
    '''Also provide subtraction method'''
    if not self and isinstance(x, Number):
      return -1 * x
    raise ValueError
    
def build_word_vector_matrix(vector_file, n_words):
  '''Return the vectors and labels for the first n_words in vector file'''
  numpy_arrays = []
  labels_array = []
  with codecs.open(vector_file, 'r', 'utf-8') as f:
    for c, r in enumerate(f):
      sr = r.split()
      labels_array.append(sr[0])
      numpy_arrays.append( numpy.array([float(i) for i in sr[1:]]) )

      if c == n_words:
        return numpy.array( numpy_arrays ), labels_array

  return numpy.array( numpy_arrays ), labels_array


def find_word_clusters(labels_array, cluster_labels):
  '''Return the set of words in each cluster'''
  cluster_to_words = autovivify_list()
  for c, i in enumerate(cluster_labels):
    cluster_to_words[ i ].append( labels_array[c] )
  return cluster_to_words


# #### 1萬個單字、每個字300維、分成1000群

# In[5]:


input_vector_file = 'glove.6B.300d.txt' # Vector file input (e.g. glove.6B.300d.txt)
n_words = int('10000') # Number of words to analyze 
reduction_factor = float('.1') # Amount of dimension reduction {0,1}
n_clusters = int( n_words * reduction_factor ) # Number of clusters to make


# #### Fit model

# In[8]:


df, labels_array = build_word_vector_matrix(input_vector_file, n_words)
kmeans_model = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
kmeans_model.fit(df)


# #### 同群歸類到cluster_to_words

# In[10]:


cluster_labels  = kmeans_model.labels_
cluster_inertia   = kmeans_model.inertia_
cluster_to_words  = find_word_clusters(labels_array, cluster_labels)


# #### 取群集 100 ,200 ,500 查看關聯

# In[17]:


print (cluster_to_words[100]) #['barcelona', 'lyon', 'valencia', 'monaco', 'marseille', 'lens', 'bordeaux']


# #### 巴塞隆納(西班牙)、里昂(法國)、瓦倫西亞(西班牙)、摩納哥(法國)、馬賽(法國)、朗斯(法國)、波爾多(法國)。地理位置接近

# In[18]:


print (cluster_to_words[200]) #['militia', 'paramilitary', 'militias', 'gangs', 'militiamen']


# #### 爵士、獨奏、弦、吉他、藍調、低音、鋼琴、三人(重奏)、二人(重奏)、喇叭、儀器的、韻律、鼓... 皆為聲樂關聯。

# In[19]:


print (cluster_to_words[500])


# #### 計劃、計劃、計劃、建議、批准、建議、計劃、草案、批准、考慮、建議、批准、草擬、揭開、概述、建議。提案申請相關
