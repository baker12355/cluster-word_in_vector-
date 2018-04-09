from __future__ import division
from keras.models import Sequential
from keras.layers import Dense,Dropout
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import codecs, numpy
from textblob import TextBlob  #情感分類
import math
from sklearn.preprocessing import scale

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

def poparity(a):         #回傳情感值 [-1,1],1表好,-1表差
    blob=TextBlob(a)
    return blob.polarity

def preprocessing(df,words): #回傳情感非0的那些單字與向量,好的為1,差的為0,
    label=[]
    data=[]
    word=[]
    for i in range(len(words)):
        if (poparity(words[i])!=0):
            a=poparity(words[i])
            data.append(df[i])
            word.append(labels_array[i])
            if a==-1:
                label.append(0)
            else:
                label.append(math.ceil(a))
    return word,data,label


def show_train_history(train_history, train, validation):  
    plt.plot(train_history.history[train])  
    plt.plot(train_history.history[validation])  
    plt.title('Train History')  
    plt.ylabel(train)  
    plt.xlabel('Epoch')  
    plt.legend(['train', 'validation'], loc='upper left')  
    plt.show()
    

input_vector_file = 'glove.6B.300d.txt' # Vector file input (e.g. glove.6B.300d.txt)
df, labels_array = build_word_vector_matrix(input_vector_file, 10000)
w,d,l=preprocessing(df,labels_array) # w為帶有情感的單字 , d 代表該單字的向量 , l 代表帶有的情感
d=scale(d)

# Split the `digits` data into training and test sets
X_train, X_test, y_train, y_test =train_test_split(d, l , test_size=0.1, random_state=42)



model = Sequential()

model.add(Dense(units=600,
                input_dim=300,
                kernel_initializer='uniform',
                activation='relu'))

model.add((Dropout(0.7)))
model.add(Dense(units=400,
                kernel_initializer='uniform',
                activation='relu'))


model.add((Dropout(0.8)))

model.add(Dense(units=200,
                kernel_initializer='uniform',
                activation='relu'))

model.add((Dropout(0.8)))

model.add(Dense(units=10,
                kernel_initializer='uniform',
                activation='relu'))


model.add(Dense(units=1,
                kernel_initializer='uniform',
                activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

train_history=model.fit(x=X_train,
                        y=y_train,
                        validation_split=0.2,
                        epochs=1000,
                        batch_size=50,
                        verbose=0)

show_train_history(train_history,'acc','val_acc')

score=model.evaluate(x=X_test,
                     y=y_test)

print ('準確度:',score[1])



