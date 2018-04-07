# cluster_word_in_vector


本次實驗採用的資料集為: GloVe(Global Vectors for Word Representation)，裡頭的單字已轉換為向量形勢，轉換方式


# GloVe

>>下載 : glove.6B.zip http://www-nlp.stanford.edu/data/
>>打開文件glove.6B.50d.txt，內容紀錄的是英文單字與其向量

>>資料型態如下: 

| words  |   v1  |   v2    | ... |
| :----- |:-----:| :-----: | --: |
| the    | 0.418 | -0.41242| ... |
| that   | 0.88387 |   $12 | ... |
| ...    | ...   | ...     | ... | 


glove.6B 的內容是紀錄一萬個英文單字，維度分別有50、100、200、300

執行 cluster.py 將對這些向量進行K-means分群，並可觀察單字分群的相關性。
