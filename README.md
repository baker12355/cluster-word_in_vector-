# cluster_word_in_vector


本次實驗採用的資料集為: GloVe，其中的單字已轉換為向量形式，其向量有以下特性:

| nearest neighbors of <br/> <em>frog</em> | Litoria             |  Leptodactylidae | Rana | Eleutherodactylus |
| --- | ------------------------------- | ------------------- | ---------------- | ------------------- |
| Pictures | <img src="http://nlp.stanford.edu/projects/glove/images/litoria.jpg"></img> | <img src="http://nlp.stanford.edu/projects/glove/images/leptodactylidae.jpg"></img> | <img src="http://nlp.stanford.edu/projects/glove/images/rana.jpg"></img> | <img src="http://nlp.stanford.edu/projects/glove/images/eleutherodactylus.jpg"></img> |

| Comparisons | man -> woman             |  city -> zip | comparative -> superlative |
| --- | ------------------------|-------------------------|-------------------------|
| GloVe Geometry | <img src="http://nlp.stanford.edu/projects/glove/images/man_woman_small.jpg"></img>  | <img src="http://nlp.stanford.edu/projects/glove/images/city_zip_small.jpg"></img> | <img src="http://nlp.stanford.edu/projects/glove/images/comparative_superlative_small.jpg"></img> |

若是找到單字'frog'，則'frog'附近的鄰居(幾何距離)也會是'類'青蛙的單字，此外也存在向量中的線性關係，郵政編碼的對應、邏輯關係man -> woman ≒ king -> queen ≒ uncle -> aunt，以及形容詞、比較級、最高級的向量關係。

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
