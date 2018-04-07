# cluster_word_in_vector



>關於本次實驗 : 

>1.分群演算法(K-means) on GloVe

>2.降維視覺化(PCA) on 創意dataset

>3.分類演算法(SVM) on 創意dataset

# 1.分群演算法(K-means) on GloVe

jupyter :http://localhost:8888/notebooks/Desktop/K-means_cluster.ipynb

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
>>打開文件glove.6B.300d.txt，內容紀錄的是英文單字與其向量

>>資料型態如下: 

| words  |   v1  |   v2    | ... |
| :----- |:-----:| :-----: | --: |
| the    | 0.418 | -0.41242| ... |
| that   | 0.88387 | 0.3011 | ... |
| ...    | ...   | ...     | ... | 

>> glove.6B 的內容是紀錄一萬個英文單字，維度分別有50、100、200、300。已知資料集有的特性(幾何相似、線性相關)，對這些向量進行K-means分群，可以預想到結果應該是這樣子的: 同群的單字間具表達某種'關係'。



有些群有著明顯的共同點，舉例而言:'ski', 'skating', 'skiing', 'alpine', 'nordic' 滑雪、滑冰、高山、挪威分成一群；'evil', 'dragon', 'monster', 'spirits', 'alien', 'beings', 'creatures', 'gods', 'insects' 惡魔、龍、怪物、靈魂、外星人、生物、神、蟲為一群；'carbon', 'emissions', 'pollution', 'greenhouse', 'dioxide', 'gases', 'hydrogen' 碳、排放、汙染、溫室、二氧化碳、氣體、氫為一群。有些群並沒有辦法只出他們之間確切的關聯性，如:
'1-0', '2-1', '2-0', '3-0', '3-1', '3-2', 'unbeaten', 'halftime' ，這一群其實是球類新聞，半場、不敗、比數等..單看數字是不容易聯想到對應到的是比數；我猜測是原生向量的產生會與句子中字詞出現的相關性有關，也就是說常常一起出現的詞會有著類似的向量，像是'amazon'與'internet', 'computer', 'web', 'networking', 'multimedia', 'websites' 分成一類。


# 2.降維視覺化(PCA) on 創意dataset & 3. SVM

jupyter:https://github.com/baker12355/cluster-word_in_vector-/blob/master/PCA%20%26%20SVM.ipynb

創意Dataset是使用TextBlob對我的詞來進行情感分析，輸入字串將回傳 [-1,1] 區間的數值，越靠近-1代表情緒越差、越靠近1代表情緒越好、0表示沒有代表情緒，因此，我有10000個詞，會得到10000個分數，為了方便起見我用二分法區別，0代表情緒差、1代表情緒好。並使用SVM監督式學習進行分類，並在testing_Data上有86%的準確率。而在進行分類之前我先將資料降維觀察。

PCA是最簡單的以特徵量分析多元統計分布的方法。其結果可以理解為對原數據中的變異數做出解釋：哪一個方向上的數據值對變異數的影響最大？換而言之，PCA提供了一種降低數據維度的有效辦法；如果分析者在原數據中除掉最小的特徵值所對應的成分，那麼所得的低維度數據必定是最優化的（也即，這樣降低維度必定是失去訊息最少的方法）。主成分分析在分析複雜數據時尤為有用，比如人臉識別。由於資料維度很大(300維), 使用PCA適合用來降維並且視覺化。

![GITHUB](https://github.com/baker12355/cluster-word_in_vector-/blob/master/distribution.JPG)

由圖可知這兩個資料不是分得很開，我猜測是因為資料集擁有不少相同的特徵如:大多同為形容詞、副詞(雨天,藍色等抽象含意並不會影響分數)。既然詞性相同則在語句中的相對位置也會相同，應當是主要的原因。分析完畢後便使用SVM分類，在此要注意到SVM中有許多參數是可以調整的，而透過網格搜索（Grid search）或交叉驗證（Cross validation），皆可以得到較佳參數，在這邊我使用Grid search。



