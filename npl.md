---
layout: page
title: Natural Language Processing
permalink: /npl
---



## [Imports & Custom Functions](#imports)

## [Basic Text Features](#basic_npl)  
> - Lower Case  
- Remove Punctuation  
- Remove Stopwords  
- Term Frequency  
- Spelling Correction  
- Stemming - Removal of Suffices  
- Lemmatization - Converting to Root Word  



## [Advanced Text Processing](#advanced_npl)  
> - ### [N-grams](#ngrams)  
- ### [TF/IDF: Term Frequency + Inverse Document Frequency](#idf)
- ### [Sentiment Analysis Features](#sentiment)
- ### [Word2Vec](#word2vec)  





# <a id='imports'>Imports & Custom Functions</a>


```python
import numpy as np
import pandas as pd

# Natural Language Processing Imports
from nltk.corpus import stopwords # stopwords
from nltk.stem import PorterStemmermmer # 2.6 Stemming - Removal of suffices like 'ing','ly','s', etc.
from textblob import Word # 2.7 Lemmatization - Converting to root word
from textblob import TextBlob # 3.1 N-grams (also for spell correction)
from sklearn.feature_extraction.text import TfidfVectorizer # 3.2 Term Frequency, 3.3 Inverse Document Frequency
from gensim.models import KeyedVectors # 3.5 Word2Vec

```


```python
def toDataFrame(list_of_dictionaries,columns_to_get):

    Dataset=pd.DataFrame()

    for column in columns_to_get:
            Dataset[column]=[dictionary[column] for dictionary in list_of_dictionaries]

    return Dataset



# Add metadata to DataFrame object
def to_Dictionary_List(df_source, nested_object, dict_keys):#, drop=False):
    print("Length of source dataset is     : ", len(df_source))
    dictionary_list=[]
    error_indices_list=[]

    for i,item in enumerate(df_source[nested_object]):
        try:
            dictionary_list.append(dict(eval(item)))
        except:
            error_indices_list.append(i)
            print('Drop index',i)

    if len(error_indices_list)!=0:
        print("WAIT! Try again after dropping these indices from original dataframe:", error_indices_list)

    return (dictionary_list, nested_object, dict_keys, error_indices_list)


# Add result from function to_Dictionary_List as columns to destination dataframe
def add_Nested_Columns(df_destination, tup):
    dictionary_list=tup[0]
    nested_object= tup[1]
    dict_keys = tup[2]
    error_indices_list=tup[3]
    assert tup[3]==[], "Drop error rows first."
    print("No error rows detected. Proceeding...")
    assert len(df_destination)==len(dictionary_list), "Number of rows don't match."

    for j,key in enumerate(dict_keys):
        try:
            df_destination[nested_object+'_'+key]=[d[key] for d in dictionary_list]
        except:
            print('ERROR')
            break

    print("Success. The destination dataframe had ", len(dict_keys)," columns added")


def add_metadata(df_destination, df_source, nested_object, dict_keys):
    add_Nested_Columns(df_destination,to_Dictionary_List(df_source, nested_object, dict_keys))



def select(val, col, df):
    df_result=df[df[col]==val]
    print("Length of original dataset:", len(df))
    print("Length of selection dataset:", len(df_result))
    return df_result


def getNumericColumns(df, numeric_columns=None):
    if numeric_columns==None:
        numeric_columns=['id','retweet_count','favorite_count',
                 'user_verified','user_followers_count','user_friends_count',
                 'user_listed_count','user_favourites_count','user_statuses_count'
                 ,'user_geo_enabled'
                ]
    return df[numeric_columns]

def getNonNumericColumns(df, non_numerics=None):
    if not non_numerics:
        numeric_columns=['id','retweet_count','favorite_count',
                 'user_verified','user_followers_count','user_friends_count',
                 'user_listed_count','user_favourites_count','user_statuses_count'
                 ,'user_geo_enabled'
                ]
        non_numerics= [col for col in df.columns if col not in numeric_columns]
    return df[non_numerics]


def getOneValueColumns(df):
    result=[]
    for col in df.columns:
        if len(df[col].unique())==1:
            result.append(col)
    print("Single value columns are:", result)
    return result

def dropColumns(df, columns_to_drop):
    df.drop(columns_to_drop, axis=1, inplace=True)
    print(columns_to_drop, "have been dropped successfully.")
    return None

def how_many_uniques(df):
    for col in df.columns:
        try:
            print(col,len(df[col].unique()))
        except:
            print(col, "length undefined")
    return None

def which_are_unique(df):
    result = []
    for col in df.columns:
        if len(df[col].unique())==1:
            result.append(col)
    return result

def how_many_na(df):
    for col in df.columns:
        try:
            print(col,sum(df[col].isna()))
        except:
            print(col, "length undefined")
    return None

def which_have_too_many_na(df):
    result=[]
    for col in df.columns:
        if sum(df[col].isna())>len(df)/2:
            result.append(col)
    return result
```


```python
# Function that calculates the average length of a word in the text
def avg_word(sentence):
    try:
        words = sentence.split()
    except:
        print('error', sentence)
        return np.nan
    return (sum(len(word) for word in words)/len(words))

# Getting stopwords
stop = stopwords.words('english')
```


```python
# Creating column of word count in each tweet
df_text['word_count']  = df_text['text'].apply(lambda x: len(str(x).split(" ")))

# Creating column of character count in each tweet
df_text['char_count']  = df_text['text'].str.len()

# Creating column of average word length in each tweet
df_text['avg_word']    = df_text['text'].apply(lambda x: avg_word(x))

# Getting number of stopwords in tweet
df_text['stopwords']   = df_text['text'].apply(lambda x: len([x for x in x.split() if x in stop]))

# Getting hashtags in tweets
#df_text['hashtags']    = df_text['text'].apply(lambda x: [x for x in x.split() if x.startswith('#')])

# Counting number of  hashtags in tweets
df_text['num_hashtags']= df_text['text'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))

# Getting number of numerics in tweet
df_text['numerics']    = df_text['text'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))

# Getting number of upper case words in tweet
df_text['upper']       = df_text['text'].apply(lambda x: len([y for y in x.split() if y.isupper()]))
```

    /anaconda/lib/python3.6/site-packages/ipykernel_launcher.py:2: SettingWithCopyWarning:
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

    /anaconda/lib/python3.6/site-packages/ipykernel_launcher.py:5: SettingWithCopyWarning:
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      """
    /anaconda/lib/python3.6/site-packages/ipykernel_launcher.py:8: SettingWithCopyWarning:
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

    /anaconda/lib/python3.6/site-packages/ipykernel_launcher.py:11: SettingWithCopyWarning:
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      # This is added back by InteractiveShellApp.init_path()
    /anaconda/lib/python3.6/site-packages/ipykernel_launcher.py:17: SettingWithCopyWarning:
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    /anaconda/lib/python3.6/site-packages/ipykernel_launcher.py:20: SettingWithCopyWarning:
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    /anaconda/lib/python3.6/site-packages/ipykernel_launcher.py:23: SettingWithCopyWarning:
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy



```python
df_text['user_screen_name']=df_non_numeric['user_screen_name']

df_text['id']=df_numeric['id']
df_non_numeric['id']=df_numeric['id']
```

    /anaconda/lib/python3.6/site-packages/ipykernel_launcher.py:1: SettingWithCopyWarning:
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      """Entry point for launching an IPython kernel.
    /anaconda/lib/python3.6/site-packages/ipykernel_launcher.py:3: SettingWithCopyWarning:
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      This is separate from the ipykernel package so we can avoid doing imports until
    /anaconda/lib/python3.6/site-packages/ipykernel_launcher.py:4: SettingWithCopyWarning:
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      after removing the cwd from sys.path.



```python
df_numeric.shape
```




    (89306, 10)




```python
df_text.shape
```




    (89306, 8)




```python
df_new=df_numeric.join(df_text)
```


```python
df_new.shape
```




    (89306, 18)




```python
df_text.drop('is_retweet',axis=1, inplace=True)
```

    /anaconda/lib/python3.6/site-packages/pandas/core/frame.py:3697: SettingWithCopyWarning:
    A value is trying to be set on a copy of a slice from a DataFrame

    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      errors=errors)


# <a id='basic_npl'>Basic Text Features Extraction</a>

## 2.1 Lower Case


```python
df=df_text.copy()
```


```python
df.text=df.text.apply(lambda x: ' '.join(x.lower() for x in x.split()))
```

## 2.2 Remove Punctuation


```python
df.text=df.text.str.replace('[^\w\s]','')
```

## 2.3 Remove Stopwords


```python
df.text=df.text.apply(lambda x:' '.join(y for y in x.split() if y not in stop))
```

## 2.4 <a id='tf'>Term Frequency</a>




```python
freq = pd.Series(' '.join(df.text).split()).value_counts()[:100]
```


```python
freq
```




    rt                 54589
    trump               8720
    realdonaldtrump     8281
    president           4598
    amp                 4595
    people              3640
    us                  3584
    one                 3303
    like                3300
    dont                2876
    putin               2790
    russia              2673
    would               2653
    obama               2628
    know                2599
    get                 2540
    im                  2164
    fbi                 2080
    time                2042
    potus               2027
    russian             1988
    democrats           1983
    see                 1856
    foxnews             1828
    never               1806
    american            1799
    fisa                1781
    today               1769
    america             1766
    new                 1755
                       ...  
    didnt               1134
    state               1130
    ever                1120
    world               1120
    thats               1117
    fake                1115
    could               1109
    via                 1087
    every               1086
    hes                 1084
    much                1073
    breaking            1045
    clinton             1036
    must                1023
    2                   1020
    youre               1009
    got                 1000
    god                  995
    white                993
    way                  989
    w                    960
    trumps               956
    cnn                  940
    believe              934
    support              931
    party                899
    carter               897
    keep                 894
    another              890
    let                  887
    Length: 100, dtype: int64



## 2.5 Spelling Correction


```python
# This takes a long time to do, but it supposedly has a huge impact on performance
df['text_spell_corrected']=df.text.apply(lambda x: str(TextBlob(x).correct()))
```

## 2.6 Stemming - Removal of Suffices like 'ing', 'ly', 's', etc.


```python
from nltk.stem import PorterStemmer
```


```python
st = PorterStemmer()
df.text.apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
```

## 2.7 Lemmatization - Converting to Root Word


```python
df['lemmatized'] = df.text.apply(lambda x: ' '.join([Word(word).lemmatize() for word in x.split()]))
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>word_count</th>
      <th>char_count</th>
      <th>avg_word</th>
      <th>stopwords</th>
      <th>num_hashtags</th>
      <th>numerics</th>
      <th>upper</th>
      <th>lemmatized</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>vamcs r trained murderers smart went 2 college...</td>
      <td>23</td>
      <td>140</td>
      <td>5.130435</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>10</td>
      <td>vamcs r trained murderer smart went 2 college ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>httpstcokwimj6hkrh one owns land united states...</td>
      <td>21</td>
      <td>140</td>
      <td>6.000000</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>httpstcokwimj6hkrh one owns land united state ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>rt rohll5 rt agree 44 potus harm good 8 years ...</td>
      <td>20</td>
      <td>137</td>
      <td>4.782609</td>
      <td>6</td>
      <td>4</td>
      <td>1</td>
      <td>5</td>
      <td>rt rohll5 rt agree 44 potus harm good 8 year s...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>wait next burnie love comi comey ag httpstco3p...</td>
      <td>13</td>
      <td>77</td>
      <td>5.000000</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>wait next burnie love comi comey ag httpstco3p...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>simple landlords get rich quick scheme ask don...</td>
      <td>21</td>
      <td>140</td>
      <td>5.714286</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>simple landlord get rich quick scheme ask dona...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>new member royal canadian mounting police hors...</td>
      <td>18</td>
      <td>136</td>
      <td>6.611111</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>new member royal canadian mounting police hors...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>il come visit u penitentiary comi comey httpst...</td>
      <td>10</td>
      <td>70</td>
      <td>6.100000</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>il come visit u penitentiary comi comey httpst...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>comi comey mac dogg mccabe supervisor capable ...</td>
      <td>15</td>
      <td>105</td>
      <td>6.066667</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>comi comey mac dogg mccabe supervisor capable ...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>lips r moving lieing going 2 never enter kingd...</td>
      <td>17</td>
      <td>99</td>
      <td>4.882353</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>lip r moving lieing going 2 never enter kingdo...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>judges prosecuters must replace new clowns bri...</td>
      <td>15</td>
      <td>103</td>
      <td>5.933333</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>judge prosecuters must replace new clown bring...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>evil httpstcokvexijflel</td>
      <td>3</td>
      <td>31</td>
      <td>9.666667</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>evil httpstcokvexijflel</td>
    </tr>
    <tr>
      <th>11</th>
      <td>team one worked demonrats publicans party pick...</td>
      <td>23</td>
      <td>140</td>
      <td>5.130435</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>team one worked demonrats publican party pick ...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>white house chief staff john kellyreportedly g...</td>
      <td>16</td>
      <td>140</td>
      <td>6.421053</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>white house chief staff john kellyreportedly g...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>myhtopoeic watch microwave weapon take u silen...</td>
      <td>25</td>
      <td>140</td>
      <td>4.640000</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>myhtopoeic watch microwave weapon take u silen...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>much money deals president trump get failing p...</td>
      <td>21</td>
      <td>140</td>
      <td>5.714286</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>much money deal president trump get failing pr...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>due known bad drugs used high dose 2 long kidn...</td>
      <td>28</td>
      <td>140</td>
      <td>4.185185</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>8</td>
      <td>due known bad drug used high dose 2 long kidne...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>veteranshealth sure thankful va medical system...</td>
      <td>20</td>
      <td>140</td>
      <td>6.050000</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>veteranshealth sure thankful va medical system...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>send law legislating judge poop shoot thru sli...</td>
      <td>20</td>
      <td>124</td>
      <td>5.250000</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>send law legislating judge poop shoot thru sli...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>u president nfl owner refused fine players nat...</td>
      <td>22</td>
      <td>140</td>
      <td>5.409091</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>u president nfl owner refused fine player nati...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>long one wants slave master chicken georgetta ...</td>
      <td>24</td>
      <td>140</td>
      <td>4.875000</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>long one want slave master chicken georgetta t...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>httpstcoetom1oxb2q oh lookie ny via ya owner t...</td>
      <td>22</td>
      <td>140</td>
      <td>5.666667</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>httpstcoetom1oxb2q oh lookie ny via ya owner t...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>killed many much blood hands lover warrior mea...</td>
      <td>29</td>
      <td>140</td>
      <td>3.862069</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>killed many much blood hand lover warrior mean...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>stretch king david went battle donald hang nai...</td>
      <td>21</td>
      <td>140</td>
      <td>5.714286</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>stretch king david went battle donald hang nai...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>scum sucker spying ass senator james lankford ...</td>
      <td>20</td>
      <td>140</td>
      <td>6.050000</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>scum sucker spying as senator james lankford a...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>interest paying interest httpstcorafdzusdnn</td>
      <td>6</td>
      <td>56</td>
      <td>8.500000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>interest paying interest httpstcorafdzusdnn</td>
    </tr>
    <tr>
      <th>25</th>
      <td>yo sanctions ever strong enough work httpstcok...</td>
      <td>10</td>
      <td>74</td>
      <td>6.500000</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>yo sanction ever strong enough work httpstcok5...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>httpstco5atradkwtt</td>
      <td>3</td>
      <td>32</td>
      <td>10.000000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>httpstco5atradkwtt</td>
    </tr>
    <tr>
      <th>27</th>
      <td>gave httpstcogvw3aznnmw</td>
      <td>5</td>
      <td>37</td>
      <td>6.600000</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>gave httpstcogvw3aznnmw</td>
    </tr>
    <tr>
      <th>28</th>
      <td>well like peon asses regulating oil north kore...</td>
      <td>23</td>
      <td>140</td>
      <td>5.130435</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>well like peon ass regulating oil north korea ...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>ted u r general michael hidden says interestin...</td>
      <td>16</td>
      <td>104</td>
      <td>5.562500</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>ted u r general michael hidden say interesting...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>107108</th>
      <td>rt tweetsfor45 trump never colluded russians g...</td>
      <td>15</td>
      <td>102</td>
      <td>6.285714</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>rt tweetsfor45 trump never colluded russian gu...</td>
    </tr>
    <tr>
      <th>107109</th>
      <td>rt cb618444 thousands trump supporters hit str...</td>
      <td>23</td>
      <td>139</td>
      <td>5.086957</td>
      <td>6</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>rt cb618444 thousand trump supporter hit stree...</td>
    </tr>
    <tr>
      <th>107110</th>
      <td>alyssa_milano realdonaldtrump looks like waite...</td>
      <td>13</td>
      <td>79</td>
      <td>5.153846</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>alyssa_milano realdonaldtrump look like waited...</td>
    </tr>
    <tr>
      <th>107111</th>
      <td>rt fleccas want make sure right left mad reald...</td>
      <td>26</td>
      <td>140</td>
      <td>4.600000</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>rt fleccas want make sure right left mad reald...</td>
    </tr>
    <tr>
      <th>107112</th>
      <td>foxandfriends dbongino anything try stop winni...</td>
      <td>11</td>
      <td>73</td>
      <td>5.727273</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>foxandfriends dbongino anything try stop winni...</td>
    </tr>
    <tr>
      <th>107113</th>
      <td>rt anncoulter peter strzoks wife threatened le...</td>
      <td>25</td>
      <td>140</td>
      <td>4.640000</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>rt anncoulter peter strzoks wife threatened le...</td>
    </tr>
    <tr>
      <th>107114</th>
      <td>rt pink_about_it democrats drafted bill abolis...</td>
      <td>26</td>
      <td>139</td>
      <td>4.384615</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>rt pink_about_it democrat drafted bill abolish...</td>
    </tr>
    <tr>
      <th>107115</th>
      <td>rt ladythriller69 fact people fighting keep ra...</td>
      <td>26</td>
      <td>140</td>
      <td>4.423077</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>rt ladythriller69 fact people fighting keep ra...</td>
    </tr>
    <tr>
      <th>107116</th>
      <td>rt the_trump_train anyone care explain mueller...</td>
      <td>22</td>
      <td>140</td>
      <td>5.409091</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>rt the_trump_train anyone care explain mueller...</td>
    </tr>
    <tr>
      <th>107117</th>
      <td>rt amymek media never cover protest amp march ...</td>
      <td>24</td>
      <td>144</td>
      <td>5.041667</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>rt amymek medium never cover protest amp march...</td>
    </tr>
    <tr>
      <th>107118</th>
      <td>rt education4libs london flew large blimp pres...</td>
      <td>23</td>
      <td>140</td>
      <td>4.833333</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>rt education4libs london flew large blimp pres...</td>
    </tr>
    <tr>
      <th>107119</th>
      <td>rt bushido49ers cbsnews always duh</td>
      <td>8</td>
      <td>51</td>
      <td>5.500000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>rt bushido49ers cbsnews always duh</td>
    </tr>
    <tr>
      <th>107120</th>
      <td>rt foxandfriends dems drafted bill abolish ice...</td>
      <td>16</td>
      <td>111</td>
      <td>6.000000</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>rt foxandfriends dems drafted bill abolish ice...</td>
    </tr>
    <tr>
      <th>107121</th>
      <td>rt hmmmthere never said didnt like trump suppo...</td>
      <td>16</td>
      <td>128</td>
      <td>7.062500</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>rt hmmmthere never said didnt like trump suppo...</td>
    </tr>
    <tr>
      <th>107122</th>
      <td>nbcnews heard took tag mattress discussing</td>
      <td>13</td>
      <td>71</td>
      <td>4.538462</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>nbcnews heard took tag mattress discussing</td>
    </tr>
    <tr>
      <th>107123</th>
      <td>rt jacobawohl huge peter strzok says inspector...</td>
      <td>23</td>
      <td>140</td>
      <td>5.130435</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>rt jacobawohl huge peter strzok say inspector ...</td>
    </tr>
    <tr>
      <th>107125</th>
      <td>cbsnews fine dandy need something</td>
      <td>9</td>
      <td>54</td>
      <td>5.111111</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>cbsnews fine dandy need something</td>
    </tr>
    <tr>
      <th>107137</th>
      <td>uberfacts one feae fridays fear mondafghafthhh...</td>
      <td>11</td>
      <td>91</td>
      <td>7.363636</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>uberfacts one feae friday fear mondafghafthhhv...</td>
    </tr>
    <tr>
      <th>107141</th>
      <td>imanmunal naked true</td>
      <td>3</td>
      <td>21</td>
      <td>6.333333</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>imanmunal naked true</td>
    </tr>
    <tr>
      <th>107142</th>
      <td>smile everything okay</td>
      <td>27</td>
      <td>99</td>
      <td>1.821429</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>smile everything okay</td>
    </tr>
    <tr>
      <th>107143</th>
      <td>rt arewashams every woman deserves man loves r...</td>
      <td>20</td>
      <td>132</td>
      <td>5.045455</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>rt arewashams every woman deserves man love re...</td>
    </tr>
    <tr>
      <th>107144</th>
      <td>rt tweets2motivate today great day say thank g...</td>
      <td>24</td>
      <td>126</td>
      <td>4.291667</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>rt tweets2motivate today great day say thank g...</td>
    </tr>
    <tr>
      <th>107146</th>
      <td>rt hqnigerianarmy coas lt gen ty buratai yeste...</td>
      <td>23</td>
      <td>140</td>
      <td>5.130435</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>rt hqnigerianarmy coas lt gen ty buratai yeste...</td>
    </tr>
    <tr>
      <th>107147</th>
      <td>hedankwambo businessdayng congratulations exce...</td>
      <td>5</td>
      <td>59</td>
      <td>11.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>hedankwambo businessdayng congratulation excel...</td>
    </tr>
    <tr>
      <th>107150</th>
      <td>rt cleverquotez never argue idiot theyll drag ...</td>
      <td>19</td>
      <td>112</td>
      <td>4.947368</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>rt cleverquotez never argue idiot theyll drag ...</td>
    </tr>
    <tr>
      <th>107151</th>
      <td>rt trackhatespeech fake news 6 people suspecte...</td>
      <td>23</td>
      <td>144</td>
      <td>5.000000</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>rt trackhatespeech fake news 6 people suspecte...</td>
    </tr>
    <tr>
      <th>107152</th>
      <td>madame_flowy ask questions</td>
      <td>8</td>
      <td>48</td>
      <td>5.125000</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>madame_flowy ask question</td>
    </tr>
    <tr>
      <th>107153</th>
      <td>itswarenbuffett kind regard sir</td>
      <td>4</td>
      <td>32</td>
      <td>7.250000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>itswarenbuffett kind regard sir</td>
    </tr>
    <tr>
      <th>107156</th>
      <td>posted photo httpstconzryhlilwg</td>
      <td>5</td>
      <td>43</td>
      <td>7.800000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>posted photo httpstconzryhlilwg</td>
    </tr>
    <tr>
      <th>107157</th>
      <td>rssurjewala nice</td>
      <td>2</td>
      <td>17</td>
      <td>8.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>rssurjewala nice</td>
    </tr>
  </tbody>
</table>
<p>89306 rows × 9 columns</p>
</div>



# <a id='advanced_npl'>Advanced Text Processing</a>

## <a id='ngrams'>N-grams</a>

N-grams are the combination of multiple words used together. Ngrams with N=1 are called unigrams. Similarly, bigrams (N=2), trigrams (N=3) and so on can also be used.

Unigrams do not usually contain as much information as compared to bigrams and trigrams. The basic principle behind n-grams is that they capture the language structure, like what letter or word is likely to follow the given one. The longer the n-gram (the higher the n), the more context you have to work with. Optimum length really depends on the application – if your n-grams are too short, you may fail to capture important differences. On the other hand, if they are too long, you may fail to capture the “general knowledge” and only stick to particular cases.




```python
TextBlob(df.text[0]).ngrams(2)
```




    [WordList(['vamcs', 'r']),
     WordList(['r', 'trained']),
     WordList(['trained', 'murderers']),
     WordList(['murderers', 'smart']),
     WordList(['smart', 'went']),
     WordList(['went', '2']),
     WordList(['2', 'college']),
     WordList(['college', 'dope']),
     WordList(['dope', 'sales']),
     WordList(['sales', 'classes']),
     WordList(['classes', 'murdering']),
     WordList(['murdering', 'poor']),
     WordList(['poor', 'went']),
     WordList(['went', 'wa']),
     WordList(['wa', 'httpstco4aluzu5crs'])]



## <a id='tfidf'>TF/IDF : Term Frequency + Inverse Document Frequency</a>


Term frequency is simply the ratio of the count of a word present in a sentence, to the length of the sentence.

Therefore, we can generalize term frequency as:

TF = (Number of times term T appears in the particular row) / (number of terms in that row).


The intuition behind inverse document frequency (IDF) is that a word is not of much use to us if it’s appearing in all the documents.

Therefore, the IDF of each word is the log of the ratio of the total number of rows to the number of rows in which that word is present.

IDF = log(N/n), where, N is the total number of rows and n is the number of rows in which the word was present.

So, let’s calculate IDF for the same tweets for which we calculated the term frequency.


```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word', stop_words= 'english',ngram_range=(1,1))

train_vect = tfidf.fit_transform(df.text)
```


```python
train_vect
```




    <89306x1000 sparse matrix of type '<class 'numpy.float64'>'
    	with 447640 stored elements in Compressed Sparse Row format>




```python
list(tfidf.vocabulary_.keys())
```




    ['smart',
     'went',
     'poor',
     'united',
     'states',
     'america',
     'pay',
     'rt',
     'agree',
     'potus',
     'good',
     'years',
     'country',
     'maga',
     'wait',
     'love',
     'comey',
     'rich',
     'ask',
     'donald',
     'new',
     'member',
     'police',
     'come',
     'visit',
     'going',
     'judges',
     'bring',
     'evil',
     'team',
     'worked',
     'party',
     'white',
     'house',
     'chief',
     'john',
     'gave',
     'gop',
     'light',
     'president',
     'trumps',
     'watch',
     'open',
     'ha',
     'money',
     'trump',
     'protect',
     'known',
     'bad',
     'used',
     'high',
     'long',
     'say',
     'caught',
     'sure',
     'ass',
     'criminal',
     'send',
     'law',
     'judge',
     'nfl',
     'fine',
     'players',
     'national',
     'anthem',
     'total',
     'wants',
     'break',
     'oh',
     'tells',
     'stay',
     'wont',
     'killed',
     'mean',
     'head',
     'getting',
     'witch',
     'senator',
     'james',
     'look',
     'im',
     'yo',
     'strong',
     'work',
     'like',
     'north',
     'korea',
     'tell',
     'little',
     'general',
     'michael',
     'says',
     'interesting',
     'dirty',
     'change',
     'maybe',
     'needs',
     'constitution',
     'use',
     'communist',
     'late',
     'right',
     'rights',
     'speak',
     'day',
     'thats',
     'wrong',
     'air',
     'force',
     'hi',
     'wow',
     'washington',
     'shit',
     'free',
     'cover',
     'americas',
     'time',
     'shut',
     'really',
     'shes',
     'liar',
     'great',
     'women',
     'got',
     'dollars',
     'make',
     'mueller',
     'meet',
     'hell',
     'cause',
     'death',
     'congress',
     'intelligence',
     'agencies',
     'plan',
     'worse',
     'read',
     'true',
     'voting',
     'makes',
     'treasonous',
     'home',
     'soon',
     'american',
     'government',
     'gets',
     'god',
     'usa',
     'leave',
     'idea',
     'democrats',
     'angry',
     'foreign',
     'nations',
     'election',
     'want',
     'vote',
     'need',
     'ice',
     'crooked',
     'thinks',
     'better',
     '17',
     'protecting',
     'spy',
     'man',
     'support',
     'peace',
     'military',
     'iranian',
     'illegals',
     'trust',
     'uranium',
     'deal',
     'joe',
     'queen',
     '2020',
     'child',
     'lies',
     'attacks',
     'sir',
     'dog',
     'russians',
     'lie',
     'know',
     'people',
     'voted',
     'dont',
     'hillary',
     'rest',
     'political',
     'running',
     'lets',
     'winning',
     'hillarys',
     'folks',
     'reason',
     'hes',
     'office',
     'george',
     'presidents',
     'realdonaldtrump',
     'asked',
     'help',
     'community',
     'let',
     'fbi',
     'cia',
     'clinton',
     'obama',
     'bit',
     'deep',
     'state',
     'officers',
     'share',
     'list',
     'barack',
     'mike',
     'server',
     'water',
     'th',
     'shows',
     'doj',
     'remember',
     'fake',
     'guy',
     'elected',
     'millions',
     'try',
     'definitely',
     'attack',
     '11',
     'longer',
     'said',
     'rosenstein',
     'gone',
     'comes',
     'inside',
     'paid',
     'honor',
     'truth',
     'politics',
     'sorry',
     'dead',
     'life',
     'living',
     'corruption',
     'yes',
     'hit',
     'resign',
     'led',
     'video',
     'week',
     'billion',
     'racist',
     'face',
     'boy',
     'live',
     'care',
     'big',
     'small',
     'senschumer',
     'hear',
     'emails',
     'sent',
     'congressman',
     'elections',
     'jobs',
     'way',
     'cyber',
     'threat',
     'mr',
     'proud',
     'dear',
     'cut',
     'jim',
     'act',
     'word',
     'amen',
     'world',
     'days',
     'ago',
     'id',
     'tedlieu',
     'officials',
     'person',
     'business',
     'court',
     'order',
     'release',
     'record',
     'youre',
     'best',
     'looking',
     'amp',
     'campaign',
     'illegally',
     'spied',
     'surveillance',
     'nunes',
     '100',
     'fisa',
     'warrant',
     'carter',
     'page',
     'prayingmedic',
     'coming',
     'thing',
     'thank',
     'lord',
     'things',
     'eyes',
     'check',
     'latest',
     'media',
     'early',
     'morning',
     'stand',
     'defend',
     'job',
     'july',
     '2018',
     'today',
     'london',
     'antitrump',
     'youtube',
     'officer',
     'edkrassen',
     'exactly',
     'russia',
     'impeach',
     'lying',
     'lead',
     'director',
     'fall',
     'tax',
     'shouldnt',
     'powerful',
     'actually',
     'meant',
     'wouldnt',
     'ill',
     'hey',
     'baby',
     'blue',
     'opinion',
     'stupid',
     'stop',
     'putin',
     'takes',
     'came',
     'think',
     'republican',
     'republicans',
     'enemy',
     'press',
     'conference',
     'clear',
     'interference',
     'repadamschiff',
     'attacked',
     'standing',
     'miss',
     'times',
     'class',
     'barackobama',
     'administration',
     'htt',
     'private',
     'meeting',
     'took',
     'disgusting',
     'donaldjtrumpjr',
     'speech',
     'peter',
     'strzok',
     'seen',
     'doesnt',
     'thomas1774paine',
     'democracy',
     'pretty',
     'secpompeo',
     'sounds',
     'thanks',
     'view',
     'war',
     'wwg1wga',
     'goes',
     'qanon',
     'fun',
     'looks',
     '_imperatorrex_',
     'news',
     'sense',
     'guess',
     'patriot',
     'faith',
     'reminder',
     'question',
     'family',
     'paulsperry_',
     'hate',
     'hope',
     'future',
     'past',
     'kind',
     'china',
     'showing',
     'suffer',
     'info',
     'dbongino',
     'answer',
     'yeah',
     'ready',
     'talking',
     'correct',
     'understand',
     'proof',
     'agents',
     'working',
     'didnt',
     'play',
     'facebook',
     'waiting',
     'start',
     'lol',
     'dems',
     'ok',
     'russian',
     'matter',
     'attention',
     'dnc',
     'twitter',
     'lost',
     'followers',
     'msm',
     'set',
     'google',
     'public',
     'report',
     'welcome',
     'propaganda',
     'left',
     'power',
     'swamp',
     'happy',
     'missing',
     'giving',
     'nato',
     'paul',
     'fox',
     'bernie',
     'weak',
     'wh',
     'friends',
     'economy',
     'treason',
     'imagine',
     'end',
     'patriots',
     'cnn',
     'point',
     'questions',
     'americans',
     'real',
     'rand',
     'enjoy',
     'kenya',
     'walkaway',
     'lose',
     'turned',
     'democratic',
     'possible',
     'wake',
     'knows',
     'pray',
     'investigation',
     'foundation',
     'directly',
     'old',
     'bob',
     'history',
     'loudobbs',
     'hunt',
     'agent',
     'butina',
     'forget',
     'democrat',
     'retweet',
     'iran',
     'threaten',
     'documents',
     'confirm',
     'steele',
     'dossier',
     'liberal',
     'tomfitton',
     'docs',
     'major',
     'realize',
     'realjameswoods',
     'hillaryclinton',
     'social',
     'message',
     'attorney',
     'million',
     'httpst',
     'immigration',
     'human',
     'single',
     'biggest',
     'breaking',
     'brennan',
     'security',
     'key',
     'pedophile',
     'abuse',
     'close',
     'hand',
     'arrest',
     'army',
     'laws',
     'tweetsfor45',
     'hold',
     'including',
     'allowed',
     '2016',
     'corrupt',
     'companies',
     'trade',
     'secret',
     '10',
     'leadership',
     'policy',
     'thread',
     'charliekirk11',
     'european',
     'countries',
     'completely',
     'calling',
     'bless',
     'krisparonto',
     'pedophilia',
     'sick',
     'literally',
     'obamas',
     'brian',
     'governor',
     'georgia',
     'tough',
     'crime',
     'successful',
     'greatest',
     'trying',
     'control',
     'told',
     'strzoks',
     'crimes',
     'released',
     'claims',
     'freedom',
     'intel',
     'application',
     'away',
     'jacobawohl',
     'wanted',
     'far',
     'congratulations',
     'judicialwatch',
     'collusion',
     'evidence',
     'education4libs',
     'called',
     'sold',
     'justice',
     'amazing',
     'believe',
     'andrew',
     'happen',
     'signed',
     'knew',
     'based',
     'entire',
     'funny',
     'members',
     'anymore',
     'ryanafournier',
     'important',
     'randpaul',
     'derangement',
     'syndrome',
     'senate',
     'pages',
     'records',
     'abc',
     'nice',
     'raise',
     'rape',
     'unverified',
     'research',
     'watching',
     'explain',
     'hollywood',
     '12',
     'tonight',
     'false',
     'arrested',
     'listen',
     'demand',
     'half',
     'california',
     'ca',
     'case',
     'problem',
     'candidate',
     'supports',
     'lauraloomer',
     'saracarterdc',
     'hard',
     'wasnt',
     'source',
     'httpstco',
     'devinnunes',
     'following',
     'complete',
     'clearly',
     'article',
     'thelastrefuge2',
     'tired',
     'whoopi',
     'join',
     'senatemajldr',
     'icegov',
     'isnt',
     'stuff',
     'red',
     'clapper',
     'clean',
     'boom',
     'jackposobiec',
     'totally',
     'admits',
     'started',
     'probe',
     'border',
     'socialism',
     'illegal',
     'foxnews',
     'telling',
     'liberals',
     'presidential',
     'calls',
     'marklevinshow',
     'whitehouse',
     'kavanaugh',
     'redacted',
     'saying',
     'information',
     'tv',
     'realcandaceo',
     'different',
     'johnbrennan',
     'conservatives',
     'glad',
     'https',
     'citizen',
     'photo',
     'children',
     'reading',
     'confirmed',
     'lisa',
     'according',
     'disgrace',
     'traitor',
     'sign',
     'lied',
     'aliens',
     'turn',
     'born',
     'black',
     'ocasio2018',
     'freedom_moates',
     'protest',
     'tweet',
     'federal',
     'leaders',
     'rep',
     'asking',
     'risk',
     'woman',
     'sex',
     'body',
     'babies',
     'conservative',
     'jeanine',
     'group',
     'mitchellvii',
     'rouhani',
     'consequences',
     'likes',
     'walk',
     'department',
     'marcorubio',
     'fact',
     'sunday',
     'lives',
     'effort',
     'outside',
     'saw',
     'november',
     'dc',
     'worth',
     'nearly',
     'accused',
     'lot',
     'benghazi',
     'weve',
     'comment',
     'story',
     'statement',
     'goldberg',
     'roseanne',
     'pirro',
     'whats',
     'realjack',
     'secretservice',
     'service',
     'special',
     '13',
     'bombshell',
     'dineshdsouza',
     'year',
     'spent',
     'youve',
     'young',
     'men',
     'pass',
     'fired',
     'interview',
     'kwilli1046',
     'save',
     'cernovich',
     'pedophiles',
     'sources',
     'robert',
     'judgejeanine',
     'seanhannity',
     'gunn',
     'tweets',
     'comments',
     '400',
     'jamesgunn',
     'hoax',
     'beautiful',
     'disney',
     'mind',
     '1776stonewall',
     'yesterday',
     'met',
     'theview',
     'ive',
     'favorite',
     'fraud',
     'presidency',
     'mcfaul',
     'whoopigoldberg',
     'tony',
     'podesta',
     'facts',
     'petition',
     'account',
     'immunity',
     'manafort',
     'words',
     '18',
     'govmikehuckabee',
     'finally',
     'ambassador',
     'die',
     'workers',
     'flotus',
     'students',
     'prove',
     'post',
     'parents',
     'shot',
     'weeks',
     'huge',
     'learn',
     'buy',
     'official',
     'fear',
     'given',
     'fight',
     'voice',
     'run',
     'dem',
     'foxandfriends',
     'prisonplanet',
     'likely',
     'win',
     'helsinki',
     'meddling',
     'summit',
     'making',
     'safe',
     'agenda',
     '20',
     'son',
     'actions',
     'unhinged',
     'thought',
     'heard',
     'joke',
     'presssec',
     'straight',
     'vp',
     'leader',
     'taking',
     'action',
     'regime',
     'happened',
     'tried',
     'supporter',
     'level',
     'sarah',
     'sanders',
     'wish',
     'probably',
     'ivankatrump',
     'ingrahamangle',
     'kids',
     'indictment',
     'held',
     'note',
     'mark',
     'hours',
     'todays',
     'speakerryan',
     'course',
     'doubt',
     'vladimir',
     'low',
     'msnbc',
     'behavior',
     'sad',
     'damn',
     'performance',
     'putins',
     'feel',
     'indicted',
     'using',
     'worst',
     'indictments',
     'stories',
     'arent',
     'afraid',
     'seriously',
     'theres',
     'nra',
     'thehill',
     'fuck',
     'nation',
     'talk',
     'wife',
     'follow',
     'ones',
     'krassenstein',
     'uk',
     'truly',
     'absolutely',
     'wonder',
     'hot',
     'conspiracy',
     'months',
     'forward',
     'hearing',
     'citizens',
     'texas',
     'second',
     'allow',
     'committed',
     'clintons',
     'heres',
     'warrants',
     'gun',
     'place',
     'supporters',
     'moment',
     'debate',
     'troy',
     'ohio',
     'nancy',
     'pelosi',
     'borders',
     'friend',
     'jim_jordan',
     'marklutchman',
     'voter',
     'happening',
     'trumpputin',
     'theyre',
     'instead',
     'speaking',
     'building',
     'book',
     'omg',
     'israel',
     'committee',
     'breitbartnews',
     'jesus',
     'supreme',
     'families',
     '30',
     'birthday',
     'thinking',
     'san',
     'mother',
     'michaelavenatti',
     'funder',
     'cohen',
     'voters',
     'kylegriffin1',
     'veterans',
     'awesome',
     'jail',
     'piece',
     'game',
     'crazy',
     'nbcnews',
     'girl',
     'friday',
     'hacking',
     'health',
     'the_trump_train',
     'wall',
     'thousands',
     'hacked',
     'grizzlemeister',
     'dangerous',
     'issue',
     'involved',
     'defending',
     'continue',
     'flag',
     'night',
     'taken',
     'nuclear',
     'school',
     'daily',
     'adamparkhomenko',
     '15',
     'nytimes',
     'means',
     'chance',
     'tuckercarlson',
     'davidhogg111',
     'havent',
     'socialist',
     'respect',
     'heart',
     'supporting',
     'fucking',
     'gonna',
     'york',
     '2017',
     'number',
     'union',
     'sethabramson',
     'able',
     'tomorrow',
     'maxine',
     'waters',
     'energy',
     'line',
     'releasethetrumptapes',
     'dailycaller',
     'maria',
     'phone',
     'diamondandsilk',
     'taxes',
     'city',
     'email',
     'south',
     'career',
     'company',
     'guys',
     '50',
     'prison',
     'al',
     'ocasiocortez',
     'poll',
     'lisamei62',
     'blocked',
     'abolish',
     'netflix',
     'realmagasteve',
     'cc',
     'taylorhicks',
     'csharpcorner']




```python
df_tfidf=pd.DataFrame(train_vect.toarray(), columns=list(tfidf.vocabulary_.keys()))
```


```python
df_tfidf['index']=df.index
```


```python
df_tfidf.set_index('index')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>smart</th>
      <th>went</th>
      <th>poor</th>
      <th>united</th>
      <th>states</th>
      <th>america</th>
      <th>pay</th>
      <th>rt</th>
      <th>agree</th>
      <th>potus</th>
      <th>...</th>
      <th>ocasiocortez</th>
      <th>poll</th>
      <th>lisamei62</th>
      <th>blocked</th>
      <th>abolish</th>
      <th>netflix</th>
      <th>realmagasteve</th>
      <th>cc</th>
      <th>taylorhicks</th>
      <th>csharpcorner</th>
    </tr>
    <tr>
      <th>index</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.396066</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.641251</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>107108</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107109</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107110</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107111</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107112</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107113</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107114</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107115</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107116</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.330643</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107117</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107118</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107119</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107120</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107121</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107122</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107123</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107125</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107137</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107141</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107142</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107143</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107144</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107146</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.506481</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107147</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107150</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107151</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107152</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107153</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107156</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107157</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>89306 rows × 1000 columns</p>
</div>



### The output above is the TF/IDF matrix for 1,000 words. Notice its shape is 89306 x 1000. 1000 columns for 1000 words. With more time, I would have definitely wanted to see what this sparse matrix can do. I would like to mention that I had some difficulty understanding exactly how TF/IDF could be applied to this particular project. I can see how it might be useful when dealing with books or collections of books, but I think that, perhaps with tweets, TF/IDF may not be as useful.... But I can also sort of see why this reasoning may be wrong... In any case, definitely something to think about more in the future.




```python
df_new.drop('text',axis=1,inplace=True)
```


```python
df_tfidf=df_tfidf.set_index('index')
```


```python
df_tfidf.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>smart</th>
      <th>went</th>
      <th>poor</th>
      <th>united</th>
      <th>states</th>
      <th>america</th>
      <th>pay</th>
      <th>rt</th>
      <th>agree</th>
      <th>potus</th>
      <th>...</th>
      <th>ocasiocortez</th>
      <th>poll</th>
      <th>lisamei62</th>
      <th>blocked</th>
      <th>abolish</th>
      <th>netflix</th>
      <th>realmagasteve</th>
      <th>cc</th>
      <th>taylorhicks</th>
      <th>csharpcorner</th>
    </tr>
    <tr>
      <th>index</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>107151</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107152</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107153</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107156</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107157</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1000 columns</p>
</div>




```python
df_new['text']=df.text
```

## <a id='sentiment'>Sentiment Analysis Features</a>


```python
df_new['sentiment_polarity']=df_new['text'].apply(lambda x: TextBlob(x).sentiment[0])
df_new['sentiment_subjectivity']=df_new['text'].apply(lambda x: TextBlob(x).sentiment[1])
```


```python
df_new.user_verified=df_new.user_verified.astype(int)
```


```python
df_new.user_geo_enabled=df_new.user_geo_enabled.astype(int)
```


```python
df_new.shape
```




    (89306, 20)




```python
df_tfidf.columns = ['tfidf_'+col for col in list(tfidf.vocabulary_.keys())]
```


```python
df_tfidf
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tfidf_smart</th>
      <th>tfidf_went</th>
      <th>tfidf_poor</th>
      <th>tfidf_united</th>
      <th>tfidf_states</th>
      <th>tfidf_america</th>
      <th>tfidf_pay</th>
      <th>tfidf_rt</th>
      <th>tfidf_agree</th>
      <th>tfidf_potus</th>
      <th>...</th>
      <th>tfidf_ocasiocortez</th>
      <th>tfidf_poll</th>
      <th>tfidf_lisamei62</th>
      <th>tfidf_blocked</th>
      <th>tfidf_abolish</th>
      <th>tfidf_netflix</th>
      <th>tfidf_realmagasteve</th>
      <th>tfidf_cc</th>
      <th>tfidf_taylorhicks</th>
      <th>tfidf_csharpcorner</th>
    </tr>
    <tr>
      <th>index</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.396066</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.641251</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>107108</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107109</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107110</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107111</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107112</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107113</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107114</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107115</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107116</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.330643</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107117</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107118</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107119</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107120</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107121</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107122</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107123</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107125</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107137</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107141</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107142</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107143</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107144</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107146</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.506481</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107147</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107150</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107151</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107152</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107153</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107156</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107157</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>89306 rows × 1000 columns</p>
</div>




```python
df_milestone=df_new.join(df_tfidf)
```


```python
df_milestone.to_csv('tweets_data_7-27.csv')
```


```python
df_new
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>retweet_count</th>
      <th>favorite_count</th>
      <th>user_verified</th>
      <th>user_followers_count</th>
      <th>user_friends_count</th>
      <th>user_listed_count</th>
      <th>user_favourites_count</th>
      <th>user_statuses_count</th>
      <th>user_geo_enabled</th>
      <th>word_count</th>
      <th>char_count</th>
      <th>avg_word</th>
      <th>stopwords</th>
      <th>num_hashtags</th>
      <th>numerics</th>
      <th>upper</th>
      <th>text</th>
      <th>sentiment_polarity</th>
      <th>sentiment_subjectivity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.021279e+18</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>0</td>
      <td>23</td>
      <td>140</td>
      <td>5.130435</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>10</td>
      <td>vamcs r trained murderers smart went 2 college...</td>
      <td>-0.092857</td>
      <td>0.621429</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.021276e+18</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>0</td>
      <td>21</td>
      <td>140</td>
      <td>6.000000</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>httpstcokwimj6hkrh one owns land united states...</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.021227e+18</td>
      <td>682.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>0</td>
      <td>20</td>
      <td>137</td>
      <td>4.782609</td>
      <td>6</td>
      <td>4</td>
      <td>1</td>
      <td>5</td>
      <td>rt rohll5 rt agree 44 potus harm good 8 years ...</td>
      <td>0.700000</td>
      <td>0.600000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.021226e+18</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>0</td>
      <td>13</td>
      <td>77</td>
      <td>5.000000</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>wait next burnie love comi comey ag httpstco3p...</td>
      <td>0.250000</td>
      <td>0.300000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.021110e+18</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>0</td>
      <td>21</td>
      <td>140</td>
      <td>5.714286</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>simple landlords get rich quick scheme ask don...</td>
      <td>0.236111</td>
      <td>0.535714</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.021109e+18</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>0</td>
      <td>18</td>
      <td>136</td>
      <td>6.611111</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>new member royal canadian mounting police hors...</td>
      <td>0.136364</td>
      <td>0.454545</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.021103e+18</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>0</td>
      <td>10</td>
      <td>70</td>
      <td>6.100000</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>il come visit u penitentiary comi comey httpst...</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.021102e+18</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>0</td>
      <td>15</td>
      <td>105</td>
      <td>6.066667</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>comi comey mac dogg mccabe supervisor capable ...</td>
      <td>0.400000</td>
      <td>0.650000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.021060e+18</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>0</td>
      <td>17</td>
      <td>99</td>
      <td>4.882353</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>lips r moving lieing going 2 never enter kingd...</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1.021059e+18</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>0</td>
      <td>15</td>
      <td>103</td>
      <td>5.933333</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>judges prosecuters must replace new clowns bri...</td>
      <td>0.136364</td>
      <td>0.454545</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.020898e+18</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>0</td>
      <td>3</td>
      <td>31</td>
      <td>9.666667</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>evil httpstcokvexijflel</td>
      <td>-1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1.020897e+18</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>0</td>
      <td>23</td>
      <td>140</td>
      <td>5.130435</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>team one worked demonrats publicans party pick...</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1.020889e+18</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>0</td>
      <td>16</td>
      <td>140</td>
      <td>6.421053</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>white house chief staff john kellyreportedly g...</td>
      <td>0.066667</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1.020886e+18</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>0</td>
      <td>25</td>
      <td>140</td>
      <td>4.640000</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>myhtopoeic watch microwave weapon take u silen...</td>
      <td>0.000000</td>
      <td>0.300000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1.020872e+18</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>0</td>
      <td>21</td>
      <td>140</td>
      <td>5.714286</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>much money deals president trump get failing p...</td>
      <td>0.200000</td>
      <td>0.200000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1.020871e+18</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>0</td>
      <td>28</td>
      <td>140</td>
      <td>4.185185</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>8</td>
      <td>due known bad drugs used high dose 2 long kidn...</td>
      <td>-0.203000</td>
      <td>0.476333</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1.020871e+18</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>0</td>
      <td>20</td>
      <td>140</td>
      <td>6.050000</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>veteranshealth sure thankful va medical system...</td>
      <td>0.084286</td>
      <td>0.464921</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1.020869e+18</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>0</td>
      <td>20</td>
      <td>124</td>
      <td>5.250000</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>send law legislating judge poop shoot thru sli...</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1.020868e+18</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>0</td>
      <td>22</td>
      <td>140</td>
      <td>5.409091</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>u president nfl owner refused fine players nat...</td>
      <td>0.208333</td>
      <td>0.625000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1.020866e+18</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>0</td>
      <td>24</td>
      <td>140</td>
      <td>4.875000</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>long one wants slave master chicken georgetta ...</td>
      <td>-0.112500</td>
      <td>0.362500</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1.020866e+18</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>0</td>
      <td>22</td>
      <td>140</td>
      <td>5.666667</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>httpstcoetom1oxb2q oh lookie ny via ya owner t...</td>
      <td>0.416667</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1.020825e+18</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>0</td>
      <td>29</td>
      <td>140</td>
      <td>3.862069</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>killed many much blood hands lover warrior mea...</td>
      <td>0.037500</td>
      <td>0.277500</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1.020825e+18</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>0</td>
      <td>21</td>
      <td>140</td>
      <td>5.714286</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>stretch king david went battle donald hang nai...</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1.020821e+18</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>0</td>
      <td>20</td>
      <td>140</td>
      <td>6.050000</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>scum sucker spying ass senator james lankford ...</td>
      <td>-0.300000</td>
      <td>0.600000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1.020819e+18</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>0</td>
      <td>6</td>
      <td>56</td>
      <td>8.500000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>interest paying interest httpstcorafdzusdnn</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1.020748e+18</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>0</td>
      <td>10</td>
      <td>74</td>
      <td>6.500000</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>yo sanctions ever strong enough work httpstcok...</td>
      <td>0.216667</td>
      <td>0.616667</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1.020687e+18</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>0</td>
      <td>3</td>
      <td>32</td>
      <td>10.000000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>httpstco5atradkwtt</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1.020673e+18</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>0</td>
      <td>5</td>
      <td>37</td>
      <td>6.600000</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>gave httpstcogvw3aznnmw</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1.020507e+18</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>0</td>
      <td>23</td>
      <td>140</td>
      <td>5.130435</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>well like peon asses regulating oil north kore...</td>
      <td>-0.187500</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1.020506e+18</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>0</td>
      <td>16</td>
      <td>104</td>
      <td>5.562500</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>ted u r general michael hidden says interestin...</td>
      <td>0.189583</td>
      <td>0.583333</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>107108</th>
      <td>1.018458e+18</td>
      <td>8362.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>21</td>
      <td>73</td>
      <td>0</td>
      <td>1875</td>
      <td>784</td>
      <td>0</td>
      <td>15</td>
      <td>102</td>
      <td>6.285714</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>rt tweetsfor45 trump never colluded russians g...</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>107109</th>
      <td>1.018458e+18</td>
      <td>885.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>21</td>
      <td>73</td>
      <td>0</td>
      <td>1875</td>
      <td>784</td>
      <td>0</td>
      <td>23</td>
      <td>139</td>
      <td>5.086957</td>
      <td>6</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>rt cb618444 thousands trump supporters hit str...</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>107110</th>
      <td>1.018457e+18</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>21</td>
      <td>73</td>
      <td>0</td>
      <td>1875</td>
      <td>784</td>
      <td>0</td>
      <td>13</td>
      <td>79</td>
      <td>5.153846</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>alyssa_milano realdonaldtrump looks like waite...</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>107111</th>
      <td>1.018452e+18</td>
      <td>1579.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>21</td>
      <td>73</td>
      <td>0</td>
      <td>1875</td>
      <td>784</td>
      <td>0</td>
      <td>26</td>
      <td>140</td>
      <td>4.600000</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>rt fleccas want make sure right left mad reald...</td>
      <td>-0.067857</td>
      <td>0.544921</td>
    </tr>
    <tr>
      <th>107112</th>
      <td>1.018243e+18</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>21</td>
      <td>73</td>
      <td>0</td>
      <td>1875</td>
      <td>784</td>
      <td>0</td>
      <td>11</td>
      <td>73</td>
      <td>5.727273</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>foxandfriends dbongino anything try stop winni...</td>
      <td>0.500000</td>
      <td>0.750000</td>
    </tr>
    <tr>
      <th>107113</th>
      <td>1.018243e+18</td>
      <td>20388.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>21</td>
      <td>73</td>
      <td>0</td>
      <td>1875</td>
      <td>784</td>
      <td>0</td>
      <td>25</td>
      <td>140</td>
      <td>4.640000</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>rt anncoulter peter strzoks wife threatened le...</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>107114</th>
      <td>1.018243e+18</td>
      <td>5256.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>21</td>
      <td>73</td>
      <td>0</td>
      <td>1875</td>
      <td>784</td>
      <td>0</td>
      <td>26</td>
      <td>139</td>
      <td>4.384615</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>rt pink_about_it democrats drafted bill abolis...</td>
      <td>-0.625000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>107115</th>
      <td>1.018104e+18</td>
      <td>951.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>21</td>
      <td>73</td>
      <td>0</td>
      <td>1875</td>
      <td>784</td>
      <td>0</td>
      <td>26</td>
      <td>140</td>
      <td>4.423077</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>rt ladythriller69 fact people fighting keep ra...</td>
      <td>-0.100000</td>
      <td>0.566667</td>
    </tr>
    <tr>
      <th>107116</th>
      <td>1.018102e+18</td>
      <td>3187.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>21</td>
      <td>73</td>
      <td>0</td>
      <td>1875</td>
      <td>784</td>
      <td>0</td>
      <td>22</td>
      <td>140</td>
      <td>5.409091</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>rt the_trump_train anyone care explain mueller...</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>107117</th>
      <td>1.018101e+18</td>
      <td>5780.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>21</td>
      <td>73</td>
      <td>0</td>
      <td>1875</td>
      <td>784</td>
      <td>0</td>
      <td>24</td>
      <td>144</td>
      <td>5.041667</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>rt amymek media never cover protest amp march ...</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>107118</th>
      <td>1.018096e+18</td>
      <td>9545.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>21</td>
      <td>73</td>
      <td>0</td>
      <td>1875</td>
      <td>784</td>
      <td>0</td>
      <td>23</td>
      <td>140</td>
      <td>4.833333</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>rt education4libs london flew large blimp pres...</td>
      <td>-0.261905</td>
      <td>0.809524</td>
    </tr>
    <tr>
      <th>107119</th>
      <td>1.017780e+18</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>21</td>
      <td>73</td>
      <td>0</td>
      <td>1875</td>
      <td>784</td>
      <td>0</td>
      <td>8</td>
      <td>51</td>
      <td>5.500000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>rt bushido49ers cbsnews always duh</td>
      <td>-0.300000</td>
      <td>0.600000</td>
    </tr>
    <tr>
      <th>107120</th>
      <td>1.017752e+18</td>
      <td>132.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>21</td>
      <td>73</td>
      <td>0</td>
      <td>1875</td>
      <td>784</td>
      <td>0</td>
      <td>16</td>
      <td>111</td>
      <td>6.000000</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>rt foxandfriends dems drafted bill abolish ice...</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>107121</th>
      <td>1.017752e+18</td>
      <td>6519.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>21</td>
      <td>73</td>
      <td>0</td>
      <td>1875</td>
      <td>784</td>
      <td>0</td>
      <td>16</td>
      <td>128</td>
      <td>7.062500</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>rt hmmmthere never said didnt like trump suppo...</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>107122</th>
      <td>1.017751e+18</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>21</td>
      <td>73</td>
      <td>0</td>
      <td>1875</td>
      <td>784</td>
      <td>0</td>
      <td>13</td>
      <td>71</td>
      <td>4.538462</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>nbcnews heard took tag mattress discussing</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>107123</th>
      <td>1.017747e+18</td>
      <td>6761.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>21</td>
      <td>73</td>
      <td>0</td>
      <td>1875</td>
      <td>784</td>
      <td>0</td>
      <td>23</td>
      <td>140</td>
      <td>5.130435</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>rt jacobawohl huge peter strzok says inspector...</td>
      <td>0.225000</td>
      <td>0.700000</td>
    </tr>
    <tr>
      <th>107125</th>
      <td>1.017744e+18</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>21</td>
      <td>73</td>
      <td>0</td>
      <td>1875</td>
      <td>784</td>
      <td>0</td>
      <td>9</td>
      <td>54</td>
      <td>5.111111</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>cbsnews fine dandy need something</td>
      <td>0.416667</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>107137</th>
      <td>1.017836e+18</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>0</td>
      <td>158</td>
      <td>1277</td>
      <td>2</td>
      <td>71</td>
      <td>450</td>
      <td>0</td>
      <td>11</td>
      <td>91</td>
      <td>7.363636</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>uberfacts one feae fridays fear mondafghafthhh...</td>
      <td>-0.312500</td>
      <td>0.687500</td>
    </tr>
    <tr>
      <th>107141</th>
      <td>1.021307e+18</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>203</td>
      <td>1593</td>
      <td>0</td>
      <td>547</td>
      <td>256</td>
      <td>0</td>
      <td>3</td>
      <td>21</td>
      <td>6.333333</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>imanmunal naked true</td>
      <td>0.175000</td>
      <td>0.525000</td>
    </tr>
    <tr>
      <th>107142</th>
      <td>1.021306e+18</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>203</td>
      <td>1593</td>
      <td>0</td>
      <td>547</td>
      <td>256</td>
      <td>0</td>
      <td>27</td>
      <td>99</td>
      <td>1.821429</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>smile everything okay</td>
      <td>0.400000</td>
      <td>0.300000</td>
    </tr>
    <tr>
      <th>107143</th>
      <td>1.020637e+18</td>
      <td>1098.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>203</td>
      <td>1593</td>
      <td>0</td>
      <td>547</td>
      <td>256</td>
      <td>0</td>
      <td>20</td>
      <td>132</td>
      <td>5.045455</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>rt arewashams every woman deserves man loves r...</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>107144</th>
      <td>1.020460e+18</td>
      <td>37.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>203</td>
      <td>1593</td>
      <td>0</td>
      <td>547</td>
      <td>256</td>
      <td>0</td>
      <td>24</td>
      <td>126</td>
      <td>4.291667</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>rt tweets2motivate today great day say thank g...</td>
      <td>0.450000</td>
      <td>0.575000</td>
    </tr>
    <tr>
      <th>107146</th>
      <td>1.020308e+18</td>
      <td>75.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>203</td>
      <td>1593</td>
      <td>0</td>
      <td>547</td>
      <td>256</td>
      <td>0</td>
      <td>23</td>
      <td>140</td>
      <td>5.130435</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>rt hqnigerianarmy coas lt gen ty buratai yeste...</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>107147</th>
      <td>1.020308e+18</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>203</td>
      <td>1593</td>
      <td>0</td>
      <td>547</td>
      <td>256</td>
      <td>0</td>
      <td>5</td>
      <td>59</td>
      <td>11.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>hedankwambo businessdayng congratulations exce...</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>107150</th>
      <td>1.020252e+18</td>
      <td>13.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>203</td>
      <td>1593</td>
      <td>0</td>
      <td>547</td>
      <td>256</td>
      <td>0</td>
      <td>19</td>
      <td>112</td>
      <td>4.947368</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>rt cleverquotez never argue idiot theyll drag ...</td>
      <td>-0.450000</td>
      <td>0.435417</td>
    </tr>
    <tr>
      <th>107151</th>
      <td>1.019529e+18</td>
      <td>173.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>203</td>
      <td>1593</td>
      <td>0</td>
      <td>547</td>
      <td>256</td>
      <td>0</td>
      <td>23</td>
      <td>144</td>
      <td>5.000000</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>rt trackhatespeech fake news 6 people suspecte...</td>
      <td>-0.500000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>107152</th>
      <td>1.019295e+18</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>203</td>
      <td>1593</td>
      <td>0</td>
      <td>547</td>
      <td>256</td>
      <td>0</td>
      <td>8</td>
      <td>48</td>
      <td>5.125000</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>madame_flowy ask questions</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>107153</th>
      <td>1.019293e+18</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>203</td>
      <td>1593</td>
      <td>0</td>
      <td>547</td>
      <td>256</td>
      <td>0</td>
      <td>4</td>
      <td>32</td>
      <td>7.250000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>itswarenbuffett kind regard sir</td>
      <td>0.600000</td>
      <td>0.900000</td>
    </tr>
    <tr>
      <th>107156</th>
      <td>1.017668e+18</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>203</td>
      <td>1593</td>
      <td>0</td>
      <td>547</td>
      <td>256</td>
      <td>0</td>
      <td>5</td>
      <td>43</td>
      <td>7.800000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>posted photo httpstconzryhlilwg</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>107157</th>
      <td>1.020607e+18</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>71</td>
      <td>0</td>
      <td>3</td>
      <td>6</td>
      <td>0</td>
      <td>2</td>
      <td>17</td>
      <td>8.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>rssurjewala nice</td>
      <td>0.600000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>89306 rows × 20 columns</p>
</div>



## 3.4 <a id='word2vec'>Word2Vec</a>


## Download the Stanford GloVe model at http://nlp.stanford.edu/data/glove.twitter.27B.zip


```python
from gensim.models import KeyedVectors # load the Stanford GloVe model
```


```python
from gensim.scripts.glove2word2vec import glove2word2vec
glove_input_file = 'glove.twitter.27B.100d.txt'
word2vec_output_file = 'glove.twitter.27B.100d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)

```




    (1193514, 100)




```python
filename = 'glove.twitter.27B.100d.txt.word2vec'
model = KeyedVectors.load_word2vec_format(filename, binary=False)
```


```python
!pip install gensim
```

    Collecting gensim
    [?25l  Downloading https://files.pythonhosted.org/packages/cb/d9/f5adaf1108aad2b3d32a11aceede54faa5da9dbf962e9bcff759e1d27bd3/gensim-3.5.0-cp36-cp36m-macosx_10_6_intel.macosx_10_9_intel.macosx_10_9_x86_64.macosx_10_10_intel.macosx_10_10_x86_64.whl (23.7MB)
    [K    100% |████████████████████████████████| 23.8MB 440kB/s eta 0:00:01    42% |█████████████▌                  | 10.0MB 13.5MB/s eta 0:00:02    79% |█████████████████████████▌      | 18.9MB 9.2MB/s eta 0:00:01
    [?25hRequirement already satisfied: six>=1.5.0 in /anaconda/lib/python3.6/site-packages (from gensim) (1.11.0)
    Requirement already satisfied: numpy>=1.11.3 in /anaconda/lib/python3.6/site-packages (from gensim) (1.14.1)
    Requirement already satisfied: scipy>=0.18.1 in /anaconda/lib/python3.6/site-packages (from gensim) (1.0.0)
    Collecting smart-open>=1.2.1 (from gensim)
      Downloading https://files.pythonhosted.org/packages/cf/3d/5f3a9a296d0ba8e00e263a8dee76762076b9eb5ddc254ccaa834651c8d65/smart_open-1.6.0.tar.gz
    Requirement already satisfied: boto>=2.32 in /anaconda/lib/python3.6/site-packages (from smart-open>=1.2.1->gensim) (2.46.1)
    Collecting bz2file (from smart-open>=1.2.1->gensim)
      Downloading https://files.pythonhosted.org/packages/61/39/122222b5e85cd41c391b68a99ee296584b2a2d1d233e7ee32b4532384f2d/bz2file-0.98.tar.gz
    Requirement already satisfied: requests in /anaconda/lib/python3.6/site-packages (from smart-open>=1.2.1->gensim) (2.18.4)
    Collecting boto3 (from smart-open>=1.2.1->gensim)
    [?25l  Downloading https://files.pythonhosted.org/packages/68/b6/36622b7185f1a26f837c0782cac2bbe1c25331c0915e6ca72d2326aca413/boto3-1.7.64-py2.py3-none-any.whl (128kB)
    [K    100% |████████████████████████████████| 133kB 8.9MB/s eta 0:00:01   55% |█████████████████▉              | 71kB 9.8MB/s eta 0:00:01
    [?25hRequirement already satisfied: chardet<3.1.0,>=3.0.2 in /anaconda/lib/python3.6/site-packages (from requests->smart-open>=1.2.1->gensim) (3.0.4)
    Requirement already satisfied: urllib3<1.23,>=1.21.1 in /anaconda/lib/python3.6/site-packages (from requests->smart-open>=1.2.1->gensim) (1.22)
    Requirement already satisfied: idna<2.7,>=2.5 in /anaconda/lib/python3.6/site-packages (from requests->smart-open>=1.2.1->gensim) (2.6)
    Requirement already satisfied: certifi>=2017.4.17 in /anaconda/lib/python3.6/site-packages (from requests->smart-open>=1.2.1->gensim) (2017.7.27.1)
    Collecting jmespath<1.0.0,>=0.7.1 (from boto3->smart-open>=1.2.1->gensim)
      Downloading https://files.pythonhosted.org/packages/b7/31/05c8d001f7f87f0f07289a5fc0fc3832e9a57f2dbd4d3b0fee70e0d51365/jmespath-0.9.3-py2.py3-none-any.whl
    Collecting s3transfer<0.2.0,>=0.1.10 (from boto3->smart-open>=1.2.1->gensim)
    [?25l  Downloading https://files.pythonhosted.org/packages/d7/14/2a0004d487464d120c9fb85313a75cd3d71a7506955be458eebfe19a6b1d/s3transfer-0.1.13-py2.py3-none-any.whl (59kB)
    [K    100% |████████████████████████████████| 61kB 12.3MB/s ta 0:00:01
    [?25hCollecting botocore<1.11.0,>=1.10.64 (from boto3->smart-open>=1.2.1->gensim)
    [?25l  Downloading https://files.pythonhosted.org/packages/5c/2a/088f1a5344c450adf83da43c6e9e16ef51a62dc528f6785798a2a56d188c/botocore-1.10.64-py2.py3-none-any.whl (4.4MB)
    [K    100% |████████████████████████████████| 4.4MB 3.0MB/s eta 0:00:01    31% |██████████                      | 1.4MB 15.4MB/s eta 0:00:01
    [?25hRequirement already satisfied: docutils>=0.10 in /anaconda/lib/python3.6/site-packages (from botocore<1.11.0,>=1.10.64->boto3->smart-open>=1.2.1->gensim) (0.14)
    Requirement already satisfied: python-dateutil<3.0.0,>=2.1; python_version >= "2.7" in /anaconda/lib/python3.6/site-packages (from botocore<1.11.0,>=1.10.64->boto3->smart-open>=1.2.1->gensim) (2.6.1)
    Building wheels for collected packages: smart-open, bz2file
      Running setup.py bdist_wheel for smart-open ... [?25ldone
    [?25h  Stored in directory: /Users/chrislee/Library/Caches/pip/wheels/73/f1/9b/ccf93d4ba073b6f79b1ed9df68ab5ce048d8136d0efcf90b30
      Running setup.py bdist_wheel for bz2file ... [?25ldone
    [?25h  Stored in directory: /Users/chrislee/Library/Caches/pip/wheels/81/75/d6/e1317bf09bf1af5a30befc2a007869fa6e1f516b8f7c591cb9
    Successfully built smart-open bz2file
    Installing collected packages: bz2file, jmespath, botocore, s3transfer, boto3, smart-open, gensim
    Successfully installed boto3-1.7.64 botocore-1.10.64 bz2file-0.98 gensim-3.5.0 jmespath-0.9.3 s3transfer-0.1.13 smart-open-1.6.0


### I ran out of time to go further than this. In the future I would really like to see for myself what all the hype is about Word2Vec. It seems there are several Word2Vec dictionaries on the web. These dictionaries are large in size on the order of GBs. Future work for sure.


```python
# Keys are terms, values are indicies in the feature matrix 'train_vect'
tfidf.vocabulary_
```




    {'smart': 787,
     'went': 958,
     'poor': 653,
     'united': 916,
     'states': 813,
     'america': 48,
     'pay': 627,
     'rt': 733,
     'agree': 39,
     'potus': 656,
     'good': 337,
     'years': 991,
     'country': 188,
     'maga': 527,
     'wait': 935,
     'love': 524,
     'comey': 159,
     'rich': 725,
     'ask': 65,
     'donald': 246,
     'new': 584,
     'member': 550,
     'police': 648,
     'come': 157,
     'visit': 926,
     'going': 333,
     'judges': 456,
     'bring': 112,
     'evil': 268,
     'team': 848,
     'worked': 978,
     'party': 620,
     'white': 962,
     'house': 393,
     'chief': 137,
     'john': 450,
     'gave': 321,
     'gop': 339,
     'light': 498,
     'president': 662,
     'trumps': 899,
     'watch': 949,
     'open': 612,
     'ha': 353,
     'money': 566,
     'trump': 897,
     'protect': 676,
     'known': 468,
     'bad': 78,
     'used': 921,
     'high': 379,
     'long': 514,
     'say': 749,
     'caught': 129,
     'sure': 836,
     'ass': 68,
     'criminal': 195,
     'send': 763,
     'law': 478,
     'judge': 454,
     'nfl': 586,
     'fine': 290,
     'players': 645,
     'national': 576,
     'anthem': 56,
     'total': 886,
     'wants': 943,
     'break': 107,
     'oh': 606,
     'tells': 852,
     'stay': 814,
     'wont': 974,
     'killed': 464,
     'mean': 543,
     'head': 365,
     'getting': 326,
     'witch': 970,
     'senator': 762,
     'james': 441,
     'look': 516,
     'im': 409,
     'yo': 994,
     'strong': 820,
     'work': 977,
     'like': 499,
     'north': 589,
     'korea': 470,
     'tell': 850,
     'little': 508,
     'general': 322,
     'michael': 555,
     'says': 751,
     'interesting': 426,
     'dirty': 235,
     'change': 134,
     'maybe': 541,
     'needs': 582,
     'constitution': 181,
     'use': 920,
     'communist': 165,
     'late': 475,
     'right': 726,
     'rights': 727,
     'speak': 799,
     'day': 205,
     'thats': 857,
     'wrong': 987,
     'air': 40,
     'force': 299,
     'hi': 378,
     'wow': 986,
     'washington': 947,
     'shit': 775,
     'free': 308,
     'cover': 191,
     'americas': 51,
     'time': 875,
     'shut': 780,
     'really': 702,
     'shes': 774,
     'liar': 491,
     'great': 344,
     'women': 972,
     'got': 340,
     'dollars': 245,
     'make': 529,
     'mueller': 573,
     'meet': 548,
     'hell': 372,
     'cause': 130,
     'death': 212,
     'congress': 175,
     'intelligence': 425,
     'agencies': 34,
     'plan': 643,
     'worse': 982,
     'read': 693,
     'true': 895,
     'voting': 933,
     'makes': 530,
     'treasonous': 892,
     'home': 388,
     'soon': 793,
     'american': 49,
     'government': 341,
     'gets': 325,
     'god': 331,
     'usa': 919,
     'leave': 485,
     'idea': 404,
     'democrats': 223,
     'angry': 54,
     'foreign': 300,
     'nations': 577,
     'election': 257,
     'want': 941,
     'vote': 929,
     'need': 581,
     'ice': 401,
     'crooked': 196,
     'thinks': 868,
     'better': 88,
     '17': 6,
     'protecting': 677,
     'spy': 806,
     'man': 532,
     'support': 830,
     'peace': 628,
     'military': 558,
     'iranian': 432,
     'illegals': 408,
     'trust': 900,
     'uranium': 918,
     'deal': 210,
     'joe': 449,
     'queen': 685,
     '2020': 13,
     'child': 138,
     'lies': 496,
     'attacks': 71,
     'sir': 785,
     'dog': 243,
     'russians': 738,
     'lie': 494,
     'know': 467,
     'people': 633,
     'voted': 930,
     'dont': 248,
     'hillary': 380,
     'rest': 723,
     'political': 650,
     'running': 735,
     'lets': 489,
     'winning': 968,
     'hillarys': 382,
     'folks': 295,
     'reason': 704,
     'hes': 376,
     'office': 601,
     'george': 323,
     'presidents': 664,
     'realdonaldtrump': 698,
     'asked': 66,
     'help': 373,
     'community': 166,
     'let': 488,
     'fbi': 284,
     'cia': 141,
     'clinton': 151,
     'obama': 597,
     'bit': 93,
     'deep': 214,
     'state': 811,
     'officers': 603,
     'share': 773,
     'list': 505,
     'barack': 79,
     'mike': 557,
     'server': 768,
     'water': 951,
     'th': 854,
     'shows': 779,
     'doj': 244,
     'remember': 713,
     'fake': 277,
     'guy': 351,
     'elected': 256,
     'millions': 560,
     'try': 902,
     'definitely': 217,
     'attack': 69,
     '11': 2,
     'longer': 515,
     'said': 742,
     'rosenstein': 731,
     'gone': 335,
     'comes': 158,
     'inside': 422,
     'paid': 618,
     'honor': 389,
     'truth': 901,
     'politics': 651,
     'sorry': 794,
     'dead': 209,
     'life': 497,
     'living': 511,
     'corruption': 186,
     'yes': 992,
     'hit': 384,
     'resign': 721,
     'led': 486,
     'video': 924,
     'week': 955,
     'billion': 91,
     'racist': 688,
     'face': 272,
     'boy': 106,
     'live': 509,
     'care': 125,
     'big': 89,
     'small': 786,
     'senschumer': 764,
     'hear': 367,
     'emails': 260,
     'sent': 766,
     'congressman': 176,
     'elections': 258,
     'jobs': 448,
     'way': 953,
     'cyber': 199,
     'threat': 873,
     'mr': 570,
     'proud': 679,
     'dear': 211,
     'cut': 198,
     'jim': 445,
     'act': 26,
     'word': 975,
     'amen': 47,
     'world': 981,
     'days': 206,
     'ago': 38,
     'id': 403,
     'tedlieu': 849,
     'officials': 605,
     'person': 635,
     'business': 114,
     'court': 190,
     'order': 614,
     'release': 710,
     'record': 705,
     'youre': 997,
     'best': 87,
     'looking': 517,
     'amp': 52,
     'campaign': 123,
     'illegally': 407,
     'spied': 805,
     'surveillance': 837,
     'nunes': 595,
     '100': 1,
     'fisa': 292,
     'warrant': 945,
     'carter': 127,
     'page': 616,
     'prayingmedic': 660,
     'coming': 160,
     'thing': 864,
     'thank': 855,
     'lord': 519,
     'things': 865,
     'eyes': 271,
     'check': 136,
     'latest': 476,
     'media': 547,
     'early': 251,
     'morning': 568,
     'stand': 807,
     'defend': 215,
     'job': 447,
     'july': 458,
     '2018': 12,
     'today': 878,
     'london': 513,
     'antitrump': 57,
     'youtube': 998,
     'officer': 602,
     'edkrassen': 253,
     'exactly': 269,
     'russia': 736,
     'impeach': 413,
     'lying': 526,
     'lead': 480,
     'director': 234,
     'fall': 278,
     'tax': 845,
     'shouldnt': 777,
     'powerful': 658,
     'actually': 29,
     'meant': 545,
     'wouldnt': 985,
     'ill': 405,
     'hey': 377,
     'baby': 77,
     'blue': 97,
     'opinion': 613,
     'stupid': 825,
     'stop': 816,
     'putin': 682,
     'takes': 841,
     'came': 122,
     'think': 866,
     'republican': 718,
     'republicans': 719,
     'enemy': 262,
     'press': 665,
     'conference': 171,
     'clear': 149,
     'interference': 427,
     'repadamschiff': 716,
     'attacked': 70,
     'standing': 808,
     'miss': 562,
     'times': 876,
     'class': 147,
     'barackobama': 80,
     'administration': 31,
     'htt': 394,
     'private': 670,
     'meeting': 549,
     'took': 885,
     'disgusting': 237,
     'donaldjtrumpjr': 247,
     'speech': 803,
     'peter': 636,
     'strzok': 821,
     'seen': 759,
     'doesnt': 242,
     'thomas1774paine': 869,
     'democracy': 220,
     'pretty': 667,
     'secpompeo': 755,
     'sounds': 795,
     'thanks': 856,
     'view': 925,
     'war': 944,
     'wwg1wga': 988,
     'goes': 332,
     'qanon': 684,
     'fun': 316,
     'looks': 518,
     '_imperatorrex_': 17,
     'news': 585,
     'sense': 765,
     'guess': 348,
     'patriot': 623,
     'faith': 276,
     'reminder': 714,
     'question': 686,
     'family': 281,
     'paulsperry_': 626,
     'hate': 363,
     'hope': 390,
     'future': 319,
     'past': 622,
     'kind': 465,
     'china': 140,
     'showing': 778,
     'suffer': 827,
     'info': 419,
     'dbongino': 207,
     'answer': 55,
     'yeah': 989,
     'ready': 695,
     'talking': 844,
     'correct': 184,
     'understand': 913,
     'proof': 674,
     'agents': 37,
     'working': 980,
     'didnt': 229,
     'play': 644,
     'facebook': 273,
     'waiting': 936,
     'start': 809,
     'lol': 512,
     'dems': 224,
     'ok': 608,
     'russian': 737,
     'matter': 539,
     'attention': 72,
     'dnc': 239,
     'twitter': 911,
     'lost': 521,
     'followers': 297,
     'msm': 571,
     'set': 770,
     'google': 338,
     'public': 681,
     'report': 717,
     'welcome': 957,
     'propaganda': 675,
     'left': 487,
     'power': 657,
     'swamp': 838,
     'happy': 361,
     'missing': 563,
     'giving': 329,
     'nato': 578,
     'paul': 625,
     'fox': 304,
     'bernie': 86,
     'weak': 954,
     'wh': 960,
     'friends': 313,
     'economy': 252,
     'treason': 891,
     'imagine': 410,
     'end': 261,
     'patriots': 624,
     'cnn': 154,
     'point': 647,
     'questions': 687,
     'americans': 50,
     'real': 696,
     'rand': 690,
     'enjoy': 264,
     'kenya': 461,
     'walkaway': 939,
     'lose': 520,
     'turned': 906,
     'democratic': 222,
     'possible': 654,
     'wake': 937,
     'knows': 469,
     'pray': 659,
     'investigation': 429,
     'foundation': 303,
     'directly': 233,
     'old': 609,
     'bob': 98,
     'history': 383,
     'loudobbs': 523,
     'hunt': 400,
     'agent': 36,
     'butina': 115,
     'forget': 301,
     'democrat': 221,
     'retweet': 724,
     'iran': 431,
     'threaten': 874,
     'documents': 241,
     'confirm': 172,
     'steele': 815,
     'dossier': 249,
     'liberal': 492,
     'tomfitton': 881,
     'docs': 240,
     'major': 528,
     'realize': 699,
     'realjameswoods': 701,
     'hillaryclinton': 381,
     'social': 788,
     'message': 553,
     'attorney': 73,
     'million': 559,
     'httpst': 396,
     'immigration': 411,
     'human': 399,
     'single': 784,
     'biggest': 90,
     'breaking': 108,
     'brennan': 110,
     'security': 758,
     'key': 462,
     'pedophile': 629,
     'abuse': 22,
     'close': 153,
     'hand': 357,
     'arrest': 62,
     'army': 61,
     'laws': 479,
     'tweetsfor45': 910,
     'hold': 386,
     'including': 415,
     'allowed': 44,
     '2016': 10,
     'corrupt': 185,
     'companies': 167,
     'trade': 889,
     'secret': 756,
     '10': 0,
     'leadership': 483,
     'policy': 649,
     'thread': 872,
     'charliekirk11': 135,
     'european': 266,
     'countries': 187,
     'completely': 170,
     'calling': 120,
     'bless': 95,
     'krisparonto': 472,
     'pedophilia': 631,
     'sick': 781,
     'literally': 507,
     'obamas': 598,
     'brian': 111,
     'governor': 342,
     'georgia': 324,
     'tough': 888,
     'crime': 193,
     'successful': 826,
     'greatest': 345,
     'trying': 903,
     'control': 183,
     'told': 880,
     'strzoks': 822,
     'crimes': 194,
     'released': 711,
     'claims': 145,
     'freedom': 309,
     'intel': 424,
     'application': 59,
     'away': 74,
     'jacobawohl': 439,
     'wanted': 942,
     'far': 282,
     'congratulations': 174,
     'judicialwatch': 457,
     'collusion': 156,
     'evidence': 267,
     'education4libs': 254,
     'called': 119,
     'sold': 791,
     'justice': 459,
     'amazing': 45,
     'believe': 84,
     'andrew': 53,
     'happen': 358,
     'signed': 783,
     'knew': 466,
     'based': 81,
     'entire': 265,
     'funny': 318,
     'members': 551,
     'anymore': 58,
     'ryanafournier': 739,
     'important': 414,
     'randpaul': 691,
     'derangement': 226,
     'syndrome': 839,
     'senate': 760,
     'pages': 617,
     'records': 706,
     'abc': 18,
     'nice': 587,
     'raise': 689,
     'rape': 692,
     'unverified': 917,
     'research': 720,
     'watching': 950,
     'explain': 270,
     'hollywood': 387,
     '12': 3,
     'tonight': 883,
     'false': 279,
     'arrested': 63,
     'listen': 506,
     'demand': 219,
     'half': 356,
     'california': 118,
     'ca': 117,
     'case': 128,
     'problem': 673,
     'candidate': 124,
     'supports': 834,
     'lauraloomer': 477,
     'saracarterdc': 745,
     'hard': 362,
     'wasnt': 948,
     'source': 796,
     'httpstco': 397,
     'devinnunes': 227,
     'following': 298,
     'complete': 169,
     'clearly': 150,
     'article': 64,
     'thelastrefuge2': 860,
     'tired': 877,
     'whoopi': 964,
     'join': 452,
     'senatemajldr': 761,
     'icegov': 402,
     'isnt': 433,
     'stuff': 824,
     'red': 707,
     'clapper': 146,
     'clean': 148,
     'boom': 102,
     'jackposobiec': 438,
     'totally': 887,
     'admits': 32,
     'started': 810,
     'probe': 672,
     'border': 103,
     'socialism': 789,
     'illegal': 406,
     'foxnews': 306,
     'telling': 851,
     'liberals': 493,
     'presidential': 663,
     'calls': 121,
     'marklevinshow': 537,
     'whitehouse': 963,
     'kavanaugh': 460,
     'redacted': 708,
     'saying': 750,
     'information': 420,
     'tv': 907,
     'realcandaceo': 697,
     'different': 231,
     'johnbrennan': 451,
     'conservatives': 179,
     'glad': 330,
     'https': 395,
     'citizen': 142,
     'photo': 639,
     'children': 139,
     'reading': 694,
     'confirmed': 173,
     'lisa': 503,
     'according': 23,
     'disgrace': 236,
     'traitor': 890,
     'sign': 782,
     'lied': 495,
     'aliens': 42,
     'turn': 905,
     'born': 105,
     'black': 94,
     'ocasio2018': 599,
     'freedom_moates': 310,
     'protest': 678,
     'tweet': 908,
     'federal': 286,
     'leaders': 482,
     'rep': 715,
     'asking': 67,
     'risk': 728,
     'woman': 971,
     'sex': 772,
     'body': 99,
     'babies': 76,
     'conservative': 178,
     'jeanine': 443,
     'group': 347,
     'mitchellvii': 564,
     'rouhani': 732,
     'consequences': 177,
     'likes': 501,
     'walk': 938,
     'department': 225,
     'marcorubio': 534,
     'fact': 274,
     'sunday': 829,
     'lives': 510,
     'effort': 255,
     'outside': 615,
     'saw': 748,
     'november': 591,
     'dc': 208,
     'worth': 984,
     'nearly': 580,
     'accused': 25,
     'lot': 522,
     'benghazi': 85,
     'weve': 959,
     'comment': 161,
     'story': 818,
     'statement': 812,
     'goldberg': 334,
     'roseanne': 730,
     'pirro': 641,
     'whats': 961,
     'realjack': 700,
     'secretservice': 757,
     'service': 769,
     'special': 802,
     '13': 4,
     'bombshell': 100,
     'dineshdsouza': 232,
     'year': 990,
     'spent': 804,
     'youve': 999,
     'young': 996,
     'men': 552,
     'pass': 621,
     'fired': 291,
     'interview': 428,
     'kwilli1046': 473,
     'save': 747,
     'cernovich': 132,
     'pedophiles': 630,
     'sources': 797,
     'robert': 729,
     'judgejeanine': 455,
     'seanhannity': 753,
     'gunn': 350,
     'tweets': 909,
     'comments': 162,
     '400': 15,
     'jamesgunn': 442,
     'hoax': 385,
     'beautiful': 82,
     'disney': 238,
     'mind': 561,
     '1776stonewall': 7,
     'yesterday': 993,
     'met': 554,
     'theview': 862,
     'ive': 437,
     'favorite': 283,
     'fraud': 307,
     'presidency': 661,
     'mcfaul': 542,
     'whoopigoldberg': 965,
     'tony': 884,
     'podesta': 646,
     'facts': 275,
     'petition': 637,
     'account': 24,
     'immunity': 412,
     'manafort': 533,
     'words': 976,
     '18': 8,
     'govmikehuckabee': 343,
     'finally': 289,
     'ambassador': 46,
     'die': 230,
     'workers': 979,
     'flotus': 294,
     'students': 823,
     'prove': 680,
     'post': 655,
     'parents': 619,
     'shot': 776,
     'weeks': 956,
     'huge': 398,
     'learn': 484,
     'buy': 116,
     'official': 604,
     'fear': 285,
     'given': 328,
     'fight': 288,
     'voice': 928,
     'run': 734,
     'dem': 218,
     'foxandfriends': 305,
     'prisonplanet': 669,
     'likely': 500,
     'win': 967,
     'helsinki': 374,
     'meddling': 546,
     'summit': 828,
     'making': 531,
     'safe': 741,
     'agenda': 35,
     '20': 9,
     'son': 792,
     'actions': 28,
     'unhinged': 914,
     'thought': 870,
     'heard': 368,
     'joke': 453,
     'presssec': 666,
     'straight': 819,
     'vp': 934,
     'leader': 481,
     'taking': 842,
     'action': 27,
     'regime': 709,
     'happened': 359,
     'tried': 893,
     'supporter': 831,
     'level': 490,
     'sarah': 746,
     'sanders': 744,
     'wish': 969,
     'probably': 671,
     'ivankatrump': 436,
     'ingrahamangle': 421,
     'kids': 463,
     'indictment': 417,
     'held': 371,
     'note': 590,
     'mark': 536,
     'hours': 392,
     'todays': 879,
     'speakerryan': 800,
     'course': 189,
     'doubt': 250,
     'vladimir': 927,
     'low': 525,
     'msnbc': 572,
     'behavior': 83,
     'sad': 740,
     'damn': 202,
     'performance': 634,
     'putins': 683,
     'feel': 287,
     'indicted': 416,
     'using': 922,
     'worst': 983,
     'indictments': 418,
     'stories': 817,
     'arent': 60,
     'afraid': 33,
     'seriously': 767,
     'theres': 861,
     'nra': 592,
     'thehill': 859,
     'fuck': 314,
     'nation': 575,
     'talk': 843,
     'wife': 966,
     'follow': 296,
     'ones': 611,
     'krassenstein': 471,
     'uk': 912,
     'truly': 896,
     'absolutely': 21,
     'wonder': 973,
     'hot': 391,
     'conspiracy': 180,
     'months': 567,
     'forward': 302,
     'hearing': 369,
     'citizens': 143,
     'texas': 853,
     'second': 754,
     'allow': 43,
     'committed': 163,
     'clintons': 152,
     'heres': 375,
     'warrants': 946,
     'gun': 349,
     'place': 642,
     'supporters': 832,
     'moment': 565,
     'debate': 213,
     'troy': 894,
     'ohio': 607,
     'nancy': 574,
     'pelosi': 632,
     'borders': 104,
     'friend': 312,
     'jim_jordan': 446,
     'marklutchman': 538,
     'voter': 931,
     'happening': 360,
     'trumpputin': 898,
     'theyre': 863,
     'instead': 423,
     'speaking': 801,
     'building': 113,
     'book': 101,
     'omg': 610,
     'israel': 434,
     'committee': 164,
     'breitbartnews': 109,
     'jesus': 444,
     'supreme': 835,
     'families': 280,
     '30': 14,
     'birthday': 92,
     'thinking': 867,
     'san': 743,
     'mother': 569,
     'michaelavenatti': 556,
     'funder': 317,
     'cohen': 155,
     'voters': 932,
     'kylegriffin1': 474,
     'veterans': 923,
     'awesome': 75,
     'jail': 440,
     'piece': 640,
     'game': 320,
     'crazy': 192,
     'nbcnews': 579,
     'girl': 327,
     'friday': 311,
     'hacking': 355,
     'health': 366,
     'the_trump_train': 858,
     'wall': 940,
     'thousands': 871,
     'hacked': 354,
     'grizzlemeister': 346,
     'dangerous': 203,
     'issue': 435,
     'involved': 430,
     'defending': 216,
     'continue': 182,
     'flag': 293,
     'night': 588,
     'taken': 840,
     'nuclear': 593,
     'school': 752,
     'daily': 200,
     'adamparkhomenko': 30,
     '15': 5,
     'nytimes': 596,
     'means': 544,
     'chance': 133,
     'tuckercarlson': 904,
     'davidhogg111': 204,
     'havent': 364,
     'socialist': 790,
     'respect': 722,
     'heart': 370,
     'supporting': 833,
     'fucking': 315,
     'gonna': 336,
     'york': 995,
     '2017': 11,
     'number': 594,
     'union': 915,
     'sethabramson': 771,
     'able': 19,
     'tomorrow': 882,
     'maxine': 540,
     'waters': 952,
     'energy': 263,
     'line': 502,
     'releasethetrumptapes': 712,
     'dailycaller': 201,
     'maria': 535,
     'phone': 638,
     'diamondandsilk': 228,
     'taxes': 846,
     'city': 144,
     'email': 259,
     'south': 798,
     'career': 126,
     'company': 168,
     'guys': 352,
     '50': 16,
     'prison': 668,
     'al': 41,
     'ocasiocortez': 600,
     'poll': 652,
     'lisamei62': 504,
     'blocked': 96,
     'abolish': 20,
     'netflix': 583,
     'realmagasteve': 703,
     'cc': 131,
     'taylorhicks': 847,
     'csharpcorner': 197}



### The following three cells helped me identify the most frequently used term to be 'rt'. Note that 'RT' is an acronym for "reply to", so this makes sense.


```python
np.argmin(tfidf.idf_)
```




    733




```python
list(tfidf.vocabulary_.values()).index(733)
```




    7




```python
list(tfidf.vocabulary_.keys())[7]
```




    'rt'




```python
tf1=(df.text[0]).apply(lambda x: pd.value_counts(x.split(' '))).sum(axis=0)
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-126-95f400db1e88> in <module>()
    ----> 1 tf1=(df.text[0]).apply(lambda x: pd.value_counts(x.split(' '))).sum(axis=0)


    AttributeError: 'str' object has no attribute 'apply'



```python
tf1.columns=['words','tf']
```


```python
tf1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>went</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>trained</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>r</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>dope</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>murdering</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>sales</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>wa</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>classes</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>murderers</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>vamcs</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>college</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>httpstco4aluzu5crs</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>smart</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>poor</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>states</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>u</td>
      <td>36.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>httpstcokwimj6hkrh</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>pay</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>cor</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>land</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>united</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>corporation</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>one</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>owns</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>httpstcozs4reg46iq</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>must</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>america</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>rt</td>
      <td>36.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>potus</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1355</th>
      <td>httpstcohlkttmfeja</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1356</th>
      <td>httpstcosfewfr0hxy</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1357</th>
      <td>claremlopez</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1358</th>
      <td>heshmatalavi</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1359</th>
      <td>thomaswictor</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1360</th>
      <td>kjtorrance</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1361</th>
      <td>drawandstrike</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1362</th>
      <td>debbieaaldrich</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1363</th>
      <td>candicemalcolm</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1364</th>
      <td>hnijohnmiller</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1365</th>
      <td>background</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1366</th>
      <td>thegreatawakening</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1367</th>
      <td>flynns</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1368</th>
      <td>relevant</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1369</th>
      <td>q</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1370</th>
      <td>role</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1371</th>
      <td>common</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1372</th>
      <td>focus</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1373</th>
      <td>potential</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1374</th>
      <td>httpstcoe5gamqpyxt</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1375</th>
      <td>denominator</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1376</th>
      <td>flynn</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1377</th>
      <td>theater</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1378</th>
      <td>sooner</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1379</th>
      <td>later</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1380</th>
      <td>guess</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1381</th>
      <td>httpstcoaw9d7adlyu</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1382</th>
      <td>ended</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1383</th>
      <td>patriot</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1384</th>
      <td>arrows</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>1385 rows × 2 columns</p>
</div>




```python
stop = stopwords.words('english')
```


    ---------------------------------------------------------------------------

    LookupError                               Traceback (most recent call last)

    /anaconda/lib/python3.6/site-packages/nltk/corpus/util.py in __load(self)
         79             except LookupError as e:
    ---> 80                 try: root = nltk.data.find('{}/{}'.format(self.subdir, zip_name))
         81                 except LookupError: raise e


    /anaconda/lib/python3.6/site-packages/nltk/data.py in find(resource_name, paths)
        652     resource_not_found = '\n%s\n%s\n%s' % (sep, msg, sep)
    --> 653     raise LookupError(resource_not_found)
        654


    LookupError:
    **********************************************************************
      Resource 'corpora/stopwords.zip/stopwords/' not found.  Please
      use the NLTK Downloader to obtain the resource:  >>>
      nltk.download()
      Searched in:
        - '/Users/chrislee/nltk_data'
        - '/usr/share/nltk_data'
        - '/usr/local/share/nltk_data'
        - '/usr/lib/nltk_data'
        - '/usr/local/lib/nltk_data'
    **********************************************************************


    During handling of the above exception, another exception occurred:


    LookupError                               Traceback (most recent call last)

    <ipython-input-61-3b5c63e2db5c> in <module>()
    ----> 1 stop = stopwords.words('english')


    /anaconda/lib/python3.6/site-packages/nltk/corpus/util.py in __getattr__(self, attr)
        114             raise AttributeError("LazyCorpusLoader object has no attribute '__bases__'")
        115
    --> 116         self.__load()
        117         # This looks circular, but its not, since __load() changes our
        118         # __class__ to something new:


    /anaconda/lib/python3.6/site-packages/nltk/corpus/util.py in __load(self)
         79             except LookupError as e:
         80                 try: root = nltk.data.find('{}/{}'.format(self.subdir, zip_name))
    ---> 81                 except LookupError: raise e
         82
         83         # Load the corpus.


    /anaconda/lib/python3.6/site-packages/nltk/corpus/util.py in __load(self)
         76         else:
         77             try:
    ---> 78                 root = nltk.data.find('{}/{}'.format(self.subdir, self.__name))
         79             except LookupError as e:
         80                 try: root = nltk.data.find('{}/{}'.format(self.subdir, zip_name))


    /anaconda/lib/python3.6/site-packages/nltk/data.py in find(resource_name, paths)
        651     sep = '*' * 70
        652     resource_not_found = '\n%s\n%s\n%s' % (sep, msg, sep)
    --> 653     raise LookupError(resource_not_found)
        654
        655


    LookupError:
    **********************************************************************
      Resource 'corpora/stopwords' not found.  Please use the NLTK
      Downloader to obtain the resource:  >>> nltk.download()
      Searched in:
        - '/Users/chrislee/nltk_data'
        - '/usr/share/nltk_data'
        - '/usr/local/share/nltk_data'
        - '/usr/lib/nltk_data'
        - '/usr/local/lib/nltk_data'
    **********************************************************************



```python
from nltk.corpus import stopwords
```


```python
stop=set(stopwords.words('english'))
```


    ---------------------------------------------------------------------------

    LookupError                               Traceback (most recent call last)

    /anaconda/lib/python3.6/site-packages/nltk/corpus/util.py in __load(self)
         79             except LookupError as e:
    ---> 80                 try: root = nltk.data.find('{}/{}'.format(self.subdir, zip_name))
         81                 except LookupError: raise e


    /anaconda/lib/python3.6/site-packages/nltk/data.py in find(resource_name, paths)
        652     resource_not_found = '\n%s\n%s\n%s' % (sep, msg, sep)
    --> 653     raise LookupError(resource_not_found)
        654


    LookupError:
    **********************************************************************
      Resource 'corpora/stopwords.zip/stopwords/' not found.  Please
      use the NLTK Downloader to obtain the resource:  >>>
      nltk.download()
      Searched in:
        - '/Users/chrislee/nltk_data'
        - '/usr/share/nltk_data'
        - '/usr/local/share/nltk_data'
        - '/usr/lib/nltk_data'
        - '/usr/local/lib/nltk_data'
    **********************************************************************


    During handling of the above exception, another exception occurred:


    LookupError                               Traceback (most recent call last)

    <ipython-input-67-6b0174d93592> in <module>()
    ----> 1 stop=set(stopwords.words('english'))


    /anaconda/lib/python3.6/site-packages/nltk/corpus/util.py in __getattr__(self, attr)
        114             raise AttributeError("LazyCorpusLoader object has no attribute '__bases__'")
        115
    --> 116         self.__load()
        117         # This looks circular, but its not, since __load() changes our
        118         # __class__ to something new:


    /anaconda/lib/python3.6/site-packages/nltk/corpus/util.py in __load(self)
         79             except LookupError as e:
         80                 try: root = nltk.data.find('{}/{}'.format(self.subdir, zip_name))
    ---> 81                 except LookupError: raise e
         82
         83         # Load the corpus.


    /anaconda/lib/python3.6/site-packages/nltk/corpus/util.py in __load(self)
         76         else:
         77             try:
    ---> 78                 root = nltk.data.find('{}/{}'.format(self.subdir, self.__name))
         79             except LookupError as e:
         80                 try: root = nltk.data.find('{}/{}'.format(self.subdir, zip_name))


    /anaconda/lib/python3.6/site-packages/nltk/data.py in find(resource_name, paths)
        651     sep = '*' * 70
        652     resource_not_found = '\n%s\n%s\n%s' % (sep, msg, sep)
    --> 653     raise LookupError(resource_not_found)
        654
        655


    LookupError:
    **********************************************************************
      Resource 'corpora/stopwords' not found.  Please use the NLTK
      Downloader to obtain the resource:  >>> nltk.download()
      Searched in:
        - '/Users/chrislee/nltk_data'
        - '/usr/share/nltk_data'
        - '/usr/local/share/nltk_data'
        - '/usr/lib/nltk_data'
        - '/usr/local/lib/nltk_data'
    **********************************************************************



```python
nltk.download()
```

    showing info https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/index.xml



    ---------------------------------------------------------------------------

    UnicodeDecodeError                        Traceback (most recent call last)

    <ipython-input-70-a1a554e5d735> in <module>()
    ----> 1 nltk.download()


    /anaconda/lib/python3.6/site-packages/nltk/downloader.py in download(self, info_or_id, download_dir, quiet, force, prefix, halt_on_error, raise_on_error)
        659             # function should make a new copy of self to use?
        660             if download_dir is not None: self._download_dir = download_dir
    --> 661             self._interactive_download()
        662             return True
        663


    /anaconda/lib/python3.6/site-packages/nltk/downloader.py in _interactive_download(self)
        980         if TKINTER:
        981             try:
    --> 982                 DownloaderGUI(self).mainloop()
        983             except TclError:
        984                 DownloaderShell(self).run()


    /anaconda/lib/python3.6/site-packages/nltk/downloader.py in mainloop(self, *args, **kwargs)
       1715
       1716     def mainloop(self, *args, **kwargs):
    -> 1717         self.top.mainloop(*args, **kwargs)
       1718
       1719     #/////////////////////////////////////////////////////////////////


    /anaconda/lib/python3.6/tkinter/__init__.py in mainloop(self, n)
       1275     def mainloop(self, n=0):
       1276         """Call the mainloop of Tk."""
    -> 1277         self.tk.mainloop(n)
       1278     def quit(self):
       1279         """Quit the Tcl interpreter. All widgets will be destroyed."""


    UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte
