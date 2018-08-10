
### S109A Final Project Submission Group 22 (Project Website: https://mathuser0.github.io)

Christopher Lee, chl2967@g.harvard.edu  
Sriganesh Pera, srp124@g.harvard.edu  
Bo Shang, bshang@g.harvard.edu
    
****

# Part 3. Natural Language Processing

In this section, we will only be dealing with the `text` column data. The aim of this part is to extract as many meaningful features as we can from the `text` portion of a Tweet object. First we extract the low-level features from the text directly as it is. Then, we will transform the text to prepare the data for advanced features extraction. Finally, in the last section of this part, we will explore natural language processing techniques such as N-grams, TF/IDF, Sentiment Analysis, and Word2Vec.

----

## Import Libraries


```python
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# Natural Language Processing Imports
from nltk.corpus import stopwords        # stopwords
from nltk.stem import PorterStemmer  # 2.6 Stemming - Removal of suffices like 'ing','ly','s', etc.
from textblob import Word                # 2.7 Lemmatization - Converting to root word
from textblob import TextBlob            # 3.1 N-grams (also for spell correction)
from sklearn.feature_extraction.text import TfidfVectorizer # 3.2 Term Frequency, 3.3 Inverse Document Frequency
from gensim.models import KeyedVectors   # 3.5 Word2Vec
```

----

## Custom Functions & Load Saved Data


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

# Getting Saved data from Part 2
df_complete = pd.read_csv("complete.csv", index_col=0)
```

----

## Get Text Data


```python
# We will only be dealing with the text columns in this part.
df_text = df_complete[['text','user_screen_name']]
```


```python
df_text.tail()
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
      <th>user_screen_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>89299</th>
      <td>RT @TrackHateSpeech: FAKE NEWS!\n\n6 people su...</td>
      <td>Gombe1Isah</td>
    </tr>
    <tr>
      <th>89300</th>
      <td>@Madame_Flowy Why do You ask all this questions?</td>
      <td>Gombe1Isah</td>
    </tr>
    <tr>
      <th>89301</th>
      <td>@itswarenbuffett Kind regard Sir</td>
      <td>Gombe1Isah</td>
    </tr>
    <tr>
      <th>89302</th>
      <td>Just posted a photo https://t.co/NZryHlilWG</td>
      <td>Gombe1Isah</td>
    </tr>
    <tr>
      <th>89303</th>
      <td>@rssurjewala Nice</td>
      <td>Jitender_shakya</td>
    </tr>
  </tbody>
</table>
</div>



----

## Basic Feature Extraction from Text Data

The following basic features can be extracted from the text data as-is. They include features such as number of words, number of characters, average word length, number of stopwords, number of special characters, number of numerics, and number of uppercase words. The cells below describe the feature being extracted along with the methodology do get the basic feature.


```python
# Creating column of word count in each tweet
df_text['word_count']  = df_text['text'].apply(lambda x: len(str(x).split(" ")))
```


```python
# Creating column of character count in each tweet
df_text['char_count']  = df_text['text'].str.len()
```


```python
# Creating column of average word length in each tweet
df_text['avg_word']    = df_text['text'].apply(lambda x: avg_word(x))
```


```python
# Getting number of stopwords in tweet
df_text['stopwords']   = df_text['text'].apply(lambda x: len([x for x in x.split() if x in stop]))
```


```python
# Getting hashtags in tweets
df_text['hashtags']    = df_text['text'].apply(lambda x: [x for x in x.split() if x.startswith('#')])
```


```python
# Counting number of  hashtags in tweets
df_text['num_hashtags']= df_text['text'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
```


```python
# Getting number of numerics in tweet
df_text['numerics']    = df_text['text'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
```


```python
# Getting number of upper case words in tweet
df_text['upper']       = df_text['text'].apply(lambda x: len([y for y in x.split() if y.isupper()]))
```


```python
df_text.head()
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
      <th>user_screen_name</th>
      <th>word_count</th>
      <th>char_count</th>
      <th>avg_word</th>
      <th>stopwords</th>
      <th>hashtags</th>
      <th>num_hashtags</th>
      <th>numerics</th>
      <th>upper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>VAMC'S R TRAINED MURDERERS THE SMART WHO WENT ...</td>
      <td>fbibug</td>
      <td>23</td>
      <td>140</td>
      <td>5.130435</td>
      <td>4</td>
      <td>[]</td>
      <td>0</td>
      <td>1</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>https://t.co/KwImJ6hKrh  no one owns land in U...</td>
      <td>fbibug</td>
      <td>21</td>
      <td>140</td>
      <td>6.000000</td>
      <td>6</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>RT @ROHLL5: 🔁 RT if you agree\n\n#44 .@POTUS\n...</td>
      <td>fbibug</td>
      <td>20</td>
      <td>137</td>
      <td>4.782609</td>
      <td>6</td>
      <td>[#44, #ObamaLegacy, #PatriotsUnite, #MAGA…]</td>
      <td>4</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Wait next is Burnie you will love comi COMEY a...</td>
      <td>fbibug</td>
      <td>13</td>
      <td>77</td>
      <td>5.000000</td>
      <td>4</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>It's simple landlords get rich quick scheme. A...</td>
      <td>fbibug</td>
      <td>21</td>
      <td>140</td>
      <td>5.714286</td>
      <td>4</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



----

## Text Pre-Processing for Advanced Feature Extraction

In order to extract meaningful high-level features, we want to clean the data so that our acquisition of data isn't hindered by slight differences in the text. For example, we want 'Dog' and 'dog' to be considered equal. Therefore, we pre-process the text data before applying advanced extraction methods as follows. 

### Lower Case

To avoid multiple copies of the same word.


```python
# Split the text, lower case each word, and then rejoin
df_text.text=df_text.text.apply(lambda x: ' '.join(x.lower() for x in x.split()))
```

### Remove Punctuation

Punctuation doesn't add any extra information while treating text data.


```python
# Use regular expressions to replace any character that is not a word character or whitespace with an empty string
df_text.text=df_text.text.str.replace('[^\w\s]','')
```

### Remove Stopwords
Stop words (commonly occurring words) should be removed. Notice that we defined the variable `stop` to be a list of stopwords above.


```python
# Remove words that are in our list of stopwords
df_text.text=df_text.text.apply(lambda x:' '.join(y for y in x.split() if y not in stop))
```

### Words Frequency

Removing frequent words in our text is different from removing stopwords which occur commonly in general.


```python
# Get the 10 most frequent words
freq = pd.Series(' '.join(df_text.text).split()).value_counts()[:10]
```


```python
freq
```




    rt                 54588
    trump               8719
    realdonaldtrump     8281
    president           4598
    amp                 4595
    people              3640
    us                  3584
    one                 3303
    like                3300
    dont                2873
    dtype: int64



### Spelling Correction

Correcting mis-spelled words can have a significant impact on analysis performance. However, it takes a really long time to do, so this is left for future work.


```python
# This takes a long time to do. Note that we use the textblob library to do spelling correction.
df_text['text_spell_corrected']=df_text.text.apply(lambda x: str(TextBlob(x).correct()))
```

### Stemming 
Removal of Suffices like 'ing', 'ly', 's', etc. We can use PorterStemmer() from the NLTK library.


```python
# We do not apply this in our current project, but this should be something to do in the future.
st = PorterStemmer()
df_text.text.apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
```

### Lemmatization
Converting to root word. Lemmatization is more effective than stemming because it converts the word into its root word rather than just stripping the suffices. Lemmatization is usually preferred over stemming.


```python
# We use Word() from the textblob library to lemmatize
df_text['lemmatized'] = df_text.text.apply(lambda x: ' '.join([Word(word).lemmatize() for word in x.split()]))
```


```python
df_text
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
      <th>user_screen_name</th>
      <th>word_count</th>
      <th>char_count</th>
      <th>avg_word</th>
      <th>stopwords</th>
      <th>hashtags</th>
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
      <td>fbibug</td>
      <td>23</td>
      <td>140</td>
      <td>5.130435</td>
      <td>4</td>
      <td>[]</td>
      <td>0</td>
      <td>1</td>
      <td>10</td>
      <td>vamcs r trained murderer smart went 2 college ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>httpstcokwimj6hkrh one owns land united states...</td>
      <td>fbibug</td>
      <td>21</td>
      <td>140</td>
      <td>6.000000</td>
      <td>6</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>httpstcokwimj6hkrh one owns land united state ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>rt rohll5 rt agree 44 potus harm good 8 years ...</td>
      <td>fbibug</td>
      <td>20</td>
      <td>137</td>
      <td>4.782609</td>
      <td>6</td>
      <td>[#44, #ObamaLegacy, #PatriotsUnite, #MAGA…]</td>
      <td>4</td>
      <td>1</td>
      <td>5</td>
      <td>rt rohll5 rt agree 44 potus harm good 8 year s...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>wait next burnie love comi comey ag httpstco3p...</td>
      <td>fbibug</td>
      <td>13</td>
      <td>77</td>
      <td>5.000000</td>
      <td>4</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>wait next burnie love comi comey ag httpstco3p...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>simple landlords get rich quick scheme ask don...</td>
      <td>fbibug</td>
      <td>21</td>
      <td>140</td>
      <td>5.714286</td>
      <td>4</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>simple landlord get rich quick scheme ask dona...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>new member royal canadian mounting police hors...</td>
      <td>fbibug</td>
      <td>18</td>
      <td>136</td>
      <td>6.611111</td>
      <td>6</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>new member royal canadian mounting police hors...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>il come visit u penitentiary comi comey httpst...</td>
      <td>fbibug</td>
      <td>10</td>
      <td>70</td>
      <td>6.100000</td>
      <td>2</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>il come visit u penitentiary comi comey httpst...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>comi comey mac dogg mccabe supervisor capable ...</td>
      <td>fbibug</td>
      <td>15</td>
      <td>105</td>
      <td>6.066667</td>
      <td>4</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>comi comey mac dogg mccabe supervisor capable ...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>lips r moving lieing going 2 never enter kingd...</td>
      <td>fbibug</td>
      <td>17</td>
      <td>99</td>
      <td>4.882353</td>
      <td>3</td>
      <td>[]</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>lip r moving lieing going 2 never enter kingdo...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>judges prosecuters must replace new clowns bri...</td>
      <td>fbibug</td>
      <td>15</td>
      <td>103</td>
      <td>5.933333</td>
      <td>6</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>judge prosecuters must replace new clown bring...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>evil httpstcokvexijflel</td>
      <td>fbibug</td>
      <td>3</td>
      <td>31</td>
      <td>9.666667</td>
      <td>0</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>evil httpstcokvexijflel</td>
    </tr>
    <tr>
      <th>11</th>
      <td>team one worked demonrats publicans party pick...</td>
      <td>fbibug</td>
      <td>23</td>
      <td>140</td>
      <td>5.130435</td>
      <td>11</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>team one worked demonrats publican party pick ...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>white house chief staff john kellyreportedly g...</td>
      <td>fbibug</td>
      <td>16</td>
      <td>140</td>
      <td>6.421053</td>
      <td>3</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>white house chief staff john kellyreportedly g...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>myhtopoeic watch microwave weapon take u silen...</td>
      <td>fbibug</td>
      <td>25</td>
      <td>140</td>
      <td>4.640000</td>
      <td>12</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>myhtopoeic watch microwave weapon take u silen...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>much money deals president trump get failing p...</td>
      <td>fbibug</td>
      <td>21</td>
      <td>140</td>
      <td>5.714286</td>
      <td>2</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>much money deal president trump get failing pr...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>due known bad drugs used high dose 2 long kidn...</td>
      <td>fbibug</td>
      <td>28</td>
      <td>140</td>
      <td>4.185185</td>
      <td>7</td>
      <td>[]</td>
      <td>0</td>
      <td>1</td>
      <td>8</td>
      <td>due known bad drug used high dose 2 long kidne...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>veteranshealth sure thankful va medical system...</td>
      <td>fbibug</td>
      <td>20</td>
      <td>140</td>
      <td>6.050000</td>
      <td>5</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>veteranshealth sure thankful va medical system...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>send law legislating judge poop shoot thru sli...</td>
      <td>fbibug</td>
      <td>20</td>
      <td>124</td>
      <td>5.250000</td>
      <td>8</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>send law legislating judge poop shoot thru sli...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>u president nfl owner refused fine players nat...</td>
      <td>fbibug</td>
      <td>22</td>
      <td>140</td>
      <td>5.409091</td>
      <td>7</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>u president nfl owner refused fine player nati...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>long one wants slave master chicken georgetta ...</td>
      <td>fbibug</td>
      <td>24</td>
      <td>140</td>
      <td>4.875000</td>
      <td>9</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>long one want slave master chicken georgetta t...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>httpstcoetom1oxb2q oh lookie ny via ya owner t...</td>
      <td>fbibug</td>
      <td>22</td>
      <td>140</td>
      <td>5.666667</td>
      <td>3</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>httpstcoetom1oxb2q oh lookie ny via ya owner t...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>killed many much blood hands lover warrior mea...</td>
      <td>fbibug</td>
      <td>29</td>
      <td>140</td>
      <td>3.862069</td>
      <td>13</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>killed many much blood hand lover warrior mean...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>stretch king david went battle donald hang nai...</td>
      <td>fbibug</td>
      <td>21</td>
      <td>140</td>
      <td>5.714286</td>
      <td>3</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>stretch king david went battle donald hang nai...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>scum sucker spying ass senator james lankford ...</td>
      <td>fbibug</td>
      <td>20</td>
      <td>140</td>
      <td>6.050000</td>
      <td>2</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>scum sucker spying as senator james lankford a...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>interest paying interest httpstcorafdzusdnn</td>
      <td>fbibug</td>
      <td>6</td>
      <td>56</td>
      <td>8.500000</td>
      <td>1</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>interest paying interest httpstcorafdzusdnn</td>
    </tr>
    <tr>
      <th>25</th>
      <td>yo sanctions ever strong enough work httpstcok...</td>
      <td>fbibug</td>
      <td>10</td>
      <td>74</td>
      <td>6.500000</td>
      <td>2</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>yo sanction ever strong enough work httpstcok5...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>httpstco5atradkwtt</td>
      <td>fbibug</td>
      <td>3</td>
      <td>32</td>
      <td>10.000000</td>
      <td>1</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>httpstco5atradkwtt</td>
    </tr>
    <tr>
      <th>27</th>
      <td>gave httpstcogvw3aznnmw</td>
      <td>fbibug</td>
      <td>5</td>
      <td>37</td>
      <td>6.600000</td>
      <td>2</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>gave httpstcogvw3aznnmw</td>
    </tr>
    <tr>
      <th>28</th>
      <td>well like peon asses regulating oil north kore...</td>
      <td>fbibug</td>
      <td>23</td>
      <td>140</td>
      <td>5.130435</td>
      <td>7</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>well like peon ass regulating oil north korea ...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>ted u r general michael hidden says interestin...</td>
      <td>fbibug</td>
      <td>16</td>
      <td>104</td>
      <td>5.562500</td>
      <td>3</td>
      <td>[]</td>
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
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>89274</th>
      <td>rt tweetsfor45 trump never colluded russians g...</td>
      <td>charlaynek</td>
      <td>15</td>
      <td>102</td>
      <td>6.285714</td>
      <td>4</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>rt tweetsfor45 trump never colluded russian gu...</td>
    </tr>
    <tr>
      <th>89275</th>
      <td>rt cb618444 thousands trump supporters hit str...</td>
      <td>charlaynek</td>
      <td>23</td>
      <td>139</td>
      <td>5.086957</td>
      <td>6</td>
      <td>[#Trump, #London, #Trump]</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>rt cb618444 thousand trump supporter hit stree...</td>
    </tr>
    <tr>
      <th>89276</th>
      <td>alyssa_milano realdonaldtrump looks like waite...</td>
      <td>charlaynek</td>
      <td>13</td>
      <td>79</td>
      <td>5.153846</td>
      <td>6</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>alyssa_milano realdonaldtrump look like waited...</td>
    </tr>
    <tr>
      <th>89277</th>
      <td>rt fleccas want make sure right left mad reald...</td>
      <td>charlaynek</td>
      <td>26</td>
      <td>140</td>
      <td>4.600000</td>
      <td>8</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>rt fleccas want make sure right left mad reald...</td>
    </tr>
    <tr>
      <th>89278</th>
      <td>foxandfriends dbongino anything try stop winni...</td>
      <td>charlaynek</td>
      <td>11</td>
      <td>73</td>
      <td>5.727273</td>
      <td>4</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>foxandfriends dbongino anything try stop winni...</td>
    </tr>
    <tr>
      <th>89279</th>
      <td>rt anncoulter peter strzoks wife threatened le...</td>
      <td>charlaynek</td>
      <td>25</td>
      <td>140</td>
      <td>4.640000</td>
      <td>11</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>rt anncoulter peter strzoks wife threatened le...</td>
    </tr>
    <tr>
      <th>89280</th>
      <td>rt pink_about_it democrats drafted bill abolis...</td>
      <td>charlaynek</td>
      <td>26</td>
      <td>139</td>
      <td>4.384615</td>
      <td>11</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>rt pink_about_it democrat drafted bill abolish...</td>
    </tr>
    <tr>
      <th>89281</th>
      <td>rt ladythriller69 fact people fighting keep ra...</td>
      <td>charlaynek</td>
      <td>26</td>
      <td>140</td>
      <td>4.423077</td>
      <td>12</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>rt ladythriller69 fact people fighting keep ra...</td>
    </tr>
    <tr>
      <th>89282</th>
      <td>rt the_trump_train anyone care explain mueller...</td>
      <td>charlaynek</td>
      <td>22</td>
      <td>140</td>
      <td>5.409091</td>
      <td>7</td>
      <td>[]</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>rt the_trump_train anyone care explain mueller...</td>
    </tr>
    <tr>
      <th>89283</th>
      <td>rt amymek media never cover protest amp march ...</td>
      <td>charlaynek</td>
      <td>24</td>
      <td>144</td>
      <td>5.041667</td>
      <td>8</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>rt amymek medium never cover protest amp march...</td>
    </tr>
    <tr>
      <th>89284</th>
      <td>rt education4libs london flew large blimp pres...</td>
      <td>charlaynek</td>
      <td>23</td>
      <td>140</td>
      <td>4.833333</td>
      <td>8</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>rt education4libs london flew large blimp pres...</td>
    </tr>
    <tr>
      <th>89285</th>
      <td>rt bushido49ers cbsnews always duh</td>
      <td>charlaynek</td>
      <td>8</td>
      <td>51</td>
      <td>5.500000</td>
      <td>1</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>rt bushido49ers cbsnews always duh</td>
    </tr>
    <tr>
      <th>89286</th>
      <td>rt foxandfriends dems drafted bill abolish ice...</td>
      <td>charlaynek</td>
      <td>16</td>
      <td>111</td>
      <td>6.000000</td>
      <td>5</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>rt foxandfriends dems drafted bill abolish ice...</td>
    </tr>
    <tr>
      <th>89287</th>
      <td>rt hmmmthere never said didnt like trump suppo...</td>
      <td>charlaynek</td>
      <td>16</td>
      <td>128</td>
      <td>7.062500</td>
      <td>0</td>
      <td>[#Strzok, #StrzokHearing]</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>rt hmmmthere never said didnt like trump suppo...</td>
    </tr>
    <tr>
      <th>89288</th>
      <td>nbcnews heard took tag mattress discussing</td>
      <td>charlaynek</td>
      <td>13</td>
      <td>71</td>
      <td>4.538462</td>
      <td>4</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>nbcnews heard took tag mattress discussing</td>
    </tr>
    <tr>
      <th>89289</th>
      <td>rt jacobawohl huge peter strzok says inspector...</td>
      <td>charlaynek</td>
      <td>23</td>
      <td>140</td>
      <td>5.130435</td>
      <td>8</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>rt jacobawohl huge peter strzok say inspector ...</td>
    </tr>
    <tr>
      <th>89290</th>
      <td>cbsnews fine dandy need something</td>
      <td>charlaynek</td>
      <td>9</td>
      <td>54</td>
      <td>5.111111</td>
      <td>3</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>cbsnews fine dandy need something</td>
    </tr>
    <tr>
      <th>89291</th>
      <td>uberfacts one feae fridays fear mondafghafthhh...</td>
      <td>yaaaq1</td>
      <td>11</td>
      <td>91</td>
      <td>7.363636</td>
      <td>2</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>uberfacts one feae friday fear mondafghafthhhv...</td>
    </tr>
    <tr>
      <th>89292</th>
      <td>imanmunal naked true</td>
      <td>Gombe1Isah</td>
      <td>3</td>
      <td>21</td>
      <td>6.333333</td>
      <td>0</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>imanmunal naked true</td>
    </tr>
    <tr>
      <th>89293</th>
      <td>smile everything okay</td>
      <td>Gombe1Isah</td>
      <td>27</td>
      <td>99</td>
      <td>1.821429</td>
      <td>2</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>smile everything okay</td>
    </tr>
    <tr>
      <th>89294</th>
      <td>rt arewashams every woman deserves man loves r...</td>
      <td>Gombe1Isah</td>
      <td>20</td>
      <td>132</td>
      <td>5.045455</td>
      <td>7</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>rt arewashams every woman deserves man love re...</td>
    </tr>
    <tr>
      <th>89295</th>
      <td>rt tweets2motivate today great day say thank g...</td>
      <td>Gombe1Isah</td>
      <td>24</td>
      <td>126</td>
      <td>4.291667</td>
      <td>8</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>rt tweets2motivate today great day say thank g...</td>
    </tr>
    <tr>
      <th>89296</th>
      <td>rt hqnigerianarmy coas lt gen ty buratai yeste...</td>
      <td>Gombe1Isah</td>
      <td>23</td>
      <td>140</td>
      <td>5.130435</td>
      <td>5</td>
      <td>[]</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>rt hqnigerianarmy coas lt gen ty buratai yeste...</td>
    </tr>
    <tr>
      <th>89297</th>
      <td>hedankwambo businessdayng congratulations exce...</td>
      <td>Gombe1Isah</td>
      <td>5</td>
      <td>59</td>
      <td>11.000000</td>
      <td>0</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>hedankwambo businessdayng congratulation excel...</td>
    </tr>
    <tr>
      <th>89298</th>
      <td>rt cleverquotez never argue idiot theyll drag ...</td>
      <td>Gombe1Isah</td>
      <td>19</td>
      <td>112</td>
      <td>4.947368</td>
      <td>9</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>rt cleverquotez never argue idiot theyll drag ...</td>
    </tr>
    <tr>
      <th>89299</th>
      <td>rt trackhatespeech fake news 6 people suspecte...</td>
      <td>Gombe1Isah</td>
      <td>23</td>
      <td>144</td>
      <td>5.000000</td>
      <td>7</td>
      <td>[]</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>rt trackhatespeech fake news 6 people suspecte...</td>
    </tr>
    <tr>
      <th>89300</th>
      <td>madame_flowy ask questions</td>
      <td>Gombe1Isah</td>
      <td>8</td>
      <td>48</td>
      <td>5.125000</td>
      <td>3</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>madame_flowy ask question</td>
    </tr>
    <tr>
      <th>89301</th>
      <td>itswarenbuffett kind regard sir</td>
      <td>Gombe1Isah</td>
      <td>4</td>
      <td>32</td>
      <td>7.250000</td>
      <td>0</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>itswarenbuffett kind regard sir</td>
    </tr>
    <tr>
      <th>89302</th>
      <td>posted photo httpstconzryhlilwg</td>
      <td>Gombe1Isah</td>
      <td>5</td>
      <td>43</td>
      <td>7.800000</td>
      <td>1</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>posted photo httpstconzryhlilwg</td>
    </tr>
    <tr>
      <th>89303</th>
      <td>rssurjewala nice</td>
      <td>Jitender_shakya</td>
      <td>2</td>
      <td>17</td>
      <td>8.000000</td>
      <td>0</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>rssurjewala nice</td>
    </tr>
  </tbody>
</table>
<p>89304 rows × 11 columns</p>
</div>



----

## Advanced Text Processing

Now are ready to extract advanced features using Natural Language Processing techniques.

### N-grams

N-grams are the combination of multiple words used together. Ngrams with N=1 are called unigrams. Similarly, bigrams (N=2), trigrams (N=3) and so on can also be used.

Unigrams do not usually contain as much information as compared to bigrams and trigrams. The basic principle behind n-grams is that they capture the language structure, like what letter or word is likely to follow the given one. The longer the n-gram (the higher the n), the more context you have to work with. Optimum length really depends on the application – if your n-grams are too short, you may fail to capture important differences. On the other hand, if they are too long, you may fail to capture the “general knowledge” and only stick to particular cases. [[1]](https://www.analyticsvidhya.com/blog/2018/02/the-different-methods-deal-text-data-predictive-python/)




```python
TextBlob(df_text.text[0]).ngrams(2)
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



### Term Frequency
Term frequency is simply the ratio of the count of a word present in a sentence, to the length of the sentence.

Therefore, we can generalize term frequency as:

TF = (Number of times term T appears in the particular row) / (number of terms in that row)


```python
tf1=(df_text['text'][1:2]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf1.columns=['words','tf']
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
      <th>words</th>
      <th>tf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>owns</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>united</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>httpstcokwimj6hkrh</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>corporation</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>states</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>cor</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>must</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>land</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>u</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>httpstcozs4reg46iq</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>pay</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>america</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>one</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### Term Frequency/Inverse Document Frequency (TF-IDF)
The intuition behind inverse document frequency (IDF) is that a word is not of much use to us if it’s appearing in all the documents.

Therefore, the IDF of each word is the log of the ratio of the total number of rows to the number of rows in which that word is present.

IDF = log(N/n), where, N is the total number of rows and n is the number of rows in which the word was present.

TF-IDF is the product of the TF and IDF. We can use the sklearn library to calculate the TF-IDF.


```python
# Use the TfidfVectorizer transformer to fit_transform our text data
tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word', stop_words= 'english',ngram_range=(1,1))
train_vect = tfidf.fit_transform(df_text.text)
```


```python
# The result is a sparse matrix - only 447,626 elements are non-zero out of 89,304,000.
train_vect
```




    <89304x1000 sparse matrix of type '<class 'numpy.float64'>'
    	with 447626 stored elements in Compressed Sparse Row format>




```python
# Displaying the first 15 words in our
list(tfidf.vocabulary_.keys())[:15]
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
     'wait']




```python
df_tfidf=pd.DataFrame(train_vect.toarray(), columns=list(tfidf.vocabulary_.keys()))
```


```python
df_tfidf['index']=df_text.index
```


```python
df_tfidf.set_index('index')
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
      <th>poll</th>
      <th>lisamei62</th>
      <th>blocked</th>
      <th>abolish</th>
      <th>netflix</th>
      <th>realmagasteve</th>
      <th>cc</th>
      <th>taylorhicks</th>
      <th>csharpcorner</th>
      <th>index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>89299</th>
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
      <td>89299</td>
    </tr>
    <tr>
      <th>89300</th>
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
      <td>89300</td>
    </tr>
    <tr>
      <th>89301</th>
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
      <td>89301</td>
    </tr>
    <tr>
      <th>89302</th>
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
      <td>89302</td>
    </tr>
    <tr>
      <th>89303</th>
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
      <td>89303</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1001 columns</p>
</div>



### Sentiment Analysis
The textblob library is able to get the sentiment of a tweet text for us. There are two sentiment analysis values which are sentiment polarity and sentiment subjectivity. We use these NLP features in our models in Part 5. 


```python
df_text['sentiment_polarity']=df_text['text'].apply(lambda x: TextBlob(x).sentiment[0])
df_text['sentiment_subjectivity']=df_text['text'].apply(lambda x: TextBlob(x).sentiment[1])
```

### Word Embeddings (Word2Vec)
The Stanford GloVe model can be downloaded at http://nlp.stanford.edu/data/glove.twitter.27B.zip. 


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
df_word2vec=df.text.apply(lambda x: sum([model[y] for y in x.split()])/len(x.split()))
```

## This was as far as I was able to get with extracting advanced features. What struck me as surprising was how similar the libraries for extracting NLP features seemed to be to the regression/classification models we learned in class.  I have no doubt that, given more time, I would not fail at extracting all of the features described here in this section and figuring out a way to put them to use. 

## Please note that due to limited time and resources, our models only consider sentiment polarity and sentiment subjectivity as advanced text features NLP. I look forward to re-visiting NLP techniques in the very near future. 


```python
# Just for the record...
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

