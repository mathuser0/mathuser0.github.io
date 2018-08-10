
### S109A Final Project Submission Group 22 (Project Website: https://mathuser0.github.io)

Christopher Lee, chl2967@g.harvard.edu  
Sriganesh Pera, srp124@g.harvard.edu  
Bo Shang, bshang@g.harvard.edu
    
****

# Part 2. Data Pre-Processing

----

The main goal of data processing is to prepare a dataset that our models can be trained on. In particular, our aim is to accomplish the following: 

1. Rid the dataset of error entries  
1. Simplify dataset where we can  
1. Turn the dataset into a numerical dataset  

Note that we will keep all raw data that is already in numerical format. In the next section, NLP, we will extract additional features using natural language processing techniques on the tweets' text data.

Here, we extract the raw data features that are already in numerical format, identify error rows, and reduce the dataset to English tweets data only. Then, at the last step, we create a label column for our pre-processed features dataset. 

 

<hr>

 ## Import Libraries & Define Custom Functions


```python
import pandas as pd
import numpy as np
from IPython.display import display
import warnings
warnings.filterwarnings('ignore')

# Create DataFrame with columns of interest from a list of dictionaries
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
    (dictionary_list, nested_object, dict_keys, error_indices_list) = tup
    
    # Make sure no error rows exist
    if error_indices_list !=[]:
        print("Drop error rows first.")
        return
    
    else:
        print("No error rows detected. Proceeding...")
    
    # Make sure that the destination dataframe has the same number of observations 
    # as the number of 'user' objects in our list
    
    if len(df_destination)!=len(dictionary_list):
        print("Number of rows don't match. Aborting...")
        return
    else:
        print("Everything looks good. Proceeding to add selected nested child objects to the destination dataframe...")
    
    for j,key in enumerate(dict_keys):
        try:
            df_destination[nested_object+'_'+key]=[d[key] for d in dictionary_list]
        except:
            print('ERROR')
            break
    
    print("Success. The destination dataframe had ", len(dict_keys)," columns added")
    

# Function to add metadata (nested values) from a source dataframe to a destination dataframe    
def add_metadata(df_destination, df_source, nested_object, dict_keys):
    add_Nested_Columns(df_destination,to_Dictionary_List(df_source, nested_object, dict_keys))

    
# Select only the rows with a value of 'val' 
# for column 'col' in dataframe 'df'
def select(val, col, df):
    df_result=df[df[col]==val]
    print("Length of original dataset:", len(df))
    print("Length of selection dataset:", len(df_result))
    return df_result

# Returns Numeric Columns in DataFrame
def getNumericColumns(df, numeric_columns=None):
    if numeric_columns==None:
        numeric_columns=['id','retweet_count','favorite_count',
                 'user_verified','user_followers_count','user_friends_count',
                 'user_listed_count','user_favourites_count','user_statuses_count'
                 ,'user_geo_enabled'
                ]
    return df[numeric_columns]


# Returns DataFrame with Non-numeric Columns
def getNonNumericColumns(df, non_numerics=None):
    if not non_numerics:
        numeric_columns=['id','retweet_count','favorite_count',
                 'user_verified','user_followers_count','user_friends_count',
                 'user_listed_count','user_favourites_count','user_statuses_count'
                 ,'user_geo_enabled'
                ]
        non_numerics= [col for col in df.columns if col not in numeric_columns]
    return df[non_numerics]


# Returns predictors that have only one value (single-valued predictors)
def getOneValueColumns(df):
    result=[]
    for col in df.columns:
        if len(df[col].unique())==1:
            result.append(col)
    print("Single value columns are:", result)
    return result


# Drops columns in place
def dropColumns(df, columns_to_drop):
    df.drop(columns_to_drop, axis=1, inplace=True)
    print(columns_to_drop, "have been dropped successfully.")
    return None


# Finds out how many unique values are in each column of the input dataframe
def how_many_uniques(df):
    for col in df.columns:
        try:
            print(col,len(df[col].unique()))
        except:
            print(col, "length undefined")
    return None


# Returns the column names of single-valued columns
def which_are_unique(df):
    result = []
    for col in df.columns:
        if len(df[col].unique())==1:
            result.append(col)
    return result


# Finds out how many NaN values exist for each column
def how_many_na(df):
    dataf=pd.DataFrame(index=df.columns,columns=['NaNs'])
    for i,col in enumerate(df.columns):
        try:
            dataf.loc[col,'NaNs']=sum(df[col].isna())
        except:
            dataf.loc[col,'NaNs']='length undefined'
    display(dataf)
    return None


# Finds columns whose majority of values are NaNs
def which_have_too_many_na(df):
    result=[]
    for col in df.columns:
        if sum(df[col].isna())>len(df)/2:
            result.append(col)
    return result
```


```python
# Loading the raw_features.csv and raw_labels.csv datasets from Part 1.
df_features=pd.read_csv("raw_features.csv", low_memory=False)
df_labels = pd.read_csv("raw_labels.csv", index_col=0)
```

<hr>

## Feature Engineering (non-NLP)

The cell below shows the many predictors we can get from a Tweet object. We have commented out the predictors that we determined to be infeasible for this project. The columns names in color are the predictors of interest to us. 


```python
# Keys in Tweet object that we are interested in
columns_from_tweets=['created_at',             # Time the tweet was created
                     'id',                     # ID of this Tweet
                     'in_reply_to_screen_name',# If this Tweet is a reply, contains original Tweet's author
                     'lang',                   # Machine-detected language of the Tweet text
                     'possibly_sensitive',     # Indicates that this Tweet contains a link
                     'source',                 # Utility used to post this Tweet
                     'is_quote_status',        # Indicates whether this is a quoted Tweet
                     'retweet_count',          # Number of times this Tweet has been retweeted
                     'favorite_count',         # Indicates how many times this Tweet has been liked by Twitter users
                     'truncated',              # Indicates whether the value of the text parameter was truncated
                     'text'                    # The text content of the Tweet
                   # 'quote_count',            # Number of times this Tweet has been quoted
                   # 'reply_count',            # Number of times this Tweet has been replied to
                    ]

# Keys in Tweet object whose values are nested dictionaries
nested_dictionary_columns=[
    'user',                    # Data on user who posted this Tweet
#    'coordinates',            # Represents that geographic location of this Tweet 
#    'place',                  # When present, indicates that the tweet is associated with a place
#    'metadata',               # Contains 
#    'retweeted_status',       # Contains Tweet object of original Tweet
#    'entities',               # Entities which have been parsed out of the text of the Tweet
#    'extended_entities'       # Contains an array 'media' metadata
#    'quoted_status',          # Contains Tweet object of original Tweet that was quoted  
#    'retweeted_status',       # Contains Tweet object of original Tweet that was retweeted  
    ]



# Keys of nested dictionaries

############################################################
# Keys in User object
user_object_keys=[
#    "id",                       # unique identifier for this User
    "id_str",                   # string representation of the unique identifier for this User
    "name",                     # Display name of user
    "screen_name",              # Screen name of user
    "location",                 # user-defined location for this account
    "url",                      # URL provided by the user in association with their profile
    "description",              # User-defined UTF-8 string describing their account
    "verified",                 # When true, indicates that the user has a verified account
    "followers_count",          # Number of followers this account currently has. Under duress may display 0
    "friends_count",            # Number of users this account is following
    "listed_count",             # Number of public lists that this user is a member of
    "favourites_count",         # Number of Tweets this user has liked in the account’s lifetime
    "statuses_count",           # Number of Tweets (including retweets) issued by the user
    "created_at",               # UTC datetime that the user account was created on Twitter
#    "utc_offset",               # Value will be set to null.
#    "time_zone",                # Value will be set to null.
    "geo_enabled",              # When true, indicates that the user has enabled the possibility of geotagging their Tweets
    "lang",                     # User’s self-declared user interface language
    "profile_image_url_https"   # A HTTP-based URL pointing to the user’s profile image
]


############################################################
# Keys in Coordinates object
coordinates_object_keys=[
#    'coordinates',
#    'type'
]


############################################################
# Keys in Place object
place_object_keys=[
#    'attributes',
#    'bounding_box',  
#    'country',
    'country_code', # Two letter country code
#    'full_name',
#    'id',
#    'name',
#    'place_type',   # Indicates type of place (ex. city...)
#    'url'
]


############################################################
# Keys in Entities object
entities_object_keys=[
    "hashtags",         # List of hashtag data:
                        #   'indices': [start position of hashtag (#), stop position of hashtag]
                        #   'text'   : name of hashtag
    "urls",
    "user_mentions",    # List of other Twitter users mentioned in text data:
                        #   'name'       : Display name of other user
                        #   'indices'    : [start position of user mention (@), stop position of user mention]
                        #   'screen_name': Screen name of other user
                        #   'id'         : id of other user
                        #   'id_str'     : id_str of other user
#    "media",
#    "symbols",
#    "polls"
]


############################################################
# Keys in Extended Entities object
extended_entities_object_keys=[
    "media"
]
```

<hr>

### Getting Root-Level Features from Tweet Objects

Below are the root-level attributes of Tweet objects that we want to use as features. The first portion of our features dataset is gathered here at the root-level. 


```python
# Displaying the columns names of interest to us
columns_from_tweets
```




    ['created_at',
     'id',
     'in_reply_to_screen_name',
     'lang',
     'possibly_sensitive',
     'source',
     'is_quote_status',
     'retweet_count',
     'favorite_count',
     'truncated',
     'text']




```python
# Selecting just the columns of interest to us
df_clean = df_features[columns_from_tweets]
```


```python
# Displaying the first three rows
df_clean.tail(3)
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
      <th>created_at</th>
      <th>id</th>
      <th>in_reply_to_screen_name</th>
      <th>lang</th>
      <th>possibly_sensitive</th>
      <th>source</th>
      <th>is_quote_status</th>
      <th>retweet_count</th>
      <th>favorite_count</th>
      <th>truncated</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>107155</th>
      <td>Tue Jul 17 09:58:16 +0000 2018</td>
      <td>1.019159e+18</td>
      <td>th_saleem</td>
      <td>in</td>
      <td>NaN</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>@th_saleem Hallaka kobo at gombe</td>
    </tr>
    <tr>
      <th>107156</th>
      <td>Fri Jul 13 07:13:41 +0000 2018</td>
      <td>1.017668e+18</td>
      <td>NaN</td>
      <td>en</td>
      <td>False</td>
      <td>&lt;a href="http://instagram.com" rel="nofollow"&gt;...</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>Just posted a photo https://t.co/NZryHlilWG</td>
    </tr>
    <tr>
      <th>107157</th>
      <td>Sat Jul 21 09:50:09 +0000 2018</td>
      <td>1.020607e+18</td>
      <td>rssurjewala</td>
      <td>en</td>
      <td>NaN</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>@rssurjewala Nice</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Notice how 'df_clean' has only 11 columns at this point
df_clean.shape
```




    (107158, 11)




```python
# Showing column names of our current clean feature set
list(df_clean.columns)
```




    ['created_at',
     'id',
     'in_reply_to_screen_name',
     'lang',
     'possibly_sensitive',
     'source',
     'is_quote_status',
     'retweet_count',
     'favorite_count',
     'truncated',
     'text']



<hr>

### Getting Nested-Level Features from Tweet Objects

Here we are going to extract the features in the nested dictionaries of the `user` column. Recall that the `user` column is one of the root-level features in a Tweet object. The `user` column takes on values that are dictionaries. In the next cell, we declare the keys in the `user`-nested dictionaries that we want to include as columns in our features dataset.


```python
# Note that we are also interested in the following 
# child objects that are nested in the 'user' column values
user_object_keys
```




    ['id_str',
     'name',
     'screen_name',
     'location',
     'url',
     'description',
     'verified',
     'followers_count',
     'friends_count',
     'listed_count',
     'favourites_count',
     'statuses_count',
     'created_at',
     'geo_enabled',
     'lang',
     'profile_image_url_https']




```python
# We use our custom function to extract the 'user'-nested child objects.
# Note that for this particular run, there is only one item ('user') in 'nested_dictionary_columns'.
# However, it is worthwhile mentioning that the code below can handle several nested dictionary columns at a time.

for item in nested_dictionary_columns:
    print("Adding "+item)
    add_metadata(df_clean,df_features,item, eval(item+"_object_keys"))
```

    Adding user
    Length of source dataset is     :  107158
    Drop index 103723
    Drop index 103724
    WAIT! Try again after dropping these indices from original dataframe: [103723, 103724]
    Drop error rows first.



> <font color='red'>Notice the warning message telling us to drop error rows first.  
The `add_meta_data()` function also tells us which indices are error rows. Let's remove them and try again. </font>




```python
# Dropping the two error rows found at indices [103723, 103724]
df_features.drop(index=[103723,103724], inplace=True)
```


```python
# Making sure they got dropped from our destination dataframe as well
df_clean = df_features[columns_from_tweets]
```


```python
# Let's try that again now. 
for item in nested_dictionary_columns:
    print("Adding "+item)
    add_metadata(df_clean,df_features,item, eval(item+"_object_keys"))
```

    Adding user
    Length of source dataset is     :  107156
    No error rows detected. Proceeding...
    Everything looks good. Proceeding to add selected nested child objects to the destination dataframe...
    Success. The destination dataframe had  16  columns added


> <font color='blue'>Great! It worked, and now we should have 16 more columnns from the user-nested child objects added to our features dataset, `df_clean`. </font>


```python
# Let's check that claim...
df_clean.head(3)
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
      <th>created_at</th>
      <th>id</th>
      <th>in_reply_to_screen_name</th>
      <th>lang</th>
      <th>possibly_sensitive</th>
      <th>source</th>
      <th>is_quote_status</th>
      <th>retweet_count</th>
      <th>favorite_count</th>
      <th>truncated</th>
      <th>...</th>
      <th>user_verified</th>
      <th>user_followers_count</th>
      <th>user_friends_count</th>
      <th>user_listed_count</th>
      <th>user_favourites_count</th>
      <th>user_statuses_count</th>
      <th>user_created_at</th>
      <th>user_geo_enabled</th>
      <th>user_lang</th>
      <th>user_profile_image_url_https</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mon Jul 23 06:22:31 +0000 2018</td>
      <td>1.021279e+18</td>
      <td>NaN</td>
      <td>en</td>
      <td>False</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>True</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>Wed Aug 30 18:28:30 +0000 2017</td>
      <td>False</td>
      <td>en</td>
      <td>https://pbs.twimg.com/profile_images/903036839...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Mon Jul 23 06:09:39 +0000 2018</td>
      <td>1.021276e+18</td>
      <td>NaN</td>
      <td>en</td>
      <td>False</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>Wed Aug 30 18:28:30 +0000 2017</td>
      <td>False</td>
      <td>en</td>
      <td>https://pbs.twimg.com/profile_images/903036839...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mon Jul 23 02:52:48 +0000 2018</td>
      <td>1.021227e+18</td>
      <td>NaN</td>
      <td>en</td>
      <td>NaN</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>682.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>Wed Aug 30 18:28:30 +0000 2017</td>
      <td>False</td>
      <td>en</td>
      <td>https://pbs.twimg.com/profile_images/903036839...</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 27 columns</p>
</div>




```python
# A better way might be to just check the column names.
df_clean.columns
```




    Index(['created_at', 'id', 'in_reply_to_screen_name', 'lang',
           'possibly_sensitive', 'source', 'is_quote_status', 'retweet_count',
           'favorite_count', 'truncated', 'text', 'user_id_str', 'user_name',
           'user_screen_name', 'user_location', 'user_url', 'user_description',
           'user_verified', 'user_followers_count', 'user_friends_count',
           'user_listed_count', 'user_favourites_count', 'user_statuses_count',
           'user_created_at', 'user_geo_enabled', 'user_lang',
           'user_profile_image_url_https'],
          dtype='object')



> NOTE: The recently added features above have the prefix `user_`. That is because they are child objects nested in the `user` column. If we had chosen the `entities` column to extract nested child objects from, the prefix would have been `entities_`. 

<hr>

### English Data Only! 

Remember that we are using the `english` score from the Botometer API. Thus, we want only English tweets to comprise our dataset. In our current features data, there are two predictors that have language-related values: `lang` from the root-level, and `user_lang` that used to be a user-nested child object.


```python
# Using our custom-defined function, select(), we select only the 
# rows that have 'en' as their 'lang' attribute. 
# Notice how the size of the original and reduced datasets are displayed in the output.

df_clean=select('en','lang',df_clean)
```

    Length of original dataset: 107156
    Length of selection dataset: 91423



```python
# Just like above, we select only the rows that have 'en' as their 'user_lang' value.
df_clean=select('en','user_lang',df_clean)
```

    Length of original dataset: 91423
    Length of selection dataset: 89306



```python
# Now we check to see which columns are single-valued, so that we can get rid of them.
which_are_unique(df_clean)
```




    ['lang', 'user_lang']




```python
# Notice how the columns we used to select only 'en' values are shown to be single-valued. 
# Good. That means it works. Again, using another custom-defined function, dropColumns(), 
# we drop the unique valued columns. 
dropColumns(df_clean, which_are_unique(df_clean))
```

    ['lang', 'user_lang'] have been dropped successfully.



```python
# Now, we decided we don't want to deal with NaNs for this project. 
# We already have plans for getting plenty of data that aren't NaNs.
# Our custom function, which_have_too_many_na(), shows us the columns whose majority of values are NaNs.
which_have_too_many_na(df_clean)
```




    ['in_reply_to_screen_name', 'possibly_sensitive', 'user_url']




```python
# Using dropColumns() again, this time we get rid of the NaN-infested columns.
dropColumns(df_clean,which_have_too_many_na(df_clean))
```

    ['in_reply_to_screen_name', 'possibly_sensitive', 'user_url'] have been dropped successfully.



```python
# Just to show you what our features dataset looks like at this point,
df_clean.head(3)

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
      <th>created_at</th>
      <th>id</th>
      <th>source</th>
      <th>is_quote_status</th>
      <th>retweet_count</th>
      <th>favorite_count</th>
      <th>truncated</th>
      <th>text</th>
      <th>user_id_str</th>
      <th>user_name</th>
      <th>...</th>
      <th>user_description</th>
      <th>user_verified</th>
      <th>user_followers_count</th>
      <th>user_friends_count</th>
      <th>user_listed_count</th>
      <th>user_favourites_count</th>
      <th>user_statuses_count</th>
      <th>user_created_at</th>
      <th>user_geo_enabled</th>
      <th>user_profile_image_url_https</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mon Jul 23 06:22:31 +0000 2018</td>
      <td>1.021279e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>True</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>True</td>
      <td>VAMC'S R TRAINED MURDERERS THE SMART WHO WENT ...</td>
      <td>902961240691101696</td>
      <td>prepare mathew 25</td>
      <td>...</td>
      <td>Follower of YESHUA  JESUS the CHRIST</td>
      <td>False</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>Wed Aug 30 18:28:30 +0000 2017</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/903036839...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Mon Jul 23 06:09:39 +0000 2018</td>
      <td>1.021276e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>True</td>
      <td>https://t.co/KwImJ6hKrh  no one owns land in U...</td>
      <td>902961240691101696</td>
      <td>prepare mathew 25</td>
      <td>...</td>
      <td>Follower of YESHUA  JESUS the CHRIST</td>
      <td>False</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>Wed Aug 30 18:28:30 +0000 2017</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/903036839...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mon Jul 23 02:52:48 +0000 2018</td>
      <td>1.021227e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>682.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>RT @ROHLL5: 🔁 RT if you agree\n\n#44 .@POTUS\n...</td>
      <td>902961240691101696</td>
      <td>prepare mathew 25</td>
      <td>...</td>
      <td>Follower of YESHUA  JESUS the CHRIST</td>
      <td>False</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>Wed Aug 30 18:28:30 +0000 2017</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/903036839...</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 22 columns</p>
</div>




```python
# Since we haven't seen the custom function, how_many_na(), in use yet, let's do it here.
# The column names in our dataset are listed on the left, 
# and the number of NaN values in each columns is displayed on the right.
how_many_na(df_clean)
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
      <th>NaNs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>created_at</th>
      <td>0</td>
    </tr>
    <tr>
      <th>id</th>
      <td>0</td>
    </tr>
    <tr>
      <th>source</th>
      <td>0</td>
    </tr>
    <tr>
      <th>is_quote_status</th>
      <td>0</td>
    </tr>
    <tr>
      <th>retweet_count</th>
      <td>0</td>
    </tr>
    <tr>
      <th>favorite_count</th>
      <td>0</td>
    </tr>
    <tr>
      <th>truncated</th>
      <td>0</td>
    </tr>
    <tr>
      <th>text</th>
      <td>0</td>
    </tr>
    <tr>
      <th>user_id_str</th>
      <td>0</td>
    </tr>
    <tr>
      <th>user_name</th>
      <td>0</td>
    </tr>
    <tr>
      <th>user_screen_name</th>
      <td>0</td>
    </tr>
    <tr>
      <th>user_location</th>
      <td>0</td>
    </tr>
    <tr>
      <th>user_description</th>
      <td>0</td>
    </tr>
    <tr>
      <th>user_verified</th>
      <td>0</td>
    </tr>
    <tr>
      <th>user_followers_count</th>
      <td>0</td>
    </tr>
    <tr>
      <th>user_friends_count</th>
      <td>0</td>
    </tr>
    <tr>
      <th>user_listed_count</th>
      <td>0</td>
    </tr>
    <tr>
      <th>user_favourites_count</th>
      <td>0</td>
    </tr>
    <tr>
      <th>user_statuses_count</th>
      <td>0</td>
    </tr>
    <tr>
      <th>user_created_at</th>
      <td>0</td>
    </tr>
    <tr>
      <th>user_geo_enabled</th>
      <td>0</td>
    </tr>
    <tr>
      <th>user_profile_image_url_https</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Hmm... They all have 0 NaNs... 
# Just to show you that this wasn't true when we began this section, 
# let's use it on our raw_features dataset for comparison. 
how_many_na(df_features)
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
      <th>NaNs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Unnamed: 0</th>
      <td>0</td>
    </tr>
    <tr>
      <th>contributors</th>
      <td>107156</td>
    </tr>
    <tr>
      <th>coordinates</th>
      <td>107146</td>
    </tr>
    <tr>
      <th>created_at</th>
      <td>0</td>
    </tr>
    <tr>
      <th>entities</th>
      <td>0</td>
    </tr>
    <tr>
      <th>extended_entities</th>
      <td>95617</td>
    </tr>
    <tr>
      <th>favorite_count</th>
      <td>0</td>
    </tr>
    <tr>
      <th>favorited</th>
      <td>0</td>
    </tr>
    <tr>
      <th>geo</th>
      <td>107146</td>
    </tr>
    <tr>
      <th>id</th>
      <td>0</td>
    </tr>
    <tr>
      <th>id_str</th>
      <td>0</td>
    </tr>
    <tr>
      <th>in_reply_to_screen_name</th>
      <td>76938</td>
    </tr>
    <tr>
      <th>in_reply_to_status_id</th>
      <td>79440</td>
    </tr>
    <tr>
      <th>in_reply_to_status_id_str</th>
      <td>79440</td>
    </tr>
    <tr>
      <th>in_reply_to_user_id</th>
      <td>76938</td>
    </tr>
    <tr>
      <th>in_reply_to_user_id_str</th>
      <td>76938</td>
    </tr>
    <tr>
      <th>is_quote_status</th>
      <td>0</td>
    </tr>
    <tr>
      <th>lang</th>
      <td>0</td>
    </tr>
    <tr>
      <th>metadata</th>
      <td>0</td>
    </tr>
    <tr>
      <th>place</th>
      <td>105665</td>
    </tr>
    <tr>
      <th>possibly_sensitive</th>
      <td>73906</td>
    </tr>
    <tr>
      <th>quoted_status</th>
      <td>100968</td>
    </tr>
    <tr>
      <th>quoted_status_id</th>
      <td>94556</td>
    </tr>
    <tr>
      <th>quoted_status_id_str</th>
      <td>94556</td>
    </tr>
    <tr>
      <th>retweet_count</th>
      <td>0</td>
    </tr>
    <tr>
      <th>retweeted</th>
      <td>0</td>
    </tr>
    <tr>
      <th>retweeted_status</th>
      <td>45142</td>
    </tr>
    <tr>
      <th>source</th>
      <td>0</td>
    </tr>
    <tr>
      <th>text</th>
      <td>0</td>
    </tr>
    <tr>
      <th>truncated</th>
      <td>0</td>
    </tr>
    <tr>
      <th>user</th>
      <td>0</td>
    </tr>
    <tr>
      <th>withheld_in_countries</th>
      <td>107064</td>
    </tr>
  </tbody>
</table>
</div>


<hr>

> That's it for extracting features from our list of Tweet objects! We extract more features from the `text` column in the next chapter, NLP (Natural Language Processing).

----

# Label Values (Botometer Scores)

In this last section of Data Pre-Processing, we merge the features data with the labels data. We do this by matching the Botometer scores (bot_score) and predicted class representations with each Tweet object (each row) using the author's screen name as the primary key. Notice that the Twitter account screen name is the value in the `user_screen_name` column in our features dataset, and it is the value in the `follower` column in our labels dataset. We merge `df_clean` with `df_labels` as follows. 


```python
# First, let's take a look at our raw_labels data
df_labels.tail()
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
      <th>follower</th>
      <th>bot_score</th>
      <th>bot_or_not</th>
      <th>bot_or_not_or_what</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3195</th>
      <td>empress_farah</td>
      <td>0.888218</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3196</th>
      <td>Gombe1Isah</td>
      <td>0.067835</td>
      <td>0.0</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>3197</th>
      <td>Justnug</td>
      <td>0.944833</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3198</th>
      <td>Scullership</td>
      <td>0.956753</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3199</th>
      <td>Jitender_shakya</td>
      <td>0.896414</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# What we want is to match each tweet to the bot_or_not value of its author. 
# Fortunately, each row in our features dataset has a value corresponding 
# to the screen name of the author in the column named `user_screen_name`.

df_complete = pd.merge(df_clean, df_labels, how='left', left_on='user_screen_name', right_on='follower')
```


```python
df_complete.tail(3)
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
      <th>created_at</th>
      <th>id</th>
      <th>source</th>
      <th>is_quote_status</th>
      <th>retweet_count</th>
      <th>favorite_count</th>
      <th>truncated</th>
      <th>text</th>
      <th>user_id_str</th>
      <th>user_name</th>
      <th>...</th>
      <th>user_listed_count</th>
      <th>user_favourites_count</th>
      <th>user_statuses_count</th>
      <th>user_created_at</th>
      <th>user_geo_enabled</th>
      <th>user_profile_image_url_https</th>
      <th>follower</th>
      <th>bot_score</th>
      <th>bot_or_not</th>
      <th>bot_or_not_or_what</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>99728</th>
      <td>Tue Jul 17 18:50:30 +0000 2018</td>
      <td>1.019293e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>False</td>
      <td>@itswarenbuffett Kind regard Sir</td>
      <td>1249642280</td>
      <td>Isah Yunusa Adamu</td>
      <td>...</td>
      <td>0</td>
      <td>547</td>
      <td>256</td>
      <td>Thu Mar 07 17:35:25 +0000 2013</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/101410137...</td>
      <td>Gombe1Isah</td>
      <td>0.067835</td>
      <td>0.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>99729</th>
      <td>Fri Jul 13 07:13:41 +0000 2018</td>
      <td>1.017668e+18</td>
      <td>&lt;a href="http://instagram.com" rel="nofollow"&gt;...</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>Just posted a photo https://t.co/NZryHlilWG</td>
      <td>1249642280</td>
      <td>Isah Yunusa Adamu</td>
      <td>...</td>
      <td>0</td>
      <td>547</td>
      <td>256</td>
      <td>Thu Mar 07 17:35:25 +0000 2013</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/101410137...</td>
      <td>Gombe1Isah</td>
      <td>0.067835</td>
      <td>0.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>99730</th>
      <td>Sat Jul 21 09:50:09 +0000 2018</td>
      <td>1.020607e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>@rssurjewala Nice</td>
      <td>2858148919</td>
      <td>Jitender Shakya</td>
      <td>...</td>
      <td>0</td>
      <td>3</td>
      <td>6</td>
      <td>Thu Oct 16 13:51:29 +0000 2014</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/102061090...</td>
      <td>Jitender_shakya</td>
      <td>0.896414</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 26 columns</p>
</div>



> Something is wrong here. There shouldn't be 99730 rows. 


```python
# But wait... how did the index become 99730? 
len(df_clean), len(df_complete)
```




    (89306, 99731)




```python
# With some thought, it becomes obvious that this has to do with duplicate entries... 
# Looking back, we never handled duplicate values. 
# So, let's get rid of all duplicates in the features as well as in the labels!
df_clean.drop_duplicates(subset='id')
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
      <th>created_at</th>
      <th>id</th>
      <th>source</th>
      <th>is_quote_status</th>
      <th>retweet_count</th>
      <th>favorite_count</th>
      <th>truncated</th>
      <th>text</th>
      <th>user_id_str</th>
      <th>user_name</th>
      <th>...</th>
      <th>user_description</th>
      <th>user_verified</th>
      <th>user_followers_count</th>
      <th>user_friends_count</th>
      <th>user_listed_count</th>
      <th>user_favourites_count</th>
      <th>user_statuses_count</th>
      <th>user_created_at</th>
      <th>user_geo_enabled</th>
      <th>user_profile_image_url_https</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mon Jul 23 06:22:31 +0000 2018</td>
      <td>1.021279e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>True</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>True</td>
      <td>VAMC'S R TRAINED MURDERERS THE SMART WHO WENT ...</td>
      <td>902961240691101696</td>
      <td>prepare mathew 25</td>
      <td>...</td>
      <td>Follower of YESHUA  JESUS the CHRIST</td>
      <td>False</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>Wed Aug 30 18:28:30 +0000 2017</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/903036839...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Mon Jul 23 06:09:39 +0000 2018</td>
      <td>1.021276e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>True</td>
      <td>https://t.co/KwImJ6hKrh  no one owns land in U...</td>
      <td>902961240691101696</td>
      <td>prepare mathew 25</td>
      <td>...</td>
      <td>Follower of YESHUA  JESUS the CHRIST</td>
      <td>False</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>Wed Aug 30 18:28:30 +0000 2017</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/903036839...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mon Jul 23 02:52:48 +0000 2018</td>
      <td>1.021227e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>682.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>RT @ROHLL5: 🔁 RT if you agree\n\n#44 .@POTUS\n...</td>
      <td>902961240691101696</td>
      <td>prepare mathew 25</td>
      <td>...</td>
      <td>Follower of YESHUA  JESUS the CHRIST</td>
      <td>False</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>Wed Aug 30 18:28:30 +0000 2017</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/903036839...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Mon Jul 23 02:51:58 +0000 2018</td>
      <td>1.021226e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>True</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>Wait next is Burnie you will love comi COMEY a...</td>
      <td>902961240691101696</td>
      <td>prepare mathew 25</td>
      <td>...</td>
      <td>Follower of YESHUA  JESUS the CHRIST</td>
      <td>False</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>Wed Aug 30 18:28:30 +0000 2017</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/903036839...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sun Jul 22 19:10:14 +0000 2018</td>
      <td>1.021110e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>True</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>It's simple landlords get rich quick scheme. A...</td>
      <td>902961240691101696</td>
      <td>prepare mathew 25</td>
      <td>...</td>
      <td>Follower of YESHUA  JESUS the CHRIST</td>
      <td>False</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>Wed Aug 30 18:28:30 +0000 2017</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/903036839...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Sun Jul 22 19:07:18 +0000 2018</td>
      <td>1.021109e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>True</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>He will be a new member of the royal Canadian ...</td>
      <td>902961240691101696</td>
      <td>prepare mathew 25</td>
      <td>...</td>
      <td>Follower of YESHUA  JESUS the CHRIST</td>
      <td>False</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>Wed Aug 30 18:28:30 +0000 2017</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/903036839...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Sun Jul 22 18:40:51 +0000 2018</td>
      <td>1.021103e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>Il come visit u it the penitentiary COMI COMEY...</td>
      <td>902961240691101696</td>
      <td>prepare mathew 25</td>
      <td>...</td>
      <td>Follower of YESHUA  JESUS the CHRIST</td>
      <td>False</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>Wed Aug 30 18:28:30 +0000 2017</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/903036839...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Sun Jul 22 18:39:21 +0000 2018</td>
      <td>1.021102e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>True</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>Or if comi COMEY and Mac dogg McCabe had been ...</td>
      <td>902961240691101696</td>
      <td>prepare mathew 25</td>
      <td>...</td>
      <td>Follower of YESHUA  JESUS the CHRIST</td>
      <td>False</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>Wed Aug 30 18:28:30 +0000 2017</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/903036839...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Sun Jul 22 15:50:45 +0000 2018</td>
      <td>1.021060e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>True</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>If his lips R MOVING HE IS LIEING and going 2 ...</td>
      <td>902961240691101696</td>
      <td>prepare mathew 25</td>
      <td>...</td>
      <td>Follower of YESHUA  JESUS the CHRIST</td>
      <td>False</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>Wed Aug 30 18:28:30 +0000 2017</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/903036839...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Sun Jul 22 15:46:35 +0000 2018</td>
      <td>1.021059e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>True</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>Judges and prosecuters all must be replace wit...</td>
      <td>902961240691101696</td>
      <td>prepare mathew 25</td>
      <td>...</td>
      <td>Follower of YESHUA  JESUS the CHRIST</td>
      <td>False</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>Wed Aug 30 18:28:30 +0000 2017</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/903036839...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Sun Jul 22 05:06:40 +0000 2018</td>
      <td>1.020898e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>So evil https://t.co/kvEXIjfleL</td>
      <td>902961240691101696</td>
      <td>prepare mathew 25</td>
      <td>...</td>
      <td>Follower of YESHUA  JESUS the CHRIST</td>
      <td>False</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>Wed Aug 30 18:28:30 +0000 2017</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/903036839...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Sun Jul 22 05:03:45 +0000 2018</td>
      <td>1.020897e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>True</td>
      <td>It's a team and this one worked for demonrats ...</td>
      <td>902961240691101696</td>
      <td>prepare mathew 25</td>
      <td>...</td>
      <td>Follower of YESHUA  JESUS the CHRIST</td>
      <td>False</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>Wed Aug 30 18:28:30 +0000 2017</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/903036839...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Sun Jul 22 04:29:41 +0000 2018</td>
      <td>1.020889e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>True</td>
      <td>White House chief of staff John Kellyreportedl...</td>
      <td>902961240691101696</td>
      <td>prepare mathew 25</td>
      <td>...</td>
      <td>Follower of YESHUA  JESUS the CHRIST</td>
      <td>False</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>Wed Aug 30 18:28:30 +0000 2017</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/903036839...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Sun Jul 22 04:18:23 +0000 2018</td>
      <td>1.020886e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>True</td>
      <td>@myhtopoeic Watch for a microwave weapon from ...</td>
      <td>902961240691101696</td>
      <td>prepare mathew 25</td>
      <td>...</td>
      <td>Follower of YESHUA  JESUS the CHRIST</td>
      <td>False</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>Wed Aug 30 18:28:30 +0000 2017</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/903036839...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Sun Jul 22 03:24:29 +0000 2018</td>
      <td>1.020872e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>True</td>
      <td>How much money and deals did PRESIDENT TRUMP G...</td>
      <td>902961240691101696</td>
      <td>prepare mathew 25</td>
      <td>...</td>
      <td>Follower of YESHUA  JESUS the CHRIST</td>
      <td>False</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>Wed Aug 30 18:28:30 +0000 2017</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/903036839...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Sun Jul 22 03:18:53 +0000 2018</td>
      <td>1.020871e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>True</td>
      <td>Due to known bad drugs used and to high a dose...</td>
      <td>902961240691101696</td>
      <td>prepare mathew 25</td>
      <td>...</td>
      <td>Follower of YESHUA  JESUS the CHRIST</td>
      <td>False</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>Wed Aug 30 18:28:30 +0000 2017</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/903036839...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Sun Jul 22 03:18:52 +0000 2018</td>
      <td>1.020871e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>True</td>
      <td>@VeteransHealth I am sure thankful for the VA ...</td>
      <td>902961240691101696</td>
      <td>prepare mathew 25</td>
      <td>...</td>
      <td>Follower of YESHUA  JESUS the CHRIST</td>
      <td>False</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>Wed Aug 30 18:28:30 +0000 2017</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/903036839...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Sun Jul 22 03:10:02 +0000 2018</td>
      <td>1.020869e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>True</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>Send this law legislating judge to the poop sh...</td>
      <td>902961240691101696</td>
      <td>prepare mathew 25</td>
      <td>...</td>
      <td>Follower of YESHUA  JESUS the CHRIST</td>
      <td>False</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>Wed Aug 30 18:28:30 +0000 2017</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/903036839...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Sun Jul 22 03:07:59 +0000 2018</td>
      <td>1.020868e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>True</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>True</td>
      <td>If u were President and NFL owner refused to f...</td>
      <td>902961240691101696</td>
      <td>prepare mathew 25</td>
      <td>...</td>
      <td>Follower of YESHUA  JESUS the CHRIST</td>
      <td>False</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>Wed Aug 30 18:28:30 +0000 2017</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/903036839...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Sun Jul 22 02:59:53 +0000 2018</td>
      <td>1.020866e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>True</td>
      <td>Not long now no one wants to be slave to maste...</td>
      <td>902961240691101696</td>
      <td>prepare mathew 25</td>
      <td>...</td>
      <td>Follower of YESHUA  JESUS the CHRIST</td>
      <td>False</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>Wed Aug 30 18:28:30 +0000 2017</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/903036839...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Sun Jul 22 02:59:53 +0000 2018</td>
      <td>1.020866e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>True</td>
      <td>https://t.co/etom1Oxb2Q oh LOOKIE NY via ya ow...</td>
      <td>902961240691101696</td>
      <td>prepare mathew 25</td>
      <td>...</td>
      <td>Follower of YESHUA  JESUS the CHRIST</td>
      <td>False</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>Wed Aug 30 18:28:30 +0000 2017</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/903036839...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Sun Jul 22 00:16:20 +0000 2018</td>
      <td>1.020825e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>True</td>
      <td>He has not killed to many not to much blood on...</td>
      <td>902961240691101696</td>
      <td>prepare mathew 25</td>
      <td>...</td>
      <td>Follower of YESHUA  JESUS the CHRIST</td>
      <td>False</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>Wed Aug 30 18:28:30 +0000 2017</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/903036839...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Sun Jul 22 00:16:19 +0000 2018</td>
      <td>1.020825e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>True</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>True</td>
      <td>What a stretch. No king David went into battle...</td>
      <td>902961240691101696</td>
      <td>prepare mathew 25</td>
      <td>...</td>
      <td>Follower of YESHUA  JESUS the CHRIST</td>
      <td>False</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>Wed Aug 30 18:28:30 +0000 2017</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/903036839...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Sun Jul 22 00:01:13 +0000 2018</td>
      <td>1.020821e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>True</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>True</td>
      <td>So is scum sucker spying ass SENATOR JAMES LAN...</td>
      <td>902961240691101696</td>
      <td>prepare mathew 25</td>
      <td>...</td>
      <td>Follower of YESHUA  JESUS the CHRIST</td>
      <td>False</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>Wed Aug 30 18:28:30 +0000 2017</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/903036839...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Sat Jul 21 23:54:31 +0000 2018</td>
      <td>1.020819e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>True</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>Your interest in paying interest https://t.co/...</td>
      <td>902961240691101696</td>
      <td>prepare mathew 25</td>
      <td>...</td>
      <td>Follower of YESHUA  JESUS the CHRIST</td>
      <td>False</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>Wed Aug 30 18:28:30 +0000 2017</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/903036839...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Sat Jul 21 19:10:56 +0000 2018</td>
      <td>1.020748e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>True</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>Yo sanctions have. Ever been strong enough to ...</td>
      <td>902961240691101696</td>
      <td>prepare mathew 25</td>
      <td>...</td>
      <td>Follower of YESHUA  JESUS the CHRIST</td>
      <td>False</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>Wed Aug 30 18:28:30 +0000 2017</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/903036839...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Sat Jul 21 15:09:23 +0000 2018</td>
      <td>1.020687e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>True</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>Not. was https://t.co/5AtRADKwtT</td>
      <td>902961240691101696</td>
      <td>prepare mathew 25</td>
      <td>...</td>
      <td>Follower of YESHUA  JESUS the CHRIST</td>
      <td>False</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>Wed Aug 30 18:28:30 +0000 2017</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/903036839...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Sat Jul 21 14:12:02 +0000 2018</td>
      <td>1.020673e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>True</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>He gave it up https://t.co/gVw3aZnNMW</td>
      <td>902961240691101696</td>
      <td>prepare mathew 25</td>
      <td>...</td>
      <td>Follower of YESHUA  JESUS the CHRIST</td>
      <td>False</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>Wed Aug 30 18:28:30 +0000 2017</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/903036839...</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Sat Jul 21 03:12:49 +0000 2018</td>
      <td>1.020507e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>True</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>True</td>
      <td>Well like your peon asses should be regulating...</td>
      <td>902961240691101696</td>
      <td>prepare mathew 25</td>
      <td>...</td>
      <td>Follower of YESHUA  JESUS the CHRIST</td>
      <td>False</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>Wed Aug 30 18:28:30 +0000 2017</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/903036839...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Sat Jul 21 03:09:17 +0000 2018</td>
      <td>1.020506e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>True</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>Ted u R and general Michael hidden says intere...</td>
      <td>902961240691101696</td>
      <td>prepare mathew 25</td>
      <td>...</td>
      <td>Follower of YESHUA  JESUS the CHRIST</td>
      <td>False</td>
      <td>50</td>
      <td>404</td>
      <td>0</td>
      <td>1900</td>
      <td>5172</td>
      <td>Wed Aug 30 18:28:30 +0000 2017</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/903036839...</td>
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
      <td>Sun Jul 15 11:32:37 +0000 2018</td>
      <td>1.018458e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>8362.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>RT @Tweetsfor45: Trump never colluded with the...</td>
      <td>454154505</td>
      <td>Clakeluv</td>
      <td>...</td>
      <td>xdem, #walkaway, it's my gun try to take it, i...</td>
      <td>False</td>
      <td>21</td>
      <td>73</td>
      <td>0</td>
      <td>1875</td>
      <td>784</td>
      <td>Tue Jan 03 17:47:09 +0000 2012</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/378800000...</td>
    </tr>
    <tr>
      <th>107109</th>
      <td>Sun Jul 15 11:30:55 +0000 2018</td>
      <td>1.018458e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>885.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>RT @CB618444: 🇺🇸Thousands of #Trump supporters...</td>
      <td>454154505</td>
      <td>Clakeluv</td>
      <td>...</td>
      <td>xdem, #walkaway, it's my gun try to take it, i...</td>
      <td>False</td>
      <td>21</td>
      <td>73</td>
      <td>0</td>
      <td>1875</td>
      <td>784</td>
      <td>Tue Jan 03 17:47:09 +0000 2012</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/378800000...</td>
    </tr>
    <tr>
      <th>107110</th>
      <td>Sun Jul 15 11:26:30 +0000 2018</td>
      <td>1.018457e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>@Alyssa_Milano @realDonaldTrump Looks to me li...</td>
      <td>454154505</td>
      <td>Clakeluv</td>
      <td>...</td>
      <td>xdem, #walkaway, it's my gun try to take it, i...</td>
      <td>False</td>
      <td>21</td>
      <td>73</td>
      <td>0</td>
      <td>1875</td>
      <td>784</td>
      <td>Tue Jan 03 17:47:09 +0000 2012</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/378800000...</td>
    </tr>
    <tr>
      <th>107111</th>
      <td>Sun Jul 15 11:07:41 +0000 2018</td>
      <td>1.018452e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>1579.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>RT @fleccas: Want to make sure I have this rig...</td>
      <td>454154505</td>
      <td>Clakeluv</td>
      <td>...</td>
      <td>xdem, #walkaway, it's my gun try to take it, i...</td>
      <td>False</td>
      <td>21</td>
      <td>73</td>
      <td>0</td>
      <td>1875</td>
      <td>784</td>
      <td>Tue Jan 03 17:47:09 +0000 2012</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/378800000...</td>
    </tr>
    <tr>
      <th>107112</th>
      <td>Sat Jul 14 21:18:44 +0000 2018</td>
      <td>1.018243e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>@foxandfriends @dbongino Anything to try and s...</td>
      <td>454154505</td>
      <td>Clakeluv</td>
      <td>...</td>
      <td>xdem, #walkaway, it's my gun try to take it, i...</td>
      <td>False</td>
      <td>21</td>
      <td>73</td>
      <td>0</td>
      <td>1875</td>
      <td>784</td>
      <td>Tue Jan 03 17:47:09 +0000 2012</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/378800000...</td>
    </tr>
    <tr>
      <th>107113</th>
      <td>Sat Jul 14 21:16:42 +0000 2018</td>
      <td>1.018243e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>20388.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>RT @AnnCoulter: Peter Strzok's wife threatened...</td>
      <td>454154505</td>
      <td>Clakeluv</td>
      <td>...</td>
      <td>xdem, #walkaway, it's my gun try to take it, i...</td>
      <td>False</td>
      <td>21</td>
      <td>73</td>
      <td>0</td>
      <td>1875</td>
      <td>784</td>
      <td>Tue Jan 03 17:47:09 +0000 2012</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/378800000...</td>
    </tr>
    <tr>
      <th>107114</th>
      <td>Sat Jul 14 21:16:16 +0000 2018</td>
      <td>1.018243e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>5256.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>RT @Pink_About_it: Democrats drafted a bill to...</td>
      <td>454154505</td>
      <td>Clakeluv</td>
      <td>...</td>
      <td>xdem, #walkaway, it's my gun try to take it, i...</td>
      <td>False</td>
      <td>21</td>
      <td>73</td>
      <td>0</td>
      <td>1875</td>
      <td>784</td>
      <td>Tue Jan 03 17:47:09 +0000 2012</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/378800000...</td>
    </tr>
    <tr>
      <th>107115</th>
      <td>Sat Jul 14 12:04:35 +0000 2018</td>
      <td>1.018104e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>951.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>RT @LadyThriller69: The fact that people are f...</td>
      <td>454154505</td>
      <td>Clakeluv</td>
      <td>...</td>
      <td>xdem, #walkaway, it's my gun try to take it, i...</td>
      <td>False</td>
      <td>21</td>
      <td>73</td>
      <td>0</td>
      <td>1875</td>
      <td>784</td>
      <td>Tue Jan 03 17:47:09 +0000 2012</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/378800000...</td>
    </tr>
    <tr>
      <th>107116</th>
      <td>Sat Jul 14 11:57:09 +0000 2018</td>
      <td>1.018102e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>3187.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>RT @The_Trump_Train: Anyone care to explain ho...</td>
      <td>454154505</td>
      <td>Clakeluv</td>
      <td>...</td>
      <td>xdem, #walkaway, it's my gun try to take it, i...</td>
      <td>False</td>
      <td>21</td>
      <td>73</td>
      <td>0</td>
      <td>1875</td>
      <td>784</td>
      <td>Tue Jan 03 17:47:09 +0000 2012</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/378800000...</td>
    </tr>
    <tr>
      <th>107117</th>
      <td>Sat Jul 14 11:52:22 +0000 2018</td>
      <td>1.018101e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>5780.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>RT @AmyMek: Why did the Media never cover this...</td>
      <td>454154505</td>
      <td>Clakeluv</td>
      <td>...</td>
      <td>xdem, #walkaway, it's my gun try to take it, i...</td>
      <td>False</td>
      <td>21</td>
      <td>73</td>
      <td>0</td>
      <td>1875</td>
      <td>784</td>
      <td>Tue Jan 03 17:47:09 +0000 2012</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/378800000...</td>
    </tr>
    <tr>
      <th>107118</th>
      <td>Sat Jul 14 11:34:21 +0000 2018</td>
      <td>1.018096e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>9545.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>RT @Education4Libs: London flew a large blimp ...</td>
      <td>454154505</td>
      <td>Clakeluv</td>
      <td>...</td>
      <td>xdem, #walkaway, it's my gun try to take it, i...</td>
      <td>False</td>
      <td>21</td>
      <td>73</td>
      <td>0</td>
      <td>1875</td>
      <td>784</td>
      <td>Tue Jan 03 17:47:09 +0000 2012</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/378800000...</td>
    </tr>
    <tr>
      <th>107119</th>
      <td>Fri Jul 13 14:38:03 +0000 2018</td>
      <td>1.017780e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>RT @bushido49ers: @CBSNews It always has been....</td>
      <td>454154505</td>
      <td>Clakeluv</td>
      <td>...</td>
      <td>xdem, #walkaway, it's my gun try to take it, i...</td>
      <td>False</td>
      <td>21</td>
      <td>73</td>
      <td>0</td>
      <td>1875</td>
      <td>784</td>
      <td>Tue Jan 03 17:47:09 +0000 2012</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/378800000...</td>
    </tr>
    <tr>
      <th>107120</th>
      <td>Fri Jul 13 12:46:49 +0000 2018</td>
      <td>1.017752e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>132.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>RT @foxandfriends: Dems who drafted bill to ab...</td>
      <td>454154505</td>
      <td>Clakeluv</td>
      <td>...</td>
      <td>xdem, #walkaway, it's my gun try to take it, i...</td>
      <td>False</td>
      <td>21</td>
      <td>73</td>
      <td>0</td>
      <td>1875</td>
      <td>784</td>
      <td>Tue Jan 03 17:47:09 +0000 2012</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/378800000...</td>
    </tr>
    <tr>
      <th>107121</th>
      <td>Fri Jul 13 12:44:25 +0000 2018</td>
      <td>1.017752e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>6519.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>RT @Hmmmthere: “No, I never said I didn’t like...</td>
      <td>454154505</td>
      <td>Clakeluv</td>
      <td>...</td>
      <td>xdem, #walkaway, it's my gun try to take it, i...</td>
      <td>False</td>
      <td>21</td>
      <td>73</td>
      <td>0</td>
      <td>1875</td>
      <td>784</td>
      <td>Tue Jan 03 17:47:09 +0000 2012</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/378800000...</td>
    </tr>
    <tr>
      <th>107122</th>
      <td>Fri Jul 13 12:43:15 +0000 2018</td>
      <td>1.017751e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>@NBCNews I heard he took the tag off his mattr...</td>
      <td>454154505</td>
      <td>Clakeluv</td>
      <td>...</td>
      <td>xdem, #walkaway, it's my gun try to take it, i...</td>
      <td>False</td>
      <td>21</td>
      <td>73</td>
      <td>0</td>
      <td>1875</td>
      <td>784</td>
      <td>Tue Jan 03 17:47:09 +0000 2012</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/378800000...</td>
    </tr>
    <tr>
      <th>107123</th>
      <td>Fri Jul 13 12:26:02 +0000 2018</td>
      <td>1.017747e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>6761.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>RT @JacobAWohl: HUGE! Peter Strzok says that t...</td>
      <td>454154505</td>
      <td>Clakeluv</td>
      <td>...</td>
      <td>xdem, #walkaway, it's my gun try to take it, i...</td>
      <td>False</td>
      <td>21</td>
      <td>73</td>
      <td>0</td>
      <td>1875</td>
      <td>784</td>
      <td>Tue Jan 03 17:47:09 +0000 2012</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/378800000...</td>
    </tr>
    <tr>
      <th>107125</th>
      <td>Fri Jul 13 12:15:49 +0000 2018</td>
      <td>1.017744e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>@CBSNews All fine and dandy until they need so...</td>
      <td>454154505</td>
      <td>Clakeluv</td>
      <td>...</td>
      <td>xdem, #walkaway, it's my gun try to take it, i...</td>
      <td>False</td>
      <td>21</td>
      <td>73</td>
      <td>0</td>
      <td>1875</td>
      <td>784</td>
      <td>Tue Jan 03 17:47:09 +0000 2012</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/378800000...</td>
    </tr>
    <tr>
      <th>107137</th>
      <td>Fri Jul 13 18:19:29 +0000 2018</td>
      <td>1.017836e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>False</td>
      <td>@UberFacts No one feae Fridays we only fear mo...</td>
      <td>725430236867878912</td>
      <td>ياسر</td>
      <td>...</td>
      <td>‏‏‏سبحان الله والحمد لله ولا إله إلا الله والل...</td>
      <td>False</td>
      <td>158</td>
      <td>1277</td>
      <td>2</td>
      <td>71</td>
      <td>450</td>
      <td>Wed Apr 27 21:03:41 +0000 2016</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/872596126...</td>
    </tr>
    <tr>
      <th>107141</th>
      <td>Mon Jul 23 08:10:59 +0000 2018</td>
      <td>1.021307e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>@ImanMunal Naked True</td>
      <td>1249642280</td>
      <td>Isah Yunusa Adamu</td>
      <td>...</td>
      <td>Islam is greater than your culture. \n\nQuit n...</td>
      <td>False</td>
      <td>203</td>
      <td>1593</td>
      <td>0</td>
      <td>547</td>
      <td>256</td>
      <td>Thu Mar 07 17:35:25 +0000 2013</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/101410137...</td>
    </tr>
    <tr>
      <th>107142</th>
      <td>Mon Jul 23 08:07:47 +0000 2018</td>
      <td>1.021306e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>⠀\n⠀\n⠀\n⠀\n⠀\n⠀\n⠀\n⠀\n⠀\n⠀⠀⠀                ...</td>
      <td>1249642280</td>
      <td>Isah Yunusa Adamu</td>
      <td>...</td>
      <td>Islam is greater than your culture. \n\nQuit n...</td>
      <td>False</td>
      <td>203</td>
      <td>1593</td>
      <td>0</td>
      <td>547</td>
      <td>256</td>
      <td>Thu Mar 07 17:35:25 +0000 2013</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/101410137...</td>
    </tr>
    <tr>
      <th>107143</th>
      <td>Sat Jul 21 11:49:17 +0000 2018</td>
      <td>1.020637e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>1098.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>RT @arewashams: Every woman deserves a man who...</td>
      <td>1249642280</td>
      <td>Isah Yunusa Adamu</td>
      <td>...</td>
      <td>Islam is greater than your culture. \n\nQuit n...</td>
      <td>False</td>
      <td>203</td>
      <td>1593</td>
      <td>0</td>
      <td>547</td>
      <td>256</td>
      <td>Thu Mar 07 17:35:25 +0000 2013</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/101410137...</td>
    </tr>
    <tr>
      <th>107144</th>
      <td>Sat Jul 21 00:07:07 +0000 2018</td>
      <td>1.020460e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>37.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>RT @Tweets2Motivate: Today is a great day to s...</td>
      <td>1249642280</td>
      <td>Isah Yunusa Adamu</td>
      <td>...</td>
      <td>Islam is greater than your culture. \n\nQuit n...</td>
      <td>False</td>
      <td>203</td>
      <td>1593</td>
      <td>0</td>
      <td>547</td>
      <td>256</td>
      <td>Thu Mar 07 17:35:25 +0000 2013</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/101410137...</td>
    </tr>
    <tr>
      <th>107146</th>
      <td>Fri Jul 20 14:03:38 +0000 2018</td>
      <td>1.020308e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>75.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>RT @HQNigerianArmy: The (COAS) Lt Gen TY Burat...</td>
      <td>1249642280</td>
      <td>Isah Yunusa Adamu</td>
      <td>...</td>
      <td>Islam is greater than your culture. \n\nQuit n...</td>
      <td>False</td>
      <td>203</td>
      <td>1593</td>
      <td>0</td>
      <td>547</td>
      <td>256</td>
      <td>Thu Mar 07 17:35:25 +0000 2013</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/101410137...</td>
    </tr>
    <tr>
      <th>107147</th>
      <td>Fri Jul 20 14:01:09 +0000 2018</td>
      <td>1.020308e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>@HEDankwambo @BusinessDayNg Congratulations Yo...</td>
      <td>1249642280</td>
      <td>Isah Yunusa Adamu</td>
      <td>...</td>
      <td>Islam is greater than your culture. \n\nQuit n...</td>
      <td>False</td>
      <td>203</td>
      <td>1593</td>
      <td>0</td>
      <td>547</td>
      <td>256</td>
      <td>Thu Mar 07 17:35:25 +0000 2013</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/101410137...</td>
    </tr>
    <tr>
      <th>107150</th>
      <td>Fri Jul 20 10:21:30 +0000 2018</td>
      <td>1.020252e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>13.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>RT @CleverQuotez: Never argue with an idiot th...</td>
      <td>1249642280</td>
      <td>Isah Yunusa Adamu</td>
      <td>...</td>
      <td>Islam is greater than your culture. \n\nQuit n...</td>
      <td>False</td>
      <td>203</td>
      <td>1593</td>
      <td>0</td>
      <td>547</td>
      <td>256</td>
      <td>Thu Mar 07 17:35:25 +0000 2013</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/101410137...</td>
    </tr>
    <tr>
      <th>107151</th>
      <td>Wed Jul 18 10:28:34 +0000 2018</td>
      <td>1.019529e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>True</td>
      <td>173.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>RT @TrackHateSpeech: FAKE NEWS!\n\n6 people su...</td>
      <td>1249642280</td>
      <td>Isah Yunusa Adamu</td>
      <td>...</td>
      <td>Islam is greater than your culture. \n\nQuit n...</td>
      <td>False</td>
      <td>203</td>
      <td>1593</td>
      <td>0</td>
      <td>547</td>
      <td>256</td>
      <td>Thu Mar 07 17:35:25 +0000 2013</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/101410137...</td>
    </tr>
    <tr>
      <th>107152</th>
      <td>Tue Jul 17 18:55:47 +0000 2018</td>
      <td>1.019295e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>@Madame_Flowy Why do You ask all this questions?</td>
      <td>1249642280</td>
      <td>Isah Yunusa Adamu</td>
      <td>...</td>
      <td>Islam is greater than your culture. \n\nQuit n...</td>
      <td>False</td>
      <td>203</td>
      <td>1593</td>
      <td>0</td>
      <td>547</td>
      <td>256</td>
      <td>Thu Mar 07 17:35:25 +0000 2013</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/101410137...</td>
    </tr>
    <tr>
      <th>107153</th>
      <td>Tue Jul 17 18:50:30 +0000 2018</td>
      <td>1.019293e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>False</td>
      <td>@itswarenbuffett Kind regard Sir</td>
      <td>1249642280</td>
      <td>Isah Yunusa Adamu</td>
      <td>...</td>
      <td>Islam is greater than your culture. \n\nQuit n...</td>
      <td>False</td>
      <td>203</td>
      <td>1593</td>
      <td>0</td>
      <td>547</td>
      <td>256</td>
      <td>Thu Mar 07 17:35:25 +0000 2013</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/101410137...</td>
    </tr>
    <tr>
      <th>107156</th>
      <td>Fri Jul 13 07:13:41 +0000 2018</td>
      <td>1.017668e+18</td>
      <td>&lt;a href="http://instagram.com" rel="nofollow"&gt;...</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>Just posted a photo https://t.co/NZryHlilWG</td>
      <td>1249642280</td>
      <td>Isah Yunusa Adamu</td>
      <td>...</td>
      <td>Islam is greater than your culture. \n\nQuit n...</td>
      <td>False</td>
      <td>203</td>
      <td>1593</td>
      <td>0</td>
      <td>547</td>
      <td>256</td>
      <td>Thu Mar 07 17:35:25 +0000 2013</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/101410137...</td>
    </tr>
    <tr>
      <th>107157</th>
      <td>Sat Jul 21 09:50:09 +0000 2018</td>
      <td>1.020607e+18</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>@rssurjewala Nice</td>
      <td>2858148919</td>
      <td>Jitender Shakya</td>
      <td>...</td>
      <td></td>
      <td>False</td>
      <td>1</td>
      <td>71</td>
      <td>0</td>
      <td>3</td>
      <td>6</td>
      <td>Thu Oct 16 13:51:29 +0000 2014</td>
      <td>False</td>
      <td>https://pbs.twimg.com/profile_images/102061090...</td>
    </tr>
  </tbody>
</table>
<p>89304 rows × 22 columns</p>
</div>




```python
# For the features, we can use the 'id' column to remove duplicate Tweet objects
df_clean.drop_duplicates(subset='id', inplace=True)
len(df_clean)
```




    89304



Looks like we got rid of two rows in our features data. Now lets drop duplicates for our labels data.


```python
# For the labels, we use the 'follower' columns for dropping duplicates. 
df_labels.drop_duplicates(subset='follower',inplace=True)
len(df_labels)
```




    2866



There were 334 duplicate screenames in our list (3,200 - 2,866 = 334). Now let's try merging them again.


```python
# Merge features and labels on user account names
df_complete = pd.merge(df_clean,df_labels,how='left',left_on='user_screen_name',right_on='follower')
```


```python
# We need there to be 89304 rows... Let's check.
len(df_complete)
```




    89304



### Awesome. Now our dataset has two columns that can be used as labels. The raw Botometer score in column 'bot_score' and the class prediction value in column 'bot_or_not'.


```python
# Saving complete dataset to file. 
# This dataset is "complete" in the sense that...
df_complete.to_csv("complete.csv")
```
