---
layout: page
title: 1. Data Collection
permalink: /data_collection
order: 2
---

### S109A Final Project Submission Group 22 (Project Website: http://mathuser0.github.io)

Christopher Lee, chl2967@g.harvard.edu  
Sriganesh Pera, srp124@g.harvard.edu  
Bo Shang, bshang@g.harvard.edu

****

# Part 1. Data Collection

----

## How Our Dataset Was Built

This section is about formulating a methodology to collect data from external sources and reallocating the data into Pandas DataFrame objects in raw form.

The data for this project was opted to be formulated based on official U.S. government Twitter accounts, with, in mind, how the political influence of Twitter bots has recently emerged as a major concern. The reasoning behind the selected accounts is not worth pondering about, as they just happened to be among the first to be looked at by the project members. In total, 14 official government Twitter accounts were selected to be the basis upon which our dataset was built. They are as follows:

1.	Homeland Security (@DHSgov) - Department of Homeland Security
2.	USTR (@UStraderep) - Office of the U.S. Trade Representative
3.	Energy Department (@ENERGY) – U.S. Department of Energy
4.	Donald Trump (@realDonaldTrump) - 45th President
5.	Barack Obama (@BarackObama) - 45th President
6.	Robert S. Mueller (@BobSMueller) – Head of Special Counsel investigation
7.	John Kelly (@GeneralJohnK) – White House Chief of Staff
8.	Ben Carson (@SecretaryCarson) – 17th Secretary of Housing & Urban Development
9.	Rick Perry (@Secretaryperry) – 14th Secretary of Energy
10.	Alex Acosta (@SecretaryAcosta) – 27th Secretary of Labor
11.	Kirstjen Nielsen (@SecNielsen) – 6th Secretary of Homeland Security
12.	Alex Azar (@SecAzar) – 24th Secretary of Health & Human Services
13.	Linda McMahon (@SBALinda) – 25th Administrator of SBA
14.	Nikki Haley (@nikkihaley) – 29th U.S. Ambassador to the United Nations

Using the Tweepy API, we queried for the screen names of the 200 most recent followers for each of the accounts in the list above except @nikkihaley, for which we acquired the screen names of the 600 most recent followers. In total, we gathered 2,800 screen names of Twitter users who have shown interest in government affairs. (See [code](#code_23))

It is inconceivable to think that these government related accounts are controlled by malicious Twitter bots. Rather, we are more interested in the Twitter accounts that monitor these accounts. Therefore, the 200 most recent followers for each of the 14 accounts were collected and analyzed in this project, with the exception of Nikki Haley, the 29th U.S. Ambassador to the United Nations, for whom the 600 most recent followers were acquired, due to an issue encountered (described in more detail below). In total, our dataset was built upon a list of 3,200 (= 13 x 200 + 1 x 600) Twitter accounts.

It is important to note that the machine learning techniques used in this project were limited to supervised learning methods. Therefore, the data necessarily consists of features and labels. Below, we create our raw features dataset first using the Tweepy API. Then, we use the Botometer API to create our raw labels dataset.  

----

## Features

The basic atomic building block of all things Twitter is the Tweet object, and every tweet is actually a Tweet object. The text portion of a tweet that we are used to seeing is actually just one of many attributes in an instantiated Tweet object.

We built our raw features data from the tweets created by the followers in our list. Using the Tweepy API again, we requested for the 200 most recent tweets of each follower in our list. Note that by doing this, for followers who had made less than 200 tweets total, all of their tweets were collected.

Our next step involved using the Tweepy API again, this time requesting the 200 most recent tweets for each of the 3,200 followers. It was not easy to figure out how to collect all the tweets from all the users into a single file. After spending too much time trying to get all the tweets of all 3,200 followers placed into one file in a single run, we decided it would be easier to put together 3,200 files (one for each follower) by iteration and Pandas. That is to say, our solution was to create 3,200 files and stitch them back up together afterwards.

The resulting DataFrame object is shown in the code below. Our final raw features dataset consists of 107,158 observations, where each row is an instance of a Tweet object.

----

## Labels

Our “true” response values, or labels, are derived from Botometer scores. The Botometer API is a government funded project and it was determined to be a reasonable and reliable source for benchmarking our model predictions.

The Botometer API was queried to obtain scores for each follower in our list. It should be noted that the Botometer API responds with many different scores. Figure 1 shows an example of a Botometer API response, where the highlighted score is the one we chose to use for our project.


![alt text <>](./img/Picture1.png)  
<strong>Figure 1. Example of a Botometer API response. There are categories and sub-categories of scores to choose from. Our research into the meaning behind each score led us to choose the score that is highlighted.  </strong>


The final result of our labels collection process is a dataframe containing the screen names of each of our 3,200 followers in the column named `follower`, and their corresponding Botometer `english` scores in another column. Two more columns named `bot_or_not` and `bot_or_not_or_what` were created in this dataframe, indicating a follower's class in a binomial representation and a 3-class representation, respectively.

The Botometer return a NaN score for Twitter accounts that have never created a tweet before. Based upon the assumption that a Twitter bot that does not tweet is unable to bear malice towards society, these NaN values were considered to be humans rather than bots in the binomial class representation, and they were classified as `inconclusive` in the 3-class representation. To be more specific, in the binomial class representation, Botometer scores above 0.5 were considered to be bots, and otherwise humans. In the 3-class representation, Botometer scores above 0.7 were classified as bots, scores under 0.3 were classified as humans, and all other cases were classified as inconclusive.



# Code

----

## Import Libraries


```python
# Basic libraries
import numpy as np
import pandas as pd
import json
import sys
import jsonpickle
import os

# Tweepy
import tweepy

# Botometer
import botometer
```

----

## Tweepy & Botometer API Authentication


```python
# Setting up API_KEY and API_SECRET
# Tweepy Authentication Setup
auth = tweepy.AppAuthHandler("9Gpcxva2RwolHrDLdhRhYlVln",
                             "ylLbTVYjTz6Px2AGV6W662QDKjYDdCuAO3aa6ybStlrCOguK0b")

# Tweepy.API object for easier retrieval of Twitter data
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

# Botometer Authentication Setup
mashape_key = "3Jb4MhTkKnmshUOMY639XsisVlVGp1SQgbFjsnj42uy0GsYZ1T"
twitter_app_auth = {
    'consumer_key': '9Gpcxva2RwolHrDLdhRhYlVln',
    'consumer_secret': 'ylLbTVYjTz6Px2AGV6W662QDKjYDdCuAO3aa6ybStlrCOguK0b',
    'access_token': '867610424690188289-sA0uIW3KLVMro7I7JJmygJXRDhJMWj6',
    'access_token_secret': 'eoBTKR3dGstSziJ6AXr8sdUWdYSOaJXCsbPPKr3adMBE0',
  }

# Botometer object for easier retrieval of Botometer scores
bom = botometer.Botometer(wait_on_ratelimit=True,
                              mashape_key=mashape_key,
                              **twitter_app_auth)
```

----

## Custom Functions


```python
# Function to get the 'english' Botometer score
def get_bm_score(screen_name):  
    score = bom.check_account('@'+screen_name)['scores']['english']
    return score

# Turn a list of dictionaries into a DataFrame object
def toDataFrame(list_of_dictionaries, columns_to_get):
    Dataset=pd.DataFrame()
    for column in columns_to_get:
            Dataset[column]=[dictionary[column] for dictionary in list_of_dictionaries]
    return Dataset


# Assign each observation into one of three classes (-1, 0, or 1)
def toBotNotorWhat(score):
    # Real-valued bot scores greater than 0.7 are labeled as bots, and set to class 1.
    if score>0.7:
        return 1

    # Real-valued bot scores less than 0.3 are labeled as humans, and set to class -1.
    elif score<0.3:
        return -1

    # All other bot scores are inconclusive (a what?), and their class is set to 0.
    else:
        return 0

```

----

## Using Tweepy API to get the 200 recent followers for the 14 goverment Twitter accounts


```python
# List of 14 official government Twitter account screen names
politician_list=['DHSgov',
                 'UStraderep',
                 'ENERGY',
                 'realDonaldTrump',
                 'BarackObama',
                 'BobSMueller',
                 'GeneralJohnK',
                 'SecretaryCarson',
                 'Secretaryperry',
                 'SecretaryAcosta',
                 'SecNielsen',
                 'SecAzar',
                 'SBALinda',
                 'nikkihaley']

# Get most recent 200 followers
number_of_followers_to_retrieve=200

# Iterate to create a list of the 200 most recent followers
follower_list=[]
for politician in politician_list:
    for item in tweepy.Cursor(api.followers,
                              id=politician).items(number_of_followers_to_retrieve):
        follower_list.append(item.screen_name)
```

----

## Using Tweepy API to get 200 most recent tweets for each follower

Our next step involved using the Tweepy API again, this time requesting the 200 most recent tweets for each of the 3,200 followers. Notice the search terms and parameters chosen in the first 6 lines below.  


```python
# For each follower in follower_list, create a text file and save their tweets.
# If a follower has more than 200 tweets, save only the most recent 200.

for follower in follower_list:

    searchQuery = 'From:'+follower     # Search for tweets created by the current 'follower'
    maxTweets = 200                    # Get at most 200 tweets for each follower
    tweetsPerQry = 100                 # Query 100 tweets at a time. This is the max the API permits.
    fName = 'tweets_'+follower+'.json' # Store the tweets for each follower in a unique text file.

    # If results from a specific ID onwards are reqd, set since_id to that ID.
    # else default to no lower limit, go as far back as API allows
    sinceId = None

    # If results only below a specific ID are, set max_id to that ID.
    # else default to no upper limit, start from the most recent tweet matching the search query.
    max_id = -1

    tweetCount = 0
    print("Downloading max {0} tweets".format(maxTweets))
    with open(fName, 'w') as f:
        while tweetCount < maxTweets:
            try:
                if (max_id <= 0):
                    if (not sinceId):
                        new_tweets = api.search(q=searchQuery, count=tweetsPerQry)
                    else:
                        new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                                since_id=sinceId)
                else:
                    if (not sinceId):
                        new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                                max_id=str(max_id - 1))
                    else:
                        new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                                max_id=str(max_id - 1),
                                                since_id=sinceId)
                if not new_tweets:
                    print("No more tweets found")
                    break
                for tweet in new_tweets:
                    f.write(jsonpickle.encode(tweet._json, unpicklable=False) +
                            '\n')
                tweetCount += len(new_tweets)
                print("Downloaded {0} tweets".format(tweetCount))
                max_id = new_tweets[-1].id
            except tweepy.TweepError as e:
                # Just exit if any error
                print("some error : " + str(e))
                break
```

### Compiling 3,200 files into a single Pandas DataFrame

At this point, we had 3,200 separate files - one for each item in our list of followers. Each file contains the set of tweets obtained for each follower. Our next step was to construct a single dataframe from these files as follows.


```python
# Initialize tweets dictionary
tweets={}

# For each of the 3,200 files, place each tweet in the tweets dictionary
for follower in follower_list:
    fName= 'tweets_'+follower+'.json'
    with open(fName,'r') as f:
        for i, line in enumerate(f):
            tweets[follower+'_'+str(i)]=json.loads(line)
```


```python
# Create dataframe with tweets data            
df_features=pd.DataFrame(tweets).T
```


```python
# Displaying raw features data
df_features.head(3)
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
      <th>contributors</th>
      <th>coordinates</th>
      <th>created_at</th>
      <th>entities</th>
      <th>extended_entities</th>
      <th>favorite_count</th>
      <th>favorited</th>
      <th>geo</th>
      <th>id</th>
      <th>id_str</th>
      <th>...</th>
      <th>quoted_status</th>
      <th>quoted_status_id</th>
      <th>quoted_status_id_str</th>
      <th>retweet_count</th>
      <th>retweeted</th>
      <th>retweeted_status</th>
      <th>source</th>
      <th>text</th>
      <th>truncated</th>
      <th>user</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bek90941_0</th>
      <td>None</td>
      <td>None</td>
      <td>Fri Aug 10 12:57:30 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [], 'u...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1027901755210838016</td>
      <td>1027901755210838016</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>&lt;a href="http://twitter.com" rel="nofollow"&gt;Tw...</td>
      <td>THE TURD @RealAlexJones JUST MIGHT BE THIS GUY...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>bek90941_1</th>
      <td>None</td>
      <td>None</td>
      <td>Fri Aug 10 12:34:12 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [], 'u...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1027895891565465600</td>
      <td>1027895891565465600</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>&lt;a href="http://twitter.com" rel="nofollow"&gt;Tw...</td>
      <td>@SGTreport ALEX JONES IS BILL HICKS</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>bek90941_2</th>
      <td>None</td>
      <td>None</td>
      <td>Fri Aug 10 12:24:01 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [], 'u...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1027893328656982017</td>
      <td>1027893328656982017</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>&lt;a href="http://twitter.com" rel="nofollow"&gt;Tw...</td>
      <td>@RealAlexJones HI BILLY BOY!!!!!!</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 30 columns</p>
</div>




```python
# Save raw features data to CSV file
df_features.to_csv("raw_features.csv")
```

### The raw features data is saved to ' `raw_features.csv`'.

----

## Using Botometer API to get Botometer scores

The Botometer score forms the basis for our labels, and we derive our "true" response values from this score. The code below retrieves the Botometer score we are interested in (the `english` score on a scale of 0 to 1) for each follower, and places them in a Pandas DataFrame object. If a follower doesn't have any tweets, then Botometer API responds with an error, in which case the score is saved as NaN.


```python
# Create DataFrame with the list of followers
df_labels = pd.DataFrame(follower_list, columns=['follower'])

# For each follower, retrieve their Botometer score and insert into
# the DataFrame under the column named 'bm_score'
for i in range(len(follower_list)):
    try:
        df_labels['bot_score'][i]=get_bm_score(df_labels['follower'][i])
    except:
        df_labels['bot_score'][i]=np.NAN
```

The resulting dataframe looks like this:


```python
# Display Botometer scores data
df_labels.head()
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ChrisEnerIdeas</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>fbibug</td>
      <td>0.135583</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MishaJo26609942</td>
      <td>0.135583</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Clint12Michelle</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>bustin1791</td>
      <td>0.221806</td>
    </tr>
  </tbody>
</table>
</div>



### Creating new columns for binomial and 3-class representations


```python
# Create 'bot_or_not' column classifying each row as a bot (1) or not (0)
# Only real valued bot scores that are greater than 0.5 are bots,
# and all other bot_score values are labeled as human.
df_labels['bot_or_not']=df_labels.bot_score.apply(lambda x: float(x>0.5))


# Create 'bot_or_not_or_what' column to classify each row as
# a bot (1) or not (-1) or inconclusive (0) (i.e., a what?)
# Custom function toBotNotorWhat() is applied to classify each row
df_labels['bot_or_not_or_what']=df_labels.bot_score.apply(lambda x: toBotNotorWhat(x))
```


```python
# Displaying raw labels data
df_labels.head()
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
      <th>0</th>
      <td>ChrisEnerIdeas</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>fbibug</td>
      <td>0.135583</td>
      <td>0.0</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MishaJo26609942</td>
      <td>0.135583</td>
      <td>0.0</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Clint12Michelle</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>bustin1791</td>
      <td>0.221806</td>
      <td>0.0</td>
      <td>-1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Getting summary statistics
df_labels.describe()
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
      <th>bot_score</th>
      <th>bot_or_not</th>
      <th>bot_or_not_or_what</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2365.000000</td>
      <td>3200.000000</td>
      <td>3200.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.504951</td>
      <td>0.354063</td>
      <td>-0.001563</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.349934</td>
      <td>0.478303</td>
      <td>0.765175</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.018240</td>
      <td>0.000000</td>
      <td>-1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.156855</td>
      <td>0.000000</td>
      <td>-1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.463592</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.888218</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.987588</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Save to CSV file
df_labels.to_csv("raw_labels.csv")
```

### The raw labels data is saved as '`raw_labels.csv`'.


----

## Issues Encountered

* The Tweepy API would only take 600 requests at a time. Once 600 requests are made, there is a 15 minute waiting period before requests can be made again. The first account we began experimenting with on the Tweepy API was @nikkihaley. This is why we ended up with 600 followers for Nikki Haley. For the rest of the official government accounts, we requested 200 followers each, for three accounts on each run.



* In 2.4, it was difficult to get all the tweets of all the followers placed into one single text file. Our solution was to create 3,200 separate files for each of the followers in our list, and afterwards stitch up the 3,200 files together in a single dataframe. Please note that we highly suspect there may be an easier way to do this, and any advice would be greatly appreciated.



* The Botometer scores took a really long time to get, largely due to the slow response time for each request. At one point, we received an e-mail notification from Rapid API, the provider of the Botometer API, stating that 85% of our free subscription had been consumed. This led us to be more mindful of our limited resources, and eventually we settled on 3,200 scores.


## Remarks

In this section, Data Collection, we collected and saved our features and labels as raw data. Note that our datasets, which potentially full of useful information, cannot be effectively employed in training a machine learning model in their current raw form.

In the next section, Data Pre-Processing, and the one after that, NLP (Natural Language Processing), you will see the raw data collected here get processed and transformed to become ready for use in machine learning algorithms.
