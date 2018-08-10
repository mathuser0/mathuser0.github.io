---
layout: page
title: Raw Data
permalink: /raw_data

---




# Gathering Raw Data

## 1. Section Description

This section is all about building the dataset used in this project. It should be noted that all analyses and results hereafter are limited to the dataset formulated in this section. With that, it should also be noted that the code and methodology used here are not bound to this particular dataset. That is to say, you should be able to use the code and methods set forth below to build a completely different set of data that fits your particular needs.



The data for this project was opted to be formulated based on official U.S. government Twitter accounts, with in mind how the political influence of Twitter bots has recently emerged as a major concern. The reasoning behind the selected accounts is not worth pondering about, as they just happened to be among the first to be looked at by the project members. In total, 14 official government Twitter accounts were selected to be the basis upon which our dataset was built. They are as follows:

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

It is important to note that the machine learning techniques used in this project were limited to supervised learning methods. Therefore, the data necessarily consists of two parts: features and labels.

### 1.1. Features

The basic atomic building block of all things Twitter is the [Tweet object](https://developer.twitter.com/en/docs/tweets/data-dictionary/overview/tweet-object.html), and every tweet is actually a Tweet object. The text portion of a tweet that we are used to seeing is actually just one of many attributes in an instantiated Tweet object.

We built our raw features data from the tweets created by the followers in our list. Using the Tweepy API again, we requested for the 200 most recent tweets of each follower in our list. Note that by doing this, for followers who had made less than 200 tweets total, all of their tweets were collected. (See [code](#code_24))



Our next step involved using the Tweepy API again, this time requesting the 200 most recent tweets for each of the 3,200 followers. Our search terms and parameters are shown in Figure 2 below.


Figure 2. The search terms and parameters we used to get the 200 most recent tweets for each ‘user’ in our list of 3,200 followers, ‘userlist’. It was not easy to figure out how to collect all the tweets from all the users into a single file. Our solution was to create 3,200 files and stitch them back up together afterwards.

After spending too much time trying to get all the tweets of all 3,200 followers placed into one file in a single run, we decided it would be easier to put together 3,200 files (one for each follower) by iteration and Pandas. Figure 3 shows the code we used to do this.


Figure 3. The code that created a Pandas DataFrame object containing our raw dataset.

The resulting DataFrame object is shown in Figure 4. At this point, our raw dataset consisted of 107,158 observations or Tweet objects. This concluded our acquisition of raw data to use as features in our machine learning model.


Figure 4. Our raw feature dataset. Each row is a Tweet object.

### 1.2. Building the Labels

Our next mission was to acquire response variable data. As we mentioned in Milestone #2, we will be deriving our “true” response values from Botometer scores. Please note that the Botometer API is a government funded project and we determined it to be a relatively reliable source for benchmarking our model predictions.

We used the Botometer API to obtain scores for each follower in our list. It should be noted that the Botometer API responds with many different scores. Figure 5 shows an example of a Botometer API response, where the highlighted score is the one we chose to use for our project.


Figure 5. Example of a Botometer API response. There are categories and sub-categories of scores to choose from. Our research into the meaning behind each score led us to choose the score that is highlighted.

After jumping through some hurdles, we were able to compile a DataFrame with the screen name of the followers in our list in one column, and their corresponding score in another column.

We realized at this point that a significant portion of our list of followers had scores that were NaN values. Upon further examination, we determined that a NaN score is given to Twitter accounts that have never tweeted. Since a Twitter bot that does not tweet is unable to bear malice toward society, it would not be a bot worth detecting. Therefore, we decided to assume that followers with NaN scores are all humans.

Now, a Botometer score that is not a NaN can take on continuous values between 0 and 1, and it is a measure of how likely it is that the associated Twitter account is a bot. We decided to derive our response variable as being equal to 1 if the Botometer score is greater than 0.5, and equal to 0 otherwise. Creating a new column called “bot_or_not”, our Botometer score DataFrame took on the form as shown in Figure 6.


Figure 6. Our Botometer dataset which contains the column, “bot_or_not”, indicating whether a ‘follower’ is a bot or not.



in this section  This

<a id='code'>code</a>

## <a id='python_code'>2. Python Code</a>

### <a id='code_1'>2.1 Libraries and Authentication</a>


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

### <a id='functions'>2.2 Custom Functions</a>


```python
# Function to get the 'english' Botometer score
def get_bm_score(screen_name):  
    score = bom.check_account('@'+screen_name)['scores']['english']
    return score


def toDataFrame(list_of_dictionaries, columns_to_get):
    Dataset=pd.DataFrame()
    for column in columns_to_get:
            Dataset[column]=[dictionary[column] for dictionary in list_of_dictionaries]
    return Dataset
```

### <a id='code_23'>2.3 Get List of 200 Most Recent Followers for the 14 Goverment Twitter Accounts</a>


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

    Rate limit reached. Sleeping for: 704
    Rate limit reached. Sleeping for: 865
    Rate limit reached. Sleeping for: 853
    Rate limit reached. Sleeping for: 867
    Rate limit reached. Sleeping for: 867



```python
len(follower_list)
```




    2800



### <a id='code_24'>2.4 Get Tweets from List of Followers</a>

Our next step involved using the Tweepy API again, this time requesting the 200 most recent tweets for each of the 3,200 followers. Notice the search terms and parameters chosen in the first 6 lines below.  


```python
# For each follower in follower_list, create a text file and save their tweets.
# If a follower has more than 200 tweets, save only the most recent 200.

for follower in follower_list:

    searchQuery = 'From:'+follower  # Search for tweets created by the current 'follower'
    maxTweets = 200 # Get at most 200 tweets for each follower
    tweetsPerQry = 100  # Query 100 tweets at a time. This is the max the API permits.
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

    print ("Downloaded {0} tweets, Saved to {1}".format(tweetCount, fName)) # Let us know of progress
```

    Downloading max 200 tweets
    Downloaded 3 tweets
    No more tweets found
    Downloaded 3 tweets, Saved to tweets_jarnomyohanen2.json
    Downloading max 200 tweets
    Downloaded 1 tweets
    No more tweets found
    Downloaded 1 tweets, Saved to tweets_rickysohel.json
    Downloading max 200 tweets
    Downloaded 53 tweets
    No more tweets found
    Downloaded 53 tweets, Saved to tweets_NoFunClub2112.json
    Downloading max 200 tweets
    No more tweets found
    Downloaded 0 tweets, Saved to tweets_timothygraham.json
    Downloading max 200 tweets
    Downloaded 71 tweets
    No more tweets found
    Downloaded 71 tweets, Saved to tweets_Tuhimbise_26.json
    Downloading max 200 tweets
    Downloaded 1 tweets
    No more tweets found
    Downloaded 1 tweets, Saved to tweets_Regan_4.json
    Downloading max 200 tweets
    Downloaded 100 tweets
    Downloaded 118 tweets
    No more tweets found
    Downloaded 118 tweets, Saved to tweets_csooran.json
    Downloading max 200 tweets
    No more tweets found
    Downloaded 0 tweets, Saved to tweets_goodgrieef.json
    Downloading max 200 tweets
    Downloaded 1 tweets



    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    /anaconda/lib/python3.6/site-packages/urllib3/connectionpool.py in _make_request(self, conn, method, url, timeout, chunked, **httplib_request_kw)
        379             try:  # Python 2.7, use buffering of HTTP responses
    --> 380                 httplib_response = conn.getresponse(buffering=True)
        381             except TypeError:  # Python 2.6 and older, Python 3


    TypeError: getresponse() got an unexpected keyword argument 'buffering'


    During handling of the above exception, another exception occurred:


    WantReadError                             Traceback (most recent call last)

    /anaconda/lib/python3.6/site-packages/urllib3/contrib/pyopenssl.py in recv_into(self, *args, **kwargs)
        279         try:
    --> 280             return self.connection.recv_into(*args, **kwargs)
        281         except OpenSSL.SSL.SysCallError as e:


    /anaconda/lib/python3.6/site-packages/OpenSSL/SSL.py in recv_into(self, buffer, nbytes, flags)
       1546             result = _lib.SSL_read(self._ssl, buf, nbytes)
    -> 1547         self._raise_ssl_error(self._ssl, result)
       1548


    /anaconda/lib/python3.6/site-packages/OpenSSL/SSL.py in _raise_ssl_error(self, ssl, result)
       1352         if error == _lib.SSL_ERROR_WANT_READ:
    -> 1353             raise WantReadError()
       1354         elif error == _lib.SSL_ERROR_WANT_WRITE:


    WantReadError:


    During handling of the above exception, another exception occurred:


    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-12-7656b4a54bfd> in <module>()
         31                     if (not sinceId):
         32                         new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
    ---> 33                                                 max_id=str(max_id - 1))
         34                     else:
         35                         new_tweets = api.search(q=searchQuery, count=tweetsPerQry,


    /anaconda/lib/python3.6/site-packages/tweepy/binder.py in _call(*args, **kwargs)
        248             return method
        249         else:
    --> 250             return method.execute()
        251
        252     # Set pagination mode


    /anaconda/lib/python3.6/site-packages/tweepy/binder.py in execute(self)
        188                                                 timeout=self.api.timeout,
        189                                                 auth=auth,
    --> 190                                                 proxies=self.api.proxy)
        191                 except Exception as e:
        192                     six.reraise(TweepError, TweepError('Failed to send request: %s' % e), sys.exc_info()[2])


    /anaconda/lib/python3.6/site-packages/requests/sessions.py in request(self, method, url, params, data, headers, cookies, files, auth, timeout, allow_redirects, proxies, hooks, stream, verify, cert, json)
        506         }
        507         send_kwargs.update(settings)
    --> 508         resp = self.send(prep, **send_kwargs)
        509
        510         return resp


    /anaconda/lib/python3.6/site-packages/requests/sessions.py in send(self, request, **kwargs)
        616
        617         # Send the request
    --> 618         r = adapter.send(request, **kwargs)
        619
        620         # Total elapsed time of the request (approximately)


    /anaconda/lib/python3.6/site-packages/requests/adapters.py in send(self, request, stream, timeout, verify, cert, proxies)
        438                     decode_content=False,
        439                     retries=self.max_retries,
    --> 440                     timeout=timeout
        441                 )
        442


    /anaconda/lib/python3.6/site-packages/urllib3/connectionpool.py in urlopen(self, method, url, body, headers, retries, redirect, assert_same_host, timeout, pool_timeout, release_conn, chunked, body_pos, **response_kw)
        599                                                   timeout=timeout_obj,
        600                                                   body=body, headers=headers,
    --> 601                                                   chunked=chunked)
        602
        603             # If we're going to release the connection in ``finally:``, then


    /anaconda/lib/python3.6/site-packages/urllib3/connectionpool.py in _make_request(self, conn, method, url, timeout, chunked, **httplib_request_kw)
        381             except TypeError:  # Python 2.6 and older, Python 3
        382                 try:
    --> 383                     httplib_response = conn.getresponse()
        384                 except Exception as e:
        385                     # Remove the TypeError from the exception chain in Python 3;


    /anaconda/lib/python3.6/http/client.py in getresponse(self)
       1329         try:
       1330             try:
    -> 1331                 response.begin()
       1332             except ConnectionError:
       1333                 self.close()


    /anaconda/lib/python3.6/http/client.py in begin(self)
        295         # read until we get a non-100 response
        296         while True:
    --> 297             version, status, reason = self._read_status()
        298             if status != CONTINUE:
        299                 break


    /anaconda/lib/python3.6/http/client.py in _read_status(self)
        256
        257     def _read_status(self):
    --> 258         line = str(self.fp.readline(_MAXLINE + 1), "iso-8859-1")
        259         if len(line) > _MAXLINE:
        260             raise LineTooLong("status line")


    /anaconda/lib/python3.6/socket.py in readinto(self, b)
        584         while True:
        585             try:
    --> 586                 return self._sock.recv_into(b)
        587             except timeout:
        588                 self._timeout_occurred = True


    /anaconda/lib/python3.6/site-packages/urllib3/contrib/pyopenssl.py in recv_into(self, *args, **kwargs)
        290                 raise
        291         except OpenSSL.SSL.WantReadError:
    --> 292             rd = util.wait_for_read(self.socket, self.socket.gettimeout())
        293             if not rd:
        294                 raise timeout('The read operation timed out')


    /anaconda/lib/python3.6/site-packages/urllib3/util/wait.py in wait_for_read(socks, timeout)
         31     or optionally a single socket if passed in. Returns a list of
         32     sockets that can be read from immediately. """
    ---> 33     return _wait_for_io_events(socks, EVENT_READ, timeout)
         34
         35


    /anaconda/lib/python3.6/site-packages/urllib3/util/wait.py in _wait_for_io_events(socks, events, timeout)
         24             selector.register(sock, events)
         25         return [key[0].fileobj for key in
    ---> 26                 selector.select(timeout) if key[1] & events]
         27
         28


    /anaconda/lib/python3.6/site-packages/urllib3/util/selectors.py in select(self, timeout)
        511
        512             kevent_list = _syscall_wrapper(self._kqueue.control, True,
    --> 513                                            None, max_events, timeout)
        514
        515             for kevent in kevent_list:


    /anaconda/lib/python3.6/site-packages/urllib3/util/selectors.py in _syscall_wrapper(func, _, *args, **kwargs)
         62         and recalculate their timeouts. """
         63         try:
    ---> 64             return func(*args, **kwargs)
         65         except (OSError, IOError, select.error) as e:
         66             errcode = None


    KeyboardInterrupt:


At this point, we had a separate file containing the tweets for each follower. Our next step was to construct a single dataframe from these files as follows.


```python
# Initialize tweets dictionary
tweets={}

# For each follower, place each tweet in the tweets dictionary
# with as a separate key:value entry
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

This concludes our acquisition of features data consisting of Tweet objects.


```python
df_features
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
      <th>jarnomyohanen2_0</th>
      <td>None</td>
      <td>None</td>
      <td>Fri Aug 03 09:11:13 +0000 2018</td>
      <td>{'hashtags': [{'indices': [19, 31], 'text': 'k...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1025308093188644865</td>
      <td>1025308093188644865</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>&lt;a href="https://mobile.twitter.com" rel="nofo...</td>
      <td>Piirränpä tässä jo #kuolinkirja:n jo kun apu e...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>jarnomyohanen2_1</th>
      <td>None</td>
      <td>None</td>
      <td>Wed Aug 01 18:01:23 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [], 'u...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1024716738049568774</td>
      <td>1024716738049568774</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>&lt;a href="https://mobile.twitter.com" rel="nofo...</td>
      <td>Mitä jos minä rakentaisin talon!!!</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>jarnomyohanen2_2</th>
      <td>None</td>
      <td>None</td>
      <td>Wed Aug 01 06:47:09 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [], 'u...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1024547063596572674</td>
      <td>1024547063596572674</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>In English; Minä oon vaan homo-huora,</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>rickysohel_0</th>
      <td>None</td>
      <td>None</td>
      <td>Thu Aug 02 05:26:14 +0000 2018</td>
      <td>{'hashtags': [], 'media': [{'display_url': 'pi...</td>
      <td>{'media': [{'display_url': 'pic.twitter.com/7M...</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1024889088652931072</td>
      <td>1024889088652931072</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>These dogs aren't pet , they are street dogs.....</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>NoFunClub2112_0</th>
      <td>None</td>
      <td>None</td>
      <td>Wed Aug 08 15:07:29 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [], 'u...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1027209691028512769</td>
      <td>1027209691028512769</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>793</td>
      <td>False</td>
      <td>{'contributors': None, 'coordinates': None, 'c...</td>
      <td>&lt;a href="https://mobile.twitter.com" rel="nofo...</td>
      <td>RT @JohnnieGuilbert: Ha\nHa\nHa\nHa\nHa\nHa\nH...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>NoFunClub2112_1</th>
      <td>None</td>
      <td>None</td>
      <td>Wed Aug 08 15:06:20 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [], 'u...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1027209403118829568</td>
      <td>1027209403118829568</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>&lt;a href="https://mobile.twitter.com" rel="nofo...</td>
      <td>@CIA You guys are awesome.</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>NoFunClub2112_2</th>
      <td>None</td>
      <td>None</td>
      <td>Wed Aug 08 15:04:41 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [], 'u...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1027208985026469888</td>
      <td>1027208985026469888</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>&lt;a href="https://mobile.twitter.com" rel="nofo...</td>
      <td>@rocksound @adtr Keep your hopes up high and y...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>NoFunClub2112_3</th>
      <td>None</td>
      <td>None</td>
      <td>Wed Aug 08 01:09:00 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [], 'u...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1026998680870633472</td>
      <td>1026998680870633472</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>&lt;a href="https://mobile.twitter.com" rel="nofo...</td>
      <td>@M69533368 @qanon76 I'm confused</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>NoFunClub2112_4</th>
      <td>None</td>
      <td>None</td>
      <td>Wed Aug 08 00:00:23 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [], 'u...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1026981411172040705</td>
      <td>1026981411172040705</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>&lt;a href="https://mobile.twitter.com" rel="nofo...</td>
      <td>@deathoftheparty @robwhisman One does not "Rag...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>NoFunClub2112_5</th>
      <td>None</td>
      <td>None</td>
      <td>Tue Aug 07 23:48:44 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [{'dis...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1026978479051628545</td>
      <td>1026978479051628545</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>&lt;a href="https://mobile.twitter.com" rel="nofo...</td>
      <td>@ACMESalesRep @somederekkid @deathoftheparty D...</td>
      <td>True</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>NoFunClub2112_6</th>
      <td>None</td>
      <td>None</td>
      <td>Tue Aug 07 23:44:07 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [], 'u...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1026977316084350976</td>
      <td>1026977316084350976</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3183</td>
      <td>False</td>
      <td>{'contributors': None, 'coordinates': None, 'c...</td>
      <td>&lt;a href="https://mobile.twitter.com" rel="nofo...</td>
      <td>RT @deathoftheparty: at what point in my 25 ye...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>NoFunClub2112_7</th>
      <td>None</td>
      <td>None</td>
      <td>Tue Aug 07 23:36:33 +0000 2018</td>
      <td>{'hashtags': [], 'media': [{'display_url': 'pi...</td>
      <td>{'media': [{'display_url': 'pic.twitter.com/zW...</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1026975412650430464</td>
      <td>1026975412650430464</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13559</td>
      <td>False</td>
      <td>{'contributors': None, 'coordinates': None, 'c...</td>
      <td>&lt;a href="https://mobile.twitter.com" rel="nofo...</td>
      <td>RT @JackPosobiec: Be a real shame if this pict...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>NoFunClub2112_8</th>
      <td>None</td>
      <td>None</td>
      <td>Tue Aug 07 23:36:20 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [], 'u...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1026975359735144462</td>
      <td>1026975359735144462</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>167</td>
      <td>False</td>
      <td>{'contributors': None, 'coordinates': None, 'c...</td>
      <td>&lt;a href="https://mobile.twitter.com" rel="nofo...</td>
      <td>RT @4uConservatives: There is a conservative p...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>NoFunClub2112_9</th>
      <td>None</td>
      <td>None</td>
      <td>Tue Aug 07 23:33:17 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [], 'u...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1026974589975519232</td>
      <td>1026974589975519232</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>&lt;a href="https://mobile.twitter.com" rel="nofo...</td>
      <td>@lollapalooza *Cough Cough* @RiotFest has a be...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>NoFunClub2112_10</th>
      <td>None</td>
      <td>None</td>
      <td>Tue Aug 07 14:17:39 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [], 'u...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1026834761929707521</td>
      <td>1026834761929707521</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6039</td>
      <td>False</td>
      <td>{'contributors': None, 'coordinates': None, 'c...</td>
      <td>&lt;a href="https://mobile.twitter.com" rel="nofo...</td>
      <td>RT @w_terrence: ANTIFA threw a Temper Tantrum ...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>NoFunClub2112_11</th>
      <td>None</td>
      <td>None</td>
      <td>Tue Aug 07 02:50:08 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [], 'u...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1026661741814599681</td>
      <td>1026661741814599681</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2</td>
      <td>False</td>
      <td>{'contributors': None, 'coordinates': None, 'c...</td>
      <td>&lt;a href="https://mobile.twitter.com" rel="nofo...</td>
      <td>RT @daa_vinci_: @Mikanojo @bpjauburn @ajplus @...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>NoFunClub2112_12</th>
      <td>None</td>
      <td>None</td>
      <td>Tue Aug 07 02:48:04 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [], 'u...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1026661220387115009</td>
      <td>1026661220387115009</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>&lt;a href="https://mobile.twitter.com" rel="nofo...</td>
      <td>@elevatormelba @Christianx14 I also am interes...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>NoFunClub2112_13</th>
      <td>None</td>
      <td>None</td>
      <td>Tue Aug 07 02:45:38 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [], 'u...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1026660608882692096</td>
      <td>1026660608882692096</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13101</td>
      <td>False</td>
      <td>{'contributors': None, 'coordinates': None, 'c...</td>
      <td>&lt;a href="https://mobile.twitter.com" rel="nofo...</td>
      <td>RT @w_terrence: Breaking! 72 people were shot ...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>NoFunClub2112_14</th>
      <td>None</td>
      <td>None</td>
      <td>Tue Aug 07 00:07:27 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [], 'u...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1026620802115022849</td>
      <td>1026620802115022849</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>&lt;a href="https://mobile.twitter.com" rel="nofo...</td>
      <td>@BEARTOOTHband @LoperandRandi You guys are leg...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>NoFunClub2112_15</th>
      <td>None</td>
      <td>None</td>
      <td>Tue Aug 07 00:05:36 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [], 'u...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1026620335553224705</td>
      <td>1026620335553224705</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>&lt;a href="https://mobile.twitter.com" rel="nofo...</td>
      <td>@rocksound "Attention, all planets of the Sola...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>NoFunClub2112_16</th>
      <td>None</td>
      <td>None</td>
      <td>Fri Aug 03 18:41:27 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [], 'u...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1025451596778545153</td>
      <td>1025451596778545153</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>&lt;a href="https://mobile.twitter.com" rel="nofo...</td>
      <td>@diskhomtl @FoxNews We have them everywhere, i...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>NoFunClub2112_17</th>
      <td>None</td>
      <td>None</td>
      <td>Fri Aug 03 03:28:23 +0000 2018</td>
      <td>{'hashtags': [], 'media': [{'display_url': 'pi...</td>
      <td>{'media': [{'additional_media_info': {'monetiz...</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1025221816443383808</td>
      <td>1025221816443383808</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1507</td>
      <td>False</td>
      <td>{'contributors': None, 'coordinates': None, 'c...</td>
      <td>&lt;a href="https://mobile.twitter.com" rel="nofo...</td>
      <td>RT @PattyxWalters: https://t.co/YhvAqXJZbo</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>NoFunClub2112_18</th>
      <td>None</td>
      <td>None</td>
      <td>Fri Aug 03 03:26:49 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [], 'u...</td>
      <td>NaN</td>
      <td>1</td>
      <td>False</td>
      <td>None</td>
      <td>1025221420572442625</td>
      <td>1025221420572442625</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>&lt;a href="https://mobile.twitter.com" rel="nofo...</td>
      <td>@KevinLyman @VansWarpedTour It rained at the C...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>NoFunClub2112_19</th>
      <td>None</td>
      <td>None</td>
      <td>Thu Aug 02 18:37:03 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [], 'u...</td>
      <td>NaN</td>
      <td>1</td>
      <td>False</td>
      <td>None</td>
      <td>1025088101742510080</td>
      <td>1025088101742510080</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>&lt;a href="https://mobile.twitter.com" rel="nofo...</td>
      <td>@alepolice22 @lollapalooza @QuarterPress Warpe...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>NoFunClub2112_20</th>
      <td>None</td>
      <td>None</td>
      <td>Thu Aug 02 18:19:15 +0000 2018</td>
      <td>{'hashtags': [], 'media': [{'display_url': 'pi...</td>
      <td>{'media': [{'display_url': 'pic.twitter.com/db...</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1025083623136026625</td>
      <td>1025083623136026625</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>43</td>
      <td>False</td>
      <td>{'contributors': None, 'coordinates': None, 'c...</td>
      <td>&lt;a href="https://mobile.twitter.com" rel="nofo...</td>
      <td>RT @RiotFest: Good luck this weekend @lollapal...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>NoFunClub2112_21</th>
      <td>None</td>
      <td>None</td>
      <td>Thu Aug 02 18:18:25 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [], 'u...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1025083414314147842</td>
      <td>1025083414314147842</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>&lt;a href="https://mobile.twitter.com" rel="nofo...</td>
      <td>@AltPress Emo Soldier.</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>NoFunClub2112_22</th>
      <td>None</td>
      <td>None</td>
      <td>Thu Aug 02 18:17:30 +0000 2018</td>
      <td>{'hashtags': [], 'media': [{'display_url': 'pi...</td>
      <td>{'media': [{'additional_media_info': {'call_to...</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1025083183883333632</td>
      <td>1025083183883333632</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>98</td>
      <td>False</td>
      <td>{'contributors': None, 'coordinates': None, 'c...</td>
      <td>&lt;a href="https://mobile.twitter.com" rel="nofo...</td>
      <td>RT @BEARTOOTHband: Beartooth: Greatness or Dea...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>NoFunClub2112_23</th>
      <td>None</td>
      <td>None</td>
      <td>Thu Aug 02 18:17:21 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [], 'u...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1025083144133857281</td>
      <td>1025083144133857281</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>&lt;a href="https://mobile.twitter.com" rel="nofo...</td>
      <td>@BEARTOOTHband @ReelBearMedia Looks good! Stil...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>NoFunClub2112_24</th>
      <td>None</td>
      <td>None</td>
      <td>Thu Aug 02 18:08:43 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [], 'u...</td>
      <td>NaN</td>
      <td>1</td>
      <td>False</td>
      <td>None</td>
      <td>1025080972172308487</td>
      <td>1025080972172308487</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>&lt;a href="https://mobile.twitter.com" rel="nofo...</td>
      <td>@FENDmovement Getting to see @BEARTOOTHband on...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>NoFunClub2112_25</th>
      <td>None</td>
      <td>None</td>
      <td>Thu Aug 02 08:07:52 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [], 'u...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1024929762257641472</td>
      <td>1024929762257641472</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>20113</td>
      <td>False</td>
      <td>{'contributors': None, 'coordinates': None, 'c...</td>
      <td>&lt;a href="https://mobile.twitter.com" rel="nofo...</td>
      <td>RT @w_terrence: If Donald Trump is a racist th...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
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
      <th>csooran_89</th>
      <td>None</td>
      <td>None</td>
      <td>Tue Jul 31 13:24:39 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [], 'u...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1024284708551749632</td>
      <td>1024284708551749632</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>&lt;a href="http://twitter.com" rel="nofollow"&gt;Tw...</td>
      <td>Do investors who believe in mean reversion of ...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>csooran_90</th>
      <td>None</td>
      <td>None</td>
      <td>Tue Jul 31 12:22:33 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [{'dis...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1024269080512159745</td>
      <td>1024269080512159745</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>In Harvard’s defense, they are selling a signa...</td>
      <td>True</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>csooran_91</th>
      <td>None</td>
      <td>None</td>
      <td>Tue Jul 31 11:51:19 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [], 'u...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1024261221023797248</td>
      <td>1024261221023797248</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3</td>
      <td>False</td>
      <td>{'contributors': None, 'coordinates': None, 'c...</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>RT @Sparticuszorro: On This Day\n\nJuly, 2000 ...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>csooran_92</th>
      <td>None</td>
      <td>None</td>
      <td>Mon Jul 30 23:35:14 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [{'dis...</td>
      <td>NaN</td>
      <td>2</td>
      <td>False</td>
      <td>None</td>
      <td>1024075979805863942</td>
      <td>1024075979805863942</td>
      <td>...</td>
      <td>{'contributors': None, 'coordinates': None, 'c...</td>
      <td>1023965570574766080</td>
      <td>1023965570574766080</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>Good luck @wd_eyre.  Important work. https://t...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>csooran_93</th>
      <td>None</td>
      <td>None</td>
      <td>Mon Jul 30 21:03:43 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [{'dis...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1024037847404359681</td>
      <td>1024037847404359681</td>
      <td>...</td>
      <td>{'contributors': None, 'coordinates': None, 'c...</td>
      <td>1023987359061299200</td>
      <td>1023987359061299200</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>&lt;a href="http://twitter.com" rel="nofollow"&gt;Tw...</td>
      <td>Asterix is the hardest-working ship in the Roy...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>csooran_94</th>
      <td>None</td>
      <td>None</td>
      <td>Mon Jul 30 20:41:49 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [], 'u...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1024032338286702593</td>
      <td>1024032338286702593</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>99</td>
      <td>False</td>
      <td>{'contributors': None, 'coordinates': None, 'c...</td>
      <td>&lt;a href="http://twitter.com" rel="nofollow"&gt;Tw...</td>
      <td>RT @lisaabramowicz1: A lot of Wall Street anal...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>csooran_95</th>
      <td>None</td>
      <td>None</td>
      <td>Mon Jul 30 20:40:18 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [], 'u...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1024031956588216320</td>
      <td>1024031956588216320</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>258</td>
      <td>False</td>
      <td>{'contributors': None, 'coordinates': None, 'c...</td>
      <td>&lt;a href="http://twitter.com" rel="nofollow"&gt;Tw...</td>
      <td>RT @charliebilello: It's official: the S&amp;amp;P...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>csooran_96</th>
      <td>None</td>
      <td>None</td>
      <td>Mon Jul 30 18:20:53 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [{'dis...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1023996869742092289</td>
      <td>1023996869742092289</td>
      <td>...</td>
      <td>{'contributors': None, 'coordinates': None, 'c...</td>
      <td>1023962677457182720</td>
      <td>1023962677457182720</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>&lt;a href="http://twitter.com" rel="nofollow"&gt;Tw...</td>
      <td>Boom. https://t.co/dGe9py1QuS</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>csooran_97</th>
      <td>None</td>
      <td>None</td>
      <td>Mon Jul 30 18:19:42 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [], 'u...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1023996574542778368</td>
      <td>1023996574542778368</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>&lt;a href="http://twitter.com" rel="nofollow"&gt;Tw...</td>
      <td>What if, after years of bull market, retail in...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>csooran_98</th>
      <td>None</td>
      <td>None</td>
      <td>Mon Jul 30 17:58:34 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [], 'u...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1023991255469105152</td>
      <td>1023991255469105152</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>18</td>
      <td>False</td>
      <td>{'contributors': None, 'coordinates': None, 'c...</td>
      <td>&lt;a href="http://twitter.com" rel="nofollow"&gt;Tw...</td>
      <td>RT @ReformedBroker: "I suspect it is this char...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>csooran_99</th>
      <td>None</td>
      <td>None</td>
      <td>Mon Jul 30 15:27:46 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [{'dis...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1023953303737180160</td>
      <td>1023953303737180160</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>&lt;a href="http://twitter.com" rel="nofollow"&gt;Tw...</td>
      <td>I suppose this means that everyone else is for...</td>
      <td>True</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>csooran_100</th>
      <td>None</td>
      <td>None</td>
      <td>Mon Jul 30 15:25:53 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [{'dis...</td>
      <td>NaN</td>
      <td>1</td>
      <td>False</td>
      <td>None</td>
      <td>1023952830506385408</td>
      <td>1023952830506385408</td>
      <td>...</td>
      <td>{'contributors': None, 'coordinates': None, 'c...</td>
      <td>1023952234252513281</td>
      <td>1023952234252513281</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>&lt;a href="http://twitter.com" rel="nofollow"&gt;Tw...</td>
      <td>Late-stage? https://t.co/a3VLB84sZD</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>csooran_101</th>
      <td>None</td>
      <td>None</td>
      <td>Mon Jul 30 12:55:45 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [{'dis...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1023915047955910656</td>
      <td>1023915047955910656</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>The hottest counterintelligence war you haven’...</td>
      <td>True</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>csooran_102</th>
      <td>None</td>
      <td>None</td>
      <td>Mon Jul 30 12:49:25 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [{'dis...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1023913454745411591</td>
      <td>1023913454745411591</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>DARPA invests in chip industry research in siz...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>csooran_103</th>
      <td>None</td>
      <td>None</td>
      <td>Mon Jul 30 12:44:19 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [{'dis...</td>
      <td>NaN</td>
      <td>1</td>
      <td>False</td>
      <td>None</td>
      <td>1023912172588621825</td>
      <td>1023912172588621825</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>The arguments for and against free trade thirt...</td>
      <td>True</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>csooran_104</th>
      <td>None</td>
      <td>None</td>
      <td>Mon Jul 30 12:40:46 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [{'dis...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1023911277368958976</td>
      <td>1023911277368958976</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>Canada starting to get religion on national se...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>csooran_105</th>
      <td>None</td>
      <td>None</td>
      <td>Mon Jul 30 12:37:01 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [{'dis...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1023910334602653697</td>
      <td>1023910334602653697</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>Can the flywheel go in reverse?  Is falling en...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>csooran_106</th>
      <td>None</td>
      <td>None</td>
      <td>Mon Jul 30 12:31:03 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [{'dis...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1023908831842189313</td>
      <td>1023908831842189313</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>“Some species like great whites are protected,...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>csooran_107</th>
      <td>None</td>
      <td>None</td>
      <td>Mon Jul 30 12:26:14 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [{'dis...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1023907617905754112</td>
      <td>1023907617905754112</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>Who are you going to believe?  Your eyes or th...</td>
      <td>True</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>csooran_108</th>
      <td>None</td>
      <td>None</td>
      <td>Mon Jul 30 12:20:05 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [{'dis...</td>
      <td>NaN</td>
      <td>1</td>
      <td>False</td>
      <td>None</td>
      <td>1023906072531222528</td>
      <td>1023906072531222528</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>Canadian Parmesan is very competitive, apparen...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>csooran_109</th>
      <td>None</td>
      <td>None</td>
      <td>Mon Jul 30 12:11:23 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [{'dis...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1023903882894819328</td>
      <td>1023903882894819328</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>It is not difficult to imagine someone raising...</td>
      <td>True</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>csooran_110</th>
      <td>None</td>
      <td>None</td>
      <td>Mon Jul 30 12:02:16 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [{'dis...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1023901590091120640</td>
      <td>1023901590091120640</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>False</td>
      <td>NaN</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>Interesting concept: pumped water storage of e...</td>
      <td>True</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>csooran_111</th>
      <td>None</td>
      <td>None</td>
      <td>Sun Jul 29 22:42:37 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [{'dis...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1023700349809565696</td>
      <td>1023700349809565696</td>
      <td>...</td>
      <td>{'contributors': None, 'coordinates': None, 'c...</td>
      <td>1023697219621408770</td>
      <td>1023697219621408770</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>Curbing gun demand is the ultimate gun control...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>csooran_112</th>
      <td>None</td>
      <td>None</td>
      <td>Sun Jul 29 19:14:00 +0000 2018</td>
      <td>{'hashtags': [], 'media': [{'display_url': 'pi...</td>
      <td>{'media': [{'additional_media_info': {'monetiz...</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1023647850239062017</td>
      <td>1023647850239062017</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>155</td>
      <td>False</td>
      <td>{'contributors': None, 'coordinates': None, 'c...</td>
      <td>&lt;a href="http://twitter.com" rel="nofollow"&gt;Tw...</td>
      <td>RT @CalebJHull: I had to do it https://t.co/by...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>csooran_113</th>
      <td>None</td>
      <td>None</td>
      <td>Sun Jul 29 16:32:20 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [{'dis...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1023607163326005248</td>
      <td>1023607163326005248</td>
      <td>...</td>
      <td>{'contributors': None, 'coordinates': None, 'c...</td>
      <td>1023604320347058176</td>
      <td>1023604320347058176</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>&lt;a href="http://twitter.com" rel="nofollow"&gt;Tw...</td>
      <td>This would be fantastic news for the economy. ...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>csooran_114</th>
      <td>None</td>
      <td>None</td>
      <td>Sun Jul 29 16:18:11 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [{'dis...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1023603604358422529</td>
      <td>1023603604358422529</td>
      <td>...</td>
      <td>{'contributors': None, 'coordinates': None, 'c...</td>
      <td>1023602535486169089</td>
      <td>1023602535486169089</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>&lt;a href="http://twitter.com" rel="nofollow"&gt;Tw...</td>
      <td>Boom. https://t.co/IiU3tqNoiR</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>csooran_115</th>
      <td>None</td>
      <td>None</td>
      <td>Sun Jul 29 15:39:05 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [], 'u...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1023593764189863936</td>
      <td>1023593764189863936</td>
      <td>...</td>
      <td>NaN</td>
      <td>1023008988357648384</td>
      <td>1023008988357648384</td>
      <td>2</td>
      <td>False</td>
      <td>{'contributors': None, 'coordinates': None, 'c...</td>
      <td>&lt;a href="http://twitter.com" rel="nofollow"&gt;Tw...</td>
      <td>RT @LIanMacDonald: In town or at the cottage, ...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>csooran_116</th>
      <td>None</td>
      <td>None</td>
      <td>Sun Jul 29 14:16:04 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [{'dis...</td>
      <td>NaN</td>
      <td>1</td>
      <td>False</td>
      <td>None</td>
      <td>1023572870621011968</td>
      <td>1023572870621011968</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>&lt;a href="http://twitter.com/#!/download/ipad" ...</td>
      <td>“Trigger warnings may inadvertently undermine ...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>csooran_117</th>
      <td>None</td>
      <td>None</td>
      <td>Sun Jul 29 12:38:09 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [{'dis...</td>
      <td>NaN</td>
      <td>3</td>
      <td>False</td>
      <td>None</td>
      <td>1023548231404404736</td>
      <td>1023548231404404736</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>All that Tesla has done in attacking Montana S...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>natv0l_0</th>
      <td>None</td>
      <td>None</td>
      <td>Wed Aug 01 00:05:23 +0000 2018</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [], 'u...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1024445955435896832</td>
      <td>1024445955435896832</td>
      <td>...</td>
      <td>NaN</td>
      <td>1024323501887565824</td>
      <td>1024323501887565824</td>
      <td>1</td>
      <td>False</td>
      <td>{'contributors': None, 'coordinates': None, 'c...</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>RT @RyanCostello: I visited a manufacturing fa...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
  </tbody>
</table>
<p>248 rows × 30 columns</p>
</div>



### <a id='code_25'>2.5 Get Botometer Scores for each Follower</a>

The Botometer score forms the basis for our labels, and we derive our "true" response values from this score. The code below retrieves the Botometer score we are interested in (the `english` score on a scale of 0 to 1) for each follower, and places them in a Pandas DataFrame object. If a follower doesn't have any tweets, then Botometer API responds with an error, in which case the score is saved as NaN.


```python
# Create DataFrame with the list of followers
df_labels = pd.DataFrame(follower_list, columns=['follower'])

# For each follower, retrieve their Botometer score and insert into
# the DataFrame under the column named 'bm_score'
for i in range(len(follower_list)):
    try:
        df_labels['bm_score'][i]=get_bm_score(df_labels['follower'][i])
    except:
        df_labels['bm_score'][i]=np.NAN
```

The resulting dataframe looks like this:


```python
df_labels.head()
```


```python
df_labels.describe()
```

## <a id='sec3'>3. Issues Encountered</a>

1.	The Tweepy API would only take 600 requests at a time. Once 600 requests are made, there is a 15 minute waiting period before requests can be made again. The first account we began experimenting with on the Tweepy API was @nikkihaley. This is why we ended up with 600 followers for Nikki Haley. For the rest of the official government accounts, we requested 200 followers each, for three accounts on each run. <br>

2. In 2.4, it was difficult to get all the tweets of all the followers placed into one single text file. Our solution was to create 3,200 separate files for each of the followers in our list, and afterwards stitch up the 3,200 files together in a single dataframe. Please note that we highly suspect there may be an easier way to do this, and any advice would be greatly appreciated.

2.	The Botometer scores took a really long time to get, largely due to the slow response time for each request. At one point, we received an e-mail notification from Rapid API, the provider of the Botometer API, stating that 85% of our free subscription had been consumed. This led us to be more mindful of our limited resources, and eventually we settled on 3,200 scores.


## 4. Remarks
