---
layout: page
title: 0. Project Statement
permalink: /project_statement
order: 1
---


## Background
Most Americans are exposed to a daily dose of false or misleading content — hoaxes, conspiracy theories, fabricated reports, click-bait headlines, and even satire. Such content is collectively referred to as “misinformation". There have been verified instances of misinformation and disinformation spreading on social media inflicting real harm on society, such as dangerous health decisions and stock market manipulations to name a few [[1]](https://arxiv.org/abs/1707.07592). Such events has led world leaders to identify the massive spread of digital misinformation as [a major global risk](http://reports.weforum.org/global-risks-2013/risk-case-1/digital-wildfires-in-a-hyperconnected-world/?doing_wp_cron=1533730169.0472350120544433593750). The impact of online misinformation spreading through the mechanism of malicious bots might be prevented if the activities of such bots are identified quickly and accurately [[1]](https://arxiv.org/abs/1707.07592).

## Project Statement

The aim of this project is to evaluate classification models that analyze tweets data using machine learning techniques to classify authors as human or bot. Some of the classifcation models we incorporate in this project include natural language processing techniques.

This project is consists of five parts at the high-level.
<ol>
<li>Data Collection</li>
<li>Data Pre-Processing</li>
<li>Natural Language Processing</li>
<li>Exploratory Data Analysis</li>
<li>Models</li>
</ol>

In each section, you will find a description of the methodology employed, any shortcomings and thoughts on ways to improve, complete set of python code used in that section (libraries, custom functions, and procedural methods), and remarks on noteworthy findings.

It was identified in this project that there may have been errors in the way that the data was put to use. This came to light only after seeing some models score close to perfect. While the exact nature of the surreal accuracy, precision, recall, and F1 score performance exhibited by some of the model has not been determined in full yet, it seems likely to be due to the some of the tweets data in the training and test sets originating from the same author. While this is certainly the case for some of a significant portion of the tweets data, it seems unlikely to be the sole reason for models being enabled to the degree that they were. Should you happen to have any thoughts or ideas on what might have went wrong, please share them with us. Thank you.
