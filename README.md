# <span style="color:blue">G20 Summit Public Sentiment Analysis Project</span>

## Project Overview

This project aims to analyze public sentiment during the G20 Summit in Delhi by leveraging data collected from two major social media platforms: Twitter and YouTube. The process involves data scraping, preprocessing, sentiment analysis, modeling, and visualization. The focus here is on YouTube data, and the required tools for scraping include Selenium, Python (version >= 3.6), Chromedriver, and urllib.

## YouTube Data Scraping

1. **Tools Used:**
    - Selenium
    - Python (version >= 3.6)
    - Chromedriver
    - urllib

2. **Data Storage:**
    - The scraped data is stored in a CSV file named `Youtube_scrapping_comments.csv`.

## Analysis of YouTube Data

1. **Content Summarization:**
    - [Brief description of the content summarization process.]

2. **Content Categorization:**
    - Utilized a k-means approach for categorization.
    - Employed Sentence Transformers for better analysis.

3. **Trend Analysis:**
    - [Details on how trend analysis was conducted.]

4. **Topic Search and Retrieval:**
    - [Explanation of the process for searching and retrieving topics.]

## Modeling with NLTK

### Workflow

1. **Importing Libraries:**
    - [List of main libraries imported for the modeling process.]

2. **Reading Data:**
    - Read data from `Youtube_scrapping_comments.csv`.

3. **Modeling with NLTK:**
    - Utilized NLTK library for sentiment analysis.

4. **Pipeline for Sentiment Analysis:**
    - Created a sentiment analysis pipeline using NLTK.

## Sentiment Analysis for Twitter Data

### Workflow

1. **Data Collection:**
    - Used Apify for collecting Twitter data.

2. **Unsupervised Learning:**
    - Considering it's unsupervised, applied k-means or Sentence Transformation.

3. **Classification Approach:**
    - Instead of clustering, employed a classification approach.
    - Used Hugging Face's `pipeline` for zero-shot classification.


