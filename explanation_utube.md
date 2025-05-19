# YouTube Data Analysis & Visualization Project

## Overview

This project analyzes YouTube videos and their comments related to the **G20 Summit 2023**. Using Python, it processes a dataset of video links, titles, and comments to clean the data, extract insights, and visualize findings. The goal is to understand public sentiment, identify key topics, group similar comments, and uncover patterns in the discussions.

The dataset is stored in a CSV file (`Youtube_scrapping_comments.csv`), and the analysis is performed in a Jupyter Notebook (`Youtube_Data_analysis&Visualisation.ipynb`). The project combines data cleaning, exploratory data analysis (EDA), sentiment analysis, clustering, topic modeling, and visualization to make sense of the text data.

---

## Purpose

I built this project to:
- Analyze public opinions about the G20 Summit 2023 expressed in YouTube comments.
- Apply data science techniques like sentiment analysis, clustering, and topic modeling.
- Practice using Python libraries for text processing and visualization.
- Develop skills in handling real-world, messy text data and presenting insights visually.

---

## Dataset Description

The dataset (`Youtube_scrapping_comments.csv`) contains:
- **Video Link**: URLs of YouTube videos about the G20 Summit 2023.
- **Video Title**: Titles of the videos (e.g., "World leaders arrive at the Bharat Mandapam").
- **Cleaned_Comments**: Comments from the videos, partially cleaned but still containing user mentions (e.g., "@username").
- **Item**: A label indicating all entries are related to "G20 2023".

The data is loaded into a **Pandas DataFrame** for analysis.

---

## Project Workflow and Technical Details

The project is divided into three main phases: **Exploratory Data Analysis (EDA)**, **Intermediate Data Analysis**, and **Advanced Data Analysis**. Below, I explain each step, the functions used, what they do, and why they’re important.

### 1. Exploratory Data Analysis (EDA)

#### 1.1 Initial Data Inspection
- **What I Did**: Loaded the CSV file into a Pandas DataFrame using `pd.read_csv('Youtube_scrapping_comments.csv')`. Used `df.head()` to display the first five rows and understand the columns. Ran `df.info()` to check data types and missing values, and `df.describe()` to get summary statistics (e.g., count, unique values).
- **Purpose**: To get a sense of the data’s structure, size, and quality (e.g., are there missing comments? Are columns in the right format?).
- **Example Output**: `df.head()` shows columns like `Video Link`, `Video Title`, `Cleaned_Comments`, and `Item`. `df.info()` reveals if `Cleaned_Comments` is a string type and if there are null values.

#### 1.2 Data Cleaning
- **What I Did**: Used the `re` (regular expressions) library to remove user mentions (e.g., "@mjfansumit") from the `Cleaned_Comments` column. Applied a function like `re.sub(r'@\w+', '', comment)` to each comment using `df['Cleaned_Comments'].apply()`.
- **Function Details**:
  - `re.sub(pattern, replacement, string)`: Replaces matches of the pattern (e.g., `@` followed by word characters) with an empty string.
  - Example: "@mjfansumit - I proud" becomes "I proud".
- **Purpose**: User mentions add noise and don’t contribute to sentiment or topic analysis. Cleaning ensures the text is focused on meaningful content.
- **Why Important**: Clean data improves the accuracy of downstream tasks like sentiment analysis and topic modeling.

#### 1.3 Basic Data Statistics
- **What I Did**: Used `df.describe()` to compute statistics (e.g., count of comments, unique video titles) and `df.info()` to check for null values or data types.
- **Function Details**:
  - `df.describe()`: Returns count, mean, min, max, etc., for numeric columns and count, unique, top, freq for categorical columns.
  - `df.info()`: Shows column names, non-null counts, and data types (e.g., `object` for strings).
- **Purpose**: To identify issues like missing data or outliers and confirm the dataset is ready for analysis.
- **Example**: If `df.info()` shows 100 rows but only 90 non-null comments, I’d need to handle missing values.

---

### 2. Intermediate Data Analysis

#### 2.1 Sentiment Analysis
- **What I Did**: Used the **TextBlob** library to analyze the sentiment of each comment. Applied `TextBlob(comment).sentiment.polarity` to get a score between -1 (negative) and 1 (positive), then categorized scores as:
  - Positive: > 0
  - Neutral: = 0
  - Negative: < 0
- **Function Details**:
  - `TextBlob(text).sentiment.polarity`: Computes a sentiment score based on word sentiment dictionaries.
  - Example: "I proud of you modi sir" might get a polarity of 0.8 (positive).
- **Purpose**: To understand the overall mood of the comments (e.g., are people supportive or critical of the G20 Summit?).
- **Why Important**: Sentiment analysis reveals public opinion trends, useful for event organizers or policymakers.

#### 2.2 Clustering Documents
- **What I Did**: Used **SentenceTransformer** to convert comments into numerical embeddings (vectors) and **UMAP** to reduce these vectors to 2D for clustering. Applied **KMeans** from `sklearn.cluster` to group similar comments, then visualized clusters using a scatter plot with **Matplotlib** and **Seaborn**.
- **Function Details**:
  - `SentenceTransformer('all-MiniLM-L6-v2').encode(comments)`: Converts text to 384-dimensional vectors capturing semantic meaning.
  - `umap.UMAP(n_components=2).fit_transform(embeddings)`: Reduces vectors to 2D for visualization.
  - `KMeans(n_clusters=5).fit(reduced_embeddings)`: Groups comments into 5 clusters based on similarity.
  - `plt.scatter()`: Plots clusters with different colors.
- **Purpose**: To group comments with similar themes (e.g., comments about "leadership" in one cluster).
- **Why Important**: Clustering helps identify patterns without manually reading thousands of comments.

#### 2.3 Network Analysis
- **What I Did**: Used **NetworkX** to create a graph where nodes are videos or comments, and edges represent relationships (e.g., shared keywords). Visualized the graph using `nx.draw()`.
- **Function Details**:
  - `nx.Graph()`: Creates an empty graph.
  - `G.add_node(video_title)`: Adds a video as a node.
  - `G.add_edge(node1, node2)`: Connects nodes if they share keywords (e.g., extracted with TF-IDF).
  - `nx.draw(G)`: Plots the graph with nodes and edges.
- **Purpose**: To show how videos or comments are related (e.g., which videos have similar discussions).
- **Why Important**: Network analysis reveals connections that aren’t obvious from raw data.

#### 2.4 Topic Analysis
- **What I Did**: Used **Latent Dirichlet Allocation (LDA)** from `gensim` to identify topics in comments. Preprocessed comments with **TfidfVectorizer** to create a document-term matrix, then trained an LDA model. Visualized topic distributions using a bar chart with **Matplotlib**.
- **Function Details**:
  - `TfidfVectorizer().fit_transform(comments)`: Converts comments to a matrix of word importance scores.
  - `gensim.models.LdaModel(corpus, num_topics=5)`: Identifies 5 topics, assigning words to each topic.
  - `lda_model.show_topics()`: Returns top words for each topic (e.g., "Modi, leader, summit" for Topic 1).
  - `plt.bar()`: Plots the frequency of each topic.
- **Purpose**: To summarize the main themes in comments (e.g., leadership, economy, diplomacy).
- **Why Important**: Topic modeling condenses large amounts of text into key ideas.

---

### 3. Advanced Data Analysis

#### 3.1 Content Summarization
- **What I Did**: Used **Gensim**’s `summarization.keywords` to extract important keywords from comments.
- **Function Details**:
  - `gensim.summarization.keywords(text, ratio=0.2)`: Returns the top 20% of words ranked by importance.
  - Example: For comments about the G20, keywords might be "Modi," "summit," "India."
- **Purpose**: To quickly summarize the main ideas in comments without reading everything.
- **Why Important**: Keywords provide a snapshot of the content, useful for reports or quick insights.

#### 3.2 Content Categorization
- **What I Did**: Used **TfidfVectorizer** to convert comments to numerical features and trained a **Multinomial Naive Bayes** classifier (`sklearn.naive_bayes.MultinomialNB`) to categorize comments (e.g., "praise," "criticism"). Split data with `train_test_split` and evaluated accuracy with `accuracy_score` and `classification_report`.
- **Function Details**:
  - `TfidfVectorizer().fit_transform(comments)`: Creates a matrix of word importance scores.
  - `MultinomialNB().fit(X_train, y_train)`: Trains a classifier to predict categories.
  - `accuracy_score(y_test, y_pred)`: Measures how many predictions were correct.
  - `classification_report(y_test, y_pred)`: Shows precision, recall, and F1-score per category.
- **Purpose**: To automatically group comments into meaningful categories.
- **Why Important**: Categorization organizes comments for easier analysis (e.g., how much praise vs. criticism?).

#### 3.3 Discovering Trends
- **What I Did**: Used **t-SNE** from `sklearn.manifold` to reduce comment embeddings (from SentenceTransformer) to 2D and visualize patterns with a scatter plot.
- **Function Details**:
  - `TSNE(n_components=2).fit_transform(embeddings)`: Reduces high-dimensional embeddings to 2D.
  - `plt.scatter()`: Plots points, with colors indicating clusters or categories.
- **Purpose**: To find hidden patterns or clusters in the data (e.g., are positive comments grouped together?).
- **Why Important**: t-SNE reveals relationships that other methods might miss.

#### 3.4 Enhancing Search and Retrieval
- **What I Did**: Associated comments with topics (from LDA) to improve search functionality. For example, searching "leadership" retrieves comments linked to the leadership topic.
- **Function Details**:
  - Used LDA topic assignments to tag comments with dominant topics.
  - Created a mapping (e.g., `df['topic'] = lda_model.get_document_topics(corpus)`).
- **Purpose**: To make it easier to find relevant comments based on topics.
- **Why Important**: Enhances usability for users who want specific information.

---

## Libraries and Tools Used

- **Pandas**: Loads and manipulates the CSV data (`pd.read_csv`, `df.head`, `df.apply`).
- **NumPy**: Handles numerical operations (e.g., array manipulation).
- **Matplotlib & Seaborn**: Creates visualizations like scatter plots and bar charts.
- **NetworkX**: Builds and visualizes network graphs.
- **TextBlob**: Performs sentiment analysis.
- **SentenceTransformer**: Converts text to semantic embeddings.
- **UMAP**: Reduces dimensions for clustering.
- **Gensim**: Extracts keywords and performs topic modeling (LDA).
- **Scikit-learn**: Provides TF-IDF, Naive Bayes, KMeans, t-SNE, and evaluation metrics.
- **Transformers**: Supports advanced NLP tasks (e.g., embeddings).
- **Scattertext**: Visualizes text differences (e.g., positive vs. negative comments).
- **re**: Cleans text using regular expressions.

---

## How to Run the Project

1. **Install Python**: Ensure Python 3.8+ is installed.
2. **Install Dependencies**: Run:
   ```
   pip install pandas numpy matplotlib seaborn networkx textblob sentence-transformers umap-learn gensim scikit-learn transformers scattertext
   ```
3. **Prepare Data**: Place `Youtube_scrapping_comments.csv` in the same folder as the notebook.
4. **Run the Notebook**: Open `Youtube_Data_analysis&Visualisation.ipynb` in Jupyter Notebook and execute cells sequentially.
5. **View Outputs**: Check tables, charts, and graphs generated in the notebook.

---

## Key Outputs

- **Tables**: DataFrame previews (`df.head()`), summary stats (`df.describe()`).
- **Charts**:
  - Scatter plots of comment clusters (UMAP, t-SNE).
  - Bar charts of topic distributions (LDA).
  - Network graphs of video/comment relationships.
- **Metrics**: Sentiment counts (e.g., 60% positive), classifier accuracy, topic keywords.
- **Insights**: Public sentiment, dominant topics, and comment categories.

---

## Challenges and Solutions

- **Challenge**: Comments were messy with user mentions and non-English text.
  - **Solution**: Used `re.sub` to remove mentions and focused on English comments for simplicity.
- **Challenge**: Choosing the right number of clusters/topics.
  - **Solution**: Tested different values (e.g., 3, 5, 10 clusters) and used visualizations to pick the most meaningful.
- **Challenge**: High-dimensional data was hard to visualize.
  - **Solution**: Used UMAP and t-SNE to reduce dimensions to 2D for scatter plots.

---

## What I Learned

- **Data Cleaning**: How to handle noisy text data with regular expressions.
- **Text Analysis**: Techniques like sentiment analysis, clustering, and topic modeling.
- **Visualization**: Creating clear charts and graphs to communicate insights.
- **Python Libraries**: Deepened my understanding of Pandas, Scikit-learn, Gensim, and more.
- **Problem-Solving**: Tackling real-world data challenges like missing values or multilingual text.

---

## Future Improvements

- **More Data**: Include comments from other platforms (e.g., X) or events for comparison.
- **Multilingual Support**: Use libraries like `langdetect` to handle non-English comments.
- **Interactive Dashboard**: Build a web app with Dash or Streamlit to display results.
- **Advanced Models**: Try BERT for sentiment analysis or zero-shot classification for categorization.

---

## Interview Preparation Tips

- **Explain the Big Picture**: Start with, “This project analyzes YouTube comments about the G20 Summit to understand public opinion using Python.”
- **Break Down Steps**: For each step (e.g., sentiment analysis), mention:
  - What it does (e.g., labels comments as positive/negative).
  - The function used (e.g., `TextBlob.sentiment.polarity`).
  - Why it’s useful (e.g., shows public mood).
- **Handle Deep Questions**:
  - **Q: How does `re.sub` work?** A: It searches for a pattern (e.g., `@\w+` for user mentions) and replaces it with an empty string.
  - **Q: Why use UMAP over PCA?** A: UMAP preserves local structure better, making clusters more meaningful for text data.
  - **Q: How does LDA find topics?** A: It assumes comments are mixtures of topics and assigns words to topics based on co-occurrence patterns.
- **Show Visuals**: If possible, share screenshots of your scatter plots, network graphs, or topic charts.
- **Connect to Skills**: Highlight Python, data analysis, NLP, and visualization skills relevant to the job.

---

This project was a great way to explore data science and NLP while working with real-world data. It shows how to turn messy text into actionable insights using Python!
