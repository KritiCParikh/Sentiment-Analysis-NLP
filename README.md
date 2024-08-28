# Sentiment-Analysis-NLP

## Overview
This project implements a robust sentiment analysis model using Natural Language Processing (NLP) and machine learning techniques. The model classifies text into positive or negative sentiments with high accuracy, providing valuable insights for various business applications.

## Skills & Technologies

- Python
- Natural Language Processing (NLP)
- Machine Learning
- Data Preprocessing
- Feature Engineering
- Model Training and Evaluation
- Libraries: pandas, matplotlib, seaborn, NLTK, scikit-learn
- Algorithms: Random Forest Classifier
- Techniques: Bag of Words, Word Cloud, GridSearchCV for hyperparameter tuning

### Business Impact

1. Customer Insight: Enables businesses to automatically analyze customer feedback, reviews, and social media mentions to gauge public sentiment about products or services.
2. Brand Monitoring: Helps in tracking public perception of the brand over time and identifying potential reputation risks or negative trends early.
3. Product Development: Provides valuable feedback for product teams to understand user needs and pain points.
4. Customer Service Improvement: Assists in prioritizing customer service responses based on sentiment, potentially improving customer satisfaction.
5. Marketing Strategy: Informs marketing teams about campaign effectiveness and helps in tailoring messaging for better engagement.
6. Competitive Analysis: Can be used to analyze competitor reviews and mentions, providing strategic insights.
7. Risk Management: Helps identify potential reputational risks by flagging negative sentiments early.

#### Dataset
The project uses a dataset from Kaggle, which consists of sentences labeled with their respective sentiments. The dataset is split into three files: train.txt, test.txt, and val.txt.

#### Data Preprocessing

- Text Cleaning: Remove non-alphabetic characters and convert to lowercase.
- Stopword Removal: Eliminate common words that don't contribute to sentiment.
- Lemmatization: Reduce words to their base form to unify different variations.
- Label Encoding: Convert categorical sentiment labels to binary (0 for negative, 1 for positive).

#### Feature Engineering

* Bag of Words (BoW) Model: Transform text data into numerical vectors.
* N-gram Range: Utilize bigrams (pairs of consecutive words) to capture more context.

#### Model

* Algorithm: Random Forest Classifier
* Hyperparameter Tuning: Employed GridSearchCV to find the optimal combination of parameters.

#### Key Features

- Word Cloud Visualization: Generate a visual representation of the most frequent words in the dataset.
- Custom Encoder: Implement a function to convert multi-class sentiment labels into binary classes.
- Robust Preprocessing Pipeline: Develop a comprehensive text transformation function for cleaning and preparing the input data.
- Cross-Validation: Utilize 5-fold cross-validation during hyperparameter tuning to ensure model robustness.
- Multiple Evaluation Metrics: Assess model performance using accuracy, precision, recall, F1-score, and ROC curve.
- Custom Input Prediction: Implement a function to predict sentiment for user-provided text inputs.

#### Results

* Accuracy: 96.1%
* Precision: 96.16%
* Recall: 95.33%
* F1-score: 96% (weighted average)

The model demonstrates excellent performance across all metrics, indicating its effectiveness in distinguishing between positive and negative sentiments.

### Future Improvements

1. Experiment with more advanced NLP techniques like word embeddings (Word2Vec, GloVe).
2. Implement deep learning models (LSTM, BERT) for potentially higher accuracy.
3. Extend the model to handle multi-class sentiment classification.
4. Incorporate aspect-based sentiment analysis to provide more granular insights.

Thank You. Let’s keep learning and growing together!
