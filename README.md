# ML-NLP-projects
# Tweets Sentiment Analysis

This project focuses on sentiment analysis of tweets using various machine learning models. The goal is to classify tweets as toxic or non-toxic based on their content. The project involves several steps, including data preprocessing, vectorization, train-test splitting, and the implementation of machine learning models.

## Dataset

The dataset consists of a CSV file with two columns:
- **Toxicity Indicator**: A binary label indicating whether the tweet is toxic (1) or non-toxic (0).
- **Tweet**: The text content of the tweet.

## Data Preprocessing

The following data preprocessing steps are applied to each tweet:
1. **Named Entity Recognition (NER)**: Identifying and tagging named entities in the tweets.
2. **Punctuation Removal**: Removing all punctuation marks from the tweet text.
3. **Digit Removal**: Eliminating digits from the text as they don't affect sentiment analysis.
4. **Lemmatization**: Reducing words to their base or dictionary form for consistency.

The preprocessed tweet text is then saved back to the original DataFrame.

## Vectorization

To prepare the text data for machine learning models, we perform vectorization using techniques such as TF-IDF (Term Frequency-Inverse Document Frequency) or Count Vectorization. This converts the text data into numerical form, making it suitable for model training.

## Train-Test Split

The dataset is split into training and testing sets to evaluate the model's performance. This helps ensure that the model generalizes well to unseen data.

## Machine Learning Models

Several machine learning models are implemented for tweet sentiment analysis. These models include:
- Gaussian Naive Bayes
- Decision Tree
- K-Nearest Neighbors (KNN)
- Random Forest
- Linear Support Vector Machine (SVM)

## Repository Link

Explore the project code and details in the [GitHub repository](https://github.com/svramprabu/ML-NLP-projects/tree/main/Tweets-sentimental_analysis).

## License

This project is open-source and does not have a specific license.

Feel free to contribute to this project or use it for your own sentiment analysis tasks. Your feedback and contributions are highly appreciated.

---

*Disclaimer: This project is for educational and research purposes only. The accuracy of sentiment analysis models may vary depending on the dataset and preprocessing techniques used.*
## References

- https://github.com/vinyluis/Articles/blob/main/ROC%20Curve%20and%20ROC%20AUC/ROC%20Curve%20-%20Multiclass.ipynb
- https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
