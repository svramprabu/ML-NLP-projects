import pandas as pd
import numpy as np
import streamlit as st
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
from sklearn import datasets
from sklearn import svm
import spacy
import nltk

st.set_option('deprecation.showPyplotGlobalUse', False)

def clean(doc):
    text_no_namedentities = []
    document = nlp(doc)
    ents = [e.text for e in document.ents]
    for item in document:
        if item.text in ents:
            pass
        else:
            text_no_namedentities.append(item.text)
    doc = (" ".join(text_no_namedentities))

    doc = doc.lower().strip()
    doc = doc.replace("</br>", " ")
    doc = doc.replace("-", " ")
    doc = "".join([char for char in doc if char not in string.punctuation and not char.isdigit()])
    doc = " ".join([token for token in doc.split() if token not in stopwords])
    doc = "".join([lemmatizer.lemmatize(word) for word in doc])
    return doc

def run_model(model,X_train,y_train,X_test,y_test,model_type = 1):
  model.fit(X_train, y_train)

  y_pred_train = model.predict(X_train)
  y_pred_test = model.predict(X_test)
  st.write("\nTraining Accuracy score:",accuracy_score(y_train, y_pred_train))
  st.write("Testing Accuracy score:",accuracy_score(y_test, y_pred_test))

  st.write(classification_report(y_test, y_pred_test, target_names=['not relevant', 'relevant']))

  cm = confusion_matrix(y_test, y_pred_test)
    # print('Confusion matrix\n', cm)

  cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive', 'Actual Negative'],
                        index=['Predict Positive', 'Predict Negative'])
  sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
  fig = plt.show()
  st.pyplot(fig)


  if model_type == 1:

    probs = model.predict_proba(X_test)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    fig = plt.show()
    st.pyplot(fig)

st.write("Welcome to Twitter tweets sentimental analysis")

if(st.checkbox("check this to clean the tweets adn save it to a new file")):
    data = pd.read_csv("FinalBalancedDataset.csv")
    data.drop('Unnamed: 0',axis=1,inplace=True)

    nlp = spacy.load('en_core_web_sm')

    stopwords = ENGLISH_STOP_WORDS
    lemmatizer = WordNetLemmatizer()
    nltk.download('wordnet')
    data['text'] = data['tweet'].apply(clean)
    # data['text'] = data['tweet'].iloc[1:5].apply(clean)
    data.to_csv('tweets_after_nlp.csv', index=False)
    st.write("yes")

data = pd.read_csv("tweets_after_nlp.csv")
data.drop('tweet',axis=1,inplace=True)
data.dropna(inplace=True)
if(st.sidebar.checkbox("view data")):
    st.write(data)
# st.write(data['text'].iloc[1:5])

docs = list(data['text'])
tfidf_vectorizer = TfidfVectorizer(use_idf=True, max_features = 1000)
tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(docs)
docs = tfidf_vectorizer_vectors.toarray()

X = docs
y = data['Toxicity']

SEED=123
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
# st.write(X_train.shape, y_train.shape)
# st.write(X_test.shape, y_test.shape)

model = st.radio("Gaussian NB",("Gaussian NB","Decision Tree","Random Forest","KNN","Linear SVM"),horizontal=True)

if(model == "Gaussian NB"):
    model_gnb = GaussianNB()
    run_model(model_gnb,X_train,y_train,X_test,y_test)

elif(model == "Decision Tree"):
    model_dt = DecisionTreeClassifier(random_state=SEED)
    run_model(model_dt,X_train,y_train,X_test,y_test,2)

elif(model == "Random Forest"):
    model_rf = RandomForestClassifier(n_estimators=50)
    run_model(model_rf,X_train,y_train,X_test,y_test,1)

elif(model == "KNN"):
    model_knn = KNeighborsClassifier(n_neighbors=3)
    run_model(model_knn,X_train,y_train,X_test,y_test)

elif(model == "Linear SVM"):
    model_svc = LinearSVC(class_weight='balanced')
    run_model(model_svc,X_train,y_train,X_test,y_test,2)

