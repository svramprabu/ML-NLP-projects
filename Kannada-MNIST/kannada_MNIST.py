import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score
from scipy import stats
from sklearn.metrics import RocCurveDisplay
from itertools import cycle
from sklearn.preprocessing import LabelBinarizer
import streamlit as st

import seaborn as sns
import matplotlib.pyplot as plt


st.set_option('deprecation.showPyplotGlobalUse', False)
n_comp = st.sidebar.number_input("n_components",min_value=1, max_value=784, value=10, step=1)

if(st.sidebar.checkbox("read X_test.npz and convert into a csv")):
    npz = np.load("X_kannada_MNIST_test.npz")
    df= pd.DataFrame(columns =list(range(784)))

    for i in range(len(npz['arr_0'])):
        df.loc[len(df)] = npz['arr_0'][i].flatten()
        
    df.head()    
    df.to_csv('X_kannada_MNIST_test.csv', index=False)
if(st.sidebar.checkbox("pca X_test")):
    df = pd.read_csv("X_kannada_MNIST_test.csv")  
    pca = PCA(n_components=n_comp)
    pca.fit(df)
    x_pca = pca.transform(df)
    df_pca = pd.DataFrame(x_pca,columns=[f'PC{i+1}' for i in range(10)])
    df_pca.to_csv('PCA_X_kannada_MNIST_test.csv', index=False)



if(st.sidebar.checkbox("read X_train.npz and convert into a csv")):
    npz = np.load("X_kannada_MNIST_train.npz")
    df= pd.DataFrame(columns =list(range(784)))

    for i in range(len(npz['arr_0'])-40000):
        df.loc[len(df)] = npz['arr_0'][i].flatten()
        
    for i in range(len(npz['arr_0'])-40000,len(npz['arr_0'])-20000):
        df.loc[len(df)] = npz['arr_0'][i].flatten()
        
    for i in range(len(npz['arr_0'])-20000,len(npz['arr_0'])):
        df.loc[len(df)] = npz['arr_0'][i].flatten()

    df.to_csv('X_kannada_MNIST_train.csv', index=False)

if(st.sidebar.checkbox("pca X_train")):
    df = pd.read_csv("X_kannada_MNIST_train.csv")  

    pca = PCA(n_components=10)
    pca.fit(df)
    x_pca = pca.transform(df)
    df_pca = pd.DataFrame(x_pca,columns=[f'PC{i+1}' for i in range(10)])
    df_pca.to_csv('PCA_X_kannada_MNIST_train.csv', index=False)

if(st.sidebar.checkbox("read y_test.npz and save it to csv")):
    npz = np.load("y_kannada_MNIST_test.npz")
    # print(type(npz['arr_0']))
    ydf = pd.DataFrame(npz['arr_0'])
    # for i in range(len(npz['arr_0'])):
    # np.savetxt("y_kannada_MNIST_test.csv", npz['arr_0'],delimiter = ",")
    ydf.to_csv('y_kannada_MNIST_test.csv', index=False)
    # ydf

if(st.sidebar.checkbox("read y_train.npz and save it to csv")):
    npz = np.load("y_kannada_MNIST_train.npz")
    ydf = pd.DataFrame(npz['arr_0'])

    # for i in range(len(npz['arr_0'])):
    # np.savetxt("y_kannada_MNIST_train.csv", npz['arr_0'],delimiter = ",")
    ydf.to_csv('y_kannada_MNIST_train.csv', index=False)




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

    st.write(classification_report(y_test, y_pred_test))#, target_names=['not relevant', 'relevant']))

    cm = confusion_matrix(y_test, y_pred_test)
        # print('Confusion matrix\n', cm)

    # cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive', 'Actual Negative'],
    #                         index=['Predict Positive', 'Predict Negative'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
    fig = plt.show()
    st.pyplot(fig)

    classes = model.classes_
    # classes

    if model_type == 1:

        label_binarizer = LabelBinarizer().fit(y_train)
        y_onehot_test = label_binarizer.transform(y_test)
        y_onehot_test.shape  # (n_samples, n_classes)
        
        y_proba = model.predict_proba(X_test)
        
        fig, ax = plt.subplots(figsize=(6, 6))



        colors = cycle(["aqua", "darkorange", "cornflowerblue","crimson","purple","lawngreen","navy","teal","darkslategrey","yellow"])
        for class_id, color in zip(range(len(classes)), colors):
            RocCurveDisplay.from_predictions(
                y_onehot_test[:, class_id],
                y_proba[:, class_id],
                name=f"ROC curve for {class_id}",
                color=color,
                ax=ax,
                # plot_chance_level=(class_id == 2),
            )

        plt.axis("square")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass")
        plt.legend()
        fig = plt.show()
        st.pyplot(fig)

# rand_state = 123




if __name__ == "__main__":
    st.write("Welcome to Kannada MNIST classification problem")

    X_train = pd.read_csv("PCA_X_kannada_MNIST_train.csv")
    X_test = pd.read_csv("PCA_X_kannada_MNIST_test.csv")
    y_train = pd.read_csv("y_kannada_MNIST_train.csv")
    y_test = pd.read_csv("y_kannada_MNIST_test.csv")


    SEED=123
    # X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    # st.write(X_train.shape, y_train.shape)
    # st.write(X_test.shape, y_test.shape)

    model = st.radio("Gaussian NB",("Gaussian NB","Decision Tree","Random Forest","KNN","Linear SVM"),horizontal=True)

    if(model == "Gaussian NB"):
        model_gnb = GaussianNB()
        run_model(model_gnb,X_train,y_train.values.ravel(),X_test,y_test.values.ravel())

    elif(model == "Decision Tree"):
        model_dt = DecisionTreeClassifier(random_state=SEED)
        run_model(model_dt,X_train,y_train,X_test,y_test,2)

    elif(model == "Random Forest"):
        model_rf = RandomForestClassifier(n_estimators=50)
        run_model(model_rf,X_train,y_train.values.ravel(),X_test,y_test.values.ravel())

    elif(model == "KNN"):
        neighbors = st.number_input("n_neighbors",min_value=3,max_value=10,value=3,step=1)
        model_knn = KNeighborsClassifier(n_neighbors=neighbors)
        run_model(model_knn,X_train,y_train.values.ravel(),X_test,y_test.values.ravel())

    elif(model == "Linear SVM"):
        model_svc = LinearSVC(class_weight='balanced')
        run_model(model_svc,X_train,y_train.values.ravel(),X_test,y_test.values.ravel(),2)

