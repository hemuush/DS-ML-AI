import streamlit as st 
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA 
import numpy as np
import matplotlib.pyplot as plt

st.title("Classifier app")

st.write('''
# Explore different classifier
which one is the best??
''')

dataset_name = st.selectbox("set dataset", ("Iris", "Breast Cancer", "Wine"))
#dataset_name = st.sidebar.selectbox("set dataset", ("Iris", "Breast Cancer", "wine"))
#st.write(dataset_name)

classifier_name = st.selectbox("Select Classifier", ("KNN","SVM","Random Forest"))

def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    x = data.data
    y = data.target
    return x,y

def add_parameter(clf_name):
    paramas = dict()
    if clf_name == "KNN":
        K = st.slider("K",1,15)
        paramas["K"] = K
    elif clf_name == "SVM":
        C = st.slider("C", 0.01, 10.0)
        paramas["C"] = C
    else:
        max_depth = st.slider("max_depth", 2, 15)
        no_of_estimators = st.slider("n_estimators", 1,100)
        paramas["max_depth"] = max_depth
        paramas["no_of_estimators"] = no_of_estimators
    return paramas

paramas = add_parameter(classifier_name)

def get_classifier(classifier_name, paramas):
    if classifier_name == "KNN":
        classifier = KNeighborsClassifier(n_neighbors= paramas["K"])
    elif classifier_name == "SVM":
        classifier = SVC(C= paramas["C"])
    else:
        classifier = RandomForestClassifier(n_estimators= paramas["no_of_estimators"], max_depth= paramas["max_depth"], random_state= 1234)
    return classifier    

x,y = get_dataset(dataset_name)

clf = get_classifier(classifier_name, paramas)

#st.write("shape of dataset :" , x.shape)
#st.write("no. of classes :", len(np.unique(y)))

#clasification
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size= 0.20, random_state= 1243)

clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(Y_test, y_pred)

st.write(f"classifier : {classifier_name}")
st.write(f"accuracy : {acc}")

#PLOT
pca = PCA(2)
x_projected = pca.fit_transform(x)

x1 = x_projected[:, 0]
x2 = x_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c= y, alpha= 0.8, cmap= "viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()
#plt.show()

st.pyplot(fig)