import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import joblib
import pickle

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

def init():
    df = pd.read_csv(r'dataset/sql_30k.csv')  # Read the dataset.
    df.head(10)

    df.drop(["Unnamed: 2", "Unnamed: 3"], axis=1, inplace=True)

    df.dropna(inplace=True)

    df['Label'] = df['Label'].apply(pd.to_numeric, errors='ignore')

    X = df['Sentence']
    y = df['Label']

    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform(X.values.astype('U')).toarray()
    # X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(X_tfidf, y, test_size=0.2)
    # print("X_train_tfidf.shape=", X_train_tfidf.shape)
    # print("y_train_tfidf.shape=", y_train_tfidf.shape)
    # print("X_test_tfidf.shape=", X_test_tfidf.shape)
    # print("y_test_tfidf.shape=", y_test_tfidf.shape)

    # # create XGBoost model instance
    # xgb_clf_tfidf = XGBClassifier(n_estimators=50, max_depth=5, learning_rate=0.01, alpha=10,
    #                               objective='binary:logistic')
    # # fit XGBoost model
    # xgb_clf_tfidf.fit(X_train_tfidf, y_train_tfidf)
    # # make predictions
    # y_pred_tfidf = xgb_clf_tfidf.predict(X_test_tfidf)
    # print(f"Accuracy of XGBClassifier on test set : {accuracy_score(y_pred_tfidf, y_test_tfidf)}")
    # print(f"F1 Score of XGBClassifier on test set : {f1_score(y_pred_tfidf, y_test_tfidf)}")
    #
    # # Save the model as a pickle in a file
    # joblib.dump(xgb_clf_tfidf, r'C:\Users\snehal\PycharmProjects\SQLInjectionMLProject\models_bkp\xgb_clf_tfidf.pkl')

    # # Load the model from the file
    # xtree_clf_from_joblib = joblib.load('/models_bkp/xtree_clf.pkl')

    return tfidf_vectorizer

# def modelTraining():


def getPrediction(text):
    tfidf_vectorizer = init()
    v0 = tfidf_vectorizer.transform([text]).toarray()
    # print(v0)
    Pkl_Filename = r'models_bkp/xgb_clf_tfidf.pkl'

    # # Load the model from the file
    # with open(Pkl_Filename, 'rb') as file:
    #     xtree_clf_tfidf = pickle.load(file)

    xgb_clf_tfidf = joblib.load(open(Pkl_Filename, 'rb'))
    pred = xgb_clf_tfidf.predict(v0)
    print("xgb_clf_tfidf =", xgb_clf_tfidf)
    print("Predicted Output = ", pred)
    return pred
