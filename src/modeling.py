from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline



def Pipeline_creation(model = 'rf', vector = 'tfid', n_estimator = 42, max_features=3000):
    if model == 'rf':
        mdl = RandomForestClassifier(n_estimators = n_estimator)

    if vector == 'tfid':
        vec = TfidfVectorizer(max_features = max_features)

    pipeline = Pipeline([ ('vector', vec),
        ('classifier',mdl)])
    return pipeline


def train_model(X_train,X_test,y_train,y_test,pipline,return_scores= False):
    pipline.fit(X_train,y_train)
    y_pred = pipline.predict(X_test)

    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    f1_scor = f1_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)

    if return_scores:
        return accuracy, precision, recall, f1_scor

    print("Accuracy : ", accuracy)
    print("Precision : ", precision)
    print("f1_score : ", f1_scor)
    print("recall_score : ", recall)
    print("Report : ", classification_report(y_test,y_pred))
    