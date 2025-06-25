from src.modeling import Pipeline_creation, train_model
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv("data/train.csv")
X = data["text"]
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


pipe = Pipeline_creation(model = 'rf', vector = 'tfid', n_estimator = 42, max_features=3000)
train_model(X_train, X_test, y_train, y_test, pipe)