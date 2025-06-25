from sklearn.model_selection import train_test_split
import pandas as pd
import pytest
import sys
import os

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if base_dir not in sys.path:
    sys.path.append(base_dir)

from src.modeling import Pipeline_creation, train_model

@pytest.fixture
def first_fixture():
    data = pd.read_csv("data/train.csv")
    return data


# test : Vérifier que le pipeline s'entraîne sans erreur sur un jeu d’exemple
def test_pipeline(first_fixture):
    df = first_fixture
    X_train, _, y_train, _ = train_test_split(df["text"], df.target, test_size=0.2, random_state=0)
    pipeline = Pipeline_creation(model = 'rf', vector = 'tfid', n_estimator = 42, max_features=3000)
    try:
        pipeline.fit(X_train, y_train)
    except Exception as e:
        pytest.fail(f"Le pipeline a échoué pendant l'entraînement : {e}")


# test : Vérifier la forme des prédictions
def test_prédictions():
    data = ["This is a space article", "This is a sports article"]
    labels = [0, 1]
    pipeline = Pipeline_creation()
    pipeline.fit(data, labels)
    preds = pipeline.predict(data)
    assert len(preds) == len(data)


# test : Vérifier que les métriques sont retournées correctement
def test_métriques():
    df = ["space news", "sports news"]
    labels = [0, 1]
    pipeline = Pipeline_creation()
    acc, prec, rec, f1 = train_model(df, df, labels, labels, pipeline, return_scores=True)
    assert all(isinstance(m, float) for m in [acc, prec, rec, f1])

