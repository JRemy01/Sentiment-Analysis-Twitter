import sys
import os
import pytest
import pandas as pd
from unittest.mock import patch

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if base_dir not in sys.path:
    sys.path.append(base_dir)

from src.preprocessing import Preprocessing


@pytest.fixture
def first_fixture():
    text = "#AFRICANBAZE: Breaking news:Nigeria flag set ablaze in Aba. http://t.co/2nndBGwyEi"
    return text

@pytest.fixture
def second_fixture():
    data = pd.read_csv("data/train.csv")
    return data


# test Nettoie correctement texte vide, ponctuation, chiffres, mots courts
def test_clean_text(first_fixture):
    prep = Preprocessing()
    data = first_fixture
    result = prep.clean_text(data)
    assert result.strip() == "breaking news nigeria flag set ablaze in aba"


# Test des token
def test_token(first_fixture):
    prep = Preprocessing()
    data = first_fixture
    clean = prep.clean_text(data)
    result = prep.tokenize_word(clean)
    assert result == ["breaking", "news", "nigeria", "flag", "set", "ablaze", "in", "aba"]


# Vérification de l’impact du stemming/lemmatisation
def test_lammatizer():
    prep = Preprocessing()
    with patch("src.preprocessing.Preprocessing.tokenize_word", return_value = ["breaking", "news", "nigeria", "flag", "set", "ablaze", "in", "aba"]):
        result = prep.lemmatize_word(["breaking", "news", "nigeria", "flag", "set", "ablaze", "in", "aba"])
        assert result == ["break", "news", "nigeria", "flag", "set", "ablaze", "aba"]


# Vérification que tous les tokens ont plus de 2 lettres
def test_token_taille(first_fixture):
    prep = Preprocessing()
    data = first_fixture
    data_1 = prep.clean_text(data)
    token = prep.tokenize_word(data_1)
    result = prep.lemmatize_word(token)
    assert all(len(tok) > 2 for tok in result)


# Vérification que les stopwords sont supprimés
def test_stopwords():
    prep = Preprocessing()
    with patch("src.preprocessing.Preprocessing.tokenize_word", return_value = ["breaking", "news", "nigeria", "flag", "set", "ablaze", "in", "aba"]):
        result = prep.lemmatize_word(["breaking", "news", "nigeria", "flag", "set", "ablaze", "in", "aba"])
        assert result == ["break", "news", "nigeria", "flag", "set", "ablaze", "aba"]  # "in" est supprimé


# Test que le vocabulaire diminue bien après nettoyage
def baisse_vocabulaire(first_fixture):
    prep = Preprocessing()
    text = first_fixture    # len(text) = 82
    result = len(prep.clean_text(text))
    assert result == 44


# Vérification de la présence et des types des colonnes ( text , target )
def test_presence_colonne(second_fixture):
    df = second_fixture
    result = list(df.columns)
    assert result == ['id', 'keyword', 'location', 'text', 'target']


# tets Détection automatique de valeurs manquantes 
def test_detection_manquantes(second_fixture):
    data = second_fixture
    result = sum(data.isna().sum())
    assert result == 2594


# Vérification que tous les textes sont non-vides
def test_text_non_vide(second_fixture):
    prep = Preprocessing()
    data = second_fixture
    df = prep.drop_donne_manquant(data)
    result = sum(df.isna().sum())
    assert result == 0


# test Validation du nombre de classes possibles
def test_nbr_classe(second_fixture):
    data = second_fixture
    df = data["target"].nunique()
    assert df == 2

