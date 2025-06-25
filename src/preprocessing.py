import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


class Preprocessing:
    def __init__(self):
        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        

    def clean_text(self,text):
        text = re.sub(r"http\S+|www\.\S+", "", text.lower())  # suppression des liens
        text = re.sub(r"@\w+","",text.lower())    # suppression des mentions
        text = re.sub(r"#\w+","",text.lower())    # suppression des hashtags
        text = re.sub(r"[0-9]@+","",text.lower())   # suppression des chiffres
        text = re.sub(r"[',\-_!;?.:'0-9]", " ", text.lower())   # suppression des symbole, ponctuations
        return text
    
    def tokenize_word(self,text): 
        return nltk.word_tokenize(text)
    
    def lemmatize_word(self,text):
        return [self.lemmatizer.lemmatize(t,pos='v') for t in text if not t in self.stopwords]
    
    def drop_donne_manquant(self,df):
        return df.dropna()
