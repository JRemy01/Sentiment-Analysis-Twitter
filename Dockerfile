FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

RUN python -m nltk.downloader stopwords wordnet punkt_tab omw-1.4


CMD ["python", "main.py"]