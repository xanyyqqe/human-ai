FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /human_ai

COPY requirements.txt .
COPY models/ ./models/
COPY data_prepare.ipynb .
COPY making_predicts.py .
COPY process_texts.py .

RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
RUN python -m spacy download en_core_web_sm && \
    python -m spacy download ru_core_news_sm
 
ENTRYPOINT ["python"]
CMD ["making_predicts.py"]