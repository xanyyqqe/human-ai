# -*- coding: utf-8 -*-
import string
import nltk
from textstat import flesch_kincaid_grade, flesch_reading_ease
from process_texts import ExtractComponents
import spacy
from nltk.corpus import stopwords
import flet as ft
import joblib
import os
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 

vectorizer = joblib.load('models/vectorizer.joblib')
model_one_class = joblib.load('models/oneclass_svm.joblib')
rfc = joblib.load('models/rfc.joblib')

nlp_en = spacy.load("en_core_web_sm")
nlp_ru = spacy.load("ru_core_news_sm")
nltk.download('stopwords')
stop_words = set(stopwords.words('russian')).union(set(stopwords.words('english'))) 

def process_txt(txt):
    print('started prediction . . .')
    txt_list = []
    chunk_size = 500
    for i in range(0, len(txt), chunk_size):
        txt_list.append(txt[i:i + chunk_size])

    
    predictions = []
    for t in txt_list:
        extractor = ExtractComponents(t, vectorizer, model_one_class,
                                      nlp_ru, nlp_en, stop_words)
        features = extractor.zz_get_all_features()
        predict = rfc.predict_proba([features])
        predictions.append(predict[0][0])

    predictions = predictions[:-1] if (len(predictions) > 1 and len(predictions[:-1]) < 100) else predictions
    result = sum(predictions)/len(predictions)
    if result > 0.5:
        return f'ai text with probability {result:.2f}'
    else:
        return f'human text with probability {1-result:.2f}'

def main(page: ft.Page):
    page.title = "AI/Human Text Classifier"
    page.scroll = ft.ScrollMode.AUTO
    
    text_input = ft.TextField(
        label="Введите текст для анализа",
        multiline=True,
        min_lines=10,
        max_lines=20,
        width=600
    )
    
    result_text = ft.Text("", size=20, weight=ft.FontWeight.BOLD)
    
    def analyze_text(e):
        if not text_input.value.strip():
            result_text.value = "Пожалуйста, введите текст"
        else:
            result_text.value = "Анализируем..."
            page.update()
            
            result = process_txt(text_input.value)
            result_text.value = f"Результат: {result}"
        
        page.update()
    
    page.add(
        ft.Column([
            ft.Text("Классификатор текста: AI vs Human", size=28, weight=ft.FontWeight.BOLD),
            ft.Divider(),
            text_input,
            ft.ElevatedButton(
                "Анализировать текст",
                on_click=analyze_text,
                width=200
            ),
            ft.Divider(),
            result_text
        ], spacing=20, horizontal_alignment=ft.CrossAxisAlignment.CENTER)
    )

# Запуск приложения
if __name__ == "__main__":
    ft.app(target=main, view=ft.WEB_BROWSER, port = 8081)