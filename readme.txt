# Human-AI Text Detector

Данный проект представляет собой инструмент, позволяющий анализировать научные статьи с целью выявления текста, сгенерированного искусственным интеллектом.

Для реализации проекта использовались инструменты для машинного обучения и анализа текста, предоставляемые модулями scikit-learn, nltk, spacy, textstat.
С помощью ансамбля деревьев текста разделяются на 2 класса (написанные человеком, сгенерированные AI) на основе 28 признаков, определяемых в process.py (в том числе с помощью заранее обученных моделей TfidfVectorizer, OneClassSVM).

Датасет для обучения моделей состоит из 2-х частей:
1. Сгенерированные искусственным интеллектом текста: небольшие параграфы, сгенерированные с помощью saiga_mistral_7b_lora, deepseek, Le Chat Mistral. Некоторые образцы представляют собой перефразированный с помощью ИИ человеческий текст.
2. Взятые с eLibrary научные статьи, разделенные для обучения на параграфы по 500-550 слов. Датасет с необработанными статьями собран DOMOVENOK KUZYA и доступен по ссылке: https://www.kaggle.com/datasets/ergkerg/russian-scientific-articles

Собранный для проекта датасет доступен на Kaggle: https://www.kaggle.com/datasets/yaroslavsokoloff/humanai-paragraphs

## Использование

Для использования необходимо сначала собрать образ Docker (~3 минуты) и запустить контейнер.
Это можно сделать, например, следующим образом*:

```bash
docker build -t human-ai https://github.com/xanyyqqe/human-ai.git && docker run --rm -p 8081:8081 human-ai
# Не стоит менять порт - так как приложение написано на Flet с настроенным портом 8081
```
Или, предварительно скачав репозиторий:
```
docker build -t human-ai:v1 .
docker run -p 8081:8081 human-ai:v1
```
Далее необходимо открыть браузер и перейти по http://localhost:8081. В открывшемся окне можно ввести для анализа текст любого объема(в разумных пределах). Текст для анализа должен быть научного стиля и обязательно на русском языке.

---------------------------------------------------

This project is a tool designed to analyze scientific articles to identify AI-generated text. It utilizes machine learning and text analysis tools provided by the scikit-learn, nltk, spacy, and textstat modules. Through an ensemble of decision trees, texts are classified into two categories (human-written or AI-generated) based on 28 features determined in `process.py` (including the use of pre-trained TfidfVectorizer and OneClassSVM models).

The dataset for model training consists of two parts:
1. **AI-generated texts**: Short paragraphs generated using saiga_mistral_7b_lora, deepseek, and Le Chat Mistral. Some samples represent human text paraphrased by AI.
2. **Scientific articles** from eLibrary, split into paragraphs of 500–550 words for training. The raw article dataset was collected by DOMOVENOK KUZYA and is available at: https://www.kaggle.com/datasets/ergkerg/russian-scientific-articles

The assembled dataset for this project is available on Kaggle: https://www.kaggle.com/datasets/yaroslavsokoloff/humanai-paragraphs

## Usage

To use the tool, first build a Docker image (~3 minutes) and run the container. This can be done, for example, as follows*:

```
docker build -t human-ai https://github.com/xanyyqqe/human-ai.git && docker run --rm -p 8081:8081 human-ai
# It is not recommended to change the port, as the application is built with Flet configured to use port 8081
```
Alternatively, after downloading the repository:
```
docker build -t human-ai:v1 .
docker run -p 8081:8081 human-ai:v1
```
Next, open your browser and navigate to http://localhost:8081. In the opened window, you can input text of any reasonable length for analysis. The text for analysis should be in a scientific style and must be in Russian.

## Contributing

PRs accepted.

## License

MIT © Iaroslav Sokolov
