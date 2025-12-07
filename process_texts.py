import string
import nltk
from textstat import flesch_kincaid_grade, flesch_reading_ease


class ExtractComponents:

    def __init__(self, text:str, vectorizer, model_one_class, 
                 nlp_ru, nlp_en, stop_words):
        self.sentences = nltk.tokenize.sent_tokenize(text)
        self.text = text
        self.words = []
        for word in self.text.split():
            clean_word = word.strip(string.punctuation)
            if len(clean_word) > 0:
                self.words.append(clean_word)
        self.vectorizer = vectorizer
        self.model_one_class = model_one_class
        self.nlp_ru = nlp_ru
        self.nlp_en = nlp_en
        self.stop_words = stop_words

    def sentense_avg_quantity(self):  # 1
        return len(self.sentences)

    def sentense_avg(self):  # 2
        if not self.sentences:
            return 0
        sentence_lengths = [len(nltk.word_tokenize(sent)) for sent in self.sentences]
        return sum(sentence_lengths) / len(sentence_lengths)

    def word_avg(self):  # 3
        if not self.words:
            return 0
        word_lengths = [len(word) for word in self.words]
        return sum(word_lengths) / len(word_lengths)
    
    def count_foreign(self):  # 4
        k = 0
        for w in self.words:
            lang = 'ru' if any('\u0400' <= c <= '\u04FF' for c in w) else 'en'
            if lang == 'en': 
                k += 1
        return k

    def parts_of_speech_avg(self):  # 5-22
        ALL_POS_TAGS = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 
                       'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
        parts_from_text = {pos: 0 for pos in ALL_POS_TAGS}

        for w in self.words:
            lang = 'ru' if any('\u0400' <= c <= '\u04FF' for c in w) else 'en'
            try:
                doc = self.nlp_ru(w) if lang == 'ru' else self.nlp_en(w)
                part_of_speech = doc[0].pos_ if doc else 'X'
                parts_from_text[part_of_speech] = parts_from_text.get(part_of_speech, 0) + 1
            except:
                parts_from_text['X'] += 1
        
        return [parts_from_text.get(pos, 0) for pos in ALL_POS_TAGS]

    def words_divercity(self):  # 23
        words_without_stop = []
        for w in self.words:
            if not(w in self.stop_words): words_without_stop.append(w)
        if not self.words:
            return 0
        return len(set(words_without_stop)) / len(words_without_stop)
    
    def text_ease(self):  # 24-25
        return [flesch_reading_ease(self.text), flesch_kincaid_grade(self.text)]
    
    def text_punctuation_per_word(self):  # 26
        punctuation_count = len([i for i in self.text if i in string.punctuation])
        #считаем только "внутреннюю" пунктуацию
        sentence_endings = {'.', '!', '?', '…'}
        end_punctuation_count = sum(1 for char in self.text if char in sentence_endings)
        internal_punctuation = punctuation_count - end_punctuation_count
        
        if internal_punctuation <= 0 or not self.words:
            return 0
        
        return len(self.words) / internal_punctuation

    def avg_dependency_path(self):  # 27
        #выбираем модель по преобладающему языку
        russian_chars = sum(1 for char in self.text if '\u0400' <= char <= '\u04FF')
        total_chars = len(self.text)
        
        if total_chars > 0 and russian_chars / total_chars >= 0.5:
            doc = self.nlp_ru(self.text)
        else:
            doc = self.nlp_en(self.text)
            
        distances = []
        for token in doc:
            if token.is_punct: 
                continue 
            elif token.head == token: 
                depth = 1
            else: 
                depth = abs(token.i - token.head.i)
            distances.append(depth)
            
        return sum(distances) / len(distances) if distances else 0
    
    def unique_rate(self):#28
        text_vector = self.vectorizer.transform([self.text])
        prediction = self.model_one_class.predict(text_vector)
        return float(prediction)
    
    def zz_get_all_features(self): ##zz because i need this method be called last
        features = []
        method_names = [
        'sentense_avg_quantity', 'sentense_avg', 'word_avg', 'count_foreign',
        'parts_of_speech_avg', 'words_divercity', 'text_ease', 
        'text_punctuation_per_word', 'avg_dependency_path', 'unique_rate'
        ]
        methods = [method for method in dir(self) if method in method_names
                    and callable(getattr(self, method))
                      ]
        for method_name in methods:
            method = getattr(self, method_name)
            result = method()
            if isinstance(result, list):
                features.extend(result)
            else:
                features.append(result)
        return features     