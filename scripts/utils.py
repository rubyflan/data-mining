import pandas as pd
from sklearn import preprocessing 
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from bs4 import BeautifulSoup
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer




def clean_data(data):
    vectorizer = TfidfVectorizer(stop_words='english')


    #nlp = spacy.load("en_core_web_sm")
    w_tokenizer = WhitespaceTokenizer()
    lemmatizer = WordNetLemmatizer()
    data['review'] = (data['review']
                    #.apply(lambda text: BeautifulSoup(text, "html.parser").get_text())
                    .apply(lambda text: ' '.join([word for word in text.split() if not re.search(r'\d', word)]))
                    .apply(lambda text: re.sub(r'[^a-zA-Z\s]', '', text))
                    #.apply(lambda text: ' '.join([token.lemma_ for token in nlp(text)]))
                    .apply(lambda text: ' '.join([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)])
)
)   
    


    tfidf_matrix = vectorizer.fit_transform(data['review'])

    tfidf_array = tfidf_matrix.toarray()

    feature_names = vectorizer.get_feature_names_out()
    data['review'] = [
    ' '.join([feature_names[i] for i in tfidf_matrix[row].nonzero()[1] if tfidf_matrix[row, i] >= 0.33])
    for row in range(tfidf_matrix.shape[0])
]

    return data


def read_data(file_path, encoding='utf-8'):

    data = pd.read_csv(file_path, header=0, names=['review', 'label'])
    
    label_encoder = preprocessing.LabelEncoder() 
    data['label']= label_encoder.fit_transform(data['label']) 
  
    data['label'].unique() 
    

    data = data.dropna()
    clean_data(data)
    
    return data['review'], data['label'] 
