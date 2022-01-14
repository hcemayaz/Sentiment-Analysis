import re
import string
import time
import nltk
import streamlit as st
from snowballstemmer import TurkishStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.word2vec import Word2Vec
from gensim.models.fasttext import FastText
import multiprocessing
from sklearn import utils
from sklearn import linear_model, metrics, svm
from sklearn import ensemble
import numpy as np
from sklearn.model_selection import train_test_split
from gensim.models.doc2vec import TaggedDocument
from collections import Counter
from nltk.corpus import stopwords as stop
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

if False:
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
warnings.filterwarnings(action='ignore')

wpt = nltk.WordPunctTokenizer()
PorterStemmer = PorterStemmer()
SnowballStemmer = TurkishStemmer()
lemmatizer = WordNetLemmatizer()
my_file = open("/Users/cemayaz/Desktop/Sentiment_analysis/turkish.txt", "r")
content = my_file.read()
stop_words = content.split(",")
my_file.close()



def handle_emojis(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)', ' EMO_POS ', tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', tweet)
    # Sad -- :-(, : (, :(, ):, )-:
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', tweet)
    return tweet


def remove_hyperlink(sentence: str) -> str:

    sentence = re.sub(r"\S*@\S*\s?", " ", sentence)
    sentence = re.sub(r"www\S+", " ", sentence)
    sentence = re.sub(r"http\S+", " ", sentence)
    sentence = re.sub(r'\brt\b', ' ', sentence)
    sentence = re.sub(r'((@[\S]+)|(#[\S]+))', ' ', sentence)
    return sentence.strip()


def to_lower(sentence: str) -> str:

    result = sentence.lower()
    return result


def remove_number(sentence: str) -> str:

    result = re.sub(r'\S*\d\S*', ' ', sentence)
    return result


def remove_punctuation(sentence: str) -> str:

    result = sentence.translate(str.maketrans(dict.fromkeys(string.punctuation)))
    return result


def remove_whitespace(sentence: str) -> str:

    result = sentence.strip()
    return result


def replace_special_chars(sentence: str) -> str:

    chars_to_remove = ['\t', '\n', ';', "!", '"', "#", "%", "&", "'", "(", ")",
                       "+", ",", "-", "/", ":", ";", "<",
                       "=", ">", "?", "@", "[", "\\", "]", "^", "_",
                       "`", "{", "|", "}", "~", "–", '”', '“', '’']
    for ch in chars_to_remove:
        sentence = sentence.replace(ch, ' ')
    # replace ascii chars with symbol 8
    sentence = sentence.replace(u'\ufffd', ' ')
    return sentence.strip()


def remove_stopwords(sentence: str) -> str:

    stop_words_list = list(stop_words)
    stop_words_list += ["avea", "vodafone", "superonline", "turkcellden", "turkcell"]
    tokens = sentence.split()
    filtered_tokens = [token for token in tokens if
                       token not in stop_words_list and token.startswith("turkcell") is False]
    sentence = ' '.join(filtered_tokens)
    return sentence


def apply_stemmer(sentence: str, stemmer_name=SnowballStemmer) -> str:

    tokens = sentence.split()
    tokens = pos_tag(tokens)
    # don't apply proper names
    stemmed_tokens = [stemmer_name.stemWord(key.lower()) for key, value in tokens if value != 'NNP']
    sentence = ' '.join(stemmed_tokens)
    return sentence


def apply_lemmatizer(sentence: str) -> str:

    tokens = sentence.split()
    lemmatize_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    sentence = ' '.join(lemmatize_tokens)
    return sentence


def remove_less_than_two(sentence: str) -> str:

    tokens = sentence.split()
    filtered_tokens = [token for token in tokens if len(token) > 2]
    sentence = ' '.join(filtered_tokens)
    return sentence


def tokenize_sentence(sentence: str) -> str:

    return wpt.tokenize(sentence)


def tokenize_list_of_sentences(sentences: list) -> list:

    return [tokenize_sentence(sentence=sentence) for sentence in sentences]


def replace_turkish_chars(sentence: str) -> str:

    sentence = sentence.replace("ü", "u")
    sentence = sentence.replace("ı", "i")
    sentence = sentence.replace("ö", "o")
    sentence = sentence.replace("ü", "u")
    sentence = sentence.replace("ş", "s")
    sentence = sentence.replace("ç", "c")
    sentence = sentence.replace("ğ", "g")

    return sentence


def classification_report(x_train, x_test, y_train, y_test):
    models = [('LogisticRegression', linear_model.LogisticRegression(solver='newton-cg', multi_class='multinomial')),
              ('RandomForest', ensemble.RandomForestClassifier(n_estimators=100)),
              ('SVM', svm.SVC())]

    for name, model in models:
        clf = model
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        st.subheader(f"{name}:")
        res = pd.DataFrame.from_dict({'Accuracy': [metrics.accuracy_score(y_pred=y_pred, y_true=y_test)],
                                      'Precision': [
                                          metrics.precision_score(y_pred=y_pred, y_true=y_test, average="macro")],
                                      'Recall': [metrics.recall_score(y_pred=y_pred, y_true=y_test, average="macro")],
                                      'F1_score': [
                                          metrics.recall_score(y_pred=y_pred, y_true=y_test, average="macro")]})
        st.dataframe(res)


def get_word_counts(data):
    words = data.tweet.to_string().split()
    return Counter(words)


def labelize_tweets_ug(tweets, label):
    result = []
    prefix = label
    for i, t in zip(tweets.index, tweets):
        result.append(TaggedDocument(t.split(), [prefix + '_%s' % i]))
    return result


def get_mean_vector(model, words):
    # remove out-of-vocabulary words
    words = [word for word in words if word in model.wv]
    if len(words) >= 1:
        return np.mean(model[words], axis=0)
    else:
        return np.zeros((1, model.vector_size))


def get_vectors(model, corpus):
    vectors = []
    for sentence in corpus:
        vec = get_mean_vector(model, sentence)
        vectors.append(vec)
    return vectors


def train_doc2vec(corpus, n_epoch, name_corpus, vector_size, negative, window, min_count, alpha, min_alpha):
    cores = multiprocessing.cpu_count()
    model = Doc2Vec(size=vector_size, negative=negative, window=window, min_count=min_count, workers=cores, alpha=alpha,
                    min_alpha=min_alpha)
    model.build_vocab(corpus)

    for epoch in range(n_epoch):
        model.train(utils.shuffle(corpus), total_examples=len(corpus), epochs=1)
        model.alpha -= 0.002
        model.min_alpha = model.alpha

    return model


def train_fasText(corpus, n_epoch, name_corpus, sg, vector_size, negative, window, min_count, alpha, min_n, max_n):
    cores = multiprocessing.cpu_count()
    model = FastText(sg=sg, negative=negative, window=window, min_count=min_count, workers=cores,
                     alpha=alpha, min_n=min_n, max_n=max_n)
    model.build_vocab([x.words for x in corpus])

    for epoch in range(n_epoch):
        model.train(utils.shuffle([x.words for x in corpus]), total_examples=len(corpus), epochs=1)
        model.alpha -= 0.002
        model.min_alpha = model.alpha

    return model


def train_word2vec(corpus, n_epoch, name_corpus, sg, vector_size, negative, window, min_count, alpha, min_alpha):
    cores = multiprocessing.cpu_count()
    model = Word2Vec(sg=sg, negative=negative, window=window, min_count=min_count, workers=cores,
                     alpha=alpha, min_alpha=min_alpha)
    model.build_vocab([x.words for x in corpus])

    for epoch in range(n_epoch):
        model.train(utils.shuffle([x.words for x in corpus]), total_examples=len(corpus), epochs=1)
        model.alpha -= 0.002
        model.min_alpha = model.alpha

    return model


st.title('Sentiment Analysis')

import base64

file_ = open("/Users/cemayaz/Desktop/Sentiment_analysis/shutterstock_1073953772.jpg", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()

page_bg_img = '''
<img src="/Users/cemayaz/Desktop/Sentiment_analysis/shutterstock_1073953772.jpg">
<style>
body {
    color: #fff;
    background-color: #FFFFFF;
}
.stButton>button {
    color: #4F8BF9;
}

.stTextInput>div>div>input {
    color: #4F8BF9;
}
</style>
'''

st.markdown(f'<img src="data:image/gif;base64,{data_url}">', unsafe_allow_html=True)




OPERATIONS = {"lower": to_lower, "remove hyperlink": remove_hyperlink, "remove number": remove_number,
              "remove punctuation": remove_punctuation,
              "remove whitespace": remove_whitespace,
              "replace special chars": replace_special_chars, "remove stopwords": remove_stopwords,
              "apply stemmer": apply_stemmer,
              "remove less than two": remove_less_than_two,
              "handle_emoji": handle_emojis,
              "remove turkish chars": replace_turkish_chars}
uploaded_file_csv = None
uploaded_file_excel = None

ex = st.selectbox("Select file extension:",
                  ["excel", "csv"])
if ex == "excel":
    uploaded_file_excel = st.file_uploader("Choose a Excel file")
elif ex == "csv":
    uploaded_file_csv = st.file_uploader("Choose a CSV file")

if uploaded_file_csv is not None or uploaded_file_excel is not None:
    st.write("Uploaded data frame:")
    if uploaded_file_csv:
        df = pd.read_csv(uploaded_file_csv)
    else:
        df = pd.read_excel(uploaded_file_excel)
    st.dataframe(df)
    if st.button('Create Data Quality Report'):
        print_quality_result(df)

    df_selected = df.dropna()
    df_selected = df_selected.drop_duplicates()
    st.header("Preprocess Operations")
    column = st.selectbox("Select column to apply preprocess operation:",
                          df.columns)
    if df[column].dtype == np.object:
        operations = st.multiselect("Select operations to apply:",
                                    (
                                        'lower', 'remove hyperlink', "handle_emoji", "remove number",
                                        "remove punctuation",
                                        "remove whitespace",
                                        "replace special chars", "remove stopwords", "apply stemmer",
                                        "remove less than two",
                                        "remove turkish chars"))

        operations_list = list()
        for op in operations:
            operations_list.append(OPERATIONS[op])
        start = time.time()
        my_bar = st.progress(0)
        with st.spinner('Wait for it...'):
            for operation, name in zip(operations_list, operations):
                df_selected[column] = df_selected[column].apply(operation)
                with st.spinner(f"Applied operation is {name}"):
                    for i in range(100):
                        my_bar.progress(i + 1)
        st.success(f"Processed {len(df_selected)} samples and it's took {(time.time() - start) / 60} minutes.")
        st.write("After operations data frame looks like:")
        st.dataframe(df_selected)
    else:
        st.warning('Please select textual column for operations.')

    # Add a selectbox to the sidebar:
    add_select_box_side = st.sidebar.selectbox(
        'Which Embedding Method You Want To Use?',
        ('Word2Vec', 'Doc2Vec', 'FastText')
    )

    if add_select_box_side == 'Word2Vec':
        sentiment = st.selectbox("Select sentiment column:",
                                 df.columns)

        column = st.selectbox("Select column to apply embedding operation:",
                              df.columns)

        df_selected.dropna(inplace=True)
        df_selected.reset_index(drop=True, inplace=True)

        x = df_selected[column]
        y = df_selected[sentiment].values

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        corpus = labelize_tweets_ug(x, 'all')
        corpus_train = pd.DataFrame(x_train)[column].apply(lambda x: x.split())
        corpus_test = pd.DataFrame(x_test)[column].apply(lambda x: x.split())

        lr = st.slider('Learning Rate', min_value=0.0001, max_value=1.0)
        window = st.slider('Window Size', min_value=2, max_value=10)
        size = st.slider('Vector Size', min_value=20, max_value=1000)
        epoch = st.slider('Number of Epoch', min_value=1, max_value=10)

        if st.button('Get results'):
            model = train_word2vec(corpus=corpus,
                                   n_epoch=epoch,
                                   name_corpus="NOE",
                                   sg=0,
                                   negative=5,
                                   alpha=lr,
                                   min_alpha=0.065,
                                   window=window,
                                   vector_size=size,
                                   min_count=3)

            vectors_train = get_vectors(model=model,
                                        corpus=corpus_train)
            vectors_test = get_vectors(model=model,
                                       corpus=corpus_test)

            X_train = np.array(vectors_train)
            X_train = np.vstack(X_train)
            X_test = np.array(vectors_test)
            X_test = np.vstack(X_test)

            classification_report(x_train=X_train,
                                  x_test=X_test,
                                  y_train=y_train,
                                  y_test=y_test)
            if st.button("Word2Vec Find Similar Words"):
                word = st.text_input("word to find similar")
                st.write(model.most_similar(word))

    if add_select_box_side == 'Doc2Vec':
        sentiment = st.selectbox("Select sentiment column:",
                                 df.columns)

        column = st.selectbox("Select column to apply embedding operation:",
                              df.columns)

        df_selected.dropna(inplace=True)
        df_selected.reset_index(drop=True, inplace=True)

        x = df_selected[column]
        y = df_selected[sentiment].values

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        corpus = labelize_tweets_ug(x, 'all')
        corpus_train = pd.DataFrame(x_train)[column].apply(lambda x: x.split())
        corpus_test = pd.DataFrame(x_test)[column].apply(lambda x: x.split())

        lr = st.slider('Learning Rate', min_value=0.0001, max_value=1.0)
        window = st.slider('Window Size', min_value=2, max_value=10)
        size = st.slider('Vector Size', min_value=20, max_value=1000)
        epoch = st.slider('Number of Epoch', min_value=1, max_value=10)

        if st.button('Get results'):
            model = train_doc2vec(corpus=corpus,
                                  n_epoch=epoch,
                                  name_corpus="Noe",
                                  negative=5,
                                  alpha=lr,
                                  min_alpha=0.065,
                                  window=window,
                                  vector_size=size,
                                  min_count=2)

            vectors_train = get_vectors(model=model,
                                        corpus=corpus_train)
            vectors_test = get_vectors(model=model,
                                       corpus=corpus_test)

            X_train = np.array(vectors_train)
            X_train = np.vstack(X_train)
            X_test = np.array(vectors_test)
            X_test = np.vstack(X_test)

            classification_report(x_train=X_train,
                                  x_test=X_test,
                                  y_train=y_train,
                                  y_test=y_test)
    if add_select_box_side == 'FastText':
        sentiment = st.selectbox("Select sentiment column:",
                                 df.columns)

        column = st.selectbox("Select column to apply embedding operation:",
                              df.columns)

        df_selected.dropna(inplace=True)
        df_selected.reset_index(drop=True, inplace=True)

        x = df_selected[column]
        y = df_selected[sentiment].values

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        corpus = labelize_tweets_ug(x, 'all')
        corpus_train = pd.DataFrame(x_train)[column].apply(lambda x: x.split())
        corpus_test = pd.DataFrame(x_test)[column].apply(lambda x: x.split())

        lr = st.slider('Learning Rate', min_value=0.0001, max_value=1.0)
        window = st.slider('Window Size', min_value=2, max_value=10)
        size = st.slider('Vector Size', min_value=20, max_value=1000)
        min_n = st.slider('Character NGram minimum length', min_value=2, max_value=10)
        max_n = st.slider('Character NGram maximum length', min_value=2, max_value=10)
        epoch = st.slider('Number of Epoch', min_value=1, max_value=10)

        if st.button('Get results'):
            model = train_fasText(corpus=corpus,
                                  n_epoch=epoch,
                                  name_corpus="No",
                                  sg=1,
                                  negative=5,
                                  alpha=lr,
                                  min_n=min_n,
                                  max_n=max_n,
                                  window=window,
                                  vector_size=size,
                                  min_count=2)

            vectors_train = get_vectors(model=model,
                                        corpus=corpus_train)
            vectors_test = get_vectors(model=model,
                                       corpus=corpus_test)

            X_train = np.array(vectors_train)
            X_train = np.vstack(X_train)
            X_test = np.array(vectors_test)
            X_test = np.vstack(X_test)

            classification_report(x_train=X_train,
                                  x_test=X_test,
                                  y_train=y_train,
                                  y_test=y_test)
