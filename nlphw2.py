import gensim
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel
from wefe.metrics.WEAT import WEAT
from wefe.utils import run_queries
from wefe.datasets import load_weat
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
from datasets import load_dataset
dataset = load_dataset("wikipedia", "20220301.simple",trust_remote_code=True)


nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def normalize_text(text):
    tokens = word_tokenize(text)
    tokenized_text = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
    return tokenized_text

pre_trained_word_embeddings_text = 'glove.42B.300d/glove.42B.300d.txt'
pre_trained_word_embeddings_text_two = 'glove.6B/glove.6B.300d.txt'
word_embeddings_text_written = 'glove.42B.300d/glove_word_to_vec_format.txt'
word_embeddings_text_written_two = 'glove.6B/glove_word_to_vec_format_two.txt'
glove2word2vec(pre_trained_word_embeddings_text, word_embeddings_text_written)
glove2word2vec(pre_trained_word_embeddings_text_two, word_embeddings_text_written_two)

normalized_dataset = []
for index in dataset['train']:
    normalized_text = normalize_text(index['text'])
    normalized_dataset.append(normalized_text)

#skip-gram = sg where sg = 0 means Continous Bag of Words Model is invoked
#window = size of the context window
#min_count = minimum frequency count for a word to be considered (usually less than this)
#vector_size = the dimensionality of the word vectors

continuous_bag_of_words_model = Word2Vec(sentences=normalized_dataset, vector_size=200, window=5, min_count=1, sg=0)
continuous_bag_of_words_model.save("cbow.model")

#skip-gram = sg where sg = 1 indicates skip_gram Model is being utilized
#window = size of the context window
#min_count = minimum frequency count for a word to be considered
#vector_size = the dimensionality of the word vectors
skip_gram_model = Word2Vec(sentences=normalized_dataset, vector_size=200, window=5, min_count=1, sg=1)
skip_gram_model.save("skipgram.model")

sample_query_word_one = 'quest'
sample_query_word_two = 'triangle'
sample_query_word_three = 'spring'
sample_query_word_four = 'ring'
sample_query_word_five = 'emerald'

def query_one(model):
    if isinstance(model, gensim.models.Word2Vec):
        similar_words_word = model.wv.most_similar(sample_query_word_one, topn=10)
    elif isinstance(model, gensim.models.keyedvectors.KeyedVectors):
        similar_words_word = model.most_similar(sample_query_word_one, topn=10)
    else:
        print("Unsupported model type:", type(model))
        return
    print('Sample Query One Top 10 Similar Words: ', similar_words_word)
    print('')

def query_two(model):
    if isinstance(model, gensim.models.Word2Vec):
        word_analogy_word = model.wv.most_similar(positive=['circle', sample_query_word_two], negative=['square'], topn=1)
    elif isinstance(model, gensim.models.keyedvectors.KeyedVectors):
        word_analogy_word = model.most_similar(positive=['circle', sample_query_word_two], negative=['square'], topn=1)
    else:
        print("Unsupported model type:", type(model))
        return
    print('Word Analogy for Sample Query Two: ', word_analogy_word)
    print('')

def query_three(model):
    if isinstance(model, gensim.models.Word2Vec):
        odd_one_out_word = model.wv.doesnt_match([sample_query_word_three,sample_query_word_four,sample_query_word_five,'helicopter'])
    elif isinstance(model, gensim.models.keyedvectors.KeyedVectors):
        odd_one_out_word = model.doesnt_match([sample_query_word_three,sample_query_word_four,sample_query_word_five,'helicopter'])
    else:
        print("Unsupported model type:", type(model))
        return
    print('Sample Query Odd One Out: ', odd_one_out_word)
    print('')

def query_four(model):
    if isinstance(model, gensim.models.Word2Vec):
        composite_result_word = model.wv.doesnt_match([sample_query_word_four,sample_query_word_five])
    elif isinstance(model, gensim.models.keyedvectors.KeyedVectors):
        composite_result_word = model.doesnt_match([sample_query_word_four,sample_query_word_five])
    else:
        print("Unsupported model type:", type(model))
        return
    print('Sample Query Compositie Result: ', composite_result_word)
    print('')

def query_five(model):
    if isinstance(model, gensim.models.Word2Vec):
        word_distance_word_one = model.wv.similarity(sample_query_word_four,sample_query_word_five)
        print('Word Distance for Sample Query Four: ', word_distance_word_one)
        print('')
        word_distance_word_two = model.wv.similarity(sample_query_word_four,'coin')
        print('Word Distance for other Query of newley inputted word: ', word_distance_word_two)
        print('')
    elif isinstance(model, gensim.models.keyedvectors.KeyedVectors):
        word_distance_word_one = model.similarity(sample_query_word_four,sample_query_word_five)
        print('Word Distance for Sample Query Four: ', word_distance_word_one)
        print('')
        word_distance_word_two = model.similarity(sample_query_word_four,'coin')
        print('Word Distance for other Query of newley inputted word: ', word_distance_word_two)
        print('')
    else:
        print("Unsupported model type:", type(model))
        return

skip_gram_model = Word2Vec.load("skipgram.model")

print('Results of Skip-Gram Model: ')
print('')
query_one(skip_gram_model)
query_two(skip_gram_model)
query_three(skip_gram_model)
query_four(skip_gram_model)
query_five(skip_gram_model)
print('--------------------------------------------------------------------------------------------------------------')
print('')

continuous_bag_of_words_model = Word2Vec.load("cbow.model")

print('Results of Bag To Words Model: ')
print('')
query_one(continuous_bag_of_words_model)
query_two(continuous_bag_of_words_model)
query_three(continuous_bag_of_words_model)
query_four(continuous_bag_of_words_model)
query_five(continuous_bag_of_words_model)
print('--------------------------------------------------------------------------------------------------------------')
print('')

pre_trained_word_embeddings_model = KeyedVectors.load_word2vec_format(word_embeddings_text_written, binary=False)

print('Results of The First Pre-Trained Word Embeddings Model: ')
print('')
query_one(pre_trained_word_embeddings_model)
query_two(pre_trained_word_embeddings_model)
query_three(pre_trained_word_embeddings_model)
query_four(pre_trained_word_embeddings_model)
query_five(pre_trained_word_embeddings_model)
print('--------------------------------------------------------------------------------------------------------------')
print('')

pre_trained_word_embeddings_model_two = KeyedVectors.load_word2vec_format(word_embeddings_text_written_two, binary=False)

print('Results of The Second Pre-Trained Word Embeddings Model: ')
print('')
query_one(pre_trained_word_embeddings_model_two)
query_two(pre_trained_word_embeddings_model_two)
query_three(pre_trained_word_embeddings_model_two)
query_four(pre_trained_word_embeddings_model_two)
query_five(pre_trained_word_embeddings_model_two)
print('--------------------------------------------------------------------------------------------------------------')

cbow_wefe = WordEmbeddingModel(continuous_bag_of_words_model.wv, 'CBOW')
skip_gram_wefe = WordEmbeddingModel(skip_gram_model.wv, 'Skip-Gram')
pre_trained_word_embeddings_wefe_1 = WordEmbeddingModel(pre_trained_word_embeddings_model, 'Pretrained Word Embeddings 1')
pre_trained_word_embeddings_wefe_2 = WordEmbeddingModel(pre_trained_word_embeddings_model_two, 'Pretrained Word Embeddings 2')

models = [skip_gram_wefe, cbow_wefe, pre_trained_word_embeddings_wefe_1, pre_trained_word_embeddings_wefe_2]
word_sets = load_weat()

target_sets = [
    ["seniors","pensioner","elders","experienced","retiree","aging","elderly","masters","ancient","dignitaries"],
    ["adolescents","pubescent","youngsters","pupils","seedlings","youthful","fledglings","novices","fresh","apprentices"]
]

attribute_sets = [
    ["seriousness", "professionalism", "responsibility", "precision", "expertise", "competence", "dedication", "skill"],
    ["joy", "excitement", "happiness", "enthusiasm", "celebration", "laughter", "optimism", "cheerfulness"]
]

queries = [
    Query(
        target_sets=target_sets,
        attribute_sets=attribute_sets,
        target_sets_names=["Old Folks", "Younger Folks"],
        attribute_sets_names=["Professionalism", "Positive Emotions"]
    )
]
wefe_results = run_queries(
    WEAT,
    queries,
    models,
    metric_params={
        'preprocessors': [
            {},
            {'lowercase': True, 'strip_accents': True}
        ]
    },
    warn_not_found_words=True,
    lost_vocabulary_threshold=0.2
).T.round(4)

print(wefe_results)

positive_train_files = glob.glob('reviews_dataset/positive_files/*')
negative_train_files = glob.glob('reviews_dataset/negative_files/*')

num_files_per_class = 1000
all_train_files = positive_train_files[:num_files_per_class] + negative_train_files[:num_files_per_class]
labels = [1] * num_files_per_class + [0] * num_files_per_class

vectorizer = TfidfVectorizer(input="filename", stop_words="english")
vectors = vectorizer.fit_transform(all_train_files)
X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(vectors, labels, test_size=0.25, random_state=42)

model = LogisticRegression(random_state=42)
model.fit(X_train_tfidf,y_train_tfidf)
y_pred = model.predict(X_test_tfidf)

print("TF-IDF Accuracy:", accuracy_score(y_test_tfidf, y_pred))
print("TF-IDF F1-Score:", f1_score(y_test_tfidf, y_pred))

def document_to_avg_vectors(filename):
    with open(filename, 'r', encoding='utf-8', errors='replace') as file:
        doc = file.read()
    words = doc.split()
    vectors = [pre_trained_word_embeddings_model[word] for word in words if word in pre_trained_word_embeddings_model]
    if len(vectors) == 0:
        return np.zeros(pre_trained_word_embeddings_model.vector_size)
    else:
        return np.mean(vectors, axis=0)
    
X_pre_trained_word_embeddings = np.array([document_to_avg_vectors(doc) for doc in all_train_files])
X_train_pre_trained_word_embeddings, X_test_pre_trained_word_embeddings, y_train_pre_trained_word_embeddings, y_test_pre_trained_word_embeddings = train_test_split(X_pre_trained_word_embeddings, labels, test_size=0.25, random_state=42)

model_pre_trained_word_embeddings = LogisticRegression(max_iter=1000)
model_pre_trained_word_embeddings.fit(X_train_pre_trained_word_embeddings, y_train_pre_trained_word_embeddings)


predictions_pre_trained_word_embeddings = model_pre_trained_word_embeddings.predict(X_test_pre_trained_word_embeddings)
print("Pretrained_Word_Embeddings Accuracy:", accuracy_score(y_test_pre_trained_word_embeddings, predictions_pre_trained_word_embeddings))
print("Pretrained_Word_Embeddings F1-Score:", f1_score(y_test_pre_trained_word_embeddings, predictions_pre_trained_word_embeddings))