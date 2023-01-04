import wget, os, re
import pandas as pd
from tqdm import tqdm
from os.path import exists

# LANGUAGE PROCESSOR
import fasttext

# WORD ANALYZING
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer

FILE_PATH = ""

if not os.path.exists('/tmp/lid.176.bin'):
    print('Downloading model...')
    wget.download("https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin", out="/tmp/lid.176.bin")

PRETRAINED_MODEL_PATH = '/tmp/lid.176.bin'

model = fasttext.load_model(PRETRAINED_MODEL_PATH)

sid = SentimentIntensityAnalyzer()
lemmantizer = WordNetLemmatizer()

ID_FEATURE = ['listing_id']
IMP_FEATURES = ['comments']
FEATURES = ID_FEATURE + IMP_FEATURES


class ReviewsPreprocessing:
    def __init__(self, feature_columns=None, auto_process=False, max_features=150, sentiment_analysis=False):
        if feature_columns is None:
            feature_columns = FEATURES

        self.reviews = pd.read_csv(f"{FILE_PATH}reviews.csv")
        self.feature_columns = feature_columns

        self.drop_columns()

        self.reviews_tfidf = None
        self.feature_names = None

        self.max_features = max_features
        self.stop_words_processed = False
        self.sentiment_analysis = sentiment_analysis

        self.polarity_df = pd.DataFrame([])
        self.tfidf_df = pd.DataFrame([])

        self.auto_process = auto_process
        if self.auto_process:
            self.drop_comments_with_len(_len=10)
            self.get_comment_lang()
            self.keep_comment_lang(lang='en')
            self.remove_emojis()
            self.remove_stopwords()
            self.calculate_comment_polarity()
            self.vectorize_words()
            self.listings_mean_data()
            self.merge_dataframes()
            self.save_file()

    def drop_columns(self):
        self.reviews.drop(columns=self.reviews.columns.difference(self.feature_columns), axis=1, inplace=True)

    def drop_comments_with_len(self, _len=10):
        self.reviews = self.reviews[self.reviews.comments.str.len() > _len]

    def get_comment_lang(self):
        if not exists('reviews_lang.csv'):
            self.reviews['lang'] = range(0, len(self.reviews))
            for index, row in tqdm(self.reviews.iterrows(), desc="Determining Language: ", total=self.reviews.shape[0]):
                if type(row['comments']) == type('s'):
                    predictions = model.predict(row['comments'])
                    l = predictions[0][0].split('__label__')[1]
                    if l != 'ceb' and l != 'nds' and l != 'war' and l != 'wuu':
                        self.reviews['lang'][index] = l

            self.reviews.to_csv('reviews_lang.csv')
        else:
            self.reviews = pd.read_csv('reviews_lang.csv')

    def keep_comment_lang(self, lang='en'):
        self.reviews = self.reviews[self.reviews.lang == lang]

    def remove_emojis(self):
        self.reviews.astype(str).apply(lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii'))

    def remove_stopwords(self):
        stop_words = set(stopwords.words('english'))

        self.reviews['comments'] = self.reviews['comments'].apply(
            lambda x: re.sub('[^a-zA-Z]', ' ', x))

        self.reviews['comments'] = self.reviews['comments'].apply(
            lambda x: re.sub(' br ', ' ', x))

        self.reviews['comments'] = self.reviews['comments'].apply(
            lambda x: " ".join([lemmantizer.lemmatize(item) for item in x.lower().split() if item not in stop_words]))

        self.stop_words_processed = True

    def calculate_comment_polarity(self):
        if not exists('reviews_polarity.csv'):
            self.reviews['compound'] = range(0, len(self.reviews))
            self.reviews['pos'] = range(0, len(self.reviews))
            self.reviews['neu'] = range(0, len(self.reviews))
            self.reviews['neg'] = range(0, len(self.reviews))

            for index, row in tqdm(self.reviews.iterrows(), desc="Calculating Sentiment: ",
                                   total=self.reviews.shape[0]):
                ss = sid.polarity_scores(row['comments'])

                self.reviews['compound'][index] = ss['compound']
                self.reviews['pos'][index] = ss['pos']
                self.reviews['neu'][index] = ss['neu']
                self.reviews['neg'][index] = ss['neg']

            self.reviews.to_csv('reviews_polarity.csv')

    def vectorize_words(self):
        self.reviews = pd.read_csv('reviews_polarity.csv')
        self.reviews = self.reviews.fillna('none', axis=1)

        _len = len(self.reviews)
        data_without_stopwords = self.reviews.comments[:_len]

        vectorizer = TfidfVectorizer(analyzer='word', max_features=self.max_features)
        reviews_tfidf = vectorizer.fit_transform(data_without_stopwords)
        reviews_tfidf = reviews_tfidf.toarray()

        self.feature_names = vectorizer.get_feature_names()

        self.reviews_tfidf = pd.DataFrame(data=reviews_tfidf)
        self.reviews_tfidf.to_csv(f'vectors_tfidf_{self.max_features}.csv', index=False)

        self.reviews_tfidf = pd.read_csv(f'vectors_tfidf_{self.max_features}.csv')
        self.reviews_tfidf.columns = [f"tfidf_{x}" for x in self.feature_names]
        self.reviews_tfidf['listing_id'] = self.reviews['listing_id']

    def listings_mean_data(self):
        for key in self.feature_names:
            self.tfidf_df[f'tfidf_{key}'] = self.reviews_tfidf.groupby(by="listing_id")[f'tfidf_{key}'].mean()

        self.tfidf_df.to_csv(f'vectors_tfidf_{self.max_features}.csv', index=False)

        for key in ['compound', 'pos', 'neg', 'neu']:
            self.polarity_df[key] = self.reviews.groupby(by="listing_id")[key].mean()

    def merge_dataframes(self):
        self.reviews = pd.merge(left=self.tfidf_df, right=self.polarity_df, left_on='listing_id', right_on='listing_id')

    def save_file(self, file_name='reviews_data.csv'):
        self.reviews.to_csv(file_name)
