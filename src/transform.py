import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def format_text(str):
    # Preprocess and lemm
    str = str.lower()
    str = str.replace("-", " ")
    tokens = re.sub(r"[^A-Za-z\s]+", "", str).split()
    return " ".join(
        [
            lemmatizer.lemmatize(token)
            for token in tokens
            if token not in stop_words and len(token) >= 3
        ]
    )


# def format_text_separate(text):

#     for col in x_feature_cols:
#         X_col = df[col].apply(
#             lambda x: ' '.join([word.lower() for word in word_tokenize(
#                 re.sub(r'[^A-Za-z\s]+', '', str(x))
#                 .translate(str.maketrans('', '', string.punctuation))
#                 .lower()
#             ) if word not in set(stopwords.words('english'))])
#         )

#         # Reinsert cols to the feature matrix
#         X = pd.concat(
#             [
#                 X, pd.DataFrame(X_text_col.toarray(),
#                 columns=tfidf_vectorizer.get_feature_names_out(),
#                 index=df.index)
#             ],
#             axis=1
#         )
