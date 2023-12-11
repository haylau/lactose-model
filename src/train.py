import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from joblib import dump
from collections import Counter

from transform import format_text

# from query import DB_CONFIG

# Import Data
df = pd.read_csv("lact_sample.csv")

# Vectorize text data, try with merging all text cols
X_concat_cols = df["name"] + " " + df["category"] + " " + df["description"]
y_target_col = "lactose"

# Remove non-english characters, punctuation, and numbers
X_concat_cols = X_concat_cols.apply(format_text)

# vectorizer = CountVectorizer()
vectorizer = TfidfVectorizer()
X_text = vectorizer.fit_transform(X_concat_cols)
print(f"Token array: {vectorizer.get_feature_names_out()}")

# Separate features and target
X = pd.DataFrame(
    X_text.toarray(), columns=vectorizer.get_feature_names_out(), index=df.index
)
y = df[y_target_col].astype(int)

# Set up test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=12
)

# Test models
linearsvc_model = LinearSVC(dual="auto")
linearsvc_model.fit(X_train, y_train)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

mnnb_model = MultinomialNB()
mnnb_model.fit(X_train, y_train)

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

# Make predictions
linearsvc_y_pred = linearsvc_model.predict(X_test)
nb_y_pred = nb_model.predict(X_test)
mnnb_y_pred = mnnb_model.predict(X_test)
dt_pred = dt_model.predict(X_test)

# Evaluate the model
linearsvc_accuracy = accuracy_score(y_test, linearsvc_y_pred)
nb_accuracy = accuracy_score(y_test, nb_y_pred)
mnnb_accuracy = accuracy_score(y_test, mnnb_y_pred)
dt_model_accuracy = accuracy_score(y_test, dt_pred)
print(f"Linear SVC Accuracy: {linearsvc_accuracy}")
print(f"Naïve Bayes Accuracy: {nb_accuracy}")
print(f"Multinomial Naïve Bayes Accuracy: {mnnb_accuracy}")
print(f"Decision Tree Accuracy: {dt_model_accuracy}")
print(f"-------------------------------------------------")

dump_path = "best_model.joblib"
dump_model = {"model": None, "vectorizer": vectorizer}
best = max(linearsvc_accuracy, nb_accuracy, mnnb_accuracy)

if best == linearsvc_accuracy:
    print(f"Linear SVC is best: {linearsvc_accuracy}")
    dump_model["model"] = linearsvc_model
    coefficients = linearsvc_model.coef_[0]
    feature_coefficients = dict(zip(vectorizer.get_feature_names_out(), coefficients))
    sorted_features = sorted(
        feature_coefficients.items(), key=lambda x: x[1], reverse=True
    )
    top_n = 10
    print(f"Top Linear SVC Features:")
    for feature, coefficient in sorted_features[:top_n]:
        print(f"Feature: {feature}, Coefficient: {coefficient:.3f}")
elif best == nb_accuracy:
    print(f"Naïve Bayes is best: {nb_accuracy}")
    dump_model["model"] = nb_model
elif best == mnnb_accuracy:
    print(f"Multinomial Naïve Bayes is best: {mnnb_accuracy}")
    c = mnnb_model
elif best == dt_model_accuracy:
    print(f"Decision Tree is best: {dt_model_accuracy}")
    dump_model["model"] = dt_model
# Use the rest of the data to train
dump_model["model"].fit(X, y)
dump(dump_model, dump_path)

true_tokens = [
    token
    for tokens in X_concat_cols[X_test.index[y_test == 1]].str.split()
    for token in tokens
]
freq_tokens = Counter(true_tokens)
