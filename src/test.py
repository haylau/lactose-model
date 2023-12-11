import pandas as pd
from sklearn.metrics import accuracy_score
from joblib import load

from transform import format_text

model_path = "best_model.joblib"
best_model = load(model_path)
model = best_model["model"]
vectorizer = best_model["vectorizer"]
feature_list = vectorizer.get_feature_names_out()
df = pd.read_csv("lact_testmodel.csv")

X_concat_cols_original = df["name"] + " " + df["category"] + " " + df["description"]
y_target_col = "lactose"

# Remove non-english characters, punctuation, and numbers

X_concat_cols = X_concat_cols_original.apply(format_text)
X_text = vectorizer.transform(X_concat_cols)

# Separate features and target
X = pd.DataFrame(
    X_text.toarray(), columns=vectorizer.get_feature_names_out(), index=df.index
)
y = df[y_target_col].astype(int)

X_feature_list = set(token for string in X_concat_cols for token in string.split())
print(f"{len(feature_list)}|{len(X_feature_list)}")

unseen_at_fit_time = set(X_feature_list) - set(feature_list)
seen_missing_at_fit_time = set(feature_list) - set(X_feature_list)
print(f"Missing: {len(unseen_at_fit_time)} words")
for word in list(unseen_at_fit_time)[:50]:
    print(word)

predict = model.predict(X)
accuracy = accuracy_score(y, predict)
print(accuracy)

for original_input, prediction, actual_value in zip(X_concat_cols, predict, y):
    print(f"Prediction: {prediction}, Actual: {actual_value}, Input: {original_input}")
