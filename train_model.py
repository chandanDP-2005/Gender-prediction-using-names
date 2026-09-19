import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATASET = "names_dataset.csv"


def train_and_save_model():
    df = pd.read_csv(DATASET).dropna()
    df["name"] = df["name"].astype(str).str.lower().str.strip()

    X = df["name"]
    y = df["gender"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # NLP: character-level TF-IDF captures spelling/orthographic patterns.
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            lowercase=True,
            sublinear_tf=True
        )),
        ("classifier", LogisticRegression(max_iter=2000))
    ])

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)

    print("\n===== NLP MODEL EVALUATION =====")
    print(f"Dataset size : {len(df)}")
    print(f"Training size: {len(X_train)}")
    print(f"Testing size : {len(X_test)}")
    print(f"Accuracy     : {accuracy*100:.2f}%\n")
    print(classification_report(y_test, pred))

    with open("gender_nlp_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Save evaluation values for the web dashboard.
    metrics = {
        "accuracy": round(accuracy * 100, 2),
        "dataset_size": int(len(df)),
        "training_size": int(len(X_train)),
        "testing_size": int(len(X_test)),
        "classes": sorted(y.unique().tolist()),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist()
    }
    with open("metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)

    print("Saved: gender_nlp_model.pkl")
    print("Saved: metrics.pkl")
    return model, metrics


if __name__ == "__main__":
    train_and_save_model()
