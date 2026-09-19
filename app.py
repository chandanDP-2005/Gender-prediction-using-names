from flask import Flask, render_template, request, jsonify
import csv
import re
import math
import os
from collections import Counter

app = Flask(__name__)

DATA_PATH = os.path.join("data", "names.csv")
FALLBACK_DATA_PATH = "names_dataset.csv"


def resolve_data_path():
    if os.path.exists(DATA_PATH):
        return DATA_PATH
    if os.path.exists(FALLBACK_DATA_PATH):
        return FALLBACK_DATA_PATH
    raise FileNotFoundError(
        "Training data not found. Expected 'data/names.csv' or 'names_dataset.csv'."
    )


class NameNLPModel:

    def __init__(self):
        self.class_counts = Counter()

        self.ngram_counts = {
            "Male": Counter(),
            "Female": Counter()
        }

        self.total_ngrams = {
            "Male": 0,
            "Female": 0
        }

        self.vocabulary = set()

        self.train()

    def normalize(self, name):
        return re.sub(r"[^a-z]", "", name.lower().strip())

    def generate_ngrams(self, name):

        # Add beginning and ending markers
        text = "^" + name + "$"

        grams = []

        # Character 2-grams
        for i in range(len(text) - 1):
            grams.append(text[i:i + 2])

        # Character 3-grams
        for i in range(len(text) - 2):
            grams.append(text[i:i + 3])

        return grams

    def train(self):

        data_path = resolve_data_path()

        with open(data_path, "r", encoding="utf-8") as file:

            reader = csv.DictReader(file)

            for row in reader:

                name = self.normalize(row["name"])
                gender = row["gender"]

                if not name:
                    continue

                self.class_counts[gender] += 1

                grams = self.generate_ngrams(name)

                self.ngram_counts[gender].update(grams)

                self.total_ngrams[gender] += len(grams)

                self.vocabulary.update(grams)

    def predict(self, original_name):

        name = self.normalize(original_name)

        if not name:
            raise ValueError("Please enter a valid name.")

        labels = ["Male", "Female"]

        total_names = sum(self.class_counts.values())

        scores = {}

        grams = self.generate_ngrams(name)

        vocabulary_size = len(self.vocabulary)

        for label in labels:

            # Prior probability
            probability = math.log(
                (self.class_counts[label] + 1) /
                (total_names + len(labels))
            )

            denominator = self.total_ngrams[label] + vocabulary_size

            for gram in grams:

                probability += math.log(
                    (self.ngram_counts[label][gram] + 1) /
                    denominator
                )

            scores[label] = probability

        # Softmax
        maximum = max(scores.values())

        exponentials = {
            label: math.exp(score - maximum)
            for label, score in scores.items()
        }

        total = sum(exponentials.values())

        probabilities = {
            label: exponentials[label] / total
            for label in labels
        }

        prediction = max(
            probabilities,
            key=probabilities.get
        )

        confidence = probabilities[prediction] * 100

        # NLP features
        padded = "^" + name + "$"

        bigrams = [
            padded[i:i + 2]
            for i in range(len(padded) - 1)
        ]

        trigrams = [
            padded[i:i + 3]
            for i in range(len(padded) - 2)
        ]

        prefixes = [
            name[:i]
            for i in range(1, min(4, len(name)) + 1)
        ]

        suffixes = [
            name[-i:]
            for i in range(1, min(4, len(name)) + 1)
        ]

        return {

            "input": original_name,

            "normalized": name,

            "prediction": prediction,

            "confidence": round(confidence, 2),

            "probabilities": {
                "Male": round(probabilities["Male"] * 100, 2),
                "Female": round(probabilities["Female"] * 100, 2)
            },

            "features": {

                "length": len(name),

                "first_character": name[0].upper(),

                "last_character": name[-1].upper(),

                "prefixes": prefixes,

                "suffixes": suffixes,

                "bigrams": bigrams[:10],

                "trigrams": trigrams[:10]
            }
        }


try:
    model = NameNLPModel()
except FileNotFoundError as exc:
    model = None
    app.config["MODEL_ERROR"] = str(exc)
else:
    app.config["MODEL_ERROR"] = None


@app.route("/")
def home():

    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return jsonify({
            "success": False,
            "error": app.config.get("MODEL_ERROR", "Model initialization failed.")
        }), 500

    data = request.get_json(silent=True) or {}

    name = data.get("name", "")

    try:

        result = model.predict(name)

        return jsonify({
            "success": True,
            "result": result
        })

    except ValueError as error:

        return jsonify({
            "success": False,
            "error": str(error)
        }), 400


@app.route("/stats")
def stats():

    if model is None:
        return jsonify({
            "total": 0,
            "male": 0,
            "female": 0
        })

    return jsonify({

        "total": sum(model.class_counts.values()),

        "male": model.class_counts["Male"],

        "female": model.class_counts["Female"]

    })


if __name__ == "__main__":

    app.run(debug=True)