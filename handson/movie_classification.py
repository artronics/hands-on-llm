import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import Pipeline, pipeline
from transformers.pipelines.pt_utils import KeyDataset


def load_dataset():
    from datasets import load_dataset

    return load_dataset("cornell-movie-review-data/rotten_tomatoes")


# p116
def classifier_model() -> Pipeline:
    model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    return pipeline(
        model=model_path,
        tokenizer=model_path,
        return_all_scores=True,
    )


def test_classifier_model(dataset):
    y_pred = []
    pipe = classifier_model()
    for output in tqdm(pipe(KeyDataset(dataset["test"], "text")), total=len(dataset["test"])):
        neg_scores = output[0]["score"]
        pos_scores = output[2]["score"]
        assignment = np.argmax([neg_scores, pos_scores])
        y_pred.append(assignment)

    return y_pred


# p122
def embedding_model(data):
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    train_embeddings = model.encode(data["train"]["text"], show_progress_bar=True)
    test_embeddings = model.encode(data["test"]["text"], show_progress_bar=True)

    # Train a logistic regression on our train embeddings
    clf = LogisticRegression(random_state=42)
    clf.fit(train_embeddings, data["train"]["label"])

    y_pred = clf.predict(test_embeddings)

    return y_pred


# p123
def cosine_similarity(data):
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    labeled_embeddings = model.encode(["A negative review", "A positive review"])
    test_embeddings = model.encode(data["test"]["text"], show_progress_bar=True)

    sim_matrix = cosine_similarity(test_embeddings, labeled_embeddings)
    y_pred = np.argmax(sim_matrix, axis=1)
    return y_pred


# p128
def text2text_generation(data):
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
    )
    prompt = "is the following sentence positive or negative?"
    data = data.map(lambda example: {"t5": prompt + example["text"]})

    y_pred = []
    for output in tqdm(pipe(KeyDataset(data["test"], "t5")), total=len(data["test"])):
        text = output[0]["generated_text"]
        y_pred.append(0 if text == "negative" else 1)

    return y_pred

def evaluate_performance(y_true, y_pred):
    """Create and print classification report"""
    from sklearn.metrics import classification_report

    performance = classification_report(
        y_true=y_true,
        y_pred=y_pred,
        target_names=["Negative Review", "Positive Review"],
    )
    print(performance)


def main():
    ds = load_dataset()
    # y_pred = test_classifier_model(ds)
    # y_pred = embedding_model(ds)
    # y_pred = cosine_similarity(ds)
    y_pred = text2text_generation(ds)
    evaluate_performance(ds["test"]["label"], y_pred)


if __name__ == '__main__':
    main()
