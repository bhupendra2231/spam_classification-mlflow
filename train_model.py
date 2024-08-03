import argparse
import os
import numpy as np
import logging
import warnings
import pickle
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.models import infer_signature
import dagshub

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred, pos_label=1)
    recall = recall_score(actual, pred, pos_label=1)
    f1 = f1_score(actual, pred, pos_label=1)
    return accuracy, precision, recall, f1

def main(alpha):
    # Load dataset
    df = pd.read_csv('data/filtered_spam.csv')

    # Handle missing values
    if df['transformed_text'].isnull().any():
        print("Missing values found in 'transformed_text' column")
        df['transformed_text'] = df['transformed_text'].fillna('')

    # Ensure the required columns are present
    required_columns = {'transformed_text', 'target'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Missing one of the required columns: {required_columns - set(df.columns)}")

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df['transformed_text'], df['target'], test_size=0.2, random_state=42)

    # Vectorize text data
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train a Multinomial Naive Bayes model with the given alpha
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train_tfidf, y_train)

    # Make predictions
    predictions = model.predict(X_test_tfidf)

    # Evaluate the model
    accuracy, precision, recall, f1 = eval_metrics(y_test, predictions)

    # Print evaluation metrics
    print(f"Alpha: {alpha}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    # Initialize Dagshub
    dagshub.init(
        repo_owner='bhupendra2231',
        repo_name='spam_classification-mlflow',
        mlflow=True
    )

    # Set MLflow tracking URI to Dagshub
    mlflow.set_tracking_uri("https://dagshub.com/bhupendra2231/spam_classification-mlflow.mlflow")

    # Log everything with MLflow
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("vectorizer", "TF-IDF")
        mlflow.log_param("model_type", "MultinomialNB")
        mlflow.log_param("alpha", alpha)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Log vectorizer
        vectorizer_path = "vectorizer.pkl"
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        mlflow.log_artifact(vectorizer_path, artifact_path="vectorizer")

        # Log the model signature
        signature = infer_signature(X_train_tfidf, predictions)
        mlflow.sklearn.log_model(model, "model", signature=signature)

        # Remove the local vectorizer pickle file after logging
        os.remove(vectorizer_path)

    print("Model training and logging completed.")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha parameter for MultinomialNB")
    args = parser.parse_args()

    np.random.seed(40)

    main(args.alpha)
