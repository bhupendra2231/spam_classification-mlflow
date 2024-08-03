import mlflow
from mlflow.tracking import MlflowClient
import mlflow.pyfunc
import os
import pickle

# Set the tracking URI if needed
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Initialize MLflow client
client = MlflowClient()

# Specify the model name, version, and run ID
model_name = "model_name"
version = "version"
run_id = "run_id"

# Get model version information
model_version_info = client.get_model_version(model_name, version)
model_uri = model_version_info.source
print(f"Model URI: {model_uri}")

# Construct the artifact URI for the vectorizer
vectorizer_uri = f"runs:/{run_id}/vectorizer/vectorizer.pkl"
print(f"Vectorizer URI: {vectorizer_uri}")

# Define custom directories
base_dir = "final_model_vectorizer"
os.makedirs(base_dir, exist_ok=True)

# Define paths for the downloaded model and vectorizer
model_download_dir = os.path.join(base_dir, "model")
vectorizer_download_dir = os.path.join(base_dir, "vectorizer")
os.makedirs(model_download_dir, exist_ok=True)
os.makedirs(vectorizer_download_dir, exist_ok=True)

# Download and save the model
try:
    # Download the model artifacts
    model_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=model_download_dir)
    print(f"Model downloaded to: {model_path}")

    # Load the model
    model = mlflow.pyfunc.load_model(model_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error downloading or loading model: {e}")

# Download and save the vectorizer
try:
    # Download the vectorizer artifact
    vectorizer_path = mlflow.artifacts.download_artifacts(artifact_uri=vectorizer_uri, dst_path=vectorizer_download_dir)
    print(f"Vectorizer downloaded to: {vectorizer_path}")

    # Load the vectorizer
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    print("Vectorizer loaded successfully")
except Exception as e:
    print(f"Error downloading or loading vectorizer: {e}")
