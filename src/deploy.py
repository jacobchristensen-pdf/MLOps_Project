import mlflow
import mlflow.pytorch


def deploy(passed_evaluation):
    """Deploy model if it passed evaluation"""
    if passed_evaluation:
        print("\n🚀 Deploying model...")

        model_uri = "models:/CatsDogsClassifier/latest"
        model = mlflow.pytorch.load_model(model_uri)

        mlflow.set_tag("deployed", "true")
        mlflow.set_tag("environment", "production")

        print("✅ Model deployed successfully!")
        return model
    else:
        print("❌ Model not deployed - did not meet criteria!")
        return None
