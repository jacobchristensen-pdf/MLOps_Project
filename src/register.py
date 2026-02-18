import mlflow
import mlflow.pytorch


def register_model(model, passed_evaluation):
    """Add model to MLflow registry if evaluation passed"""
    if passed_evaluation:
        print("\n📝 Registering model in MLflow...")

        mlflow.pytorch.log_model(
            model,
            "model",
            registered_model_name="CatsDogsClassifier",
        )
        mlflow.set_tag("deployment_status", "registered")
        print("✅ Model registered as CatsDogsClassifier!")
    else:
        mlflow.set_tag("deployment_status", "rejected")
        print("❌ Model not registered - did not meet criteria!")
