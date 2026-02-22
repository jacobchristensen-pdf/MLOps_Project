import mlflow

from .deploy import deploy
from .evaluate import evaluate
from .model_card import create_model_card
from .register import register_model
from .train import main as train_model


def run_pipeline():
    """Run the full MLOps pipeline"""
    print("Starting MLOps Pipeline...")

    mlflow.set_tracking_uri("http://172.24.198.42:5000")
    mlflow.set_experiment("CatsDogs")

    with mlflow.start_run(run_name="Full_MLOps_Pipeline"):

        # Step 1: Train
        print("\n[1/5] TRAINING...")
        model, accuracy, loss, config = train_model()

        # Step 2: Evaluate
        print("\n[2/5] EVALUATING...")
        passed = evaluate(accuracy, loss)

        # Step 3: Register
        print("\n[3/5] REGISTERING...")
        register_model(model, passed)

        # Step 4: Deploy
        print("\n[4/5] DEPLOYING...")
        deploy(passed)

        # Step 5: Model Card
        print("\n[5/5] MODEL CARD...")
        create_model_card(accuracy, loss, config)

        print("\nPipeline completed successfully!")


if __name__ == "__main__":
    run_pipeline()
