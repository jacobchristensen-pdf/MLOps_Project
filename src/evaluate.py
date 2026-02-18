import mlflow


def evaluate(best_val_accuracy, best_val_loss):
    """Automatic evaluation of trained model"""
    threshold = 0.80
    print("\n📊 Evaluating model...")
    print(f"   Accuracy: {best_val_accuracy:.4f}")
    print(f"   Threshold: {threshold}")

    mlflow.log_metric("final_accuracy", best_val_accuracy)
    mlflow.log_metric("final_loss", best_val_loss)

    passed = best_val_accuracy >= threshold

    if passed:
        print("✅ Model passed evaluation!")
    else:
        print("❌ Model failed evaluation!")

    return passed
