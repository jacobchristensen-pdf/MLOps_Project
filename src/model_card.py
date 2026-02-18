from pathlib import Path

import mlflow


def create_model_card(best_val_accuracy, best_val_loss, config):
    """Create and store model card in MLflow"""
    print("\n📋 Creating model card...")

    model_card = f"""# Model Card - Cats vs Dogs Classifier

## Performance
- **Accuracy**: {best_val_accuracy:.4f}
- **Loss**: {best_val_loss:.4f}
- **Passed Evaluation**: {best_val_accuracy >= 0.80}

## Training Configuration
- **Learning Rate**: {config['training']['learning_rate']}
- **Epochs**: {config['training']['epochs']}
- **Batch Size**: {config['training']['batch_size']}

## Intended Use
Classifies images as cats or dogs.
For educational and MLOps demonstration purposes.
"""

    model_card_path = Path("model_card.md")
    with open(model_card_path, "w") as f:
        f.write(model_card)

    mlflow.log_artifact(str(model_card_path))
    print("✅ Model card saved to MLflow!")
