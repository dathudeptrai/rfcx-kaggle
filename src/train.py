import os
import tensorflow as tf
from models import Classifier, DeepMetricLearning


def get_model(
    saved_path="",
    pretrained_with_contrastive=False,
    pretrained_path="",
):
    if pretrained_with_contrastive:
        model = DeepMetricLearning()
    else:
        model = Classifier()
        model._build()

        if pretrained_path != "":
            print(f" -> Loading weights from {pretrained_path}\n")
            model.load_weights(pretrained_path, by_name=True)

    os.makedirs(saved_path, exist_ok=True)
    return model


def get_callbacks(pretrained_with_contrastive=False, fold_id=0, saved_path=""):
    if pretrained_with_contrastive:
        return [
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(saved_path, f"pretrained_best_fold{fold_id}.h5"),
                monitor="val_loss",
                save_weights_only=True,
                save_best_only=True,
                mode="min",
                save_freq="epoch",
            ),
            tf.keras.callbacks.EarlyStopping(
                patience=30,
                monitor="val_loss",
                mode="min",
            ),
        ]

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(
            saved_path, f"fold{fold_id}", "model-{val_lwlrap:.3f}-{val_loss:.3f}.h5"
        ),
        monitor="val_lwlrap",
        save_weights_only=True,
        save_best_only=True,
        mode="max",
        save_freq="epoch",
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        patience=25, monitor="val_lwlrap", mode="max"
    )
    return [model_checkpoint, early_stopping]
