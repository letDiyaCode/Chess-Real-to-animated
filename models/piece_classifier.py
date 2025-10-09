import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm

# ========== CONFIG ==========
DATA_DIR = "data/processed"
MODEL_PATH = "models/trained_models/piece_classifier.h5"
IMG_SIZE = 128
EPOCHS = 20
BATCH_SIZE = 32
# ============================

def load_data(subset):
    """Load .npy files and labels."""
    data, labels = [], []
    subset_dir = os.path.join(DATA_DIR, subset)

    for label_name in sorted(os.listdir(subset_dir)):
        label_path = os.path.join(subset_dir, label_name)
        if not os.path.isdir(label_path):
            continue
        for file in tqdm(os.listdir(label_path), desc=f"Loading {subset}/{label_name}"):
            if file.endswith(".npy"):
                img = np.load(os.path.join(label_path, file))
                data.append(img)
                labels.append(label_name)

    return np.array(data), np.array(labels)

def build_model(num_classes):
    """Define CNN architecture."""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def plot_training(history):
    """Plot accuracy and loss curves."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training vs Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training vs Validation Loss')

    plt.tight_layout()
    plt.show()
    plt.savefig("results/training_curves.png")


def main():
    print("📦 Loading dataset...")
    X_train, y_train = load_data("train")
    X_test, y_test = load_data("test")

    print(f"\n✅ Data loaded: {X_train.shape[0]} train, {X_test.shape[0]} test")

    # Encode string labels to integers
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    num_classes = len(le.classes_)
    print(f"🔢 Classes detected: {num_classes} -> {list(le.classes_)}")

    # Build and summarize model
    model = build_model(num_classes)
    model.summary()

    # Train model
    print("\n🚀 Training model...")
    history = model.fit(
        X_train, y_train_enc,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test_enc)
    )

    # Plot learning curves
    plot_training(history)

    # Evaluate
    print("\n📊 Evaluating model...")
    test_loss, test_acc = model.evaluate(X_test, y_test_enc)
    print(f"✅ Test Accuracy: {test_acc:.4f}")

    # Classification report
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test_enc, y_pred, target_names=le.classes_))

    # Save model + encoder
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    np.save("models/trained_models/label_encoder_classes.npy", le.classes_)

    print(f"\n💾 Model saved to: {MODEL_PATH}")
    print(f"💾 Label encoder saved to: models/trained_models/label_encoder_classes.npy")

if __name__ == "__main__":
    main()
