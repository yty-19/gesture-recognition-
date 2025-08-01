import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# Load and prepare dataset that got from dataset-creator.py
def load_data(data_file='data.pickle'):
    with open(data_file, 'rb') as f:
        data_dict = pickle.load(f)

    return np.asarray(data_dict['data']), np.asarray(data_dict['labels'])


# Train and evaluate the model
def train_model(X_train, X_test, y_train, y_test):
    # Initialize model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    return model, y_pred


# Evaluate model performance
def evaluate_model(model, X_train, X_test, y_train, y_test, y_pred):
    # Calculate accuracies
    train_score = model.score(X_train, y_train)
    test_score = accuracy_score(y_test, y_pred)

    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)

    # results
    print("\nModel Performance:")
    print(f"Training Accuracy: {train_score * 100:.2f}%")
    print(f"Testing Accuracy: {test_score * 100:.2f}%")
    print(f"Cross-validation Accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 2 * 100:.2f}%)")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return train_score, test_score, cv_scores


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


# Save trained model
def save_model(model, filename='model.p'):
    with open(filename, 'wb') as f:
        pickle.dump({'model': model}, f)
    print(f"\nModel saved to {filename}")


if __name__ == "__main__":
    try:
        # Load and split data
        print("Loading data...")
        X, y = load_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train model
        print("\nTraining model...")
        model, y_pred = train_model(X_train, X_test, y_train, y_test)

        # Evaluate model
        evaluate_model(model, X_train, X_test, y_train, y_test, y_pred)

        # Plot confusion matrix
        plot_confusion_matrix(y_test, y_pred)

        # Save model
        save_model(model)

    except Exception as e:
        print(f"Error occurred: {e}")
