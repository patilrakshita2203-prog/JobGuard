import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load cleaned dataset
df = pd.read_csv("cleaned_job_data.csv")

# Features and target
X = df["cleaned_text"]
y = df["fraudulent"]

print("Dataset loaded successfully!")
print(df[["cleaned_text", "fraudulent"]].head())

# Load BERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")

# Convert text to embeddings
X_embeddings = model.encode(
    X.tolist(),
    show_progress_bar=True
)

print("Embeddings generated successfully!")
print(X_embeddings.shape)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_embeddings,
    y,
    test_size=0.2,
    random_state=42
)

# Train classifier
clf = LogisticRegression(class_weight='balanced')
clf.fit(X_train, y_train)
import joblib

joblib.dump(clf, "models/fake_job_model.pkl")

print("Model saved successfully!")
# Predict
y_pred = clf.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))