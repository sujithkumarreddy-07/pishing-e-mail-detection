import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.linear_model import LogisticRegression

# 1. Load dataset
data = pd.read_csv("emails.csv")
data["text"] = data["text"].fillna("")
data["label"] = data["label"].str.lower().str.strip()

print("Dataset loaded successfully")
print(data.head())

# 2. URL and keyword feature extraction
def extract_url_keyword_features(email):
    urls = re.findall(r'https?://\S+|www\.\S+', email)
    suspicious_words = [
        "login", "verify", "account", "password", "security",
        "urgent", "confirm", "bank", "update", "click", "suspended"
    ]
    suspicious_count = sum(word in email.lower() for word in suspicious_words)
    return len(urls), suspicious_count

url_features = data["text"].apply(extract_url_keyword_features)
data["url_count"] = url_features.apply(lambda x: x[0])
data["suspicious_keyword_count"] = url_features.apply(lambda x: x[1])

# 3. Prepare labels
label_map = {"safe": 0, "legitimate": 0, "phishing": 1}
data["target"] = data["label"].map(label_map)
data = data.dropna(subset=["target"])

X = data["text"]
y = data["target"].astype(int)

# 4. Split data
X_train, X_test, y_train, _, = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Re-create test labels correctly because the line above keeps both outputs
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. TF-IDF feature extraction
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 6. Train classifier
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# 7. Evaluate
predictions = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)

print("\nMODEL PERFORMANCE")
print("-----------------------------")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(
    y_test, predictions,
    target_names=["Safe", "Phishing"],
    zero_division=0
))
print("Confusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Safe", "Phishing"]
)
disp.plot()
plt.title("Phishing Email Detection - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=200)
plt.show()

# 8. Detect a new email
def detect_email(email):
    email_vector = vectorizer.transform([email])
    result = model.predict(email_vector)[0]
    probability = model.predict_proba(email_vector)[0][1]

    classification = "PHISHING" if result == 1 else "SAFE"

    print("\nEmail Analysis")
    print("-----------------------------")
    print("Prediction:", classification)
    print(f"Phishing Probability: {probability * 100:.2f}%")
    return classification

# Test examples
detect_email("""
URGENT! Your bank account has been suspended.
Click https://example.com/login to verify your password immediately.
""")

detect_email("""
Hi John,
The meeting is scheduled for tomorrow at 10 AM.
Please bring the project report.
Regards,
Team
""")
