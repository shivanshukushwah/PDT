
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Function to extract key features from a URL
def extract_features(url):
    features = {
        'url_length': len(url),
        'num_dots': url.count('.'),
        'num_hyphens': url.count('-'),
        'num_slashes': url.count('/'),
        'has_https': 1 if 'https' in url else 0,
        'num_subdomains': url.count('.') - 1
    }
    return features

# Example dataset (replace with a real dataset for better results)
data = [
    {"url": "http://example.com", "label": 0},
    {"url": "https://secure-site.com", "label": 0},
    {"url": "http://phishing-site.com", "label": 1},
    {"url": "http://another-phishing.com", "label": 1}
]

# Convert dataset into a DataFrame and extract features
data_df = pd.DataFrame(data)
data_df = pd.DataFrame([{
    **extract_features(row['url']),
    'label': row['label']
} for _, row in data_df.iterrows()])

# Split the dataset into training and testing sets
X = data_df.drop(columns=['label'])
y = data_df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("Detailed Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model to a file
joblib.dump(model, 'phishing_model.pkl')
print("Model has been saved as 'phishing_model.pkl'.")

# Function to predict if a URL is phishing or legitimate
def predict_url(url):
    features = extract_features(url)
    features_df = pd.DataFrame([features])
    prediction = model.predict(features_df)
    return "Phishing" if prediction[0] == 1 else "Legitimate"

# Example usage
example_url = "http://phishing-test.com"
print(f"The URL '{example_url}' is predicted as: {predict_url(example_url)}")
