import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Load the restaurant reviews dataset from a TSV file
data = pd.read_csv('Restaurant_Review.tsv', delimiter='\t')


# Split the dataset into training and testing sets
X = data['Review']
y = data['Liked']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer to convert text to numerical features
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a Support Vector Machine (SVM) classifier
svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)
svm_classifier.fit(X_train_tfidf, y_train)

# Function to predict sentiment based on user input
def predict_sentiment(input_text):
    # Preprocess the user input text using the same vectorizer
    input_tfidf = tfidf_vectorizer.transform([input_text])

    # Predict the sentiment
    prediction = svm_classifier.predict(input_tfidf)[0]

    return prediction

# Get user input and predict sentiment
user_input = input("Enter your restaurant review: ")
predicted_sentiment = predict_sentiment(user_input)

if predicted_sentiment == 0:
    print("The sentiment is negative!")
elif predicted_sentiment == 1:
    print("The sentiment is positive!")
else:
    print("The sentiment could not be determined or is neutral.")
