# File for AI_Chatbot_Report Appendix C: TF-IDF Code Snippet
# This code initializes and fits the vectorizer to the FAQ data.

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Load the FAQ data
try:
    faq = pd.read_csv('faq_dataset.csv')
except FileNotFoundError:
    print("Error: faq_dataset.csv not found. Please ensure it's in the same directory.")
    exit()

# Initialize the TF-IDF Vectorizer
# It will tokenize and compute the Inverse Document Frequency for the questions
vectorizer = TfidfVectorizer()

# Fit the vectorizer on all FAQ questions and transform them into a matrix
tfidf_matrix = vectorizer.fit_transform(faq['Question'])


print(f"Vectorizer successfully fitted on {len(faq)} questions.")
print(f"The vocabulary size (number of unique words) is: {len(vectorizer.get_feature_names_out())}")
# The vectorizer and matrix are now ready to be used for similarity scoring.
