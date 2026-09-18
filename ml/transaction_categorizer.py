import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

class TransactionCategorizer:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"Loading embedding model {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.categories = {}
        self.category_centroids = {}

    def fit(self, examples):
        """
        examples: Dictionary where keys are categories and values are lists of example descriptions.
        Example: {'Housing': ['UPI RENT PAYMENT', 'MORTGAGE'], 'Food': ['SWIGGY', 'ZOMATO']}
        """
        print("Building category centroids...")
        for category, descriptions in examples.items():
            # Create embeddings for all examples in this category
            embeddings = self.model.encode(descriptions)
            # The centroid is the average vector of all examples in the category
            self.category_centroids[category] = np.mean(embeddings, axis=0)

        self.categories = list(self.category_centroids.keys())
        print(f"Categorizer trained with {len(self.categories)} categories.")

    def predict(self, text):
        """Categorize a single transaction description."""
        if not self.categories:
            return "Uncategorized"

        # Handle NaN or non-string inputs
        if text is None or (isinstance(text, float) and np.isnan(text)):
            return "Other/Unknown"

        # Ensure input is a string
        text = str(text)

        # Embed the input text
        text_embedding = self.model.encode([text])[0]

        # Calculate cosine similarity against all centroids
        similarities = []
        for category, centroid in self.category_centroids.items():
            sim = cosine_similarity([text_embedding], [centroid])[0][0]
            similarities.append(sim)

        # Find the best match
        best_idx = np.argmax(similarities)
        best_sim = similarities[best_idx]

        # Use a threshold to avoid forced categorization of completely unrelated text
        if best_sim < 0.3:
            return "Other/Unknown"

        return self.categories[best_idx]

    def categorize_batch(self, texts):
        """Categorize a list of texts."""
        return [self.predict(t) for t in texts]

if __name__ == "__main__":
    # Test the categorizer
    examples = {
        'Housing': ['UPI RENT PAYMENT', 'HOUSE RENT', 'MORTGAGE PAYMENT'],
        'Income': ['NEFT SALARY CREDIT', 'DIRECT DEPOSIT SALARY', 'BONUS CREDIT'],
        'Food': ['SWIGGY ORDER', 'ZOMATO DELIVERY', 'STARBUCKS COFFEE'],
        'Cash': ['ATM CASH WITHDRAWAL', 'CASH DEPOSIT ATM'],
        'Transport': ['UBER TRIP', 'OLA CAB', 'METRO RECHARGE']
    }

    categorizer = TransactionCategorizer()
    categorizer.fit(examples)

    test_texts = ["Swiggy for dinner", "Salary credited for Oct", "Rent for November", "Uber to office"]
    results = categorizer.categorize_batch(test_texts)

    for text, cat in zip(test_texts, results):
        print(f"'{text}' -> {cat}")
