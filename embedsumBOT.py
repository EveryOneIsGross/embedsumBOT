import pickle
import numpy as np
import math
from collections import Counter
from scipy.spatial.distance import cosine
from gpt4all import GPT4All, Embed4All
from nltk.sentiment import SentimentIntensityAnalyzer
from rake_nltk import Rake
import os

# Get the current working directory
cwd = os.getcwd()

# Initialize the models
gpt_model = GPT4All("orca-mini-3b.ggmlv3.q4_0.bin")
embedder = Embed4All()
sentiment_analyzer = SentimentIntensityAnalyzer()
keyword_extractor = Rake()

# Load the database from a pickle file
try:
    with open('summary_embeddings.pkl', 'rb') as f:
        database = pickle.load(f)
except FileNotFoundError:
    database = {}
from scipy.spatial.distance import cosine

#def euclidean_distance(c1, c2):
#    # Convert the Counters to vectors
#    terms = set(c1).union(c2)
#    vec1 = [c1.get(k, 0) for k in terms]
#    vec2 = [c2.get(k, 0) for k in terms]
#
#    # Compute the Euclidean distance
#    return math.sqrt(sum((x - y) ** 2 for x, y in zip(vec1, vec2)))

def cosine_similarity(c1, c2):
    set1 = set(c1)
    set2 = set(c2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if not union:
        return 0.0
    else:
        return float(len(intersection)) / len(union)

while True:
    # Get the user's input
    user_input = input("Enter a command: ")

    # If the user wants to summarize a document
    # Import the os library


    # If the user wants to summarize a local .txt file
    if user_input.startswith("summarize"):
        # Get the file name from the user input
        file_name = user_input[len("summarize"):].strip()

        # Check if the file exists and has a .txt extension
        if os.path.exists(file_name) and file_name.endswith(".txt"):
            # Open the file and read its content
            with open(file_name, "r") as f:
                document = f.read()

            # Generate the summary
            summary = gpt_model.generate(f"Summarize: {document}", max_tokens=500)

            # Embed the summary
            embedding = embedder.embed(summary)

            # Analyze the sentiment of the summary
            sentiment = sentiment_analyzer.polarity_scores(summary)

            # Extract keywords from the summary
            keyword_extractor.extract_keywords_from_text(summary)
            keywords = keyword_extractor.get_ranked_phrases()

            # Save the summary, its embedding, its sentiment, and its keywords to the database
            database[summary] = {
                'embedding': embedding,
                'sentiment': sentiment,
                'keywords': Counter(keywords)  # Save keywords as a Counter for euclidean distance
            }

            print(f"Summary: {summary}")
            print(f"Sentiment: {sentiment}")
            print(f"Keywords: {keywords}")

        else:
            print("Invalid file name. Please enter a valid .txt file name.")

    # If the user wants to search for a summary
    elif user_input.startswith("search"):
        search_text = user_input[len("search"):].strip()

        # Embed the search text
        search_embedding = embedder.embed(search_text)

        # Extract keywords from the search text
        keyword_extractor.extract_keywords_from_text(search_text)
        search_keywords = Counter(keyword_extractor.get_ranked_phrases())

        # Find the summary with the highest cosine similarity to the search text
        best_summary = None
        best_similarity = -1

        for summary, data in database.items():
            # Compute the cosine similarity between the search text and the summary
            embedding_similarity = 1 - cosine(search_embedding, data['embedding'])

            # Or convert the summary keywords to a Counter object when retrieving them from the database
            #keyword_similarity = euclidean_distance(search_keywords, Counter(data['keywords']))
            keyword_similarity = cosine_similarity(search_keywords, data['keywords'])

            # Combine the two similarities
            similarity = (embedding_similarity + keyword_similarity) / 2  # average the two similarities

            # If this summary is more similar to the search text than the current best summary
            if similarity > best_similarity:
                best_summary = summary
                best_similarity = similarity

        # If a summary was found
        if best_summary is not None:
            print(f"Best match: {best_summary}")
            print(f"Sentiment: {database[best_summary]['sentiment']}")
            print(f"Keywords: {database[best_summary]['keywords']}")
        else:
            print("No matches found.")

    # If the user wants to quit
    elif user_input == "quit":
        break

    # If the user's input is not recognized
    else:
        print("Command not recognized. Please enter 'summarize', 'search', or 'quit'.")

# Save the database to a pickle file
with open('summary_embeddings.pkl', 'wb') as f:
    pickle.dump(database, f)
