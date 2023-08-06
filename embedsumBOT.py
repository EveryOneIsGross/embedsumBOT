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

def chunk_text(text, max_tokens=250):
    """Split the text into chunks of approximately max_tokens tokens."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        if current_length + len(word) <= max_tokens:
            current_chunk.append(word)
            current_length += len(word)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
    chunks.append(' '.join(current_chunk))
    return chunks

def filter_repeated_words_keywords(keywords):
    """Filter out keyword phrases that contain repeated words."""
    filtered_keywords = []
    for keyword in keywords:
        words = keyword.split()
        if len(words) == len(set(words)):
            filtered_keywords.append(keyword)
    return filtered_keywords

def cosine_similarity(c1, c2):
    set1 = set(c1)
    set2 = set(c2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if not union:
        return 0.0
    else:
        return float(len(intersection)) / len(union)

# Load the database from a pickle file
try:
    with open('summary_embeddings.pkl', 'rb') as f:
        database = pickle.load(f)
except FileNotFoundError:
    database = {}

while True:
    user_input = input("Enter a command: ")

    # If the user wants to summarise a local .txt file
    if user_input.startswith("summarise"):
        file_name = user_input[len("summarise"):].strip()
        if os.path.exists(file_name) and file_name.endswith(".txt"):
            with open(file_name, "r") as f:
                document = f.read()
                
                # Extract keywords from the original chunk
                keyword_extractor.extract_keywords_from_text(document)
                data = {}
                original_chunk_keywords = Counter(keyword_extractor.get_ranked_phrases())
                # Split the document into chunks
                chunks = chunk_text(document)
            
                for chunk in chunks:
                    summary = gpt_model.generate(f"Summarise: {chunk}", max_tokens=150, temp=0.3)
                    embedding = embedder.embed(summary)
                    sentiment = sentiment_analyzer.polarity_scores(summary)
                    # Combine keywords from the original chunk and the summary
                    combined_keywords = original_chunk_keywords + Counter(keyword_extractor.get_ranked_phrases())
                    data['keywords'] = combined_keywords.most_common()  # Store combined keywords in the database
                    keywords = keyword_extractor.get_ranked_phrases()
                    keywords = filter_repeated_words_keywords(keywords)
                    
                    database[summary] = {
                        'embedding': embedding,
                        'sentiment': sentiment,
                        'keywords': Counter(keywords)
                    }
                    print(f"Summary: {summary}")
                    print(f"Sentiment: {sentiment}")
                    #print(f"Keywords: {keywords}")
        else:
            print("Invalid file name. Please enter a valid .txt file name.")

    # If the user wants to memorise and summarise typed text
    elif user_input.startswith("memorise"):
        typed_text = user_input[len("memorise"):].strip()
        summary = gpt_model.generate(f"Summarise: {typed_text}", max_tokens=500, temp=0.3)
        embedding = embedder.embed(summary)
        sentiment = sentiment_analyzer.polarity_scores(summary)
        keyword_extractor.extract_keywords_from_text(summary)
        keywords = keyword_extractor.get_ranked_phrases()
        database[summary] = {
            'embedding': embedding,
            'sentiment': sentiment,
            'keywords': Counter(keywords)
        }
        print(f"Summary: {summary}")
        print(f"Sentiment: {sentiment}")
        #print(f"Keywords: {keywords}")

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
            #print(f"Keywords: {database[best_summary]['keywords']}")
        else:
            print("No matches found.")

    # If the user wants to quit
    elif user_input == "quit":
        break

    # If the user's input is not recognized
    else:
        print("Command not recognized. Please enter 'summarise', 'memorise', 'search', or 'quit'.")

# Save the database to a pickle file
with open('summary_embeddings.pkl', 'wb') as f:
    pickle.dump(database, f)
