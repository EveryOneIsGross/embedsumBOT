# embedsumBOT

EmbedSumBot is a tool that can help you summarize text and search for summaries based on text or keywords. It uses several models and libraries to perform different tasks, such as generating summaries with GPT4All1, embedding texts with Embed4All2, analyzing sentiments with nltk3, extracting keywords with rake_nltk, and measuring similarities with euclidean distance. Local and private. 💅

![embedsumBOT](https://github.com/EveryOneIsGross/embedsumBOT/assets/23621140/f9257814-e4a2-4c63-ada1-5788173c1c99)

## Search Algorithm:

This search algorithm uniquely integrates both semantic and keyword-based approaches to provide relevant results:

**Semantic Similarity:** Utilizes embeddings to capture the underlying meaning of the text. The cosine similarity between embeddings indicates how closely the content aligns in terms of meaning.

**Keyword Overlap:** Employs the Jaccard similarity to measure the overlap between sets of keywords extracted from the search query and the stored documents. This ensures that explicit terms from the user's query are taken into account.

**Combined Metrics:** The algorithm smartly averages the results from both the semantic and keyword-based approaches. This balance ensures that the search results are both semantically relevant and directly aligned with the user's query terms.

This dual approach aims to offer a more comprehensive and nuanced search experience, bridging the gap between deep semantic understanding and precise keyword matching.

---

## Usage:

To use EmbedSumBot, you need to run the script embedsumBOT.py in your terminal. You will be prompted to enter a command. There are three commands available:

**summarise:** followed by entering the name of a local .txt file that you want to summarize. It will generate a summary of the file content as chunks, embed it, analyze its sentiment, extract its keywords, and save them to a database.

**memorise:** This command allows you to enter text directly to be summarised and that summary remembered.

**search:** enter text or a keyword that you want to search for. It will embed it, extract its keywords, and find the most similar summary in the database based on euclidean distance.

**quit:** This command will exit the program and save the database to a pickle file.


