# embedsumBOT

EmbedSumBot is a tool that can help you summarize text and search for summaries based on text or keywords. It uses several models and libraries to perform different tasks, such as generating summaries with GPT4All1, embedding texts with Embed4All2, analyzing sentiments with nltk3, extracting keywords with rake_nltk, and measuring similarities with euclidean distance.

![embedsumBOT](https://github.com/EveryOneIsGross/embedsumBOT/assets/23621140/f9257814-e4a2-4c63-ada1-5788173c1c99)

---
## Usage:

To use EmbedSumBot, you need to run the script embedBOT_comparesearch.py in your terminal. You will be prompted to enter a command. There are three commands available:

**summarize:** This command will ask you to enter the name of a local .txt file that you want to summarize. It will generate a summary of the file content, embed it, analyze its sentiment, extract its keywords, and save them to a database.

**search:** This command will ask you to enter a text or a keyword that you want to search for. It will embed it, extract its keywords, and find the most similar summary in the database based on euclidean distance.

**quit:** This command will exit the program and save the database to a pickle file.
