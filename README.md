# Basic Search Engine

This project is a basic implementation of a search engine in Python. It uses various Natural Language Processing (NLP) techniques to process and analyze text data, and then uses this processed data to perform search operations.

## Code Explanation

The code in `main.py` can be divided into several parts:

1. **Tokenization**: The code reads text files from a directory, tokenizes the content (breaks the text down into individual words), removes stop words (common words that do not contain important meaning), and stems the words (reduces them to their root form).

2. **Positional Index**: The code creates a positional index, which is a dictionary where each key is a word and the value is a list containing the frequency of the word and a dictionary of the document numbers and their positions in the document.

3. **Phrase Query**: The code takes a user input query, stems the query, and then finds the documents where the words in the query appear in the same order.

4. **Term Frequency (TF)**: The code calculates the term frequency, which is the number of times a word appears in a document.

5. **TF-Weight**: The code calculates the TF-Weight, which is 1 plus the logarithm of the term frequency.

6. **Inverse Document Frequency (IDF)**: The code calculates the inverse document frequency, which measures how important a term is. It is calculated as the logarithm of the total number of documents divided by the number of documents containing the term.

7. **TF-IDF**: The code calculates the TF-IDF, which is the product of the term frequency and the inverse document frequency. This gives a weight to each word in each document.

8. **Document Length**: The code calculates the length of each document, which is the square root of the sum of the squares of the TF-IDF values.

9. **Normalized TF-IDF**: The code normalizes the TF-IDF values by dividing each TF-IDF value by the document length.

10. **Similarity between Query and Each Document**: The code calculates the similarity between the query and each document using the cosine similarity method.

## Equations

1. **TF-Weight**: w_tf = 1 + log(tf)
2. **IDF**: idf = log(n / df)
3. **TF-IDF**: tf-idf = tf-weight * idf
4. **Document Length**: doc_length = sqrt(sum(tf-idf^2))
5. **Normalized TF-IDF**: normalized_tf-idf = tf-idf / doc_length
6. **Cosine Similarity**: cosine_similarity = sum(query * matched_docs) / (query_length * doc_length)

## Dependencies

This project uses the following Python libraries:

- nltk
- os
- pandas
- math
- numpy
- warnings
- natsort

## Usage

To use this search engine, run the `main.py` file and enter your search query when prompted.