import mechanicalsoup
import nltk
import networkx as nx
import numpy as np




def request():
  browser = mechanicalsoup.Browser()
  pages = ["https://en.wikipedia.org/wiki/Tea", "https://www.teaclass.com/lesson_0101.html", "https://www.arborteas.com/what-is-tea/"]
  result = ""
  for page in pages:
    page = browser.get(page)
    tag = page.soup.select("p")
    for i in tag:
      result += i.text

  return result

def preprocess(text):
  # Split text into sentences
  sentences = nltk.sent_tokenize(text)
  # Tokenize each sentence into words
  words = [nltk.word_tokenize(sent) for sent in sentences]
  # Lowercase and remove punctuation
  words = [[w.lower() for w in sent if w.isalnum()] for sent in words]
  return sentences, words

# Define a function to compute sentence similarity based on word overlap
def similarity(sent1, sent2):
  # Get the set of unique words in each sentence
  set1 = set(sent1)
  set2 = set(sent2)
  # Compute the intersection and union of the sets
  intersect = set1.intersection(set2)
  union = set1.union(set2)
  # Return the Jaccard similarity coefficient
  return len(intersect) / len(union)

# Define a function to build a similarity matrix from a list of sentences
def build_matrix(sentences):
  # Initialize an empty matrix
  matrix = []
  # Loop through each pair of sentences
  for i in range(len(sentences)):
    row = []
    for j in range(len(sentences)):
      # Compute the similarity score and append to the row
      score = similarity(sentences[i], sentences[j])
      row.append(score)
    # Append the row to the matrix
    matrix.append(row)
  # Return the matrix as a numpy array
  return np.array(matrix)

# Define a function to extract the most important sentences from a text
def summarize(text, n):
  # Preprocess the text
  sentences, words = preprocess(text)
  # Build the similarity matrix
  matrix = build_matrix(words)
  # Create a graph from the matrix
  graph = nx.from_numpy_array(matrix)
  # Compute the PageRank scores for each sentence
  scores = nx.pagerank(graph)
  # Sort the sentences by their scores
  ranked = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
  # Extract the top n sentences as the summary
  summary = " ".join([s for _, s in ranked[:n]])
  return summary



summary = summarize(request(), n=4)

# Print the summary
print(summary)

