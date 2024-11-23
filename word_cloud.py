from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from collections import Counter
import re

def word_cloud(text_data):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off') 
    plt.show()

def word_frequency_pie_chart(text, include_others=True, n=10):
    words = [word for word in nltk.word_tokenize(text) if re.match(r'\w+', word)]
    word_counts = Counter(words)

    most_common = word_counts.most_common(n)
    total_words = sum(word_counts.values())

    if include_others:
        other_count = total_words - sum(count for word, count in most_common)
        most_common.append(("Others", other_count))

    labels = [word for word, _ in most_common]
    sizes = [count / total_words * 100 for _, count in most_common]

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Word Frequency Distribution (Pie Chart)')
    plt.show()

def word_frequency_bar_chart(text, include_others=True, n=10):
    words = [word for word in nltk.word_tokenize(text) if re.match(r'\w+', word)]
    word_counts = Counter(words)

    most_common = word_counts.most_common(n)
    total_words = sum(word_counts.values())

    if include_others:
        other_count = total_words - sum(count for word, count in most_common)
        most_common.append(("Others", other_count))

    labels = [word for word, _ in most_common]
    sizes = [count / total_words * 100 for _, count in most_common]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, sizes)
    plt.xlabel('Words')
    plt.ylabel('Percentage of Occurrences')
    plt.title('Word Frequency Distribution (Bar Chart)')
    plt.xticks(rotation=45)
    plt.show()

text_data = """
Text mining, text data mining (TDM) or text analytics is the process of deriving high-quality information from text. It involves "the discovery by computer of new, previously unknown information, by automatically extracting information from different written resources."[1] Written resources may include websites, books, emails, reviews, and articles. High-quality information is typically obtained by devising patterns and trends by means such as statistical pattern learning. According to Hotho et al. (2005), there are three perspectives of text mining: information extraction, data mining, and knowledge discovery in databases (KDD).[2] Text mining usually involves the process of structuring the input text (usually parsing, along with the addition of some derived linguistic features and the removal of others, and subsequent insertion into a database), deriving patterns within the structured data, and finally evaluation and interpretation of the output. 'High quality' in text mining usually refers to some combination of relevance, novelty, and interest. Typical text mining tasks include text categorization, text clustering, concept/entity extraction, production of granular taxonomies, sentiment analysis, document summarization, and entity relation modeling (i.e., learning relations between named entities).

Text analysis involves information retrieval, lexical analysis to study word frequency distributions, pattern recognition, tagging/annotation, information extraction, data mining techniques including link and association analysis, visualization, and predictive analytics. The overarching goal is, essentially, to turn text into data for analysis, via the application of natural language processing (NLP), different types of algorithms and analytical methods. An important phase of this process is the interpretation of the gathered information.

A typical application is to scan a set of documents written in a natural language and either model the document set for predictive classification purposes or populate a database or search index with the information extracted. The document is the basic element when starting with text mining. Here, we define a document as a unit of textual data, which normally exists in many types of collections.[3]
"""

word_cloud(text_data)
word_frequency_pie_chart(text_data, include_others=True, n=10)
word_frequency_bar_chart(text_data, include_others=True, n=10)
