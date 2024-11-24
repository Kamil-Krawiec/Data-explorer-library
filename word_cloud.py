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