import json
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords

# Download Russian stopwords
nltk.download('stopwords', quiet=True)

def process_tenders(file_path):
    # Read the JSON file
    with open(file_path, 'r', encoding='utf-8') as f:
        tenders = json.load(f)

    # Combine all tender descriptions
    all_text = ' '.join(tenders.values())

    # Convert to lowercase and remove punctuation
    all_text = re.sub(r'[^\w\s]', '', all_text.lower())

    # Split into words
    words = all_text.split()

    # Remove Russian stopwords
    try:
        stop_words = set(stopwords.words('russian'))
    except LookupError:
        # If stopwords aren't downloaded, use a basic list
        stop_words = {'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб', 'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'жизнь', 'чуть', 'первый', 'тогда', 'тем', 'через', 'said', 'моя', 'yours', 'whose', 'whom'}

    # Filter out stopwords and short words
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]

    # Count word frequencies
    word_counts = Counter(filtered_words)

    # Print top 20 most frequent words
    print("Top 20 most frequent words in tenders:")
    for word, count in word_counts.most_common(20):
        print(f"{word}: {count}")

# Path to the tenders summary JSON file
tenders_file = 'resources/tenders_data/tenders_summary.json'

# Process the tenders
process_tenders(tenders_file)