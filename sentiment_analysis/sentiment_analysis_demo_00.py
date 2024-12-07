######################## Demo 01
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download necessary NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('vader_lexicon', quiet=True)

url_data = "https://raw.githubusercontent.com/kirenz/twitter-tweepy/main/tweets-obama.csv"
df = pd.read_csv(url_data)	# Load the dataset

# Preprocess text
df['text'] = df['text'].astype(str).str.lower()

# Tokenize text
regexp = RegexpTokenizer('\w+')
df['text_token'] = df['text'].apply(regexp.tokenize)

# Remove stopwords
stopwords_list = nltk.corpus.stopwords.words("english")
my_stopwords = ['https']
stopwords_list.extend(my_stopwords)
df['text_token'] = df['text_token'].apply(lambda x: [item for item in x if item not in stopwords_list])

# Rejoin tokens into a string
df['text_string'] = df['text_token'].apply(lambda x: ' '.join([item for item in x if len(item) > 2]))

# Lemmatization
wordnet_lem = WordNetLemmatizer()
df['text_string_lem'] = df['text_string'].apply(wordnet_lem.lemmatize)

# Sentiment analysis
analyzer = SentimentIntensityAnalyzer()
df['polarity'] = df['text_string_lem'].apply(lambda x: analyzer.polarity_scores(x))

# Split polarity into columns
df = pd.concat([df.drop(['polarity'], axis=1), df['polarity'].apply(pd.Series)], axis=1)

# Plotting the Sentiment Analysis (Polarity Scores)
plt.figure(figsize=(10, 6))
plt.hist(df['compound'], bins=20, color='skyblue', edgecolor='black')
plt.title('Sentiment Distribution of Obama Tweets')
plt.xlabel('Compound Sentiment Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

# Print some additional insights
print("Sentiment Score Statistics:")
print(df['compound'].describe())

# Count tweets by sentiment
def categorize_sentiment(score):
    if score > 0.05:
        return 'Positive'
    elif score < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment_category'] = df['compound'].apply(categorize_sentiment)
sentiment_counts = df['sentiment_category'].value_counts()
print("\nSentiment Distribution:")
print(sentiment_counts)

"""
Sentiment Score Statistics:
count    5.000000
mean     0.205920
std      0.625353
min     -0.648600
25%     -0.128000
50%      0.226300
75%      0.648600
max      0.931300
Name: compound, dtype: float64

Sentiment Distribution:
sentiment_category
Positive    3
Negative    2
Name: count, dtype: int64
"""

######################## Demo 02
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

def preprocess_text(text):
    """
    Preprocess text by lowercasing, tokenizing, removing stopwords, and lemmatizing
    """
    # Convert to lowercase
    text = str(text).lower()
    
    # Tokenize
    regexp = RegexpTokenizer(r'\w+')
    tokens = regexp.tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english') + ['https', 'rt'])
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

def main():
    url_data = "https://raw.githubusercontent.com/kirenz/twitter-tweepy/main/tweets-obama.csv"
    df = pd.read_csv(url_data)	# Load the dataset
    print("Dataset loaded. Shape:", df.shape)

    # Preprocess text
    df['processed_text'] = df['text'].apply(preprocess_text)

    # Sentiment Analysis
    analyzer = SentimentIntensityAnalyzer()
    df['sentiment'] = df['processed_text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    
    # Create multi-plot figure
    plt.figure(figsize=(15, 10))
    
    # Sentiment Distribution Subplot
    plt.subplot(2, 1, 1)
    plt.hist(df['sentiment'], bins=20, color='skyblue', edgecolor='black')
    plt.title('Sentiment Distribution of Obama Tweets')
    plt.xlabel('Compound Sentiment Score')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Emotion Classification
    try:
        # Load pre-trained model for emotion classification
        tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
        model = TFAutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
        
        # Prepare inputs for all tweets
        inputs = tokenizer(df['processed_text'].tolist(), return_tensors="tf", padding=True, truncation=True)
        
        # Predict emotions
        outputs = model(inputs)
        predictions = tf.nn.softmax(outputs.logits, axis=-1)
        
        # Emotion labels
        emotion_labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
        
        # Get emotion predictions
        predicted_emotions = tf.argmax(predictions, axis=-1).numpy()
        emotion_confidences = np.max(predictions.numpy(), axis=1)
        
        # Emotion Distribution Subplot
        plt.subplot(2, 1, 2)
        emotion_counts = [np.sum(predicted_emotions == i) for i in range(len(emotion_labels))]
        plt.bar(emotion_labels, emotion_counts, color='lightgreen')
        plt.title('Emotion Distribution of Obama Tweets')
        plt.xlabel('Emotions')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.grid(True)

        # Tight layout and save
        plt.tight_layout()
        plt.savefig('obama_tweets_analysis.png')
        
        # Print some statistics
        print("\nEmotion Distribution:")
        for label, count in zip(emotion_labels, emotion_counts):
            print(f"{label}: {count}")
        
        print("\nSentiment Statistics:")
        print(df['sentiment'].describe())
    
    except Exception as e:
        print("Error in emotion classification:", e)
        print("Proceeding without emotion classification.")

if __name__ == "__main__":
    main()

######################## Demo 03
def analyze_tweets(csv_path):
    """
    Analyze tweets for sentiment and emotion
    """
    # Load the dataset
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

    # Preprocess text
    df['processed_text'] = df['text'].apply(preprocess_text)

    # Sentiment Analysis
    analyzer = SentimentIntensityAnalyzer()
    df['sentiment_score'] = df['processed_text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

    # Categorize sentiment
    def categorize_sentiment(score):
        if score > 0.05:
            return 'Positive'
        elif score < -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    df['sentiment_category'] = df['sentiment_score'].apply(categorize_sentiment)

    # Emotion Classification
    try:
        # Load pre-trained model for emotion classification
        tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
        model = TFAutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
        
        # Prepare inputs for all tweets
        inputs = tokenizer(df['processed_text'].tolist(), return_tensors="tf", padding=True, truncation=True)
        
        # Predict emotions
        outputs = model(inputs)
        predictions = tf.nn.softmax(outputs.logits, axis=-1)
        
        # Emotion labels
        emotion_labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
        
        # Get emotion predictions
        predicted_emotions = tf.argmax(predictions, axis=-1).numpy()
        df['predicted_emotion'] = [emotion_labels[e] for e in predicted_emotions]

        # Visualization
        plt.figure(figsize=(15, 10))

        # Sentiment Distribution Subplot
        plt.subplot(2, 1, 1)
        df['sentiment_score'].hist(bins=20, color='skyblue', edgecolor='black')
        plt.title('Sentiment Distribution of Tweets')
        plt.xlabel('Compound Sentiment Score')
        plt.ylabel('Frequency')
        plt.grid(True)

        # Emotion Distribution Subplot
        plt.subplot(2, 1, 2)
        emotion_counts = df['predicted_emotion'].value_counts()
        emotion_counts.plot(kind='bar', color='lightgreen')
        plt.title('Emotion Distribution of Tweets')
        plt.xlabel('Emotions')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the plot
        plt.savefig('tweet_analysis.png')

        # Print summary statistics
        print("\nSentiment Distribution:")
        print(df['sentiment_category'].value_counts())

        print("\nEmotion Distribution:")
        print(df['predicted_emotion'].value_counts())

        return df

    except Exception as e:
        print(f"Error in emotion classification: {e}")
        return df

def main():
    csv_path = url_data
    result_df = analyze_tweets(csv_path)
    
    if result_df is not None:
        print("\nAnalysis complete. Check 'tweet_analysis.png' for visualizations.")

if __name__ == "__main__":
    main()

"""
Sentiment Distribution:
sentiment_category
Positive    3
Negative    2
Name: count, dtype: int64

Emotion Distribution:
predicted_emotion
joy        2
fear       1
anger      1
sadness    1
Name: count, dtype: int64

Analysis complete. Check 'tweet_analysis.png' for visualizations.
"""
