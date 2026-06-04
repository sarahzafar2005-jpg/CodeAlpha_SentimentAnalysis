import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

os.makedirs('images', exist_ok=True)

# --- Step 1: Load dataset ---
print("Loading dataset...")
df = pd.read_csv('test.csv', header=None, names=['label', 'title', 'review'])
print("Dataset loaded! Shape:", df.shape)

# Keep only 2000 rows for speed
df = df.head(2000)

# --- Step 2: Clean data ---
df['review'] = df['review'].fillna('')
df['title'] = df['title'].fillna('')
df['full_text'] = df['title'] + ' ' + df['review']

# --- Step 3: VADER Sentiment Analysis ---
print("\nRunning sentiment analysis...")
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = analyzer.polarity_scores(str(text))['compound']
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def get_score(text):
    return analyzer.polarity_scores(str(text))['compound']

df['sentiment'] = df['full_text'].apply(get_sentiment)
df['score'] = df['full_text'].apply(get_score)

# --- Step 4: Results ---
print("\n=== Sentiment Distribution ===")
print(df['sentiment'].value_counts())

print("\n=== Average Sentiment Score ===")
print(df['score'].mean().round(3))

print("\n=== Sample Positive Review ===")
print(df[df['sentiment']=='Positive']['review'].iloc[0][:200])

print("\n=== Sample Negative Review ===")
print(df[df['sentiment']=='Negative']['review'].iloc[0][:200])

# --- Chart 1: Sentiment Distribution ---
plt.figure(figsize=(7, 5))
colors = ['#2ecc71', '#e74c3c', '#95a5a6']
ax = sns.countplot(x='sentiment', data=df,
                   order=['Positive', 'Negative', 'Neutral'],
                   palette=colors)
ax.set_title('Amazon Reviews — Sentiment Distribution',
             fontsize=14, pad=12)
ax.set_xlabel('Sentiment')
ax.set_ylabel('Number of Reviews')
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width()/2, p.get_height()),
                ha='center', va='bottom', fontsize=12)
plt.tight_layout()
plt.savefig('images/01_sentiment_distribution.png', dpi=150)
plt.show()
print("Chart 1 done!")

# --- Chart 2: Sentiment Score Distribution ---
plt.figure(figsize=(9, 5))
sns.histplot(data=df, x='score', hue='sentiment',
             bins=30, kde=True,
             palette={'Positive':'#2ecc71',
                      'Negative':'#e74c3c',
                      'Neutral':'#95a5a6'})
plt.title('Sentiment Score Distribution',
          fontsize=14, pad=12)
plt.xlabel('Compound Score (-1 to +1)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('images/02_score_distribution.png', dpi=150)
plt.show()
print("Chart 2 done!")

# --- Chart 3: Sentiment by Original Label ---
plt.figure(figsize=(8, 5))
label_map = {1: 'Negative Review', 2: 'Positive Review'}
df['original_label'] = df['label'].map(label_map)
sns.countplot(x='original_label', hue='sentiment', data=df,
              palette={'Positive':'#2ecc71',
                       'Negative':'#e74c3c',
                       'Neutral':'#95a5a6'})
plt.title('VADER Sentiment vs Original Labels',
          fontsize=14, pad=12)
plt.xlabel('Original Label')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('images/03_vader_vs_labels.png', dpi=150)
plt.show()
print("Chart 3 done!")

print("\n=== Sentiment Analysis Complete! ===")