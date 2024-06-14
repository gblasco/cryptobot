import praw
import pandas as pd
from textblob import TextBlob
import re
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
load_dotenv()  # This loads the environment variables from a .env file

reddit = praw.Reddit(client_id=os.getenv('REDDIT_API_KEY'),
                     client_secret=os.getenv('REDDIT_API_SECRET'),
                     user_agent='script:scraper:v1.0 (by /u/********)', # sustituir * por Username en reddit
                     username=os.getenv('REDDIT_USER'),
                     password=os.getenv('REDDIT_PASSWD'))

# limpiar texto
def prepare_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # Eliminar urls en los mensajes
    text = re.sub(r'@\S+', '', text)  # Eliminar menciones en el mensaje
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Eliminar caracteres especiales y numeros
    return text

# polaridad
def addPolarity(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Extraer los ultimos 100 posts de reddit en el canal de r/bitcoin
btc_reddit = reddit.subreddit('bitcoin')
posts = btc_reddit.new(limit=100)
data = {'Date': [], 'Time': [], 'User': [], 'Title': [], 'Text': [], 'Sentiment': [], 'Polarity': []}

for post in posts:
    datepost = post.created_utc
    title = prepare_text(post.title)
    text = prepare_text(post.selftext)
    # Calcular la polaridad
    polarity_title = addPolarity(title)
    polarity_text = addPolarity(text)
    if text == '':
        polarity = polarity_title
    else:
        polarity = polarity_text
    
    if polarity > 0:
        str_sentiment = 'Positive'
    elif polarity == 0:
        str_sentiment = 'Neutral'
    else:
        str_sentiment = 'Negative'
    
    if post.author:
        user = post.author.name
    else:
        user = 'Unknown'
    
    # Obtener la fecha y hora del post
    time = pd.to_datetime(datepost, unit='s').time()
    date = pd.to_datetime(datepost, unit='s').date()
    data['Date'].append(date)
    data['Time'].append(time)
    data['User'].append(user)
    data['Title'].append(title)
    data['Text'].append(text)
    data['Sentiment'].append(str_sentiment)
    data['Polarity'].append(polarity)

df = pd.DataFrame(data)
df.to_csv('sentiments_bitcoin.csv', index=False)
polarity_avg = df['Polarity'].mean()
print(f'[LOG]: Media de los sentimientos!!: {polarity_avg:.4f}')

# plt.hist(df['Polarity'], bins=20, edgecolor='black')
# plt.title('Analisis de sentimientos del canal r/bitcoin')
# plt.xlabel('Polaridad')
# plt.ylabel('Frecuencia')
# plt.show()
