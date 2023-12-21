from bs4 import BeautifulSoup
import urllib.request
import urllib.parse
import urllib.error
import re
import requests
import pandas as pd
import time

import random

import nltk
#nltk.download('stopwords')
#nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def getLyrics(artist,song):
    url = f"https://www.azlyrics.com/lyrics/{artist}/{song}.html"
    page = requests.get(url)
    html = BeautifulSoup(page.content, 'html.parser')
    lyrics = html.select_one(".ringtone ~ div").get_text(strip=True, separator="\n")
    return lyrics


def get_Artist_Song(artist, song):
  artist = artist.replace(" ", "").lower()
  song = song.replace(" ", "").lower()
  artist = "".join(letter for letter in artist if letter.isalnum())
  song = "".join(letter for letter in song if letter.isalnum())
  return artist, song


def processData(lyric):
    lyric = lyric.replace("\n", "")
    return lyric


def getEmo(sen):
    if sen == "happy":
        return 1
    elif sen == "angry":
        return 2
    elif sen == "sad":
        return 3
    elif sen == "relaxed":
        return 4


def getSen(sen):
    if sen == "happy" or sen == "relaxed":
        return 0
    elif sen == "angry" or sen == "sad":
        return 1


def processLy(songs):
    cat = songs[0]
    lyrics = songs[2:]
    tokens = word_tokenize(lyrics)
    tokens = [a_word.lower() for a_word in tokens]
    tokens = [a_word for a_word in tokens if a_word.isalpha()]
    tokens = [a_word for a_word in tokens if a_word not in STOPWORDS]
    tokens = [a_word for a_word in tokens if a_word not in string.punctuation]
    tokens = [lemmatizer.lemmatize(a_word) for a_word in tokens]
    tokens = [stemmer.stem(a_word) for a_word in tokens]
    new_songs = cat + " " + " ".join(tokens) + "\n"
    return new_songs


if __name__ == "__main__":
    df = pd.read_csv('./MoodyLyrics.csv')
    file1 = open("emotion.txt", "a")
    file2 = open("sentiment.txt", "a")
    error = 0
    lyrics = ""
    for ind in df.index:
        artist, song = get_Artist_Song(df['Artist'][ind], df['Song'][ind])
        song = processLy(song)
        try:
            lyrics = getLyrics(artist, song)
        except Exception as e:
            print(e)
            print("NOTHING Found")
            error = 1
        time.sleep(5)
        if error:
            print(artist)
            print(song)
            error = 0
            continue
        lyrics = processData(lyrics)
        emotion = getEmo(df['Emotion'][ind])
        output1 = str(emotion) + " " + lyrics
        file1.write(output1)

        sense = getSen(df['Emotion'][ind])
        output2 = str(sense) + " " + lyrics
        file2.write(output2)

    file1.close()
    file2.close()

