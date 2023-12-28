# NLP: Emotion and Sentimental Analysis For Songs 
In this project, we implement two large CNN models for emotion recognition and sentiment analysis for song’s lyrics. We collected the lyrics using a web scraper, and pre-processed them using several different techniques, and used them coupled
with MoodyLyrics classification tags to train our models. We achieved a high accuracy and F1 scores for both emotion and sentiment analysis.

## Dataset
**Web Scrapper:** We obtain our primary dataset from the MoodyLyrics5 dataset which contains the song’s name, artist, and emotion. However, we still needed to collect the lyrics of the songs that were included in MoodyLyrics. For this purpose, we implemented a web scraper that searches for lyrics on azlyrics.com and genius.com. We then collected the lyrics for roughly 2,000 songs and stored our data in two files, emotion.txt and sentiment.txt, to be used for emotion recognition and sentiment analysis respectively.

**MoodyLyrics** corpus of songs is a large dataset of roughly 2,000 songs, dividing them into four categories: Happy, Sad, Angry, and Relaxed. For the sentiment analysis, we define the “Happy” and “Relaxed” categories as “Positive” categories and “Sad” and ” Anger” as “Negative”. MoodyLyrics dataset has equal distribution of all 4 categories (see Fig. 1), hence this eliminates the biases that may be related to the size of the corpus.

![emotion dataset pie chart](https://github.com/Sagarnandeshwar/Lyrics_Emotion_Recognition/blob/main/image/Emopie.png)
![sentiment dataset pie chart](https://github.com/Sagarnandeshwar/Lyrics_Emotion_Recognition/blob/main/image/snetimentpie.png)

**Embedding**: We used GloVe, which are Global Vectors for Word Representation. It help to map words into a meaningful space where the distance between words is related to semantic similarity.

**Preprocessing**: In the preprocessing phase, several important steps are employed to prepare the lyrics data for emotion classification. 
-  Tokenization is applied using NLTK’s word_tokenize, breaking down sentences into individual words, for instance, transforming "I like a cat" into ["I", "like", "a", "cat"].
-  Detokenization follows to convert lists of tokens back into continuous text strings, using NLTK’s TreebankWordDetokenizer, ensuring the coherent representation of the processed data.
- Stop words, common but semantically less meaningful words like "and" or "the", are also removed using NLTK stopwords to improve the signal strength of the important words of the text. Punctuation is eliminated using Python’s string.punctuation.
- Lemmatization is implemented with NLTK’s WordNetLemmatizer to reduce words to their base forms (lemmas), aiming to standardize the vocabulary.
- Stemming is performed by utilizing NLTK’s PorterStemmer and involves removing morphological affixes to retain only the word stems.
- Before being passed on to our model, lyrical text documents are converted into feature vectors using sklearn’s CountVectorizer, representing the data as a matrix of token counts, and normalization using sklearn’s
- TfidfTransformer to transform the count matrix into a normalized TF (Term frequency) or TF-IDF (Term frequence-Inverse Document Frequency) representation. At the end of these preprocessing steps, we have prepared our raw lyrics data for downstream emotion classification analysis.

We performed a couple of small-scale experiments to decide what combination of these preprocessing steps is more beneficial for our purpose. We ended up using removal of stop-words, removal of punctuations, and lemmatizer, and avoided
stemmer and POS tagging based on our experimental observations, as well as prior knowledge that we have acquired (most notably from the first and second programming assignments of the course).

## Models
**Emotion Classification Model:** For the emotion classification, we implemented a large neural network (NN) that consists of several layers, as listed below, in order:
- Dropout layer: Applies dropout to the input features. The Dropout layer randomly sets input units to 0 with a frequency of the rate at each step during training time, helping to prevent overfitting. We have selected the rate to be 0.2.
- Embedding layer: Turns positive integers (indexes) into dense vectors of fixed size.
- 1D convolution layers: This layer creates a convolution kernel that is convolved with the input layer over a single spatial (or temporal) dimension to produce a tensor of outputs. We used the ReLU (Rectified Linear Unit) activation function for 1D convolution layers.
- GlobalMaxPooling1D layer: Performs global max pooling operation for temporal data.
- Dense layers: 3 or 4 densely connected NN layers are the final layers of our NN. We again used the ReLU (Rectified Linear Unit) activation function for the Dense layers. However, for the last dense layer, we used the softmax function which has the same shape as the input.

We compiled the model with the Adam optimizer, which is a stochastic gradient descent (SGD) method that is based on adaptive estimation of first-order and second-order moments, and binary cross-entropy loss, which measures the difference between two probability distributions: the true distribution and the predicted distribution. We trained the model with 10 epochs and a batch size of 16.

![](https://github.com/Sagarnandeshwar/Lyrics_Emotion_Recognition/blob/main/image/emodiagram.png)
![](https://github.com/Sagarnandeshwar/Lyrics_Emotion_Recognition/blob/main/image/emo%20model.png)

**Sentiment Classification:** Similar to our emotion classification model, here we also used a neural network structure with several layers as listed before. For the last dense layer we used sigmoid function instead of softmax, in order to get binary output. The training was also performed by the Adam Optimizer. We trained the model with 10 epochs and a batch size of 16. The slight differences between our models - mainly in layer numbers - can be seen in Fig. 2. Further details are available in Fig. 3.

![](https://github.com/Sagarnandeshwar/Lyrics_Emotion_Recognition/blob/main/image/Sentimentdia.png)
![](https://github.com/Sagarnandeshwar/Lyrics_Emotion_Recognition/blob/main/image/sendtiment.png)

## Result 

**Emotion Model**
![](https://github.com/Sagarnandeshwar/Lyrics_Emotion_Recognition/blob/main/image/Emo%20accuracy%20mat.png)
![](https://github.com/Sagarnandeshwar/Lyrics_Emotion_Recognition/blob/main/image/Emo%20f1%20mat.png)
![](https://github.com/Sagarnandeshwar/Lyrics_Emotion_Recognition/blob/main/image/Emo%20confusion%20mat.png)
![](https://github.com/Sagarnandeshwar/Lyrics_Emotion_Recognition/blob/main/image/Emo%20loss%20mat.png)

**Sentiment Model**
![](https://github.com/Sagarnandeshwar/Lyrics_Emotion_Recognition/blob/main/image/Sentiment%20accuracy%20mat.png)
![](https://github.com/Sagarnandeshwar/Lyrics_Emotion_Recognition/blob/main/image/Sentiment%20f1%20mat.png)
![](https://github.com/Sagarnandeshwar/Lyrics_Emotion_Recognition/blob/main/image/Sentiment%20confusion%20mat.png)
![](https://github.com/Sagarnandeshwar/Lyrics_Emotion_Recognition/blob/main/image/Sentiment%20loss%20mat.png)

Our Results can be seen in Figures 4 and 5. For emotion classification, our model can perform with an accuracy of %90 on the training set and %80 on the test set on the last epoch, which is higher than some of the related works that we have reviewed.
From a critical point of view, this could be a result of not having enough data points (despite having almost 2000 songs) for our training, which could lead to problems such as overfitting and/or bias. However, we could also argue that our model is
performing quite well because of the way we have structured our neural network. For sentiment analysis, our model performs well too, with an accuracy of higher than %95 on the training set and more than %90 on the test set. The same points that
we made for our emotion classification model hold for this model as well. As for the F1 score, we get 0.80 for the emotion model and 0.93 for the sentiment model. From the confusion matrix for emotion classification, we observe that for most of the
songs we accurately classify, the diagonals have high values. Similarly for sentiment models, we have high values in diagonal, confirming that the model has high accuracy.

## Conclusion 
In this project, we implemented a large Convolutional Neural Network (CNN) model for emotion recognition and sentiment analysis for song lyrics. We collected the lyrics using a web scraper that we implemented ourselves and pre-processed the data
for model training. In our results, we demonstrate that our models were capable of achieving high accuracies for both of the models. One potential reason behind this could be having a relatively large dataset of 2000 songs, compared to just 4 categories
of emotions, for the emotion recognition model. The same principle also applies to our second model with only 2 categories of sentiments for sentiment analysis. The way that we have decided to construct our CNN network model also provides us with a 
huge number of trainable parameters because of the sheer size of the network, which probably further helps in the analysis of the data and learning the patterns. Despite starting quite late and not having much time, we used a paid GPU to train our
models in a feasible amount of time. Finally, we believe the preprocessing decisions that we have made - such as lemmatization and removing the stop words - have also played a key role in feature extraction, which has led to a significant performance of
our models.

For future work, we were hoping for our model to incorporate the data about the artist. We believe that most of the artists
have a genre that they follow regularly. This would be a piece of very beneficial information for both emotion and sentiment
analysis.







