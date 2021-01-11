
from trax import layers as tl
import re


from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os
import string
import numpy as np
import json



class NLPPredict:
    def __init__(self,):
        self.vocab = None
        self.dir = os.path.dirname(os.path.abspath(__file__))

        if ( os.path.isfile(self.dir+'/dictionary.json')):
            with open(self.dir+'/dictionary.json', 'r') as fp:
                self.vocab = json.load(fp)
        self.stemmer = PorterStemmer()
        self.stopwords_english = stopwords.words("english")
        self.model = self.classifier()

    def process_tweet(self, tweet):
    # remove stock market tickers like $GE
        tweet = re.sub(r'\$\w*', '', tweet)
        # remove old style retweet text "RT"
        tweet = re.sub(r'^RT[\s]+', '', tweet)
        # remove hyperlinks
        tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
        # remove hashtags
        # only removing the hash # sign from the word
        tweet = re.sub(r'#', '', tweet)
        # tokenize tweets
        tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
        tweet_tokens = tokenizer.tokenize(tweet)
        tweets_clean = []
        for word in tweet_tokens:
            if word not in self.stopwords_english and word not in string.punctuation:
                    stem_word = self.stemmer.stem(word)
                    tweets_clean.append(stem_word)
        return tweets_clean


    def tweet_to_tensor(self, tweet, vocab, unk_token='__UNK__'):
        tensor = []
        unk_ID = vocab[unk_token]
        tweet = self.process_tweet(tweet)
        for word in tweet:
            tensor.append(vocab.get(word, unk_ID))
            #tensors.append(word)
        return tensor

    def predict(self, sentence):
        self.model = self.classifier()
        self.model.init_from_file(self.dir+'/model.pkl.gz')
        inputs = np.array(self.tweet_to_tensor(sentence, vocab=self.vocab))
        
        # Batch size 1, add dimension for batch, to work with the model
        inputs = inputs[None, :]  
        
        # predict with the model
        preds_probs = self.model(inputs)
        
        # Turn probabilities into categories
        preds = int(preds_probs[0, 1] > preds_probs[0, 0])
        
        print("The Sentiment of the sentence: \n",sentence,'\n')
        if preds == 1:
            sentiment = 'positive'
            print("It is positive")
        else:
            sentiment = "negative"        
            print("I is negative")

        return preds, sentiment


    def classifier(self,vocab_size=None, embedding_dim=256, output_dim=2, mode='train'):
        if vocab_size is None:
            vocab_size = len(self.vocab)
        
        model = tl.Serial(
            tl.Embedding(vocab_size = vocab_size, d_feature = embedding_dim),   
            tl.Mean(axis = 1),
            tl.Dense(n_units=output_dim),
            tl.LogSoftmax()
        )
        
        return model


#sentence = "“10% of voters would have changed their vote if they knew about Hunter Biden.” Miranda Devine  @nypost @TuckerCarlson But I won anyway!"
#sentence = "STOCK MARKETS AT NEW ALL TIME HIGHS!!!"
#sentence = "Gee, what a surprise. Has anyone informed the so-called (says he has no power to do anything!) Governor @BrianKempGA & his puppet Lt. Governor @GeoffDuncanGA, that they could easily solve this mess, & WIN. Signature verification & call a Special Session. So easy!"
#p = NLPPredict()
#p.predict(sentence)