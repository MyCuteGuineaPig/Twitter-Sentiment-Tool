
import trax
import trax.fastmath.numpy as fastnp
from trax import layers as tl
from trax.supervised import training
import numpy as np

import random
import contextlib

import os

import random
import re
import string
import numpy as np

import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords, twitter_samples
from nltk.stem import PorterStemmer
import os
import json

class NLPClass:
    def __init__(self):
        self.dir =  os.path.dirname(os.path.abspath(__file__))
        nltk.data.path.append(self.dir+'/../env/nltk_data/')


        if not( os.path.isfile(self.dir+'/../env/nltk_data/corpora/twitter_samples/positive_tweets.json') and os.path.isfile(self.dir+'/../env/nltk_data/corpora/twitter_samples/negative_tweets.json')):
            nltk.download('twitter_samples',download_dir=self.dir+'/../env/nltk_data/')

        if not  os.path.isfile('/../env/nltk_data/corpora/stopwords/english'):
            nltk.download('stopwords',download_dir=self.dir+'/../env/nltk_data/')


        self.stemmer = PorterStemmer()
        self.stopwords_english = stopwords.words("english")
        self.vocab = None
        self.model = None
        if ( os.path.isfile(self.dir+'/dictionary.json')) and os.path.isfile(self.dir+'/model.pkl.gz'):
            with open(self.dir+'/dictionary.json', 'r') as fp:
                self.vocab = json.load(fp)
            self.model = self.classifier()
            self.model.init_from_file(self.dir+'/model.pkl.gz')



    def load_tweets(self):
        all_positive_tweets = twitter_samples.strings("positive_tweets.json")
        all_negative_tweets = twitter_samples.strings("negative_tweets.json")
        return all_positive_tweets,all_negative_tweets


    def process_tweet(self,tweet):
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

    def clean(self,X):
        clean_tweet = []
        for tweet in X:
            clean_tweet.append(self.process_tweet(tweet))
        return clean_tweet


    def build_vocabulary(self,X):
        Vocab = {'__PAD__': 0, '__</e>__': 1, '__UNK__': 2} 
        for tweet in X:
            tweet_clean = self.process_tweet(tweet)
            for word in tweet_clean:
                if word not in Vocab:
                    Vocab[word] = len(Vocab)
        with open(self.dir+'/dictionary.json', 'w') as fp:
            json.dump(Vocab, fp, indent=4)
        return Vocab

    def tweet_to_tensor(self,tweet, vocab, unk_token='__UNK__'):
        tensor = []
        unk_ID = vocab[unk_token]
        tweet = self.process_tweet(tweet)
        for word in tweet:
            tensor.append(vocab.get(word, unk_ID))
            #tensors.append(word)
        return tensor

    def data_generator(self,data_pos, data_neg, batch_size, loop, vocab, shuffle):
        assert batch_size % 2 == 0
        n_to_take = batch_size // 2
        
        pos_index = 0 
        neg_index = 0
        pos_index_lines = list(range(len(data_pos)))
        neg_index_lines = list(range(len(data_neg)))
        if shuffle:
            random.shuffle(pos_index_lines)
            random.shuffle(neg_index_lines)
        stop = False
        
        while not stop:
            batch = []
            for i in range(n_to_take):
                if pos_index >= len(data_pos):
                    if not loop:
                        stop = True
                        break
                
                    pos_index = 0
                    if shuffle:
                        random.shuffle(pos_index_lines)
                tweet = self.tweet_to_tensor(data_pos[pos_index_lines[pos_index]],vocab)
                batch.append(tweet)
                pos_index += 1
                
            for i in range(n_to_take):
                if neg_index >= len(data_neg):
                    if not loop:
                        stop = True
                        break
                
                    neg_index = 0
                    if shuffle:
                        random.shuffle(neg_index_lines)
                    
                tweet = self.tweet_to_tensor(data_neg[neg_index_lines[neg_index]],vocab)
                batch.append(tweet)
                neg_index += 1
            if stop:
                break
            
            pos_index += n_to_take
            neg_index += n_to_take
            max_len = max([len(t) for t in batch])
            
            tensor_pad_l = [ b + [0,]*(max_len - len(b)) for b in batch]
            tensor_pad_l = np.array(tensor_pad_l)
            targets = [1,]*n_to_take + [0,]*n_to_take
            targets = np.array(targets)
            
            weights = np.ones_like(targets)
        
            yield tensor_pad_l, targets, weights
        

    def classifier(self, vocab_size=None, embedding_dim=256, output_dim=2, mode='train'):
        if vocab_size is None:
            vocab_size = len(self.vocab)
        
        model = tl.Serial(
            tl.Embedding(vocab_size = vocab_size, d_feature = embedding_dim),   
            tl.Mean(axis = 1),
            tl.Dense(n_units=output_dim),
            tl.LogSoftmax()
        )
        
        return model


    def training(self,steps, ratio):
        if os.path.isfile(self.dir+'/metrics.txt'):
            os.remove(self.dir+"/metrics.txt")
        if os.path.isfile(self.dir+'/finish.txt'):
            os.remove(self.dir+"/finish.txt")
        if os.path.isfile(self.dir+'/model.pkl.gz'):
            os.remove(self.dir+"/model.pkl.gz")

        all_positive_tweets,all_negative_tweets = self.load_tweets()
        
        tot_positive, tot_negative = len(all_positive_tweets), len(all_negative_tweets)
        count_pos = int(ratio*tot_positive)
        count_neg = int(ratio*tot_negative)

        train_pos, train_neg = all_positive_tweets[:count_pos],  all_negative_tweets[:count_neg]
        train_x = np.array(train_pos + train_neg)

        train_y = np.array([1 for _ in range(count_pos)] + [0 for _ in range(count_neg)] )

        val_pos, val_neg = all_positive_tweets[count_pos:], all_negative_tweets[count_neg:]
        val_x =  np.array(val_pos+ val_neg)
        val_y = np.array([1,]*len(val_pos)+ [0,]*len(val_neg) )


        self.vocab = self.build_vocabulary(train_x)

        def train_generator( batch_size, shuffle = False):
            return self.data_generator(train_pos, train_neg, batch_size, True, self.vocab, shuffle)
    
        def val_generator(batch_size, shuffle = False):
            return self.data_generator(val_pos, val_neg, batch_size, True, self.vocab, shuffle)
    
        def test_generator(batch_size, shuffle = False):
            return self.data_generator(val_pos, val_neg, batch_size, False, self.vocab, shuffle)





        random.seed(271)
        batch_size = 16

        #%% Training 

        train_task = training.TrainTask(
            labeled_data = train_generator(batch_size = batch_size, shuffle=True),
            loss_layer = tl.CrossEntropyLoss(),
            optimizer = trax.optimizers.Adam(0.01),
            n_steps_per_checkpoint = 10,
        )

        my_eval_task = training.EvalTask(
            labeled_data = val_generator(batch_size = batch_size, shuffle=True),
            metrics = [tl.CrossEntropyLoss(), tl.Accuracy()]
        )

        self.model = self.classifier()
        #display(model)


        training_loop = training.Loop(
            self.model,
            train_task,
            eval_tasks = my_eval_task,
            output_dir = self.dir,
        )

        with open(self.dir+'/metrics.txt', 'w') as f:
            with contextlib.redirect_stdout(f):
                training_loop.run(n_steps = steps)
        
        
        with open(self.dir+"/finish.txt",'w') as f:
            f.write('vocab_size='+str(len(self.vocab))+'\n')
            f.write('train_size='+str(count_pos+count_neg)+'\n')
            f.write('val_size='+str(tot_positive+ tot_negative-count_pos-count_neg)+'\n')
        
        return len(self.vocab), count_pos+count_neg, tot_positive+ tot_negative-count_pos-count_neg
        
    def predict(self, sentence):
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

#nlp = NLPClass()
#nlp.training(100,0.9)
#sentence = "STOCK MARKETS AT NEW ALL TIME HIGHS!!!"
#nlp.predict(sentence)

"""

#sentence = "“10% of voters would have changed their vote if they knew about Hunter Biden.” Miranda Devine  @nypost @TuckerCarlson But I won anyway!"
sentence = "STOCK MARKETS AT NEW ALL TIME HIGHS!!!"
sentence = "Gee, what a surprise. Has anyone informed the so-called (says he has no power to do anything!) Governor @BrianKempGA & his puppet Lt. Governor @GeoffDuncanGA, that they could easily solve this mess, & WIN. Signature verification & call a Special Session. So easy!"
predict(sentence,training_loop.eval_model)

"""