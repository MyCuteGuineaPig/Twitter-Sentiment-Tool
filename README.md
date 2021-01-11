# Twitter-Sentiment-Tool

The model consists of Embedding layer, Mean layer, Fully-connected layer and LogSoftmax layer. Data is from [nltk.corpus.twitter_samples](https://www.nltk.org/howto/corpus.html). User can adjust train validation ratio and training steps, and make prediction after model finish training

## Initialize environment 


```
./run.sh # if on linux environment
```

Above scirpt does the following steps:

- create virtual environment
- download required library 
- activate virtual environment
- start Django server on local computer

Then open a browser, go to [127.0.0.1:8000/](http://127.0.0.1:8000/)

## Train Model 

When open the webpage at the first time, it shows the model status as untrained. 

![untrained](img/untrained.png)

Then can ajdust *training steps* and *training test split ratio*, and click *Train button* to train the model. 

![untrained](img/training.png)

After training, charts, traing size, validation size, vocabulary size, and status bar will be updated. 


![untrained](img/trained.png)


## Make Prediction

After training the model, can input text and click to predict as follow

![untrained](img/positive.png)

![untrained](img/negative.png)

