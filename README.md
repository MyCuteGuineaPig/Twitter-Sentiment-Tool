# Twitter-Sentiment-Tool

[![PyPI
version](https://badge.fury.io/py/trax.svg)](https://badge.fury.io/py/trax)

[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)


[django](https://github.com/django/django)
[trax](https://github.com/google/trax)

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

