from django.shortcuts import render
from django.http import JsonResponse,HttpResponse
from django.views.generic import View

#from rest_framework.views import APIView
#from rest_framework.response import Response
import os
from django.contrib import messages
from .NLPClass import NLPClass

# Create your views here.
nlp = NLPClass()
dirname = os.path.dirname(os.path.abspath(__file__))

def index(request):
    if not (os.path.isfile(dirname+'/metrics.txt') and os.path.isfile(dirname+'/finish.txt')):
            return render(request = request,
                  template_name = 'main/index.html',
                  context = {})

    else:
        steps, trainloss, evalLoss, evalAccuracy = ReadMetrics()
        train_size, val_size, vocab_size = readFinishConfig()
        return render(request = request,
                template_name = 'main/index.html',
                context = {"steps": steps, "trainloss":trainloss,"evalLoss":evalLoss,"evalAccuracy":evalAccuracy,
                            "train_size":train_size, "val_size":val_size,"vocab_size":vocab_size })

    
class HomeView(View):
    def get(self, request):
        return render(request = request,
                  template_name = 'main/index.html')


def get_data(request, *args, **kwargs):
    data = {
        "sales":100,
        "customers": 10,
    }
    return JsonResponse(data)

def ReadMetrics():
    steps, trainloss, evalLoss, evalAccuracy = [], [], [], []
    with open(dirname+'/metrics.txt') as f:
        i = 0
        for line in f:
            line = line.strip().replace(':','').split()
            if '|' in line:
                if i == 0:
                    steps.append(int(line[1]))
                    trainloss.append(float(line[line.index('|')+1]))
                elif i == 1:
                    evalLoss.append(float(line[line.index('|')+1]))
                elif i == 2:
                    evalAccuracy.append(float(line[line.index('|')+1]))
                    i = -1
    
                i += 1
    return steps, trainloss, evalLoss, evalAccuracy

def readFinishConfig():
    train_size = val_size = vocab_size = 0
    with open(dirname+'/finish.txt','r') as f:
        for line in f:
            line = line.strip().split('=')
            if line[0] == 'vocab_size':
                vocab_size = float(line[1])
            elif line[0] == 'train_size':
                train_size = float(line[1])
            elif line[0] == 'val_size':
                val_size = float(line[1])
    return train_size, val_size, vocab_size

def train(request):
    if request.method == "POST":
        steps = request.POST['steps']
        ratio = request.POST['ratio']


        with open(dirname+"/trainingconfig.txt",'w') as f:
            f.write('steps='+str(steps)+'\n')
            f.write('ratio='+str(ratio)+'\n')

        vocab_size, train_size, val_size = nlp.training(steps= 100, ratio=0.9)
        #vocab_size, train_size, val_size =  100,100,100 #For debug

        
        if not os.path.isfile(dirname+'/metrics.txt') :
            print("Error")
            messages.error(request, "Metrics file not generated. Error for training")
        else:
            steps, trainloss, evalLoss, evalAccuracy = ReadMetrics()

            data = {
            "steps":steps,
            "trainloss": trainloss,
            "evalLoss": evalLoss,
            "evalAccuracy":evalAccuracy,
            "train_size": train_size,
            "val_size": val_size,
            "vocab_size":vocab_size
            }
            return JsonResponse(data)
    return HttpResponse('')

def predict(request):
    if request.method == "POST":
        sentence = request.POST['sentence']

        preds, sentiment = nlp.predict(sentence=sentence)
        #preds = 1  #For debug
        data = {
            "prediction":preds
        }
        return JsonResponse(data)
    return HttpResponse('')
