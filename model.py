import json
import os
import os.path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
#import preprocess_lstm
import re
import os
import sys
import pickle as cPickle
import utils
import time
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence
#import data
import data_lstm_check


if __name__ == '__main__': 
    start = time.time()
    
    
    dataset = data_lstm_check.VQA(('v2_OpenEnded_mscoco_train2014_questions.json','v2_mscoco_train2014_annotations.json','train2014'))
    batch_size = 10000
    train_loader = DataLoader(dataset, batch_size ,shuffle=True, num_workers=0,drop_last=True)
    question_dict_len = data_lstm_check.VQA.get_question_dic_len(dataset)
    filter_answer_dict_len = data_lstm_check.VQA.get_filter_answer_dic_len(dataset)
    question_vector_size = 20 
    
    
    # class VGG(nn.Module):
    #     def __init__(self, n_classes=1000):
    #         super(VGG, self).__init__()
    
    #         # Conv blocks (BatchNorm + ReLU activation added in each block)
    #         in_list = [3,64,64,128,128,256,256,256,512,512,512,512,512]
    #         out_list = in_list[1:]
    #         out_list.append(512)
    #         kernels = 3
    #         stride = 1
    #         pool=2
    #         pool_k=2
    #         layers=[]
    
    #         i=0
    #         layers.append(nn.Conv2d(in_list[i], out_list[i], kernel_size=kernels, padding=kernels // 2))
    #         layers.append(nn.BatchNorm2d(out_list[i]))
    #         layers.append(nn.ReLU())
    #         i = 1
    #         layers.append(nn.Conv2d(in_list[i], out_list[i], kernel_size=kernels, padding=kernels // 2))
    #         layers.append(nn.BatchNorm2d(out_list[i]))
    #         layers.append(nn.ReLU())
    #         layers.append(nn.Dropout(0.5))
    #         layers.append(nn.MaxPool2d(kernel_size=pool_k, stride=pool_k))
    
    #         i = 2
    #         layers.append(nn.Conv2d(in_list[i], out_list[i], kernel_size=kernels, padding=kernels // 2))
    #         layers.append(nn.BatchNorm2d(out_list[i]))
    #         layers.append(nn.ReLU())
    #         i = 3
    #         layers.append(nn.Conv2d(in_list[i], out_list[i], kernel_size=kernels, padding=kernels // 2))
    #         layers.append(nn.BatchNorm2d(out_list[i]))
    #         layers.append(nn.ReLU())
    #         layers.append(nn.Dropout(0.5))
    #         layers.append(nn.MaxPool2d(kernel_size=pool_k, stride=pool_k))
    
    #         i = 4
    #         layers.append(nn.Conv2d(in_list[i], out_list[i], kernel_size=kernels, padding=kernels // 2))
    #         layers.append(nn.BatchNorm2d(out_list[i]))
    #         layers.append(nn.ReLU())
    #         i = 5
    #         layers.append(nn.Conv2d(in_list[i], out_list[i], kernel_size=kernels, padding=kernels // 2))
    #         layers.append(nn.BatchNorm2d(out_list[i]))
    #         layers.append(nn.ReLU())
    #         i = 6
    #         layers.append(nn.Conv2d(in_list[i], out_list[i], kernel_size=kernels, padding=kernels // 2))
    #         layers.append(nn.BatchNorm2d(out_list[i]))
    #         layers.append(nn.ReLU())
    #         layers.append(nn.Dropout(0.5))
    #         layers.append(nn.MaxPool2d(kernel_size=pool_k, stride=pool_k))
    
    #         i = 7
    #         layers.append(nn.Conv2d(in_list[i], out_list[i], kernel_size=kernels, padding=kernels // 2))
    #         layers.append(nn.BatchNorm2d(out_list[i]))
    #         layers.append(nn.ReLU())
    #         i = 8
    #         layers.append(nn.Conv2d(in_list[i], out_list[i], kernel_size=kernels, padding=kernels // 2))
    #         layers.append(nn.BatchNorm2d(out_list[i]))
    #         layers.append(nn.ReLU())
    #         i = 9
    #         layers.append(nn.Conv2d(in_list[i], out_list[i], kernel_size=kernels, padding=kernels // 2))
    #         layers.append(nn.BatchNorm2d(out_list[i]))
    #         layers.append(nn.ReLU())
    #         layers.append(nn.Dropout(0.5))
    #         layers.append(nn.MaxPool2d(kernel_size=pool_k, stride=pool_k))
    
    #         i = 10
    #         layers.append(nn.Conv2d(in_list[i], out_list[i], kernel_size=kernels, padding=kernels // 2))
    #         layers.append(nn.BatchNorm2d(out_list[i]))
    #         layers.append(nn.ReLU())
    #         i = 11
    #         layers.append(nn.Conv2d(in_list[i], out_list[i], kernel_size=kernels, padding=kernels // 2))
    #         layers.append(nn.BatchNorm2d(out_list[i]))
    #         layers.append(nn.ReLU())
    #         i = 12
    #         layers.append(nn.Conv2d(in_list[i], out_list[i], kernel_size=kernels, padding=kernels // 2))
    #         layers.append(nn.BatchNorm2d(out_list[i]))
    #         layers.append(nn.ReLU())
    #         layers.append(nn.Dropout(0.5))
    #         layers.append(nn.MaxPool2d(kernel_size=pool_k, stride=pool_k))
    
    #         self.seq1 = nn.Sequential(*layers)
    
    
    #         layers2=[]
    #         layers2.append(nn.Linear(7 * 7 * out_list[-1], 4096))
    #         layers2.append(nn.BatchNorm1d(4096))
    #         layers2.append(nn.ReLU())
    #         layers2.append(nn.Linear(4096, 4096))
    #         layers2.append(nn.BatchNorm1d(4096))
    #         layers2.append(nn.ReLU())
    #         layers2.append(nn.Linear(4096, n_classes))
    
    #         self.seq2 = nn.Sequential(*layers2)
    
    
    #     def forward(self, x):
    #         vgg_features = self.seq1(x)
    #         out1 = vgg_features.view(x.size(0), -1)
    #         out = self.seq2(out1)
    
    #         return vgg_features, out
    
    
    
    class LSTM_VQA(nn.Module):
        ''' this model will process the question vector'''
        def __init__(self):
            super(LSTM_VQA,self).__init__()
            
            self.lstm_features = 512
            self.hidden_size2 = 1024
            self.num_layers = 1
            self.tanh = nn.Tanh()
            self.relu = nn.ReLU()
            self.embedding = torch.nn.Embedding(question_dict_len+10,question_vector_size,padding_idx=question_vector_size)
            self.drop = nn.Dropout(0.3)
            self.lstm = nn.LSTM(question_vector_size,self.lstm_features,self.num_layers,batch_first = True)
            self.fc = nn.Linear(self.lstm_features,1024)
            self._init_lstm(self.lstm.weight_ih_l0)
            self._init_lstm(self.lstm.weight_hh_l0)
            self.lstm.bias_ih_l0.data.zero_()
            self.lstm.bias_hh_l0.data.zero_()
    
            init.xavier_uniform(self.embedding.weight)

        def _init_lstm(self, weight):
            for w in weight.chunk(4, 0):
                init.xavier_uniform(w)

        def forward(self,question):
            embeds = self.embedding(question) #output:[input,question_vector_size]
            #embeds = self.drop(embeds)
            tan_embeds =self.tanh(embeds) #(batch_size,question vector size,embedded features)
            # tan_embeds = tan_embeds.cuda()
            pack = pack_padded_sequence(tan_embeds,lengths,batch_first=True)
            output , (hidden,c) = self.lstm(pack)
            c = self.fc(c.squeeze(0))
            
            # c = c.cuda()
            return c

    class MLP(nn.Module):
        ''' combine the features from the LSTM and CNN '''
        def __init__(self):
            super(MLP,self).__init__()
            # self.lstm_model = LSTM_VQA().cuda()
            self.lstm_model = LSTM_VQA()
            #self.vgg_model = VGG().cuda()
            self.fc1 = nn.Linear(1024,1000)
            self.drop = nn.Dropout(0.5)
            self.fc2 = nn.Linear(1000,filter_answer_dict_len+1)
            self.relu = nn.ReLU()
            #self.softmax  = nn.Softmax()
        
        def forward(self,question):
            #image = data
            #question = data
            question_features = self.lstm_model(question)
            image_featues = torch.randn(batch_size,1024)
            # image_featues = image_featues.cuda()
            output = torch.mul(question_features,image_featues)
            output = self.relu(output)
            output = self.fc1(output)
            output = self.drop(output)
            output = self.fc2(output)
            return output
            
            
            
    vqa_model = MLP()
    # vqa_model = MLP()
    criterion = nn.CrossEntropyLoss() #because classification problem
    optimizer = torch.optim.Adam(vqa_model.parameters(),lr=0.0001)
    #creating list of lengths of each question
    lengths =[]
    for i in range(batch_size):
        lengths.append(20)


    for i,(data) in enumerate(train_loader):
        hs = 0
        X = data[0]
        y = data[1]
        # y = y.cuda()
        # X =X.cuda()
        optimizer.zero_grad()
        preds = vqa_model.forward(X.long())
       
        loss = criterion(preds,y)
        #loss.backward()
        #optimizer.step()
        print(loss)
         
        
    features = torch.ones(1000)
        
    end = time.time()
    print(f"run time {end-start:.4}")
   