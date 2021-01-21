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
import matplotlib.pyplot as plt
import time
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence
#import data
import pickle

import training_dataset_class
import val_dataset_class


def batch_accuracy(predicted, true):
    _, predicted_index = predicted.max(dim=1, keepdim=True)
    # print('predicted size= ', predicted.size())
    # print('true size= ', true.size())
    predicted_index = torch.squeeze(predicted_index.view(true.size(0),-1))
    # predicted_index = predicted_index.view(-1,true.size(1))
    # print('predicted_index size= ', predicted_index.size())
    # acc = min(torch.eq(true, predicted_index).sum().item()* 0.3, 3)
    acc = 100*torch.eq(true, predicted_index).sum().item() / len(true)
    # print('answer = ',answer)
    # print('pred_idx= ',predicted_index)
    # agreeing = true.gather(dim=1, index=predicted_index)
    return acc
    # return (agreeing * 0.3).clamp(max=1)


if __name__ == '__main__':
    start = time.time()
        
    train_dataset = training_dataset_class.VQA_train(('/datashare/v2_OpenEnded_mscoco_train2014_questions.json','/datashare/v2_mscoco_train2014_annotations.json','/datashare/train2014'))
    val_dataset = val_dataset_class.VQA_val(('/datashare/v2_OpenEnded_mscoco_val2014_questions.json','/datashare/v2_mscoco_val2014_annotations.json','/datashare/val2014'))
    batch_size = 96
    train_loader = DataLoader(train_dataset, batch_size ,shuffle=True, num_workers=0,drop_last=True)
    test_loader = DataLoader(val_dataset, batch_size ,shuffle=True, num_workers=0,drop_last=True)
    question_dict_len = training_dataset_class.VQA_train.get_question_dic_len(train_dataset)
    filter_answer_dict_len =training_dataset_class.VQA_train.get_filter_answer_dic_len(train_dataset)
    question_vector_size = 20


    class VGG(nn.Module):
        def __init__(self, n_classes=1024):
            super(VGG, self).__init__()

            # Conv blocks (BatchNorm + ReLU activation added in each block)
            in_list = [3, 64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512]
            out_list = in_list[1:]
            out_list.append(512)
            kernels = 3
            stride = 1
            pool = 2
            pool_k = 2
            layers = []

            i = 0
            layers.append(nn.Conv2d(in_list[i], out_list[i], kernel_size=kernels, padding=kernels // 2))
            layers.append(nn.BatchNorm2d(out_list[i]))
            layers.append(nn.ReLU())
            i = 1
            layers.append(nn.Conv2d(in_list[i], out_list[i], kernel_size=kernels, padding=kernels // 2))
            layers.append(nn.BatchNorm2d(out_list[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
            layers.append(nn.MaxPool2d(kernel_size=pool_k, stride=pool_k))

            i = 2
            layers.append(nn.Conv2d(in_list[i], out_list[i], kernel_size=kernels, padding=kernels // 2))
            layers.append(nn.BatchNorm2d(out_list[i]))
            layers.append(nn.ReLU())
            i = 3
            layers.append(nn.Conv2d(in_list[i], out_list[i], kernel_size=kernels, padding=kernels // 2))
            # layers.append(nn.BatchNorm2d(out_list[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
            layers.append(nn.MaxPool2d(kernel_size=pool_k, stride=pool_k))

            i = 4
            layers.append(nn.Conv2d(in_list[i], out_list[i], kernel_size=kernels, padding=kernels // 2))
            layers.append(nn.BatchNorm2d(out_list[i]))
            layers.append(nn.ReLU())
            i = 5
            # layers.append(nn.Conv2d(in_list[i], out_list[i], kernel_size=kernels, padding=kernels // 2))
            # layers.append(nn.BatchNorm2d(out_list[i]))
            # layers.append(nn.ReLU())
            i = 6
            layers.append(nn.Conv2d(in_list[i], out_list[i], kernel_size=kernels, padding=kernels // 2))
            layers.append(nn.BatchNorm2d(out_list[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
            layers.append(nn.MaxPool2d(kernel_size=pool_k, stride=pool_k))

            i = 7
            layers.append(nn.Conv2d(in_list[i], out_list[i], kernel_size=kernels, padding=kernels // 2))
            layers.append(nn.BatchNorm2d(out_list[i]))
            layers.append(nn.ReLU())
            i = 8
            layers.append(nn.Conv2d(in_list[i], out_list[i], kernel_size=kernels, padding=kernels // 2))
            layers.append(nn.BatchNorm2d(out_list[i]))
            layers.append(nn.ReLU())
            # i = 9
            # layers.append(nn.Conv2d(in_list[i], out_list[i], kernel_size=kernels, padding=kernels // 2))
            # layers.append(nn.BatchNorm2d(out_list[i]))
            # layers.append(nn.ReLU())
            # layers.append(nn.Dropout(0.5))
            # layers.append(nn.MaxPool2d(kernel_size=pool_k, stride=pool_k))
            #
            # i = 10
            # layers.append(nn.Conv2d(in_list[i], out_list[i], kernel_size=kernels, padding=kernels // 2))
            # layers.append(nn.BatchNorm2d(out_list[i]))
            # layers.append(nn.ReLU())
            i = 11
            layers.append(nn.Conv2d(in_list[i], out_list[i], kernel_size=kernels, padding=kernels // 2))
            layers.append(nn.BatchNorm2d(out_list[i]))
            layers.append(nn.ReLU())
            i = 12
            layers.append(nn.Conv2d(in_list[i], out_list[i], kernel_size=kernels, padding=kernels // 2))
            layers.append(nn.BatchNorm2d(out_list[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
            layers.append(nn.MaxPool2d(kernel_size=pool_k, stride=pool_k))

            self.seq1 = nn.Sequential(*layers)

            layers2 = []
            dim1 = 2048
            dim1 = 8192
            dim2 = 4096  #

            # print('========')
            # print('linear dimensions   ',dim1, dim2)
            # print('========')
            layers2.append(nn.Linear(dim1, dim2))
            layers2.append(nn.BatchNorm1d(dim2))
            layers2.append(nn.ReLU())
            # layers2.append(nn.Linear(dim2, dim2))
            # layers2.append(nn.BatchNorm1d(dim2))
            # layers2.append(nn.ReLU())
            layers2.append(nn.Linear(dim2, n_classes))

            self.seq2 = nn.Sequential(*layers2)

        def forward(self, x):
            # print('x_size   ', x.size())
            vgg_features = self.seq1(x)
            out1 = vgg_features.view(x.size(0), -1)
            # print('========')
            # print('out1 size   ',out1.size())
            # print('========')

            out = self.seq2(out1)

            return out
    
    
    
    class LSTM_VQA(nn.Module):
        ''' this model will process the question vector'''
        def __init__(self):
            super(LSTM_VQA,self).__init__()
            
            self.lstm_features = 512
            self.hidden_size2 = 1024
            self.num_layers = 1
            self.tanh = nn.Tanh()
            self.relu = nn.ReLU()
            # self.embedding = torch.nn.Embedding(question_dict_len+10,question_vector_size,padding_idx=question_vector_size)
            self.embedding = torch.nn.Embedding(1002,300,padding_idx=question_vector_size)
            self.drop = nn.Dropout(0.3)
            self.lstm = nn.LSTM(300,self.lstm_features,self.num_layers,batch_first = True)
            # self.lstm = nn.LSTM(question_vector_size,self.lstm_features,self.num_layers,batch_first = True)
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
            self.lstm_model = LSTM_VQA().cuda()
            self.vgg_model = VGG().cuda()
            self.fc1 = nn.Linear(1024,1000)
            self.drop = nn.Dropout(0.5)
            self.fc2 = nn.Linear(1000,filter_answer_dict_len+1)
            self.relu = nn.ReLU()

        
        def forward(self,question, image):
            image_featues = self.vgg_model(image)
            question_features = self.lstm_model(question)
            output = torch.mul(question_features, image_featues)
            output = self.relu(output)
            output = self.fc1(output)
            output = self.drop(output)
            output = self.fc2(output)

            return output
            
            
            
    vqa_model = MLP()
    vqa_model = vqa_model.cuda()
    criterion = nn.CrossEntropyLoss().cuda() #because classification problem
    optimizer = torch.optim.Adam(vqa_model.parameters(),lr=0.001)
    
    #creating list of lengths of each question
    lengths =[]
    for i in range(batch_size):
        lengths.append(20)
    
    epochs = 3
    train_accuracy = np.zeros(epochs)
    test_accuracy = np.zeros(epochs)
    losses_test = []
    losses_train = []
    accs_train=[]
    accs_test=[]
    for epoch in range(epochs):

        running_loss = 0
        running_loss_test = 0
        print('epoch = ', epoch)
        for i,(data) in enumerate(train_loader):
            question = data[0].long().cuda()
            image = data[1].cuda()
            answer = data[2].cuda()
            optimizer.zero_grad()
            preds = vqa_model.forward(question,image)
            acc_train = batch_accuracy(preds, answer)
            accs_train.append(acc_train)
            loss = criterion(preds,answer)
            running_loss += loss

            loss.backward()
            optimizer.step()
            losses_train.append(loss)
            average_acc = 100*(sum(accs_train) / len(accs_train))
            average_loss = 100*(sum(losses_train) / len(losses_train))
            #print(f'batch {i} Train loss {loss:.3} accuracy{acc_train}')


        #print(f'train accuracy {max(acc_train)}')

        train_accuracy[epoch] = acc_train
        torch.save(vqa_model.state_dict(), f'model{epoch}.pkl')
        print('average train loss:', average_loss)
        print('average train accuracy:', average_acc)

        '''after the train the network will run the validation data '''
        torch.cuda.empty_cache()
        with torch.no_grad():
            for i,(data) in enumerate(test_loader):
                question = data[0].cuda()
                # question = data[0]
                image = data[1].cuda()
                # image = data[1]
                answer = data[2].cuda()
                # answer = data[2]
                preds = vqa_model.forward(question.long(),image).cuda()
                # print('preds size:', preds.size())
                ###### todo -check acc calculation
                acc_test = batch_accuracy(preds, answer)
                accs_test.append(acc_test)
                loss = criterion(preds,answer)
                losses_test.append(loss)
                average_acc = 100*(sum(accs_test) / len(accs_test))
                average_loss = 100*(sum(losses_test) / len(losses_test))
            test_accuracy[epoch] = acc_test
            #print('average test loss:', average_loss)
            print('average test accuracy:', average_acc)



    with open('accs_train.txt', 'w') as f:
        for item in accs_train:
            f.write("%s\n" % item)

    with open('losses_train.txt', 'w') as f:
        for item in losses_train:
            f.write("%s\n" % item)

    with open('accs_test.txt', 'w') as f:
        for item in accs_test:
            f.write("%s\n" % item)

    with open('losses_test.txt', 'w') as f:
        for item in losses_test:
            f.write("%s\n" % item)



    """ plot accuracy """
    plt.plot(train_accuracy,label = 'Train plot')
    plt.plot(test_accuracy,label = 'Test plot')
    plt.legend()
    plt.grid()
    plt.xlabel("epoch number")
    plt.ylabel("accuracy")



