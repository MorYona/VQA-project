''' training dataset file'''
import json
import os
import os.path
import torch
from torch.utils.data import Dataset, DataLoader
#import preprocess_lstm
import re
import os
import sys
import pickle as cPickle

import time
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from collections import Counter

#from dataset import Dictionary

'''dicsioneris for answer preprocess''' 
contractions = {

    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve":
    "could've", "couldnt": "couldn't", "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've", "didnt": "didn't", "doesnt":
    "doesn't", "dont": "don't", "hadnt": "hadn't", "hadnt've":
    "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent":
    "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve":
    "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll",
    "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im":
    "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've":
    "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
    "maam": "ma'am", "mightnt": "mightn't", "mightnt've":
    "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't",
    "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat":
    "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve":
    "she'd've", "she's": "she's", "shouldve": "should've", "shouldnt":
    "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve":
    "shouldn't've", "somebody'd": "somebodyd", "somebodyd've":
    "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll":
    "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd":
    "something'd", "somethingd've": "something'd've", "something'dve":
    "something'd've", "somethingll": "something'll", "thats":
    "that's", "thered": "there'd", "thered've": "there'd've",
    "there'dve": "there'd've", "therere": "there're", "theres":
    "there's", "theyd": "they'd", "theyd've": "they'd've", "they'dve":
    "they'd've", "theyll": "they'll", "theyre": "they're", "theyve":
    "they've", "twas": "'twas", "wasnt": "wasn't", "wed've":
    "we'd've", "we'dve": "we'd've", "weve": "we've", "werent":
    "weren't", "whatll": "what'll", "whatre": "what're", "whats":
    "what's", "whatve": "what've", "whens": "when's", "whered":
    "where'd", "wheres": "where's", "whereve": "where've", "whod":
    "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl":
    "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve":
    "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll":
    "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd":
    "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll":
    "you'll", "youre": "you're", "youve": "you've"
}
manual_map = { 'none': '0',
              'zero': '0',
              'one': '1',
              'two': '2',
              'three': '3',
              'four': '4',
              'five': '5',
              'six': '6',
              'seven': '7',
              'eight': '8',
               'nine': '9',
              'ten': '10'}
articles = ['a', 'an', 'the']
period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
comma_strip = re.compile("(\d)(\,)(\d)")
punct = [';', r"/", '[', ']', '"', '{', '}',
                '(', ')', '=', '+', '\\', '_', '-',
                '>', '<', '@', '`', ',', '?', '!']



def create_ans2label(occurence, name, cache_root):

    """Note that this will also create label2ans.pkl at the same time
    occurence: dict {answer -> whatever}

    name: prefix of the output file

    cache_root: str

    """
    ans2label = {}
    label2ans = []
    label = 0
    for answer in occurence:
        label2ans.append(answer)
        ans2label[answer] = label
        label += 1
        

    return ans2label

def process_punctuation(inText):
    outText = inText
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) \
           or (re.search(comma_strip, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = period_strip.sub("", outText, re.UNICODE)
    return outText

def questions_preprocess(questions_json):
    questions = [q['question'] for q in questions_json['questions']]
    
    for question in questions:
        yield question.lower()

def process_digit_article(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manual_map.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText

            
def preprocess_answer(answer):
    answer = process_digit_article(process_punctuation(answer))
    answer = answer.replace(',', '')
    return answer

def filter_answers(answers_dset, min_occurence):
    """This will change the answer to preprocessed version

    """
    occurence = {}
    for ans_entry in answers_dset:
        gtruth = ans_entry['multiple_choice_answer']
        gtruth = preprocess_answer(gtruth)
        if gtruth not in occurence:
            occurence[gtruth] = set()
        occurence[gtruth].add(ans_entry['question_id'])
        
    for answer in list(occurence.keys()):
        if len(occurence[answer]) < min_occurence:
            #occurence.pop(answer)
            del occurence[answer]
            


    print('Num of answers that appear >= %d times: %d' % (
        min_occurence, len(occurence)))
    return occurence




class VQA_train(Dataset):
        ''' VQA dataset for open ended questions '''
        
        def __init__(self,paths):
            
            ''' this part is creating the dict for answers and questions'''
            path = '/datashare'
            train_images_path ='/datashare/train2014'
            train_questions_file = '/datashare/v2_OpenEnded_mscoco_train2014_questions.json'
            train_answer_file = '/datashare/v2_mscoco_train2014_annotations.json'
            #
            # path = '/datashare'
            # #train_images_path ='D:\MSc\קורסים\למידה עמוקה\VQA\train'
            # train_images_path ='/datashare/train2014'
            # train_questions_file = '/datashare/v2_OpenEnded_mscoco_train2014_questions.json'
            # train_answer_file = '/datashare/v2_mscoco_train2014_annotations.json'
            # val_questions_file = '/datashare/v2_OpenEnded_mscoco_val2014_questions.json'
            # val_answer_file = '/datashare/v2_mscoco_val2014_annotations.json'

            with open(train_answer_file) as f:
                self.train_annotations = json.load(f)['annotations']
            
            with open(train_questions_file) as f:
               self.train_questions = json.load(f)['questions']

            self.filtered_answers_dict = filter_answers(self.train_annotations,9)
            self.train_entries =VQA_train._get_train_entires(self)
            self.train_tokens = VQA_train.tokenize_train(self)
            self.answer_dict = VQA_train.create_answer_dict(self)
            self.question_word_dict = VQA_train.create_question_dict(self)
            self.image_dict = VQA_train.create_image_dict(self)
            # print(self.image_dict)
            ''' start the  dataset'''
            self.path_questions,self.path_annotations,self.path_images = paths
            # print(self.path_images)
            # print(self.path_annotations)
            with open(self.path_annotations) as f:
                self.annotations = json.load(f)['annotations']
            
            with open(self.path_questions) as f:
                self.questions = json.load(f)['questions']
  
        #def _get_train_entires(self):    
            entries =[]
            for entry in range(len(self.annotations)):
                if self.annotations[entry]['multiple_choice_answer'] in self.filtered_answers_dict:
                    question = self.questions[entry]['question']                    
                    answer = self.annotations[entry]['multiple_choice_answer']
                    
                    question = preprocess_answer(question)
                    question_id = self.annotations[entry]['question_id']
                    image_id = self.annotations[entry]['image_id']
                    entries.append((question_id,image_id,question,answer))
                else:
                    pass
            self.entries = entries
            
    
        #def tokenize(self):
            ''' split the questions and answers to words'''
            
            self.tokens = []
            number_of_samples = len(self.entries)
            for entry in range(number_of_samples):
                question_token = entries[entry][2].split(" ")  
                answer_token = entries[entry][3]
                image_token = entries[entry][1]
                self.tokens.append((question_token,answer_token,image_token))
            

        def _get_train_entires(self):
            train_entries =[]
            for entry in range(len(self.train_annotations)):
                if self.train_annotations[entry]['multiple_choice_answer'] in self.filtered_answers_dict:
                    question = self.train_questions[entry]['question']                    
                    answer = self.train_annotations[entry]['multiple_choice_answer']
                    
                    question = preprocess_answer(question)
                    question_id = self.train_annotations[entry]['question_id']
                    image_id = self.train_annotations[entry]['image_id']
                    train_entries.append((question_id,image_id,question,answer))
            return train_entries

        
        def tokenize_train(self):
            ''' split the questions and answers to words'''
            
            train_tokens = []
            number_of_samples = len(self.train_entries)
            for entry in range(number_of_samples):
                question_token = self.train_entries[entry][2].split(" ")  
                answer_token = self.train_entries[entry][3]
                image_token = self.train_entries[entry][1]
                train_tokens.append((question_token,answer_token,image_token))
            train_tokens = train_tokens  
            return train_tokens

                  
        def create_question_dict(self):
            '''the question is a bag of word vector that each word is a number'''
            question_word_dict = {}
            index = 1
            question_word_list =[]
            for token in range(len(self.train_tokens)):
                #run on question answer token
                #print(self.train_tokens[token][0])
                for word in range(len(self.train_tokens[token][0])):
                    #print(self.train_tokens[token][0][word])
                    #print(entries[token][word-1])
                    question_word_list.append(self.train_tokens[token][0][word])
            
            
            common_word = Counter(question_word_list)
            common_word = common_word.most_common(1000)
            
            for i in range(len(common_word)):
                question_word_dict[common_word[i][0]] = index
                index += 1
            
            #adding key to known words
            question_word_dict['unknown'] = 1001
            self.question_word_dict  = question_word_dict 
            return question_word_dict 

        
        def create_answer_dict(self):
            ''' create dictuniry for filltered answers  only in training'''
            index = 1
            processes_answers = []
            answer_dict={}
            for entry in self.train_annotations:
                
                        x = preprocess_answer(entry['multiple_choice_answer'])
                        
                        if x not in processes_answers and x in self.filtered_answers_dict:
                            
                            answer_dict[x] = index
                            processes_answers.append(x)
                            index +=1      
            return answer_dict
        def create_image_dict(self):
            #for train images 
            '''need to change this to the correct folder'''
            ''' SHARON MAYBE YOU CAN USE!!!! self.path_images'''
            self.path = '/datashare/train2014'
            self.id_to_imagename = {}
            for image_file in os.listdir(self.path): # iterate over all the files in the path directory
                if not image_file.endswith('.jpg'):
                    continue
                image_id_jpg = image_file.split('_')[-1] # "<image_id>.jpg"
                id = int(image_id_jpg.split('.')[0]) # <image_id> in int format
                self.id_to_imagename[id] = image_file # filename sorted by the id {image_id: image_name}

            
            '''need to change this to the correct folder '''
            # self.path = r'D:\MSc\קורסים\למידה עמוקה\VQA\val' 
            # for image_file in os.listdir(self.path): # iterate over all the files in the path directory
            #     if not image_file.endswith('.jpg'):
            #         continue
            #     image_id_jpg = image_file.split('_')[-1] # "<image_id>.jpg"
            #     id = int(image_id_jpg.split('.')[0]) # <image_id> in int format
            #     self.id_to_imagename[id] = image_file # filename sorted by the id {image_id: image_name}
            
            self.image_size = 224+224
            self.central_fraction = 0.875
            self.transform = transforms.Compose([
            # transforms.Scale(int(self.image_size / self.central_fraction)),
            # transforms.CenterCrop(self.image_size),
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),])
                
            return self.id_to_imagename
        
        def __len__(self):
            return len(self.tokens)    
        
        def get_question_dic_len(self):
            return len(self.question_word_dict)
            #for the embedding layer
        
        def get_filter_answer_dic_len(self):
            return len(self.answer_dict)
        
        def __getitem__(self, index):
            #the dataloader get the data from here 
            question_vector = torch.zeros(20)
            answer_vector = self.answer_dict[self.tokens[index][1]]
            #transform the question words to index
            for word in range(len(self.tokens[index][0])):
                if self.tokens[index][0][word] in self.question_word_dict:
                  question_vector[word] = self.question_word_dict[self.tokens[index][0][word]]
                 
                else:
                    question_vector[word] = self.question_word_dict['unknown']

            image_file = self.id_to_imagename[self.train_tokens[index][2]]
            # print(self.train_tokens[index][2])
            # print(image_file)
            # print('====================================')
            # print(self.path_images)
            # print('====================================')
            # print(os.path.join(self.path_images, image_file))
            # print('====================================')
            # print('====================================')
            # print(os.path.join(self.path_images, self.id_to_imagename[self.train_tokens[index][2]]))
            # print('====================================')


            path = os.path.join(self.path_images, self.id_to_imagename[self.train_tokens[index][2]])

            img = Image.open(path).convert('RGB')
            img = self.transform(img)
            item = (question_vector,img,answer_vector)
            return item

