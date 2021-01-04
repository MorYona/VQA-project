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
import utils
import time
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

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



class VQA(Dataset):
        ''' VQA dataset for open ended questions '''
        def __init__(self):
            '''open json files'''
            path = 'D:\MSc\קורסים\למידה עמוקה\VQA'
            train_images_path ='D:\MSc\קורסים\למידה עמוקה\VQA\train'
            train_questions_file = 'v2_OpenEnded_mscoco_train2014_questions.json'
            train_answer_file = 'v2_mscoco_train2014_annotations.json'
            val_questions_file = 'v2_OpenEnded_mscoco_val2014_questions.json'
            val_answer_file = 'v2_mscoco_val2014_annotations.json'
    
            with open(train_answer_file) as f:
                train_annotations = json.load(f)['annotations']
            
            with open(train_questions_file) as f:
              train_questions = json.load(f)['questions']
              
            with open(val_answer_file) as f:
                val_annotations = json.load(f)['annotations']
            
            with open(val_questions_file) as f:
              val_questions = json.load(f)['questions']
            
            '''init self parameters'''
            self.max_question_len = 20 # known information about the dataset
            self.train_annotations = train_annotations
            self.train_questions = train_questions
            self.val_annotations = val_annotations
            self.val_questions = val_questions
            self.filtered_answers_dict = filter_answers(train_annotations,9)
            
            
             ############################
    
        def _get_entires(self,train_flag):
            ''' create a list with question,question id,image id and answer '''
            if train_flag == 1:
                train_entries =[]
                for entry in range(len(self.train_annotations)):
                    if self.train_annotations[entry]['multiple_choice_answer'] in self.filtered_answers_dict:
                        question = self.train_questions[entry]['question']                    
                        answer = self.train_annotations[entry]['multiple_choice_answer']
                        
                        question = preprocess_answer(question)
                        question_id = self.train_annotations[entry]['question_id']
                        image_id = self.train_annotations[entry]['image_id']
                        train_entries.append((question_id,image_id,question,answer))
            
                self.train_entries =train_entries
                print(self.train_entries[2])
                return train_entries
            
            elif train_flag == 0:
                val_entries =[]
                for entry in range(len(self.val_annotations)):
                    # here I cant fillter the answers
                    question = self.val_questions[entry]['question']                    
                    answer = self.val_annotations[entry]['multiple_choice_answer']
                    
                    question = preprocess_answer(question)
                    question_id = self.val_annotations[entry]['question_id']
                    image_id = self.val_annotations[entry]['image_id']
                    val_entries.append((question_id,image_id,question,answer))
                    
                self.val_entries = val_entries
                return val_entries
            
        def number_of_samples(self,train_flag):
            ''' return the number of samples in the dataset'''
            if train_flag == 1:
                return len(self.train_entries)
            
            elif train_flag == 0:
                return len(self.val_entries)
        
        
        def tokenize(self,train_flag):
            ''' split the questions and answers to words'''
            if train_flag == 1:
                train_tokens = []
                number_of_samples = self.number_of_samples(train_flag=1)
                for entry in range(number_of_samples):
                    question_token = self.train_entries[entry][2].split(" ")  
                    answer_token = self.train_entries[entry][3]
                    image_token = self.train_entries[entry][1]
                    train_tokens.append((question_token,answer_token,image_token))
                self.train_tokens = train_tokens  
                return train_tokens
            
            elif train_flag == 0:
                val_tokens = []
                number_of_samples = self.number_of_samples(train_flag=0)
                for entry in range(number_of_samples):
                    question_token = self.val_entries[entry][2].split(" ")  
                    answer_token = self.val_entries[entry][3]
                    image_token = self.val_entries[entry][1]
                    val_tokens.append((question_token,answer_token,image_token))
                self.val_tokens = val_tokens    
                return val_tokens
        
        def create_question_dict(self):
            '''the question is a bag of word vector that each word is a number'''
            question_word_dict = {}
            index = 1
            question_word_list =[]
            for token in range(len(self.train_tokens)):
                #run on question answer token
                for word in range(len(self.train_tokens[token][0])):
                    #check if the word is part of the dict
                    if self.train_tokens[token][0][word] not in question_word_list:
                        #add the word to the dict
                        question_word_list.append(self.train_tokens[token][0][word])
                        question_word_dict[self.train_tokens[token][0][word]] = index
                        index += 1
                        
            for token in range(len(self.val_tokens)):
                #run on question answer token
                for word in range(len(self.val_tokens[token][0])):
                    #check if the word is part of the dict
                    if self.val_tokens[token][0][word] not in question_word_list:
                        #add the word to the dict
                        question_word_list.append(self.val_tokens[token][0][word])
                        question_word_dict[self.val_tokens[token][0][word]] = index
                        index += 1
                        
            self.question_word_dict =question_word_dict
            return question_word_dict
        
        def create_answer_dict(self):
            ''' 
            all the answers even if 3 word will be a class with an index,
            and also need to include the unfiltered answers and validation answers 
            '''
            index = 1
            answer_dict = {}
            answer_list =[]
            processes_answers = []
            for entry in self.train_annotations:
                x = preprocess_answer(entry['multiple_choice_answer'])
                if x not in processes_answers:
                    answer_dict[x] = index
                    processes_answers.append(x)
                    index +=1
                    
            for entry in self.val_annotations:
                x = preprocess_answer(entry['multiple_choice_answer'])
                if x not in processes_answers:
                    answer_dict[x] = index
                    processes_answers.append(x)
                    index +=1
            
            self.answer_dict = answer_dict
            return answer_dict
        
        def create_image_dict(self):
                #for train images 
                self.path = r'D:\MSc\קורסים\למידה עמוקה\VQA\train'
                self.id_to_imagename = {}
                for image_file in os.listdir(self.path): # iterate over all the files in the path directory
                    if not image_file.endswith('.jpg'):
                        continue
                    image_id_jpg = image_file.split('_')[-1] # "<image_id>.jpg"
                    id = int(image_id_jpg.split('.')[0]) # <image_id> in int format
                    self.id_to_imagename[id] = image_file # filename sorted by the id {image_id: image_name}

                 #for validation images 
                self.path = r'D:\MSc\קורסים\למידה עמוקה\VQA\val'
                for image_file in os.listdir(self.path): # iterate over all the files in the path directory
                    if not image_file.endswith('.jpg'):
                        continue
                    image_id_jpg = image_file.split('_')[-1] # "<image_id>.jpg"
                    id = int(image_id_jpg.split('.')[0]) # <image_id> in int format
                    self.id_to_imagename[id] = image_file # filename sorted by the id {image_id: image_name}
                
                self.image_size = 224+224
                self.central_fraction = 0.875
                self.transform = transforms.Compose([
                transforms.Scale(int(self.image_size / self.central_fraction)),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),])
                
                
                
                id_to_imagename = self.id_to_imagename
                return id_to_imagename
            
            
            
            
            
        def __getitem__(self, index,train_flag):
            ''' the item is (image,question,answer)'''
            if train_flag == 1:
                '''create vector with zeros at max size '''
                question_vector = torch.zeros(self.max_question_len)
                answer_vector = self.answer_dict[self.train_tokens[index][1]]
                #transform the question words to index
                for word in range(len(self.train_tokens[index][0])):
                     question_vector[word] = self.question_word_dict[self.train_tokens[index][0][word]]
                 
                self.path = r'D:\MSc\קורסים\למידה עמוקה\VQA\train'
                image_file = self.id_to_imagename[self.train_tokens[index][2]]
                path = os.path.join(self.path, self.id_to_filename[self.train_tokens[index][2]])
                img = Image.open(path).convert('RGB')
                img = self.transform(img)

                
                item = (img,question_vector,answer_vector)
                return item
                
                '''sharon add here the image preprocess and what the model need to get '''
                
            elif train_flag == 0: 
                 question_vector = torch.zeros(self.max_question_len)
                 answer_vector = self.answer_dict[self.val_tokens[index][1]]
                 for word in range(len(self.val_tokens[index][0])):
                     question_vector[word] = self.question_word_dict[self.val_tokens[index][0][word]]
                     
                 self.path = r'D:\MSc\קורסים\למידה עמוקה\VQA\val'
                 image_file = self.id_to_imagename[self.train_tokens[index][2]]
                 path = os.path.join(self.path, self.id_to_filename[self.train_tokens[index][2]])
                 img = Image.open(path).convert('RGB')
                 img = self.transform(img)

                 item = (img,question_vector,answer_vector)
                 return item
                

if __name__ == '__main__': 
    start = time.time()
    dataset = VQA()
    #create entries
    mor1 = dataset._get_entires(train_flag=1)
    mor = dataset._get_entires(train_flag=0)
    
    
    sas = dataset.create_answer_dict()
    #tokenize
    niko = dataset.tokenize(train_flag=0)
    niko = dataset.tokenize(train_flag=1)
    #create the dictioneries 
    shiki = dataset.create_question_dict()
    sharon = dataset.create_image_dict()
    print(dataset.__getitem__(90,1))
    end = time.time()
    print(f"run time {end-start:.4}")
    
    



 









