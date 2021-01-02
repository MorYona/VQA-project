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
        
    # utils.create_dir(cache_root)

    # cache_file = os.path.join(cache_root, name+'_ans2label.pkl')
    # cPickle.dump(ans2label, open(cache_file, 'wb'))
    # cache_file = os.path.join(cache_root, name+'_label2ans.pkl')
    # cPickle.dump(label2ans, open(cache_file, 'wb'))

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
        ''' VQA dataset'''
        
        def __init__(self):
            '''open json files'''
            path = 'D:\MSc\קורסים\למידה עמוקה\VQA'
            train_images_path ='D:\MSc\קורסים\למידה עמוקה\VQA\train'
            train_questions_file = 'v2_OpenEnded_mscoco_train2014_questions.json'
            train_answer_file = 'v2_mscoco_train2014_annotations.json'
            # val_questions_file = 'v2_OpenEnded_mscoco_val2014_questions.json'
            # val_answer_file = 'v2_mscoco_val2014_annotations.json'
    
            with open(train_answer_file) as f:
                train_annotations = json.load(f)['annotations']
            
            with open(train_questions_file) as f:
              train_questions = json.load(f)['questions']
              
            
            # with open(val_answer_file) as f:
            #     val_annotations = json.load(f)['annotations']
            
            # with open(val_questions_file) as f:
            #   val_questions = json.load(f)['questions']
            
            self.train_annotations = train_annotations
            self.train_questions = train_questions
            
                 
            
            
    
            
        def _get_entires(self,train_flag):
            ''' create a list with question,question id,image id and answer '''
            if train_flag == 1:
                train_entries =[]
                for entry in range(len(self.train_annotations)):
                    question = self.train_questions[entry]['question']                    
                    answer = self.train_annotations[entry]['multiple_choice_answer']
                    answer = preprocess_answer(answer)
                    question = preprocess_answer(question)
                    question_id = self.train_annotations[entry]['question_id']
                    image_id = self.train_annotations[entry]['image_id']
                    train_entries.append((question_id,image_id,question,answer))
                
                self.train_entries =train_entries
                return train_entries
            
            if train_flag == 0:
                val_entries =[]
                for entry in range(len(self.train_annotations)):
                    question = self.val_questions[entry]['question']                    
                    answer = self.val_annotations[entry]['multiple_choice_answer']
                    answer = preprocess_answer(answer)
                    question = question.lower()
                    question_id = self.val_annotations[entry]['question_id']
                    image_id = self.val_annotations[entry]['image_id']
                    train_entries.append((question_id,image_id,question,answer))
                    
                self.val_entries =val_entries
                return val_entries
            
            
        def tokenize(self,train_flag):
            if train_flag == 1:
                train_tokens = []
                for entry in range(len(self.train_entries)):
                    question_token = self.train_entries[entry][2].split(" ")  
                    answer_token = self.train_entries[entry][3].split(" ")
                    train_tokens.append((question_token,answer_token))
                    
                return train_tokens
            
        def token(self):
            ''' split the questions and answers to words'''
            

if __name__ == '__main__': 
    start = time.time()
    dataset = VQA()
    mor = dataset._get_entires(train_flag=1)
    niko = dataset.tokenize(train_flag=1)
    end = time.time()
    print(f"learining time{end-start}")




















