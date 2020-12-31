import json
import os
import os.path
import torch
from torch.utils.data import Dataset, DataLoader
#import preprocess_lstm



path = 'D:\MSc\קורסים\למידה עמוקה\VQA'
images_path ='D:\MSc\קורסים\למידה עמוקה\VQA\train'
questions_path = 'v2_OpenEnded_mscoco_train2014_questions.json'
annotaions_path = 'v2_mscoco_train2014_annotations.json'

''' open json files'''
with open (questions_path) as f:
    questions_json = json.load(f)

with open (annotaions_path) as f:
    answers_json = json.load(f)


'''genarator that return the question in lower case '''
def questions_preprocess(questions_json):
    questions = [q['question'] for q in questions_json['questions']]
    
    for question in questions:
        yield question.lower()
        

def answers_preprocess(answers_json):
    for annotation in answers_json["annotations"]:
        for answer in annotation["answers"]:
            print(answer['answer'])
            yield answer['answer']
        

'''chack'''
# answers_preprocess(answers_json)
# questions_preprocess(questions_json)
