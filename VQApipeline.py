import os,base64 
import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
import pandas as pd
import csv
from pandas import DataFrame
import json
import requests
from modelscope.preprocessors.multi_modal import OfaPreprocessor
from sentence_transformers import SentenceTransformer
import torch 
from torch import nn
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import nltk

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw

#the path to the origin data
filepath = "xxx"

#the path to phrase-bert model
model = SentenceTransformer('xxx')
aug_Reversetran = naw.BackTranslationAug(device='cuda')
#keyboard error
aug_keyboard = nac.KeyboardAug(aug_char_max=1,aug_word_max=1)
def wordsimcompute(phrase_list):
    # phrase_list = [ 'play an active role', 'participate actively']
    
    phrase_embs = model.encode( phrase_list )
    [p1, p2] = phrase_embs
    # print(p1)
    cos_sim = nn.CosineSimilarity(dim=0)
    return cos_sim( torch.tensor(p1), torch.tensor(p2))

def simwordreplace(sentence):
    words=word_tokenize(sentence)
    tags = set(['NN', 'JJ', 'JJR', 'JJS', 'VB', 'VBD', 'VBG', 'VBZ', 'VBN', 'VBP', 'VBZ', 'RB', 'RBR', 'RBS'])
    pos_tags =nltk.pos_tag(words)
    ret = []
    tempsentence = ""
    for word,pos in pos_tags:
        if (pos in tags):
            simword = ""
            synonyms = []
            for syn in wordnet.synsets(word):
                for lm in syn.lemmas():
                    synonyms.append(lm.name())
            if len(synonyms) > 1:
                sym_tags = nltk.pos_tag(synonyms)
                for num,each in enumerate(synonyms):
                    templist = []
                    templist.append(each.lower())
                    templist.append(word.lower())
                    simscore = wordsimcompute(templist)
                    #and sym_tags[num][1] == pos
                    if each.lower() != word.lower() and simscore > 0.6 :
                        simword = each
                        break
                if simword == "":
                    simword = word
            elif len(synonyms) == 1:
                simword = synonyms[0].lower()
            else:
                simword = word
            tempsentence +=  simword + " " 
            # ret.append(word)
        else:
            tempsentence +=  word + " " 
    # print(tempsentence) 
    return tempsentence

uniq_id_list = []
image_id_list = []
text_list = []
region_list = []
imagebase64_list = []
with open(filepath, 'r', encoding='utf-8') as f:
    for line in f:
#       remove '\n' in the end of line 
        line = line.strip('\n').split('\t')  
        uniq_id_list.append(line[0])
        image_id_list.append(line[1])
        text_list.append(line[2])
        region_list.append(line[3])
        imagebase64_list.append(line[4])

#extract entity
#http://172.16.20.73:8000/api/EntityRelationEx/
url = "http://172.16.16.103:8000/api/EntityRelationEx/"
headers = {'Content-Type': 'application/json'}
typeveclist = []
finaltypelist = []
VQAmodel = 'damo/ofa_visual-question-answering_pretrain_huge_en'
preprocessor = OfaPreprocessor(model_dir=VQAmodel)
ofa_pipe = pipeline(
            Tasks.visual_question_answering,
            model=VQAmodel,
            preprocessor=preprocessor)
finalMRtextlist = []
finalrawtext = []
finaluniq_list = []
finalreducetextlist = []
for num, gthtext in enumerate(text_list):
    print(num)
#   extract entity
#   path to the image in JPEG format, please download from the dataset
    image = "xxx"
    datas = {"text":gthtext}
    data = json.dumps(datas)
    response = requests.post(url, data=data, headers=headers)
    result = response.json()
    entitylist = result[0]["entities"]
    entityobject = {}
    entityobjectnum = 0
    newtextlist = []
    for entity in entitylist:
        if entity["type"] == "主体":
            entityobjectnum += 1
#   construct questions and ask on the subject
    for entity in entitylist:
        if entity["type"] == "主体":
#           not deal with the situation that there are more that one subject
            if entity["value"].endswith("s") or entity["value"].endswith("es") or "men" in entity["value"] or "women" in entity["value"] or entityobjectnum > 1:
                break
            else:
                entityobject = entity
#               construct queries
                questionone = "How many" + entity["value"] + " in the image?"
                questiontwo = "Are there many" + entity["value"]  + " in the image?"
                questonthree = "Is there a reflected" + entity["value"]  + "in the image?"
#               ask VQA model
                resultlist = []
                inputone = {'image': image, 'text': questionone}
                inputtwo = {'image': image, 'text': questiontwo}
                inputthree = {'image': image, 'text': questonthree}
                resultone = ofa_pipe(inputone)
                resulttwo = ofa_pipe(inputtwo)
                resultthree = ofa_pipe(inputthree)
                resultlist.append(resultone[OutputKeys.TEXT][0])
                resultlist.append(resulttwo[OutputKeys.TEXT][0])
                resultlist.append(resultthree[OutputKeys.TEXT][0])
#               determine the uniqueness of the subject
                if ("one" in resultlist[0].lower() and "no" in resultlist[1].lower() and "no" in resultlist[2].lower())  or ("1" in resultlist[0].lower() and "no" in resultlist[1].lower() and "no" in resultlist[2].lower()):
                    newtextlist.append(entity["value"].strip())
                    break
#   joint the features of the subject which is unique randomly
    if newtextlist != [] and len(entitylist) > 2:
        for each in entitylist:
            if entity["type"] == "特征":
                if entityobject["startIndex"] < each["startIndex"]:
                    newtextlist.append(entityobject["value"].strip() + " " + each["value"])
                elif entityobject["startIndex"] > each["startIndex"]:
                    newtextlist.append(each["value"].strip() + " " + entityobject["value"].strip())
    elif newtextlist == [] and len(entitylist) > 2 and entityobjectnum == 1:
#       add features to the not unique subject and determine its uniqueness again
        for each in entitylist:
            newsentence = ""
            if entity["type"] == "特征":
                if entityobject["startIndex"] < each["startIndex"]:
                    newsentence = entityobject["value"].strip() + " " + each["value"]
                elif entityobject["startIndex"] > each["startIndex"]:
                    newsentence = each["value"].strip() + " " + entityobject["value"].strip()
#               construct queries
                questionone = "How many " + newsentence+ " in the image?"
                questiontwo = "Are there many " + newsentence  + " in the image?"
                questonthree = "Is there a reflected" + entity["value"]  + "in the image?"
                resultlist = []
                inputone = {'image': image, 'text': questionone}
                inputtwo = {'image': image, 'text': questiontwo}
                inputthree = {'image': image, 'text': questonthree}
                resultone = ofa_pipe(inputone)
                resulttwo = ofa_pipe(inputtwo)
                resultthree = ofa_pipe(inputthree)
                resultlist.append(resultone[OutputKeys.TEXT][0])
                resultlist.append(resulttwo[OutputKeys.TEXT][0])
                resultlist.append(resultthree[OutputKeys.TEXT][0])
#               determine the uniqueness of the subject
                if ("one" in resultlist[0].lower() and "no" in resultlist[1].lower() and "no" in resultlist[2].lower())  or ("1" in resultlist[0].lower() and "no" in resultlist[1].lower() and "no" in resultlist[2].lower()):
                    newtextlist.append(newsentence)
    MRtextlist = []
    
#   perturb the text at character, word, and sentence levels
    if '' not in newtextlist and newtextlist != []:
        finalreducetextlist.append(newtextlist)
#       replace the word with its synonym
        for each in newtextlist:
            augmented_text = aug_Reversetran.augment(each)
            simsentence = simwordreplace(augmented_text[0])
            try:
                augmented_text = aug_keyboard.augment(simsentence)
                MRtextlist.append(augmented_text[0])
            except IndexError:
                augmented_text = simwordreplace(each)
                MRtextlist.append(augmented_text)
            # MRtextlist.append(augmented_text[0])
    if MRtextlist != []:
        finaluniq_list.append(uniq_id_list[num])
        finalrawtext.append(text_list[num])
        finalMRtextlist.append(MRtextlist)
    
#generate new text and print them
final_list = []

for newnum,each in enumerate(finaluniq_list):
    for othernum, sentence in enumerate(finalMRtextlist[newnum]):
        templist = []
        templist.append(each)
        templist.append(finalrawtext[newnum])
        templist.append(finalreducetextlist[newnum][othernum])
        templist.append(sentence)
        final_list.append(templist)


df = DataFrame(final_list,columns=["uniq_id","rawtext","Reducetext","MRtext"])
#the output path in xlsx format
df.to_excel('xxx', sheet_name='Sheet1', index=False)
print("ok")

#the output path in tsv format
out_path = "xxx"
with open(out_path, 'w', newline='') as f:
    tsv_w = csv.writer(f, delimiter='\t')
    for newnum,each in enumerate(finaluniq_list):
        index = uniq_id_list.index(each)
        for sentence in finalMRtextlist[newnum]:
            templist = []
            templist.append(each)
            templist.append(image_id_list[index])
            templist.append(sentence)
            templist.append(region_list[index])
            templist.append(imagebase64_list[index])
            tsv_w.writerow(templist)  