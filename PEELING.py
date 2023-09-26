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

os.environ["TOKENIZERS_PARALLELISM"] = "true"


model = SentenceTransformer('/XXX/phrase-bert')
#互译
aug_Reversetran = naw.BackTranslationAug(device='cuda')
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
filepath = "XXX/refcocoplus_test.tsv"
uniq_id_list = []
image_id_list = []
text_list = []
region_list = []
imagebase64_list = []
with open(filepath, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip('\n').split('\t')  
        uniq_id_list.append(line[0])
        image_id_list.append(line[1])
        text_list.append(line[2])
        region_list.append(line[3])
        imagebase64_list.append(line[4])


object_path = "/XXX/refcocoplus_test_entity.xlsx"
try:
    pd_sheets = pd.ExcelFile(object_path)
except Exception as e:
    print("read {} file fail".format(object_path), e)
df = pd.read_excel(pd_sheets, "Sheet1", header=[0])
uniq_list = [] 
object_list = []
property_list = []
sentence_list = []
num = 0
final_list = []
for row in df.itertuples(index=True):
    row_list = list(row)
    uniq_id = row_list[1:2]
    object = row_list[2:3]
    property = row_list[3:4]
    sentence = row_list[4:5]
    uniq_list.append(uniq_id[0])
    object_list.append(object[0])
    property_list.append(property[0])
    sentence_list.append(sentence[0])

propertylistis_nan = [pd.isna(item) for item in property_list]

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
for num, uniq_id in enumerate(uniq_list):
    print(num)
    image = "/XXX/refcocoplus_data/test_image/" + str(uniq_id) + ".jpg"
  
    object = str(object_list[num]).strip()
    if propertylistis_nan[num] == False:
        tempproperty = str(property_list[num]).strip()
        properties = tempproperty.split(";")[1:]
    else:
        properties = "none"
    newtextlist = []
    if object.endswith("s") or object.endswith("es") or "men" in object or "women" in object:
        continue
    else:
        #contruct problem
        questionone = "How many " + object + " in the image?"
        questiontwo = "Are there many " + object  + " in the image?"
        questonthree = "Is there a reflected " + object  + "in the image?"
        #ask VQA
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
        #idntify uniqueness
        if ("one" in resultlist[0].lower() and "no" in resultlist[1].lower() and "no" in resultlist[2].lower())  or ("1" in resultlist[0].lower() and "no" in resultlist[1].lower() and "no" in resultlist[2].lower()):
            newtextlist.append(object)

    #Random feature splicing for the object’s only special case
    if newtextlist != [] and properties != "none":
        for each in properties:
            #Calculate the position of property and object in the sentence
            object_index = sentence_list[num].lower().find(object.lower())
            property_index = sentence_list[num].lower().find(each.lower())
            if property_index == -1:
                property_index = sentence_list[num].lower().find(each.lower().split()[0])
            if object_index < property_index:
                newtextlist.append(object + " " + each.strip())
            else :
                newtextlist.append(each.strip() + " " + object)
    
    elif newtextlist == [] and properties != "none":
        #If the object is not unique, add property to check for uniqueness
        for each in properties:     
            object_index = sentence_list[num].lower().find(object.lower())
            property_index = sentence_list[num].lower().find(each.lower())
            if property_index == -1:
                property_index = sentence_list[num].lower().find(each.lower().split()[0])
            if object_index < property_index:
                newsentence = object + " " + each.strip()
            else:
                newsentence = each.strip() + " " + object
            questionone = "How many " + newsentence+ " in the image?"
            questiontwo = "Are there many " + newsentence  + " in the image?"
            questonthree = "Is there a reflected " + newsentence  + "in the image?"
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
            if ("one" in resultlist[0].lower() and "no" in resultlist[1].lower() and "no" in resultlist[2].lower())  or ("1" in resultlist[0].lower() and "no" in resultlist[1].lower() and "no" in resultlist[2].lower()):
                newtextlist.append(newsentence)
    MRtextlist = []
    
    if '' not in newtextlist and newtextlist != []:
        finalreducetextlist.append(newtextlist)
        #Replace synonyms
        for each in newtextlist:
            if len(each) > 5:
                augmented_text = aug_Reversetran.augment(each)
                simsentence = simwordreplace(augmented_text[0])
                try:
                    augmented_text = aug_keyboard.augment(simsentence)
                    MRtextlist.append(augmented_text[0])
                except IndexError:
                    augmented_text = simwordreplace(each)
                    MRtextlist.append(augmented_text)
            else:
                augmented_text = simwordreplace(each)
                MRtextlist.append(augmented_text)
            # MRtextlist.append(augmented_text[0])
    if MRtextlist != []:
        finaluniq_list.append(uniq_id)
        finalrawtext.append(sentence_list[num])
        finalMRtextlist.append(MRtextlist)
    
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
df.to_excel('/XXX/refcocoplus_data/refcocoplus_test_3aug3q.xlsx', sheet_name='Sheet1', index=False)
out_path = "/XXX/refcocoplus_data/refcocoplus_test_3aug3q.tsv"
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
