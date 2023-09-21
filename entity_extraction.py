import json
from tkinter import E
import Levenshtein
from pandas import DataFrame
import openai
import time
import pandas as pd
from openpyxl import load_workbook
def edit_distance_score(text1, text2):
    distance = Levenshtein.distance(text1, text2)
    max_length = max(len(text1), len(text2))
    similarity_score = 1 - (distance / max_length)
    return similarity_score

openai.api_key = "your api key"

def load_test_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        test_data = json.load(file)
    return test_data

def extract_entities(uniq_list,expression_list):
    predictions = []
    for num, item in enumerate(expression_list):
        print(num)
        sentence = item
        if num < 7481:
            continue 
        if sentence == "a":
            continue

        prompt = '''
        Please do an entity extraction task, extract the subject and modifiers from the following sentence, and output it in json format and in English:
        Let me give you some example:
         example1: 
         Input:a blue tie being worn by a man
         Output:
         {
             "subject": "bag" ,
             "modifiers": ["blue", "being worn by a man"]
         }  

        example2: 
        Input:A white bird stands behind two brown birds
        Output:
        {
            "subject": "bird" ,
            "modifiers": ["white", "stands behind two brown birds"]
        }

        example3: 
        Input:a van
        Output:
        {
            "subject": "van" ,
            "modifiers": []
        }

        example4: 
        Input:A man with black hair who is wearing a black shit and pulling on the rope.
        Output:
        {
            "subject": "man" ,
            "modifiers": ["with black hair", "wearing a black shit", " pulling on the rope"]
        }

        example5: 
        Input:Man in a car talking on a cellphone
        Output:
        {
            "subject": "Man" ,
            "modifiers": ["in a car", "talking on a cellphone"]
        }

        example6: 
        Input:A ceramic top to a toilet tank
        Output:
        {
            "subject": "top" ,
            "modifiers": ["ceramic", "to a toilet tank"]
        }

        example7: 
        Input:A navy blue SUV with dark windows and a silver grill.
        Output:
        {
            "subject": "SUV" ,
            "modifiers": ["navy blue", "with dark windows and a silver grill."]
        }

        Input: 
        ''' + sentence + "\nOutput:"
        try:
            completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ]
            )
        except openai.error.RateLimitError :
            time.sleep(80)
            completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ]
            )
        except openai.error.APIError:
            time.sleep(80)
            completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ]
            )
        except openai.error.ServiceUnavailableError:
            time.sleep(80)
            completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ]
            )
        except openai.error.APIConnectionError:
            time.sleep(80)
            completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ]
            )

        output = completion.choices[0].message["content"]
        start_index = output.find('{')
        end_index = output.rfind('}') + 1   
        cleaned_string = output[start_index:end_index]
        cleaned_string = cleaned_string.split('}', 1)[0] + '}'  #The characters after the first occurrence of } are removed.
        output_string = cleaned_string.replace('\n', '')  # Remove escape characters

        try:
            output_json = json.loads(output_string)
        except json.decoder.JSONDecodeError:
            index = output_string.find('}')
            if index != -1:
                output_string = output_string[:index+1] + ',' + output_string[index+1:]
            output_json = json.loads(output_string)
        final_list = []
        templist = []
        subject_new = ""
        modifier_new = ""
        subject_new = output_json["subject"]
        for each in output_json["modifiers"]:
            modifier_new = modifier_new + ";" + each

        object_path = "XXX/refcocoplus_test_entity.xlsx"
        try:
            pd_sheets = pd.ExcelFile(object_path)
        except Exception as e:
            print("read {} file fail".format(object_path), e)
        df = pd.read_excel(pd_sheets, "Sheet1", header=[0])
        final_list = []
        for row in df.itertuples(index=True):
            templist = []
            row_list = list(row)
            uniq_id = row_list[1:2]
            subject = row_list[2:3]
            modifier = row_list[3:4]
            sentence_old = row_list[4:5]
            templist.append(uniq_id[0])
            templist.append(subject[0])
            templist.append(modifier[0])
            templist.append(sentence_old[0])
            final_list.append(templist)
        templist = []
        templist.append(uniq_list[num])
        templist.append(subject_new)
        templist.append(modifier_new)
        templist.append(sentence)
        final_list.append(templist)
        df1 = DataFrame(final_list,columns=["uniq_id","object","property","sentence"])
        df1.to_excel('XXX/refcocoplus_test_entity.xlsx', sheet_name='Sheet1', index=False)        
    return predictions

object_path = "XXX/refcocoplus_test.xlsx"
try:
    pd_sheets = pd.ExcelFile(object_path)
except Exception as e:
    print("read {} file fail".format(object_path), e)
df = pd.read_excel(pd_sheets, "Sheet1", header=[0])
uniq_list = [] 
expression_list = []
caption_list = []
label_list = []
num = 0
final_list = []
for row in df.itertuples(index=True):
    row_list = list(row)
    uniq_id = row_list[1:2]
    text = row_list[2:3]
    uniq_list.append(uniq_id[0])
    expression_list.append(text[0])

predictions = extract_entities(uniq_list,expression_list)


