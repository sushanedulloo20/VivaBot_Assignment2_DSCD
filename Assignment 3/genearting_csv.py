import os
import csv
import re

def convert_feedback_to_json(file_path):
    functionalities = []
    with open(file_path, 'r') as file:
    # Read all lines from the file
        lines = file.readlines()
        
        # Temporary dictionary to hold a single functionality block
        functionality = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith('****'):
                # If we encounter a block separator and the current functionality is not empty, save it
                if functionality:
                    functionalities.append(functionality)
                    functionality = {}
            elif ' = ' in line:
                # Split the line into key and value parts
                key, value = line.split(' = ', 1)
                # Convert numeric values to int if possible
                if value.isdigit():
                    value = int(value)
                functionality[key] = value

        # Append the last functionality if not already appended
        if functionality:
            functionalities.append(functionality)
    return functionalities

def extract_group_id(path,n):
    splitted_path=path.split("_")
    return int(splitted_path[n])

def extract_fields_TA(text):
    # Regular expression patterns to extract fields
    question_pattern = r'\*\*Question\*\*\n(.*?)\n\*\*Model Answer\*\*'
    model_answer_pattern = r'\*\*Model Answer\*\*\n(.*?)\n\*\*Student Answer\*\*'
    student_answer_pattern = r'\*\*Student Answer\*\*\n(.*?)\n\*\*Answer Evaluation By LLM'
    llm_answer=r'\*\*Answer Evaluation By LLM\*\*\n(.*?)\n\*\*TA Rating for Question Usefulness \(1 being very poor, 5 being excellent\)'
    rating_pattern = r'\*\*TA Rating for Question Usefulness \(1 being very poor, 5 being excellent\)\*\*\n(.*?)\n\*\*TA Comments for Question Usefulness'
    comments_pattern = r'\*\*TA Comments for Question Usefulness\*\*\n(.*?)\n\*\*End of TA Comments\*\*'
    comments_new_pattern=r'\*\*TA Comments for Question Usefulness\*\*\n(.*?)\n\*\*TA Rating for LLM Feedback \(1 being very poor, 5 being excellent\)\*\*'
    ta_llm_rating_pattern=r'\*\*TA Rating for LLM Feedback \(1 being very poor, 5 being excellent\)\*\*\n(.*?)\n\*\*TA Comments for LLM Feedback\*\*'
    ta_llm_feedback_pattern=r'\*\*TA Comments for LLM Feedback\*\*\n(.*?)\n\*\*TA Score for Student Answer\*\*'
    ta_score_pattern=r'\*\*TA Score for Student Answer\*\*\n(.*?)\n\*\*End of TA Score\*\*'

    # Extracting fields using regular expressions
    question = re.search(question_pattern, text, re.DOTALL).group(1).strip()
    model_answer = re.search(model_answer_pattern, text, re.DOTALL).group(1).strip()
    student_answer = re.search(student_answer_pattern, text, re.DOTALL).group(1).strip()
    llm_ans=re.search(llm_answer, text, re.DOTALL).group(1).strip()
    rating = re.search(rating_pattern, text, re.DOTALL).group(1).strip()
    comments1 = re.search(comments_new_pattern, text, re.DOTALL)
    ta_llm_rating=re.search(ta_llm_rating_pattern, text, re.DOTALL).group(1).strip()
    ta_llm_feedback=re.search(ta_llm_feedback_pattern, text, re.DOTALL).group(1).strip()
    ta_score=re.search(ta_score_pattern, text, re.DOTALL).group(1).strip()
    comments=comments1.group(1).strip()
    fields = {
        'rating': rating,
        'comments': comments,
        'ta_rating_llm': ta_llm_rating,
        'ta_llm_feedback': ta_llm_feedback,
        'ta_score': ta_score
    }
    return fields

def extract_fields_combined(text):
    # Regular expression patterns to extract fields
    question_pattern = r'\*\*Question\*\*\n(.*?)\n\*\*Model Answer\*\*'
    model_answer_pattern = r'\*\*Model Answer\*\*\n(.*?)\n\*\*Student Answer\*\*'
    student_answer_pattern = r'\*\*Student Answer\*\*\n(.*?)\n\*\*Answer Evaluation By LLM'
    llm_answer=r'\*\*Answer Evaluation By LLM\*\*\n(.*?)\n\*\*TA Rating for Question Usefulness \(1 being very poor, 5 being excellent\)'
    rating_pattern = r'\*\*TA Rating for Question Usefulness \(1 being very poor, 5 being excellent\)\*\*\n(.*?)\n\*\*TA Comments for Question Usefulness'
    comments_pattern = r'\*\*TA Comments for Question Usefulness\*\*\n(.*?)\n\*\*End of TA Comments\*\*'
    comments_new_pattern=r'\*\*TA Comments for Question Usefulness\*\*\n(.*?)\n\*\*TA Rating for LLM Feedback \(1 being very poor, 5 being excellent\)\*\*'
    ta_llm_rating_pattern=r'\*\*TA Rating for LLM Feedback \(1 being very poor, 5 being excellent\)\*\*\n(.*?)\n\*\*TA Comments for LLM Feedback\*\*'
    ta_llm_feedback_pattern=r'\*\*TA Comments for LLM Feedback\*\*\n(.*?)\n\*\*TA Score for Student Answer\*\*'
    ta_score_pattern=r'\*\*TA Score for Student Answer\*\*\n(.*?)\n\*\*End of TA Score\*\*'

    # Extracting fields using regular expressions
    question = re.search(question_pattern, text, re.DOTALL).group(1).strip()
    model_answer = re.search(model_answer_pattern, text, re.DOTALL).group(1).strip()
    student_answer = re.search(student_answer_pattern, text, re.DOTALL).group(1).strip()
    llm_ans=re.search(llm_answer, text, re.DOTALL).group(1).strip()
    rating = re.search(rating_pattern, text, re.DOTALL).group(1).strip()
    comments1 = re.search(comments_new_pattern, text, re.DOTALL)
    ta_llm_rating=re.search(ta_llm_rating_pattern, text, re.DOTALL).group(1).strip()
    ta_llm_feedback=re.search(ta_llm_feedback_pattern, text, re.DOTALL).group(1).strip()
    ta_score=re.search(ta_score_pattern, text, re.DOTALL).group(1).strip()
    comments=comments1.group(1).strip()
    fields = {
        'question': question,
        'model_answer': model_answer,
        'student_answer': student_answer,
        'rating': rating,
        'comments': comments,
        'llm_answer': llm_ans,
        'ta_rating_llm': ta_llm_rating,
        'ta_llm_feedback': ta_llm_feedback,
        'ta_score': ta_score
    }
    return fields


# Function to read the file and extract fields for each block
def read_questions_file_into_dict(file_path,isTa_only):
    with open(file_path, 'r') as file:
        text = file.read()

    # Splitting text into blocks
    blocks = text.split('******************************************************************\n\n')

    # Extracting fields from each block
    extracted_blocks = []
    for block in blocks:
        if block.strip():  # Skipping empty blocks
            if isTa_only==False:
                extracted_blocks.append(extract_fields_combined(block))
            else:
                extracted_blocks.append(extract_fields_TA(block))

    return extracted_blocks
def generate_csv_answer_eval_combined(dir_names_TA):
    all_groups_dict=[]
    for TA in dir_names_TA:
        path=f"{TA}/viva_answer_files"
        for file in os.listdir(path):
            file_path=f"{TA}/viva_answer_files/{file}"
            data=read_questions_file_into_dict(file_path,False)
            for d in data:
                d["group_id"]=extract_group_id(file,5)
                d["TA_id"]=TA
            all_groups_dict.append(data)
    json_nested_list = all_groups_dict
    flattened_list = [item for sublist in json_nested_list for item in sublist]
    csv_file_name = "ASSMT3_combined_answerEval.csv"
    header = flattened_list[0].keys()
    with open(csv_file_name, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header)
        writer.writeheader()
        for data in flattened_list:
            writer.writerow(data)
    print(f"Data has been successfully written to {csv_file_name}")

def generate_csv_answer_eval_TA(dir_names_TA):
    all_groups_dict=[]
    for TA in dir_names_TA:
        path=f"{TA}/viva_answer_files"
        for file in os.listdir(path):
            file_path=f"{TA}/viva_answer_files/{file}"
            data=read_questions_file_into_dict(file_path,True)
            for d in data:
                d["group_id"]=extract_group_id(file,5)
                d["TA_id"]=TA
            all_groups_dict.append(data)
    json_nested_list = all_groups_dict
    flattened_list = [item for sublist in json_nested_list for item in sublist]
    csv_file_name = "ASSMT3_TAonly_answerEval.csv"
    header = flattened_list[0].keys()
    with open(csv_file_name, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header)
        writer.writeheader()
        for data in flattened_list:
            writer.writerow(data)
    print(f"Data has been successfully written to {csv_file_name}")
def generate_csv_code_eval(dir_names_TA):
    all_groups_dict=[]
    for TA in dir_names_TA:
        path=f"{TA}/ta_code_evaluation_feedback"
        for file in os.listdir(path):
            file_path=f"{TA}/ta_code_evaluation_feedback/{file}"
            data=convert_feedback_to_json(file_path)
            for d in data:
                d["group_id"]=extract_group_id(file,6)
                d["TA_id"]=TA
            all_groups_dict.append(data)
    json_nested_list = all_groups_dict
    flattened_list = [item for sublist in json_nested_list for item in sublist]
    csv_file_name = "ASSMT3_codeEval.csv"
    header = flattened_list[0].keys()
    with open(csv_file_name, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header)
        writer.writeheader()
        for data in flattened_list:
            writer.writerow(data)
    print(f"Data has been successfully written to {csv_file_name}")
        

     


n=int(input("The Number of TA's: "))
dir_names_TA=[]
for i in range(n):
    dir_name=input(f"Enter the name of directory for TA {i+1} : ")
    dir_names_TA.append(dir_name)

generate_csv_answer_eval_combined(dir_names_TA)
generate_csv_answer_eval_TA(dir_names_TA)
generate_csv_code_eval(dir_names_TA)

print("All  CSV files created successfully")
