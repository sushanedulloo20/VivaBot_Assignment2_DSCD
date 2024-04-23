import textwrap

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import json
import os
import sys
import utils
from dotenv import load_dotenv

load_dotenv()


def load_questions_from_json(file_path):
    with open(file_path, 'r') as file:
        questions_data = json.load(file)
    return questions_data

def evaluate_answer(question, student_answer, model_answer, VectorStore):
    prompt = f'''
    You are a teaching assistant bot that has to evaluate the students answers for a conducted viva. 
    The question asked was: "{question}".
    
    The model answer to this question is: "{model_answer}". 
    The corresponding student answer is: "{student_answer}". 
    
    Evaluate this student answer based on the model answer. Your evaluation should be comprehensive, addressing the following criteria:

    1. Correctness: Identify and explain what is correct in the student answer. Highlight specific lines that align with the question's expectations and the model answer.
    2. Errors: Point out any errors or incorrectness in the student answer. Explain why these are errors.
    3. Omissions: Discuss any specifications that are missing in the student answer. Mention what should be added to meet the question's expectations.
    4. Overall Evaluation: Provide a concise overall assessment of the student answer, summarizing its strengths and weaknesses.
    '''
    
    docs = VectorStore.similarity_search(query=prompt, k=3)
    llm = OpenAI(api_key=os.getenv('API_KEY'))
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=prompt)
    return response
 
def main():
    # upload PDF files to be included for context
    pdfs = {'Assignemt': 'W24 CSE530 DSCD Assignment 3 - For Evaluation using LLMs.pdf', 'Map Reduce for K-Means Algorithm': 'MapReduce for K-Means Clustering.pdf'}
    sup_text = ""

    for key, value in pdfs.items():
        sup_text += f" {key} : \n" 
        if value.endswith('.txt'):
            with open(value, 'r') as file:
                sup_text += file.read()
        elif value.endswith('.pdf'):
            pdf = PdfReader(value)
            for page in pdf.pages:
                sup_text += page.extract_text() + "\n"
        sup_text += "\n\n"

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
        )
    chunks = text_splitter.split_text(text=sup_text)
 
    embeddings = OpenAIEmbeddings(api_key=os.getenv('API_KEY'))
    VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
    
    #change the path to JSON file with the path to JSON file containing the answers to be evaluated
    data_url = sys.argv[1]
    data_url = data_url.replace('\\', '/')
    data_parts = data_url.rstrip('/').split('/')
    data = data_parts[-1]
    # questions_data = load_questions_from_json(os.path.join("viva_question_files",data))
    questions_data = utils.read_questions_file_into_dict(os.path.join("viva_question_files", data))

    for i, qa in enumerate(questions_data, 1):
        evaluation = evaluate_answer(qa['question'], qa['student_answer'], qa['model_answer'], VectorStore)
        qa['evaluation'] = evaluation
        qa['TA_rating_for_LLM_feedback'] = "0"
        qa['TA_comments_for_LLM_feedback'] = "NA"
        qa['TA_score_given_to_student_answer'] = "-1"

    evaluate_answer_path = os.path.join("viva_answer_files", f"evaluated_{data}")
    # with open(evaluate_answer_path, 'w', encoding="utf-8") as file:
    #     json.dump(questions_data, file, indent=4, ensure_ascii=False)

    with open(evaluate_answer_path, "w", encoding="utf-8") as txt_file:
        for i, block in enumerate(questions_data, start=1):
            # txt_file.write(f"""Question: {textwrap.indent(textwrap.dedent(d['question']))}\n""")
            wrapper = textwrap.TextWrapper(width=100)

            txt_file.write("""******************************************************************\n\n""")
            txt_file.write(f"""**Question**\n""")
            lines = textwrap.dedent(block['question'].strip()).split("\n")
            for line in lines:
                txt_file.write(wrapper.fill(line) + "\n")
            txt_file.write("\n")

            txt_file.write(f"""**Model Answer**\n""")
            lines = textwrap.dedent(block['model_answer'].strip()).split("\n")
            for line in lines:
                txt_file.write(wrapper.fill(line) + "\n")
            txt_file.write("\n")

            txt_file.write("""**Student Answer**\n\n""")
            lines = textwrap.dedent(block['student_answer'].strip()).split("\n")
            for line in lines:
                txt_file.write(wrapper.fill(line) + "\n")
            txt_file.write("\n")

            txt_file.write("""**Answer Evaluation By LLM**\n\n""")
            lines = textwrap.dedent(block['evaluation'].strip()).split("\n")
            for line in lines:
                txt_file.write(wrapper.fill(line) + "\n")
            txt_file.write("\n")

            txt_file.write("""**TA Rating for Question Usefulness (1 being very poor, 5 being excellent)**\n\n""")
            lines = textwrap.dedent(block['rating'].strip()).split("\n")
            for line in lines:
                txt_file.write(wrapper.fill(line) + "\n")
            txt_file.write("\n")

            txt_file.write("""**TA Comments for Question Usefulness**\n\n""")
            lines = textwrap.dedent(block['comments'].strip()).split("\n")
            for line in lines:
                txt_file.write(wrapper.fill(line) + "\n")
            txt_file.write("\n")

            txt_file.write("""**TA Rating for LLM Feedback**\n\n""")
            lines = textwrap.dedent(block['TA_rating_for_LLM_feedback'].strip()).split("\n")
            for line in lines:
                txt_file.write(wrapper.fill(line) + "\n")
            txt_file.write("\n")

            txt_file.write("""**TA Comments for LLM Feedback**\n\n""")
            lines = textwrap.dedent(block['TA_comments_for_LLM_feedback'].strip()).split("\n")
            for line in lines:
                txt_file.write(wrapper.fill(line) + "\n")
            txt_file.write("\n")

            txt_file.write("""**TA Score for Student Answer**\n\n""")
            lines = textwrap.dedent(block['TA_score_given_to_student_answer'].strip()).split("\n")
            for line in lines:
                txt_file.write(wrapper.fill(line) + "\n")
            txt_file.write("\n")

            txt_file.write("""**End of TA Score**\n""")
            txt_file.write("""******************************************************************\n\n""")


if __name__ == '__main__':
    main()