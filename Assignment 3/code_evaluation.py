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
import openai
from dotenv import load_dotenv
import textwrap

load_dotenv()


def code_Evaluation(question,code_snippet,VectorStore):
    query_for_code_eval = f'''
    You are a teaching assistant bot tasked with evaluating code snippets from a student's programming assignment. Your evaluation should be comprehensive, addressing the following criteria:

    1. Correctness: Identify and explain what is implemented correctly in the code snippet. Highlight specific lines or logic that align with the assignment's requirements.
    2. Errors: Point out any syntactical or logical errors in the code. Explain why these are errors and how they affect the program's functionality.
    3. Omissions: Discuss any requirements or specifications from the assignment that are missing in the code snippet. Mention what should be added to meet the assignment's complete criteria.
    4. Overall Evaluation: Provide a concise overall assessment of the code snippet, summarizing its strengths and weaknesses.

    Rubric: {question}
    Code Snippet: {code_snippet}

    Evaluate the provided code snippet based on these criteria and provide detailed feedback on its functionality. Mention any additional considerations or improvements that could enhance the code's effectiveness or efficiency.
    '''



    llm = OpenAI(api_key=os.getenv('API_KEY'))
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    docs = VectorStore.similarity_search(query=query_for_code_eval, k=3)
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=query_for_code_eval)
    return response

def code_summarization(code_snippet):
    query_for_additional_insights = f'''
    You are a teaching assistant bot tasked with providing additional insights into a student's programming assignment submission. Your response should include the following components:

    1. Code Summary: Provide a summary of what the code is intended to do. Describe the main functionality and logic of the code snippet in 5-7 lines, focusing on key operations and the flow of data.

    2. Relevant Questions: Generate two questions based on the code snippet that test the student's understanding and ability to write similar code.

    Code Snippet: {code_snippet}

    Please provide a clear and concise summary and formulate questions that can help assess the depth of the student's understanding of the concepts implemented in the code snippet.
    '''

    llm = openai.OpenAI(api_key=os.getenv('API_KEY'))
    messages = [{'role': 'system', 'content': '''You are a teaching assistant bot tasked with providing additional insights into a student's programming assignment submission. Your response should include the following components:
    1. Code Summary: Provide a summary of what the code is intended to do. Describe the main functionality and logic of the code snippet in 5-7 lines, focusing on key operations and the flow of data.
    2. Relevant Questions: Generate two questions based on the code snippet that test the student's understanding and ability to write similar code.'''}]

    messages.append({'role': 'user', 'content': query_for_additional_insights})
    response = llm.chat.completions.create(model="gpt-4-turbo", messages=messages)

    return response.choices[0].message.content

def write_to_file(question, evaluation_response, summarization_response, output_file):
    with open("output.txt", "a") as output_file:
        wrapper = textwrap.TextWrapper(width=100)
        lines = question.split("\n")
        for line in lines:
            output_file.write(wrapper.fill(line) + "\n")
        output_file.write("\n")
        output_file.write("Code Evaluation:\n")
        lines = evaluation_response.split("\n")
        for line in lines:
            output_file.write(wrapper.fill(line) + "\n")
        output_file.write("\n")
        output_file.write("Code Summarization and questions:\n")
        lines = summarization_response.split("\n")
        for line in lines:
            output_file.write(wrapper.fill(line) + "\n")
        output_file.write("\n\n")

def main():
    rubric = 'rubric.json'

    with open(rubric,"r") as fd:
        data=json.load(fd)

    pdfs = {'Assignemt': 'W24 CSE530 DSCD Assignment 3.pdf', 'Map Reduce for K-Means Algorithm': 'MapReduce for K-Means Clustering.pdf'}

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

    for q in data:
        ques=q["question"]
        pause=input(f'''Please paste the code snippet for
                        Question:  {ques}
                        After that press [Enter] to continue...''')
        
        with open("codesnippet.txt","r") as fc:
            code=fc.read()

        output_file = open("output.txt", "a")
        evaluation_response = code_Evaluation(ques, code, VectorStore)
        summarization_response = code_summarization(code)

        write_to_file(ques, evaluation_response, summarization_response, output_file)

if __name__ == "__main__":
    main()