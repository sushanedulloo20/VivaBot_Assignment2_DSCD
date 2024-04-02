
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
from dotenv import load_dotenv

load_dotenv()

def load_questions_from_json(file_path):
    with open(file_path, 'r') as file:
        questions_data = json.load(file)
    return questions_data

def evaluate_answer(question, student_answer, model_answer, VectorStore):
    prompt = f"""You are a teaching assistant bot that has to evaluate the students answers for a conducted viva. 
    The question asked was {question} and the model answer to this question is {model_answer}. The corresponding student answer is 
    {student_answer}. Evaluate this student answer based on the model answer. Give relevant feedback on the student submission in 
    3-5 points."""
    

    docs = VectorStore.similarity_search(query=prompt, k=3)
    
    llm = OpenAI(api_key=os.getenv('api_key'))
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=prompt)
  

    return response
 
def main():
    # upload a PDF file
    assignment = 'W24 CSE530 DSCD Assignment 2.pdf'

    sup_text=""
    if assignment.endswith('.txt'):
        with open(assignment, 'r') as file:
            sup_text = file.read()
    elif assignment.endswith('.pdf'):
        pdf = PdfReader(assignment)
        for page in pdf.pages:
            sup_text += page.extract_text() + "\n"


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
        )
    chunks = text_splitter.split_text(text=sup_text)
 
    embeddings = OpenAIEmbeddings(api_key=os.getenv('api_key'))
    VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
    
    #change the path to JSON file with the path to JSON file containing the answers to be evaluated
    data = sys.argv[1]
    questions_data = load_questions_from_json(data)

    for i, qa in enumerate(questions_data, 1):
        evaluation = evaluate_answer(qa['question'], qa['student_answer'], qa['model_answer'], VectorStore)
        qa['evaluation'] = evaluation

    with open(f"evaluated_{data}", 'w') as file:
        json.dump(questions_data, file, indent=4)
            
                
if __name__ == '__main__':
    main()