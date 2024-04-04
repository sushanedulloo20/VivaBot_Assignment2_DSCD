
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from datetime import datetime

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
import time
import random
import PyPDF2
import json
def evaluate(students_answer,VectorStore,llm_res):
    
    evaluation_query=f'''You have to give marks to the student out of 6 .(Each question has 2 marks) Below are the students and teachers response.
            Students response: {students_answer}
            Teachers response:{llm_res}
            If the answer is irrelevant then give zero.Display marks given to each question and then the total final marks .'''
    grade=ask_question(evaluation_query,VectorStore)
def split(question_answer:str):
    ind_questionmark=question_answer.find("?")
    question=question_answer[:ind_questionmark+1]
    answer=question_answer[ind_questionmark+1:]
    return question,answer

# def read_pdf(pdf_path):
#     # Open the PDF file
#     with open(pdf_path, 'rb') as file:
#         # Create a PDF reader object
#         pdf_reader = PyPDF2.PdfReader(file)
        
#         # Read each page of the PDF
#         text = ''
#         for page_num in range(len(pdf_reader.pages)):
#             page = pdf_reader.pages[page_num]
#             text += page.extract_text()
        
#         return text
def read_pdf(pdf_path):
    # Open the PDF file
    with open(pdf_path, 'rb') as file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(file)
        
        # Read each page of the PDF
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        
        return text

def read_multiple_pdfs(pdf_paths):
    all_texts = []
    for pdf_path in pdf_paths:
        text = read_pdf(pdf_path)
        all_texts.append(text)
    return all_texts


    




    return grade

def format_query(query:str):
    l=query.split(', ')
    s="\n".join(l)
    return s
def choose(list_topics):
    return random.choice(list_topics)
def randomize_list(list_topics:list):
    x=list_topics
    new_list=[]
    for i in range(len(x)):
        choosed_item=choose(list_topics)
        new_list.append(choosed_item)
        list_topics.remove(choosed_item)
    topics_choosen=new_list
    # topics_choosen=[]
    # for i in range(5):

    #     topic=choose(new_list)
    #     new_list.remove(topic)
    #     topics_choosen.append(topic)
    return topics_choosen

    
def ask_question(query,VectorStore):
    
    
    docs = VectorStore.similarity_search(query=query, k=3)
    
    llm = OpenAI(api_key=os.getenv('API_KEY'))
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=query)
            
        

    return response
 
load_dotenv()
 
def main():
    print("Viva Questions")
    group_number=input("Enter the group number:")
    num_questions=int(input("Enter the number of questions you wish to generate: "))
 
 
    # upload a PDF file
    pdf = ["VivaBot - W24 CSE530 DSCD Assignment 2.pdf"]
    # pdf = ["VivaBot - W24 CSE530 DSCD Assignment 2.pdf","raft.pdf"]
    text_list=read_multiple_pdfs(pdf)
    text=''''''
    pdf_id=0
    pdf_topics=["Assignment PDF"]
    # pdf_topics=["Assignment PDF","Raft Paper PDF"]
    for t in text_list:
        # text=text+f'''PDF Topic:{pdf_topics[pdf_id]}\n\n{t}\nNext Pdf\n'''
        text=text+f'''PDF Topic:{pdf_topics[pdf_id]}\n\n{t}\n'''
        pdf_id+=1
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
        )
    chunks = text_splitter.split_text(text=text)
    embeddings = OpenAIEmbeddings(api_key=str(os.getenv('API_KEY')))
    VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

    
    import_topics=['Raft', 'Leader Election in Raft', 'Log Replication in Raft', 'RequestVote RPC', 'AppendEntry RPC', 'Leader Lease in Raft', 'Heartbeats in Raft' , 'Fault Tolerance in Raft']

    

    if import_topics!=[]:
        
            
        questions=[]
        topics_choosen=randomize_list(import_topics)
        c=0
        for i in range(num_questions):
            if c>=len(topics_choosen):
                c=0
                
            q2=f'''Generate one different,very detailed and technical question  along with their answer (IN SEPERATE LINES) STRICTLY from the text.Question should be strictly from the topic given below:
                                        TOPIC:
                                        {topics_choosen[c]}
                                .Do NOT REPEAT THE PREVIOUSLY ASKED QUESTIONS '''
            
            response=ask_question(q2,VectorStore)
            questions.append(response)
            c+=1
            # student_response=st.empty()
        al_questions="\n\n\n".join(questions)
        new_q=[]
        new_a=[]
        for q in questions:
            q1,a1=split(q)
            new_q.append(q1)
            new_a.append(a1)
        for i in range(num_questions):
            print(f"Question {i+1}:{new_q[i]}")
        final_list_dictionary=[]
        for i in range(num_questions):
            d={}
            d["question"]=new_q[i]
            d["model_answer"]=new_a[i]
            d["student_answer"]=""
            d["TA_rating_for_question_usefulness"]=""
            d["TA_comments_for_question_usefulness"]=""
            final_list_dictionary.append(d)
            timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
            question_file_path = os.path.join("question_files", f"students_answers_group_id_{group_number}_time_{timestamp}.json")

            with open(question_file_path, "w", encoding="utf-8") as json_file:
                json.dump(final_list_dictionary, json_file, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main()
