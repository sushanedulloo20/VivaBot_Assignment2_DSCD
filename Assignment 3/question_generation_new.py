from dotenv import load_dotenv
from datetime import datetime
from langchain_openai import OpenAIEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks import get_openai_callback
import os
import random
import PyPDF2
import json


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


def format_query(query: str):
    l = query.split(', ')
    s = "\n".join(l)
    return s


def choose(list_topics):
    return random.choice(list_topics)


def randomize_list(list_topics: list):
    x = list_topics
    new_list = []
    for i in range(len(x)):
        choosed_item = choose(list_topics)
        new_list.append(choosed_item)
        list_topics.remove(choosed_item)
    topics_choosen = new_list
    return topics_choosen


def prompt_llm(query, VectorStore):
    docs = VectorStore.similarity_search(query=query, k=3)
    load_dotenv()
    llm = OpenAI(api_key=os.getenv('API_KEY'))
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=query)
    return response


load_dotenv()

import textwrap

def main():
    print("Viva Questions")
    group_number = input("Enter the group number:")
    num_questions = int(input("Enter the number of questions you wish to generate: "))

    # upload a PDF file
    pdf = ["W24 CSE530 DSCD Assignment 3 - For Evaluation using LLMs.pdf"]
    # pdf = ["VivaBot - W24 CSE530 DSCD Assignment 2.pdf","raft.pdf"]
    text_list = read_multiple_pdfs(pdf)
    text = ''''''
    pdf_id = 0
    pdf_topics = ["Assignment PDF"]
    # pdf_topics=["Assignment PDF","Raft Paper PDF"]
    for t in text_list:
        # text=text+f'''PDF Topic:{pdf_topics[pdf_id]}\n\n{t}\nNext Pdf\n'''
        text = text + f'''PDF Topic:{pdf_topics[pdf_id]}\n\n{t}\n'''
        pdf_id += 1
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text=text)
    load_dotenv()
    embeddings = OpenAIEmbeddings(api_key=os.getenv('API_KEY'))
    # llm = OpenAI(api_key=os.getenv('API_KEY'))

    VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

    important_topics = ["MapReduce", "Map phase in MapReduce", "Reduce phase in MapReduce", "Partition phase in MapReduce",
                     "Shuffle phase in MapReduce", "Sort phase in MapReduce"]

    if important_topics:
        questions = []
        answers = []
        topics_choosen = randomize_list(important_topics)
        topic_index = 0
        for i in range(num_questions):
            if topic_index >= len(topics_choosen):
                topic_index = 0
            prompt_to_llm = f'''You are a teaching assistant for distributed systems class. You need to take viva of a student in which the student will be asked a question and they will be given 3 minutes to write the answer for this question. Please generate a question strictly related to the topic given below:
                TOPIC:
                {topics_choosen[topic_index]}
                .Do NOT REPEAT THE PREVIOUSLY ASKED QUESTIONS'''
            response = prompt_llm(prompt_to_llm, VectorStore)
            questions.append(response)

            prompt_to_llm = f'''Please answer the following question in at most 5-7 lines:
                {response}'''
            response = prompt_llm(prompt_to_llm, VectorStore)
            answers.append(response)

            topic_index += 1

        for i in range(len(questions)):
            print(f"Question {i + 1}:{questions[i]}")
        final_list_dictionary = []

        for i in range(len(questions)):
            d = {
                "question": questions[i],
                "model_answer": answers[i],
                "student_answer": "",
                "TA_rating_for_question_usefulness": "",
                "TA_comments_for_question_usefulness": ""
            }
            final_list_dictionary.append(d)
            timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
            question_file_path = os.path.join("viva_question_files",
                                              f"students_answers_group_id_{group_number}_time_{timestamp}.txt")
            # with open(question_file_path, "w", encoding="utf-8") as json_file:
            #     json.dump(final_list_dictionary, json_file, indent=4, ensure_ascii=False)

            with open(question_file_path, "w", encoding="utf-8") as txt_file:
                for d in final_list_dictionary:
                    # txt_file.write(f"""Question: {textwrap.indent(textwrap.dedent(d['question']))}\n""")
                    wrapper = textwrap.TextWrapper(width=100)

                    txt_file.write("""******************************************************************\n\n""")
                    txt_file.write(f"""**Question**\n""")
                    lines = textwrap.dedent(d['question'].strip()).split("\n")
                    for line in lines:
                        txt_file.write(wrapper.fill(line) + "\n")
                    txt_file.write("\n")

                    txt_file.write(f"""**Model Answer**\n""")
                    lines = textwrap.dedent(d['model_answer'].strip()).split("\n")
                    for line in lines:
                        txt_file.write(wrapper.fill(line) + "\n")
                    txt_file.write("\n")

                    txt_file.write("""**Student Answer**\n\n""")
                    txt_file.write("""**TA Rating for Question Usefulness (1 being very poor, 5 being excellent)**\n\n0\n\n""")

                    txt_file.write("""**TA Comments for Question Usefulness**\n\n""")
                    txt_file.write("""NA\n\n""")
                    txt_file.write("""**End of TA Comments**\n""")
                    txt_file.write("""******************************************************************\n\n""")

if __name__ == '__main__':
    main()
