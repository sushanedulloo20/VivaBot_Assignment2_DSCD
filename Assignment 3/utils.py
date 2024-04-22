import re
import sys
import textwrap


def extract_fields(text):
    # Regular expression patterns to extract fields
    question_pattern = r'\*\*Question\*\*\n(.*?)\n\*\*Model Answer\*\*'
    model_answer_pattern = r'\*\*Model Answer\*\*\n(.*?)\n\*\*Student Answer\*\*'
    student_answer_pattern = r'\*\*Student Answer\*\*\n(.*?)\n\*\*TA Rating for Question Usefulness'
    rating_pattern = r'\*\*TA Rating for Question Usefulness \(1 being very poor, 5 being excellent\)\*\*\n(.*?)\n\*\*TA Comments'
    comments_pattern = r'\*\*TA Comments for Question Usefulness\*\*\n(.*?)\n\*\*End of TA Comments\*\*'

    # Extracting fields using regular expressions
    question = re.search(question_pattern, text, re.DOTALL).group(1).strip()
    model_answer = re.search(model_answer_pattern, text, re.DOTALL).group(1).strip()
    student_answer = re.search(student_answer_pattern, text, re.DOTALL).group(1).strip()
    rating = re.search(rating_pattern, text, re.DOTALL).group(1).strip()
    comments = re.search(comments_pattern, text, re.DOTALL).group(1).strip()
    fields = {
        'question': question,
        'model_answer': model_answer,
        'student_answer': student_answer,
        'rating': rating,
        'comments': comments
    }
    return fields


# Function to read the file and extract fields for each block
def read_text_into_dict(file_path):
    with open(file_path, 'r') as file:
        text = file.read()

    # Splitting text into blocks
    blocks = text.split('******************************************************************\n\n')

    # Extracting fields from each block
    extracted_blocks = []
    for block in blocks:
        if block.strip():  # Skipping empty blocks
            extracted_blocks.append(extract_fields(block))
    return extracted_blocks


# Example usage
# input_file_path = sys.argv[1]
# file_path = 'example_text_file.txt'  # Replace with the path to your text file
#
# blocks = extract_blocks(input_file_path)
#
# # Printing extracted fields for each block
# with open(file_path, "w", encoding="utf-8") as txt_file:
#     for i, block in enumerate(blocks, start=1):
#         # txt_file.write(f"""Question: {textwrap.indent(textwrap.dedent(d['question']))}\n""")
#         wrapper = textwrap.TextWrapper(width=100)
#
#         txt_file.write("""******************************************************************\n\n""")
#         txt_file.write(f"""**Question**\n""")
#         lines = textwrap.dedent(block['question'].strip()).split("\n")
#         for line in lines:
#             txt_file.write(wrapper.fill(line) + "\n")
#         txt_file.write("\n")
#
#         txt_file.write(f"""**Model Answer**\n""")
#         lines = textwrap.dedent(block['model_answer'].strip()).split("\n")
#         for line in lines:
#             txt_file.write(wrapper.fill(line) + "\n")
#         txt_file.write("\n")
#
#         txt_file.write("""**Student Answer**\n\n""")
#         lines = textwrap.dedent(block['student_answer'].strip()).split("\n")
#         for line in lines:
#             txt_file.write(wrapper.fill(line) + "\n")
#         txt_file.write("\n")
#
#         txt_file.write("""**TA Rating for Question Usefulness (1 being very poor, 5 being excellent)**\n\n""")
#         lines = textwrap.dedent(block['rating'].strip()).split("\n")
#         for line in lines:
#             txt_file.write(wrapper.fill(line) + "\n")
#         txt_file.write("\n")
#
#         txt_file.write("""**TA Comments for Question Usefulness**\n\n""")
#         lines = textwrap.dedent(block['comments'].strip()).split("\n")
#         for line in lines:
#             txt_file.write(wrapper.fill(line) + "\n")
#         txt_file.write("\n")
#         txt_file.write("""**End of TA Comments**\n""")
#         txt_file.write("""******************************************************************\n\n""")
