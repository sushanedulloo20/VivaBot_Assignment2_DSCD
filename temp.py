from PyPDF2 import PdfReader

pdf = PdfReader('W24 CSE530 DSCD Assignment 2.pdf')

for page in pdf.pages:
    print(page.extract_text())
    