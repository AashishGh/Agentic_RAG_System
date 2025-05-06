#When you run this script pdf documents are converted into txt files

from pdfminer.high_level import extract_text
import os

# Root documents folder
documents_root = './documents/'

# Auto-detect all subdirectories (e.g., biofilms/, material/, etc.)
pdf_directories = [
    d for d in os.listdir(documents_root)
    if os.path.isdir(os.path.join(documents_root, d))
]

for pdf_directory in pdf_directories:
    pdf_paths=[]
    # List all files in the directory
    for filename in os.listdir('./documents/'+pdf_directory):
        if filename.endswith('.pdf'):
            # Construct the full path to the PDF file
            pdf_paths.append(os.path.join(pdf_directory, filename))
    # print(pdf_paths)

    def extract_text_from_pdf(pdf_path):
        return extract_text(pdf_path)

    dir_length=len(pdf_directory)
    text_output_paths = [pdf_path[:dir_length]+'texts/'+pdf_path[dir_length:-3]+"txt" for pdf_path in pdf_paths]  # Make sure this is different from pdf_path

    # Write the extracted text to the output file
    for text_output_path,pdf_path in zip(text_output_paths,pdf_paths):
        extracted_text = extract_text_from_pdf(pdf_path)
        with open(text_output_path, 'w', encoding='utf-8') as f:
            f.write(extracted_text)

