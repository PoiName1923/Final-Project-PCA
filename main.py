import os
import numpy as np
import pandas as pd
from src.Vectorizer.vectorizer import FeatureVectorizer
from src.Reader.reader_agent import DataExpander

# Path to the files
txt_path   = os.path.join("data_test_for_reader", "report.txt")
image_path = os.path.join("data_test_for_reader", "cat.jpg")
table_path = os.path.join("data_test_for_reader", "Network.csv")
pdf_path   = os.path.join("data_test_for_reader", "test_pdf.pdf")
docx_path  = os.path.join("data_test_for_reader", "report.docx")
json_path  = os.path.join("data_test_for_reader", "data.json")
html_path  = os.path.join("data_test_for_reader", "article.html")

def main():
    # 0. Call modules
    reader = DataExpander()
    vectorizer = FeatureVectorizer()

    # 1. Read files
    #text_data = dict(reader.expand(txt_path)[0])
    #image_data = reader.expand(image_path)
    #table_data = reader.expand(table_path)
    #pdf_data = reader.expand(pdf_path)
    #docx_data = reader.expand(docx_path)
    json_data = reader.expand(json_path)
    #html_data = reader.expand(html_path)    

    # 2. Vectorize data
    #text_vector = vectorizer.vectorize(text_data)
    #print(text_vector.shape)
    #image_vector = vectorizer.vectorize(image_data)
    #table_vector = vectorizer.vectorize(table_data)
    #pdf_vector = vectorizer.vectorize(pdf_data)
    #docx_vector = vectorizer.vectorize(docx_data)
    #html_vector = vectorizer.vectorize(html_data)
    json_vector = vectorizer.vectorize(json_data)

    # 3. Print results
    #print("text after vectorization: \n", text_vector)
    #print("image after vectorization:", image_vector)
    #print("table after vectorization: \n", table_vector)
    #print("pdf after vectorization: \n", pdf_vector)
    #print("docx after vectorization: \n", docx_vector)
    #print("html after vectorization: \n", html_vector)
    print("json after vectorization: \n", json_vector)

if __name__ == "__main__":
    main()
#json after vectorization: 
# [array([[-1.731144475553872, 0, True, ..., 0, 0, 0],
#       [-1.7299359977349862, 0, False, ..., 1, 1, 0],
#       [-1.7287275199161005, 0, True, ..., 0, 2, 0],
#       ...,
#       [1.7287275199161005, 0, True, ..., 2, 0, 0],
#       [1.7299359977349862, 0, True, ..., 2, 0, 0],
#       [1.731144475553872, 0, True, ..., 0, 1, 0]],
#      shape=(2866, 9), dtype=object)]
#
