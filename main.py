import os
from sys import implementation
import numpy as np
import pandas as pd
from src.Vectorizer.vectorizer import FeatureVectorizer
from src.Reader.reader_agent import DataExpander
import warnings
warnings.filterwarnings("ignore")

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
    table_data = reader.expand(table_path)
    #pdf_data = reader.expand(pdf_path)
    #docx_data = reader.expand(docx_path)
    #json_data = reader.expand(json_path)
    #html_data = reader.expand(html_path)    

    # 2. Vectorize data
    #text_vector = vectorizer.vectorize(text_data)
    #print(text_vector.shape)
    #image_vector = vectorizer.vectorize(image_data)
    table_vector = vectorizer.vectorize(table_data)
    #pdf_vector = vectorizer.vectorize(pdf_data)
    #docx_vector = vectorizer.vectorize(docx_data)
    #html_vector = vectorizer.vectorize(html_data)
    #json_vector = vectorizer.vectorize(json_data)

    # 3. Print results
    #print("text after vectorization: \n", text_vector)
    #print("image after vectorization:", image_vector)
    print("table after vectorization: \n", table_vector)
    print("shape of table before and after vectorization: \n", table_data[0]['content'].shape, table_vector[0].shape)
    #print("pdf after vectorization: \n", pdf_vector)
    #print("docx after vectorization: \n", docx_vector)
    #print("html after vectorization: \n", html_vector)
    #print("json after vectorization: \n", json_vector)

if __name__ == "__main__":
    main()

#table after vectorization: 
# [array([[-1.4536866955017875, -1.5711922679353918, -1.2571573752692051,
#        ..., -0.455216529497014, -0.8808441166601825,
#        -0.8841387301974818],
#       [-1.4536866955017875, -1.5711922679353918, -0.6327550954241935,
#        ..., -0.455216529497014, -0.8808441166601825,
#        -0.8841387301974818],
#       [-1.4536866955017875, -1.5711922679353918, -0.6254249066299262,
#        ..., -0.455216529497014, -0.8808441166601825,
#        -0.8841387301974818],
#       ...,
#       [1.1239011264459569, 1.5241187584099283, 0.8249598449024486, ...,
#        -0.455216529497014, 0.3641951636191139, 1.5530615776386347],
#       [1.1239011264459569, 1.5241187584099283, 1.513539454763936, ...,
#        -0.455216529497014, 1.6092344438984103, 1.5530615776386347],
#       [1.1239011264459569, 1.5241187584099283, 1.5867649865733358, ...,
#        -0.455216529497014, 1.6092344438984103, 1.5530615776386347]],
#      shape=(3822, 14), dtype=object)]
#shape of table before and after vectorization: 
# (3822, 14) (3822, 14)
