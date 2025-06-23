import os
# Path to the files
txt_path   = os.path.join("data_test_for_reader", "report.txt")
image_path = os.path.join("data_test_for_reader", "cat.jpg")
table_path = os.path.join("data_test_for_reader", "Network.csv")
pdf_path   = os.path.join("data_test_for_reader", "test_pdf.pdf")
docx_path  = os.path.join("data_test_for_reader", "report.docx")
json_path  = os.path.join("data_test_for_reader", "data.json")
html_path  = os.path.join("data_test_for_reader", "article.html")
audio_path = os.path.join("data_test_for_reader", "music.mp3")

# main.py
from src.Reader.reader_agent import DataExpander
from src.Vectorizer.vectorizer_module import FeatureVectorizer
from pca_module import PCA
from evaluate_error_module import mean_squared_error_manual
from evaluate_error_module import explained_variance

import numpy as np

def main():
    path = "Final-Project-PCA/data_test_for_reader/file.txt"
    
    # 1. read data
    data = DataExpander().expand(path)
    # print(data)
    # # 2. Vectorize 

    fv = FeatureVectorizer()
    vectorized_data = fv.vectorize(data)

    print(vectorized_data[0].shape)
    print(vectorized_data[0])
 
    # # 3.1 apply PCA for many files
    # # vectors = []
    # # for file in list_of_paths:
    # #     data_info = DataExpander().expand(file)
    # #     vector = fv.vectorize(data_info)
    # #     vectors.append(vector)

    # # X = np.vstack(vectors)  # shape: (num_samples, num_features)
    # # my_pca = PCA().fit(X)

    # 3.2: apply PCA for 1 file
    import time 
    start_time = time.time()
    my_pca = PCA().fit(vectorized_data[0])
    X_reduced = my_pca.transform(vectorized_data[0])
    X_reconstructed = my_pca.inverse_transform(X_reduced)
    end_time = time.time()
    print(f"Time taken for PCA: {end_time - start_time} seconds")
    # # 4. evaluate error
    error = mean_squared_error_manual(vectorized_data[0], X_reconstructed)
    print(error) 
    explain_var = explained_variance(vectorized_data[0], X_reconstructed)
    print(explain_var)
    # # test with sample test
    
if __name__ == "__main__":
    main()

"""
images: oke
DataFrame: oke
Text: Oke
audio: oke
"""
