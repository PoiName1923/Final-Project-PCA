# Source/main.py
from Reader.reader_agent import DataExpander
from Vectorizer.vectorizer_module import FeatureVectorizer
from Evaluate.evaluate_error_module import mean_squared_error_manual, explained_variance
from PCA.pca_module import PCA
import numpy as np
import pandas as pd
import os
import time 

def main():
    t1 = time.time()
    log_path = "log_txt.txt" # CHANGE NAME FOLLOWING THE TASK REQUIREMENTS, EX: log_csv, log_txt, ...

    with open(log_path, "w", encoding="utf-8") as log_file:
        error_mean = 0
        explain_var_mean = 0

        for i in range(1, 6):
            path = f"./Source/Test_Data/txt/file{i}.txt"  # change path here
            data = DataExpander().expand(path)
      
            fv = FeatureVectorizer()
            vectorized_data = fv.vectorize(data)

            my_pca = PCA().fit(vectorized_data[0])
            X_reduced = my_pca.transform(vectorized_data[0])
            X_reconstructed = my_pca.inverse_transform(X_reduced)

            error = mean_squared_error_manual(vectorized_data[0], X_reconstructed)
            explain_var = explained_variance(vectorized_data[0], X_reconstructed)

            error_mean += error
            explain_var_mean += explain_var

            print(f"File {i}: MSE = {error}, Explained Variance = {explain_var}")
            log_file.write(f"File {i}:\n")
            log_file.write(f"  MSE: {error}\n")
            log_file.write(f"  Explained Variance: {explain_var}\n\n")

        error_mean /= 5
        explain_var_mean /= 5 

        log_file.write(f"Mean Error (MSE): {error_mean}\n")
        log_file.write(f"Mean Explained Variance: {explain_var_mean}\n")

        print(f"Mean Error (MSE): {error_mean}")
        print(f"Mean Explained Variance: {explain_var_mean}")
        print(f"Done. Log saved to {log_path}")

    t2 = time.time()
    print(f"Total time taken: {t2 - t1:.2f} seconds")

if __name__ == "__main__":
    main()
