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
    log_path = "log_docx.txt"

    with open(log_path, "w", encoding="utf-8") as log_file:
        total_error = 0
        total_explained_var = 0
        num_files = 5

        for file_idx in range(1, num_files + 1):
            path = f"./Source/Test_Data/txt/file{file_idx}.txt"
            data = DataExpander().expand(path)

            file_error_sum = 0
            file_explained_var_sum = 0
            valid_vector_count = 0

            for i in range(len(data)):
                try:
                    fv = FeatureVectorizer()
                    vectorized_data = fv.vectorize([data[i]])

                    # Nếu vector rỗng thì bỏ qua
                    if vectorized_data[0].size == 0:
                        continue

                    my_pca = PCA().fit(vectorized_data[0])
                    X_reduced = my_pca.transform(vectorized_data[0])
                    X_reconstructed = my_pca.inverse_transform(X_reduced)

                    error = mean_squared_error_manual(vectorized_data[0], X_reconstructed)
                    explain_var = explained_variance(vectorized_data[0], X_reconstructed)

                    file_error_sum += error
                    file_explained_var_sum += explain_var
                    valid_vector_count += 1
                except Exception as e:
                    print(f"[Warning] Skipped vectorization/PCA on file{file_idx}, data[{i}] due to: {e}")
                    continue

            if valid_vector_count == 0:
                print(f"[Warning] file{file_idx} has no valid data to process.")
                continue

            avg_file_error = file_error_sum / valid_vector_count
            avg_file_explain_var = file_explained_var_sum / valid_vector_count

            total_error += avg_file_error
            total_explained_var += avg_file_explain_var

            print(f"File {file_idx}: MSE = {avg_file_error}, Explained Variance = {avg_file_explain_var}")
            log_file.write(f"File {file_idx}:\n")
            log_file.write(f"  Avg MSE: {avg_file_error}\n")
            log_file.write(f"  Avg Explained Variance: {avg_file_explain_var}\n\n")

        # Trung bình trên tất cả file
        mean_error = total_error / num_files
        mean_explained_var = total_explained_var / num_files

        log_file.write(f"Mean Error (MSE): {mean_error}\n")
        log_file.write(f"Mean Explained Variance: {mean_explained_var}\n")

        print(f"Mean Error (MSE): {mean_error}")
        print(f"Mean Explained Variance: {mean_explained_var}")
        print(f"Done. Log saved to {log_path}")

    t2 = time.time()
    print(f"Total time taken: {t2 - t1:.2f} seconds")

if __name__ == "__main__":
    main()
