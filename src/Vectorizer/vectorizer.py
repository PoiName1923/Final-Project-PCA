import numpy as np
import pandas as pd
import cv2
from .tf_idf_module import TfidfVectorizer


class FeatureVectorizer:
    """
    A class to vectorize text, arrays, dataframes to a feature vector.
    """

    def __init__(self):

        self._tfidf_vectorizer = TfidfVectorizer()


    def _text_vectorizer(self, text: str) -> np.ndarray:
        """
        Vectorize text data into a feature vector.

        Returns:
            np.ndarray: An array representing the feature vector of the text data.
        """
        return self._tfidf_vectorizer.transform(text)


    def _image_vectorizer(self, image_matrix: np.ndarray) -> np.ndarray:
        """
        Vectorize image data into a feature vector.
 
        Returns:
            np.ndarray: An array representing the feature vector of the image data.
        """

        # Check params
        if not isinstance(image_matrix, np.ndarray):
            raise TypeError("image_matrix must be a numpy array.")
        
        # Convert image to grayscale matrix
        gray_image_matrix = cv2.cvtColor(image_matrix, cv2.COLOR_BGR2GRAY)
        
        return gray_image_matrix    # Shape(m, n) 
       

    def _standard_scaler(self, series: pd.Series) -> pd.Series:
        """
        Apply standard scaling to a numeric Series.

        Args:
            series (pd.Series): The numeric column to scale.

        Returns:
            pd.Series: The scaled column.
        """

        # Check param
        if not isinstance(series, pd.Series):
            raise TypeError("Input must be a pandas Series.")

        if not pd.api.types.is_numeric_dtype(series):
            raise TypeError("Series must be of numeric dtype.")

        # Calculate mean & std
        mean = series.mean()
        std = series.std()

        return (series - mean) / std if std != 0 else series


    def _table_vectorizer(self, table_data: pd.DataFrame, length_threshold: int = 20) -> np.ndarray:
        """
        Normalize and vectorize table data into a feature vector.

        Returns:
            np.ndarray: An array representing the feature vector of the table data.
        """

        # Check params
        if not isinstance(table_data, pd.DataFrame):
            raise TypeError("table_data must be a pandas DataFrame.")

        # Traverse all cols 
        for col in table_data.columns:
            # ========== Process numeric cols ==========
            if pd.api.types.is_numeric_dtype(table_data[col]):
                # Fill missing values
                if table_data[col].isnull().any():
                    # Check standard distribution
                    skew = table_data[col].skew()

                    table_data[col] = (
                        table_data[col].fillna(table_data[col].mean()) if abs(skew) < 1 \
                        else table_data[col].fillna(table_data[col].median())
                    )

            # ========== Process text cols ==========
            # Try to convert to datetime first (if possible)
            else:

                try:
                    table_data[col] = pd.to_datetime(table_data[col], errors = 'raise')
                    
                    # Convert to Unix timestamp
                    table_data[col] = table_data[col].astype('int64') // 10**9
                    
                except Exception:
                    # If not datetime col, continue to handle text
                    pass

                # Fill null by '' and calculate avg length of text
                avg_length = table_data[col].fillna('').apply(lambda x: len(str(x))).mean()

                # Create list of unique values in the text column
                unique_values = list(table_data[col].unique())

                # Check
                # 1. If the average length is above the threshold, apply tf_idf to vectorize the text
                if avg_length > length_threshold:
                    table_data[col] = table_data[col].\
                            apply(lambda x: self._tfidf_vectorizer.transform(x) if pd.notnull(x) else np.array([0]))

                # 2. If it is a categorical column, encode using the index of unique values
                else:
                    table_data[col] = table_data[col].\
                                           apply(lambda x: unique_values.index(x)).fillna(-1).astype('int')
                    
            # Normalize data
            if pd.api.types.is_numeric_dtype(table_data[col]):
                table_data[col] = self._standard_scaler(table_data[col])
    
        return table_data.to_numpy()


    def _audio_vectorizer(self, audio_data):
        pass
        # To be continued... 
   

    def vectorize(self, list_data: list) -> np.ndarray:
        """
        Vectorize the data based on its type.
        
        Returns:
            np.ndarray: An array representing the feature vector of the data.
        """

        # Check params
        if not isinstance(list_data, list):
            raise TypeError("list_data must be a list.")
  
        # create vector to store all vectorized data
        vectorized_vector = []
            
        for data in list_data:    
            if not all(key in data for key in ['type', 'content', 'meta']):
                raise ValueError("list_data must contain 'type', 'content', and 'meta' keys.")

            if data['type'] not in ['text', 'image', 'table']:
                raise ValueError("data_type must be one of 'text', 'image', 'table'.")
       
            # If condition to vectorize appropriate data type
            if data['type'] == 'text':
                vectorized_vector.append(self._text_vectorizer(data['content']))
            
            elif data['type'] == 'image':
                vectorized_vector.append(self._image_vectorizer(data['content']))

            else: 
                vectorized_vector.append(self._table_vectorizer(data['content']))

        
        return vectorized_vector








