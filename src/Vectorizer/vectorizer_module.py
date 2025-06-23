import numpy as np
import pandas as pd
import cv2
import librosa
from tf_idf_module import TfidfVectorizer
from vector_for_text_Way1 import ManualTokenizer, auto_select_maxlen

class FeatureVectorizer:
    """
    A class to vectorize text, arrays, dataframes to a feature vector.
    """

    def __init__(self):
        self.tokenizer = ManualTokenizer(maxlen=512, scale="minmax")

    def _text_vectorizer(self, text: str) -> np.ndarray:
        """
        Vectorize text data into a 2D array using ManualTokenizer with optimal maxlen.
        
        Returns:
            np.ndarray: (num_chunks, maxlen)
        """
        if not isinstance(text, str):
            raise TypeError("Input text must be a string.")

        # 1. Tự động chọn maxlen tối ưu
        optimal_maxlen = auto_select_maxlen(text)
        self.tokenizer.maxlen = optimal_maxlen

        # 2. Build vocab nếu chưa có
        if not self.tokenizer._is_vocab_built:
            self.tokenizer.build_vocab(text)

        # 3. Vectorize
        return self.tokenizer.transform(text)

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
    
    def _extract_header_if_exists(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        Nhận diện và tách dòng đầu tiên nếu nó là header giả.
        Trả về: (DataFrame đã xử lý, meta dict chứa 'header' nếu có)
        """
        meta = {}

        # Chỉ kiểm tra khi cột là số (RangeIndex)
        if not isinstance(df.columns, pd.RangeIndex) or len(df) < 2:
            return df, meta  # Không nghi ngờ gì

        first_row = df.iloc[0]
        next_rows = df.iloc[1:6]  # lấy vài dòng sau để so sánh

        # ========== Rule 1: dòng đầu chứa toàn string có chữ ==========
        text_like_count = sum(isinstance(x, str) and any(c.isalpha() for c in str(x)) for x in first_row)
        is_text_dominated = text_like_count >= len(df.columns) * 0.5

        # ========== Rule 2: kiểu dữ liệu khác biệt rõ rệt ==========
        def get_type_list(row): return [type(x) for x in row]
        first_types = get_type_list(first_row)
        later_types = next_rows.apply(get_type_list, axis=1).values

        # Đếm số kiểu khác nhau giữa dòng đầu và các dòng sau
        type_mismatch_scores = [
            sum(ft != lt for ft, lt in zip(first_types, row)) for row in later_types
        ]
        avg_type_diff = sum(type_mismatch_scores) / max(len(type_mismatch_scores), 1)

        # ========== Rule 3: đặc trưng header (nhiều ký tự đặc biệt) ==========
        special_chars = set(":_-@/.#")
        special_score = sum(
            sum(c in special_chars for c in str(x)) for x in first_row
        ) / len(df.columns)

        # ========== Tổng hợp ==========
        if is_text_dominated and avg_type_diff >= len(df.columns) * 0.4 and special_score > 1.0:
            meta['header'] = [str(x) for x in first_row]
            df = df.iloc[1:].reset_index(drop=True)

        return df, meta

    def _is_link_like(self, s: str) -> bool:
        s = str(s).lower()
        return any(p in s for p in ['http', 'www.', '.com', '.org', '.net'])

    def _table_vectorizer(self, table_data: pd.DataFrame, length_threshold: int = 15) -> tuple[np.ndarray, dict]:
        if not isinstance(table_data, pd.DataFrame):
            raise TypeError("table_data must be a pandas DataFrame.")

        df = table_data.copy()
        df, meta = self._extract_header_if_exists(df)
        numeric_data, categorical_data, tfidf_data, date_data, bool_data = [], [], [], [], []
        meta.setdefault('ignored_link_columns', [])
        meta.setdefault('col_feature_types', {})

        for col in df.columns:
            series = df[col]

            # Handle missing values
            if series.isnull().any():
                if pd.api.types.is_numeric_dtype(series):
                    series = series.fillna(series.mean() if abs(series.skew()) < 1 else series.median())
                else:
                    series = series.fillna("missing")

            # ========== Check truly numeric ==========
            is_strict_numeric = pd.api.types.is_numeric_dtype(series) and not series.astype(str).str.contains('[a-zA-Z]', na=False).any()

            if is_strict_numeric:
                series = self._standard_scaler(series)
                numeric_data.append(series.to_numpy().reshape(-1, 1))
                meta['col_feature_types'][col] = 'numeric'

            # ========== Boolean ==========
            elif pd.api.types.is_bool_dtype(series):
                bool_data.append(series.astype(int).to_numpy().reshape(-1, 1))
                meta['col_feature_types'][col] = 'boolean'

            # ========== Datetime ==========
            else:
                try:
                    dt_series = pd.to_datetime(series, errors='raise')
                    ts = dt_series.astype('int64') // 10**9
                    ts = self._standard_scaler(ts)
                    date_data.append(ts.to_numpy().reshape(-1, 1))
                    meta['col_feature_types'][col] = 'datetime'
                    continue
                except Exception:
                    pass  # not datetime

                # ========== TEXT column ==========
                text_series = series.astype(str)
                n_unique = text_series.nunique()
                avg_length = text_series.apply(len).mean()
                ratio_special_chars = text_series.apply(lambda x: sum(1 for c in x if not c.isalnum()) / (len(x) + 1)).mean()
                is_link_col = text_series.apply(self._is_link_like).mean() > 0.3

                if is_link_col:
                    meta['ignored_link_columns'].append(col)
                    meta['col_feature_types'][col] = 'link_ignored'
                    continue  # skip processing this column

                # 💡 Categorical column
                if n_unique <= 30 and avg_length <= length_threshold:
                    unique_vals = list(text_series.unique())
                    encoded = text_series.apply(lambda x: unique_vals.index(x) if x in unique_vals else -1)
                    encoded = self._standard_scaler(encoded)
                    categorical_data.append(encoded.to_numpy().reshape(-1, 1))
                    meta['col_feature_types'][col] = 'categorical'

                # 💡 TF-IDF column
                elif avg_length > length_threshold or ratio_special_chars > 0.2:
                    self._tfidf_vectorizer.fit(text_series.tolist())
                    tfidf_matrix = np.vstack([
                        self._tfidf_vectorizer.transform(doc).reshape(1, -1)
                        for doc in text_series
                    ])
                    tfidf_data.append(tfidf_matrix)
                    meta['col_feature_types'][col] = 'tf-idf'
                # 💡 Fallback categorical
                else:
                    unique_vals = list(text_series.unique())
                    encoded = text_series.apply(lambda x: unique_vals.index(x) if x in unique_vals else -1)
                    encoded = self._standard_scaler(encoded)
                    categorical_data.append(encoded.to_numpy().reshape(-1, 1))
                    meta['col_feature_types'][col] = 'categorical_simple'

        all_parts = numeric_data + bool_data + date_data + categorical_data + tfidf_data
        vector = np.hstack(all_parts) if all_parts else np.empty((len(df), 0))

        return vector

    def _audio_vectorizer(self, audio_data: np.ndarray,
                          frame_length: int = 2048, 
                          hop_length: int = 512) -> np.ndarray:
        """
        Vectorize audio data into a 2D feature matrix by splitting audio_data into frames.

        Args:
            frame_length (int): Length of each frame in data.
            hop_length (int): Step size between frames in samples.
            audio_data (np.ndarray): 1D waveform array.

        Returns:
            np.ndarray: 
        """

        # Check params
        if not isinstance(audio_data, np.ndarray):
            raise TypeError("audio_data must be a np.ndarray.")

        # Create frames
        frames = librosa.util.frame(audio_data, frame_length = frame_length, hop_length = hop_length).T

        return frames

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

            if data['type'] not in ['text', 'image', 'table', 'audio']:
                raise ValueError("data_type must be one of 'text', 'image', 'table'.")
       
            # If condition to vectorize appropriate data type
            if data['type'] == 'text':
                vectorized_vector.append(self._text_vectorizer(data['content']))
            
            elif data['type'] == 'image':
                vectorized_vector.append(self._image_vectorizer(data['content']))

            elif data['type'] == 'audio':
                vectorized_vector.append(self._audio_vectorizer(data['content']))

            else: 
                vectorized_vector.append(self._table_vectorizer(data['content']))

        return vectorized_vector