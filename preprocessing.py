import pandas as pd
import re
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def remove_null_values(data):
    """
    Remove rows with null values from the DataFrame.
    
    Parameters:
        data (DataFrame): Input DataFrame containing the data.
        
    Returns:
        DataFrame: DataFrame with null values removed.
    """
    processed_data = data.dropna()
    return processed_data

def preprocess_text(text):
    """
    Preprocess text data by removing stop words, special characters, and blank spaces.
    
    Parameters:
        text (str): Input text data.
        
    Returns:
        str: Preprocessed text.
    """
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Join tokens back into a string
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

def preprocess_numeric(data, numeric_columns):
    """
    Preprocess numeric columns by standardizing them.
    
    Parameters:
        data (DataFrame): Input DataFrame containing the data.
        numeric_columns (list): List of column names containing numeric data.
        
    Returns:
        DataFrame: Processed DataFrame with numeric columns standardized.
    """
    processed_data = data.copy()
    scaler = StandardScaler()
    processed_data[numeric_columns] = scaler.fit_transform(processed_data[numeric_columns])
    
    return processed_data

def preprocess_textual(data, textual_columns):
    """
    Preprocess textual columns by applying the preprocess_text function.
    
    Parameters:
        data (DataFrame): Input DataFrame containing the data.
        textual_columns (list): List of column names containing textual data.
        
    Returns:
        DataFrame: Processed DataFrame with textual columns preprocessed.
    """
    processed_data = data.copy()
    for column in textual_columns:
        processed_data[column] = processed_data[column].apply(preprocess_text)
    return processed_data

# Take input of CSV file
file_path = 'input.csv'

# Load the CSV file into a DataFrame
data = pd.read_csv(file_path)

# Remove null values
data = remove_null_values(data)

# Specify the names of numeric and textual columns
numeric_columns = ['numeric_column1', 'numeric_column2']
textual_columns = ['text_column1', 'text_column2']

# Preprocess numeric data
processed_numeric_data = preprocess_numeric(data, numeric_columns)

# Preprocess textual data
processed_data = preprocess_textual(processed_numeric_data, textual_columns)

# Save the processed data to a new CSV file
processed_data.to_csv('processed_data.csv', index=False)
