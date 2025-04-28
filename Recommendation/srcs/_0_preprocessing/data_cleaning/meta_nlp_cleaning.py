import pandas as pd
import re
import os
import sys
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from logger import get_module_logger

# Initialize core components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
logger = get_module_logger("meta_nlp_cleaning")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/../../")
sys.path.append(PROJECT_ROOT)

from settings import SAMPLE_NLP_CLEANED_META_DATA_PATH


def clean_text(text):
    """
    Clean and preprocess text for NLP tasks, using regex-based tokenization.
    """
    logger.info(f"Starting text cleaning. Input type: {type(text)}")
    if isinstance(text, str):
        text = text.lower().strip()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        tokens = re.findall(r'\b\w+\b', text)
        cleaned_tokens = [
            lemmatizer.lemmatize(tok)
            for tok in tokens
            if tok not in stop_words
        ]
        cleaned = ' '.join(cleaned_tokens)
        logger.info(f"Completed text cleaning. Cleaned text length: {len(cleaned)}")
        return cleaned
    else:
        logger.warning("Received non-string input for text cleaning. Returning empty string.")
        return ""


def nlp_metadata_cleaning(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply NLP preprocessing to relevant columns and save cleaned metadata.
    """
    try:
        logger.info(f"Starting NLP preprocessing on metadata with shape: {metadata_df.shape}")

        # 1. Clean text columns
        for col in ['title', 'description', 'store']:
            if col in metadata_df.columns:
                logger.info(f"Cleaning column: {col}")
                metadata_df[col] = metadata_df[col].apply(clean_text)
            else:
                logger.warning(f"Column '{col}' not found. Skipping.")

        # 2. Encode category
        if 'main_category' in metadata_df.columns:
            logger.info("Encoding 'main_category' column.")
            encoder = LabelEncoder()
            metadata_df['main_category'] = encoder.fit_transform(metadata_df['main_category'].astype(str))
        else:
            logger.warning("'main_category' column not found. Skipping encoding.")

        # 3. Save the NLP-cleaned metadata
        os.makedirs(os.path.dirname(SAMPLE_NLP_CLEANED_META_DATA_PATH), exist_ok=True)
        metadata_df.to_csv(SAMPLE_NLP_CLEANED_META_DATA_PATH, index=False)
        logger.info(f"Saved NLP-cleaned metadata to: {SAMPLE_NLP_CLEANED_META_DATA_PATH}")

        logger.info("NLP preprocessing completed successfully.")
        return metadata_df

    except Exception as e:
        logger.error("Error during NLP preprocessing.", exc_info=True)
        raise
