
import pandas as pd


def clean_reviews(reviews_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean reviews DataFrame by removing unnecessary rows by
    ensuring we only keep reviews with a rating and user_id
    and filtering reviews that have valid ratings (1 to 5)
    """
    reviews_df = reviews_df.dropna(subset=['rating', 'user_id'])
    reviews_df = reviews_df[reviews_df['rating'].isin([1, 2, 3, 4, 5])]
    
    return reviews_df

