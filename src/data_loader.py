import pandas as pd
from typing import Optional, Dict, Any
import logging

class DataLoader:
    """Class to handle data loading and preprocessing operations."""
    
    def __init__(self, file_path: str):
        """
        Initialize the DataLoader.
        
        Args:
            file_path (str): Path to the data file
        """
        self.file_path = file_path
        self.data: Optional[pd.DataFrame] = None
        self.logger = logging.getLogger(__name__)
    
    def load_data(self) -> pd.DataFrame:
        """
        Load the data from the specified file path.
        
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            self.data = pd.read_csv(self.file_path)
            self.logger.info(f"Successfully loaded data from {self.file_path}")
            return self.data
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocess the loaded data.
        
        Returns:
            pd.DataFrame: Preprocessed data
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Convert date columns to datetime
        date_columns = self.data.select_dtypes(include=['object']).columns
        for col in date_columns:
            try:
                self.data[col] = pd.to_datetime(self.data[col])
            except:
                continue
        
        # Handle missing values
        self.data = self.data.fillna(self.data.mean(numeric_only=True))
        
        return self.data
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the data.
        
        Returns:
            Dict[str, Any]: Dictionary containing data summary
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        summary = {
            'shape': self.data.shape,
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'numeric_summary': self.data.describe().to_dict(),
            'categorical_summary': {
                col: self.data[col].value_counts().to_dict()
                for col in self.data.select_dtypes(include=['object']).columns
            }
        }
        
        return summary 