import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

class EDAAnalyzer:
    """Class to perform Exploratory Data Analysis."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the EDAAnalyzer.
        
        Args:
            data (pd.DataFrame): Input data for analysis
        """
        self.data = data
        self.logger = logging.getLogger(__name__)
    
    def calculate_loss_ratio(self, group_by: List[str] = None) -> pd.DataFrame:
        """
        Calculate loss ratio (TotalClaims / TotalPremium) with optional grouping.
        
        Args:
            group_by (List[str], optional): Columns to group by. Defaults to None.
            
        Returns:
            pd.DataFrame: Loss ratio analysis
        """
        if group_by:
            return self.data.groupby(group_by).apply(
                lambda x: x['TotalClaims'].sum() / x['TotalPremium'].sum()
            ).reset_index(name='LossRatio')
        return pd.DataFrame({
            'LossRatio': self.data['TotalClaims'].sum() / self.data['TotalPremium'].sum()
        }, index=[0])
    
    def analyze_distributions(self, columns: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Analyze distributions of specified columns.
        
        Args:
            columns (List[str]): Columns to analyze
            
        Returns:
            Dict[str, Dict[str, float]]: Distribution statistics
        """
        distributions = {}
        for col in columns:
            if col in self.data.select_dtypes(include=[np.number]).columns:
                distributions[col] = {
                    'mean': self.data[col].mean(),
                    'std': self.data[col].std(),
                    'skew': self.data[col].skew(),
                    'kurtosis': self.data[col].kurtosis()
                }
        return distributions
    
    def detect_outliers(self, columns: List[str], method: str = 'iqr') -> Dict[str, List[int]]:
        """
        Detect outliers in specified columns.
        
        Args:
            columns (List[str]): Columns to analyze
            method (str): Method to use ('iqr' or 'zscore')
            
        Returns:
            Dict[str, List[int]]: Dictionary of outlier indices by column
        """
        outliers = {}
        for col in columns:
            if method == 'iqr':
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers[col] = self.data[
                    (self.data[col] < (Q1 - 1.5 * IQR)) | 
                    (self.data[col] > (Q3 + 1.5 * IQR))
                ].index.tolist()
            elif method == 'zscore':
                z_scores = np.abs((self.data[col] - self.data[col].mean()) / self.data[col].std())
                outliers[col] = self.data[z_scores > 3].index.tolist()
        return outliers
    
    def analyze_temporal_trends(self, date_column: str, value_column: str) -> pd.DataFrame:
        """
        Analyze temporal trends in the data.
        
        Args:
            date_column (str): Name of the date column
            value_column (str): Name of the value column to analyze
            
        Returns:
            pd.DataFrame: Monthly aggregated data
        """
        return self.data.groupby(
            pd.Grouper(key=date_column, freq='M')
        )[value_column].agg(['mean', 'sum', 'count']).reset_index()
    
    def analyze_vehicle_claims(self) -> pd.DataFrame:
        """
        Analyze claims by vehicle make/model.
        
        Returns:
            pd.DataFrame: Claims analysis by vehicle
        """
        return self.data.groupby(['Make', 'Model']).agg({
            'TotalClaims': ['sum', 'mean', 'count'],
            'TotalPremium': 'sum'
        }).reset_index() 