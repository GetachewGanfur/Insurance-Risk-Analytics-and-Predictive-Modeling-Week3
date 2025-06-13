import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional
import logging

class Visualizer:
    """Class to handle data visualization."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the Visualizer.
        
        Args:
            data (pd.DataFrame): Input data for visualization
        """
        self.data = data
        self.logger = logging.getLogger(__name__)
        plt.style.use('seaborn')
    
    def plot_loss_ratio_by_category(self, category_column: str, 
                                  figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot loss ratio by category.
        
        Args:
            category_column (str): Column to group by
            figsize (Tuple[int, int]): Figure size
        """
        loss_ratio = self.data.groupby(category_column).apply(
            lambda x: x['TotalClaims'].sum() / x['TotalPremium'].sum()
        ).sort_values(ascending=False)
        
        plt.figure(figsize=figsize)
        loss_ratio.plot(kind='bar')
        plt.title(f'Loss Ratio by {category_column}')
        plt.xlabel(category_column)
        plt.ylabel('Loss Ratio')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_distribution(self, column: str, 
                         figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot distribution of a numerical column.
        
        Args:
            column (str): Column to plot
            figsize (Tuple[int, int]): Figure size
        """
        plt.figure(figsize=figsize)
        sns.histplot(data=self.data, x=column, kde=True)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
    
    def plot_temporal_trend(self, date_column: str, value_column: str,
                          figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot temporal trend of a value column.
        
        Args:
            date_column (str): Date column
            value_column (str): Value column to plot
            figsize (Tuple[int, int]): Figure size
        """
        monthly_data = self.data.groupby(
            pd.Grouper(key=date_column, freq='M')
        )[value_column].mean()
        
        plt.figure(figsize=figsize)
        monthly_data.plot()
        plt.title(f'Monthly {value_column} Trend')
        plt.xlabel('Date')
        plt.ylabel(value_column)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_matrix(self, columns: List[str],
                              figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Plot correlation matrix for specified columns.
        
        Args:
            columns (List[str]): Columns to include in correlation matrix
            figsize (Tuple[int, int]): Figure size
        """
        corr_matrix = self.data[columns].corr()
        
        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show()
    
    def plot_boxplot(self, column: str, group_by: Optional[str] = None,
                    figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot boxplot for a column, optionally grouped by another column.
        
        Args:
            column (str): Column to plot
            group_by (Optional[str]): Column to group by
            figsize (Tuple[int, int]): Figure size
        """
        plt.figure(figsize=figsize)
        if group_by:
            sns.boxplot(data=self.data, x=group_by, y=column)
            plt.xticks(rotation=45)
        else:
            sns.boxplot(data=self.data, y=column)
        plt.title(f'Boxplot of {column}')
        plt.tight_layout()
        plt.show() 