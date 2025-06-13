import logging
from data_loader import DataLoader
from eda_analyzer import EDAAnalyzer
from visualizer import Visualizer

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """Main function to run the analysis."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize data loader
        data_loader = DataLoader('data/insurance_data.csv')
        data = data_loader.load_data()
        data = data_loader.preprocess_data()
        
        # Get data summary
        summary = data_loader.get_data_summary()
        logger.info("Data Summary:")
        logger.info(f"Shape: {summary['shape']}")
        logger.info(f"Missing Values: {summary['missing_values']}")
        
        # Initialize EDA analyzer
        eda = EDAAnalyzer(data)
        
        # Calculate loss ratios
        overall_loss_ratio = eda.calculate_loss_ratio()
        logger.info(f"Overall Loss Ratio: {overall_loss_ratio['LossRatio'][0]:.2f}")
        
        # Calculate loss ratio by province
        province_loss_ratio = eda.calculate_loss_ratio(['Province'])
        logger.info("\nLoss Ratio by Province:")
        logger.info(province_loss_ratio)
        
        # Analyze distributions
        numeric_columns = ['TotalPremium', 'TotalClaims', 'CustomValueEstimate']
        distributions = eda.analyze_distributions(numeric_columns)
        logger.info("\nDistribution Analysis:")
        logger.info(distributions)
        
        # Detect outliers
        outliers = eda.detect_outliers(numeric_columns)
        logger.info("\nOutlier Analysis:")
        logger.info(outliers)
        
        # Analyze temporal trends
        temporal_trends = eda.analyze_temporal_trends('Date', 'TotalClaims')
        logger.info("\nTemporal Trends:")
        logger.info(temporal_trends)
        
        # Analyze vehicle claims
        vehicle_claims = eda.analyze_vehicle_claims()
        logger.info("\nVehicle Claims Analysis:")
        logger.info(vehicle_claims)
        
        # Initialize visualizer
        visualizer = Visualizer(data)
        
        # Create visualizations
        logger.info("\nCreating visualizations...")
        
        # Plot loss ratio by province
        visualizer.plot_loss_ratio_by_category('Province')
        
        # Plot distributions
        for col in numeric_columns:
            visualizer.plot_distribution(col)
        
        # Plot temporal trends
        visualizer.plot_temporal_trend('Date', 'TotalClaims')
        
        # Plot correlation matrix
        visualizer.plot_correlation_matrix(numeric_columns)
        
        # Plot boxplots
        for col in numeric_columns:
            visualizer.plot_boxplot(col)
            visualizer.plot_boxplot(col, group_by='Province')
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 