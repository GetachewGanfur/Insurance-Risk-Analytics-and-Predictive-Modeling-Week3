# Insurance Risk Analytics and Predictive Modeling

This project performs exploratory data analysis and risk assessment on insurance data to understand patterns in risk and profitability.

## Project Structure

```
.
├── data/                  # Data directory
├── notebooks/            # Jupyter notebooks
├── src/                  # Source code
│   ├── data_loader.py    # Data loading and preprocessing
│   ├── eda_analyzer.py   # Exploratory data analysis
│   ├── visualizer.py     # Data visualization
│   └── main.py          # Main script
├── tests/               # Test files
├── requirements.txt     # Project dependencies
└── README.md           # Project documentation
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your insurance data CSV file in the `data` directory as `insurance_data.csv`

2. Run the analysis:
```bash
python src/main.py
```

## Features

- Data loading and preprocessing
- Exploratory Data Analysis (EDA)
  - Loss ratio analysis by various categories
  - Distribution analysis
  - Outlier detection
  - Temporal trend analysis
  - Vehicle claims analysis
- Data visualization
  - Loss ratio plots
  - Distribution plots
  - Temporal trend plots
  - Correlation matrices
  - Box plots

## Key Analysis Areas

1. Loss Ratio Analysis
   - Overall portfolio loss ratio
   - Loss ratio by province
   - Loss ratio by vehicle type
   - Loss ratio by gender

2. Financial Variable Analysis
   - Distribution of key financial variables
   - Outlier detection in claims and premiums
   - Correlation analysis

3. Temporal Analysis
   - Monthly trends in claims and premiums
   - Seasonal patterns
   - Year-over-year comparisons

4. Vehicle Analysis
   - Claims by make/model
   - Premium distribution by vehicle type
   - Risk assessment by vehicle characteristics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.