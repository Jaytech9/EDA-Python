# EDA Python - Boston Housing Analysis

A comprehensive Exploratory Data Analysis (EDA) project analyzing the Boston Housing dataset using Python. This project demonstrates various data analysis techniques, visualization methods, and statistical insights with automated report generation.

## 🎯 Project Overview

This project performs an in-depth exploratory data analysis on the Boston Housing dataset, providing insights into factors that influence housing prices in Boston. The analysis includes data cleaning, statistical analysis, correlation studies, outlier detection, and comprehensive visualizations with professional HTML report generation.

## 📊 Dataset Information

The Boston Housing dataset contains information about various factors that might influence housing prices:

- **CRIM**: Per capita crime rate by town
- **ZN**: Proportion of residential land zoned for lots over 25,000 sq.ft.
- **INDUS**: Proportion of non-retail business acres per town
- **CHAS**: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
- **NOX**: Nitric oxides concentration (parts per 10 million)
- **RM**: Average number of rooms per dwelling
- **AGE**: Proportion of owner-occupied units built prior to 1940
- **DIS**: Weighted distances to five Boston employment centres
- **RAD**: Index of accessibility to radial highways
- **TAX**: Full-value property-tax rate per $10,000
- **PTRATIO**: Pupil-teacher ratio by town
- **B**: Proportion of blacks by town
- **LSTAT**: % lower status of the population
- **MEDV**: Median value of owner-occupied homes in $1000's (Target Variable)

## 🚀 Features

### Data Analysis Components
- **Basic Dataset Information**: Shape, data types, memory usage
- **Missing Data Analysis**: Detection and handling of null values
- **Distribution Analysis**: Histogram and density plots for all numerical variables
- **Correlation Analysis**: Heatmap and correlation coefficients
- **Outlier Detection**: Box plots and statistical outlier identification
- **Target Variable Analysis**: Deep dive into housing price distributions
- **Feature-Target Relationships**: Scatter plots showing relationships with price

### Visualizations Generated
- Distribution plots for all numerical variables
- Correlation heatmap
- Outlier detection box plots
- Target variable analysis plots
- Feature-target relationship scatter plots

## 📄 Report Generation

This project generates comprehensive reports in multiple formats:

### Generated Reports
- **HTML Report** (`eda_report.html`) - Interactive web-based report with embedded visualizations
- **Text Report** (`eda_report.txt`) - Detailed text summary of findings
- **Visualization Files** - PNG files for all charts and graphs

### Report Features
- Professional Bootstrap-styled HTML layout
- Embedded base64-encoded visualizations
- Key metrics dashboard
- Color-coded correlation insights
- Mobile-responsive design
- Statistical summaries in formatted tables

### Viewing Reports
```bash
# Open HTML report in browser (recommended: Chrome)
open -a "Google Chrome" eda_report.html

# View text report
cat eda_report.txt
```

## 📋 Requirements

```
numpy==1.21.6
pandas==1.5.3
matplotlib==3.6.3
seaborn==0.12.2
scipy==1.9.3
scikit-learn==1.1.3
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/Jaytech9/EDA-Python.git
cd EDA-Python
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## 🎮 Usage

### Quick Start
```bash
# Run complete EDA with report generation
python main.py
```

### Custom Analysis
```python
from main import BostonHousingEDA

# Initialize EDA
eda = BostonHousingEDA()

# Run individual components
eda.basic_info()
eda.distribution_analysis()
eda.correlation_analysis()
eda.target_analysis()

# Generate reports separately
eda.generate_html_report("custom_report.html")
eda.generate_text_report("summary.txt")
```

## 📈 Key Insights

The analysis reveals several important findings:

- **Strong Predictors**: Features like LSTAT (% lower status population) and RM (average rooms) show strong correlations with housing prices
- **Geographic Factors**: Charles River proximity and accessibility affect property values
- **Quality of Life**: Crime rates and pollution levels negatively impact housing prices
- **Socioeconomic Patterns**: Clear relationships between demographic factors and property values

## 📊 Sample Output

After running the analysis, you'll get:

### Console Output
- Dataset overview and statistics
- Missing data analysis
- Correlation insights
- Key findings summary

### Generated Files
- `distributions.png` - Feature distribution visualizations
- `correlation_heatmap.png` - Correlation matrix heatmap
- `target_analysis.png` - Target variable analysis plots
- `eda_report.html` - **Professional interactive web report**
- `eda_report.txt` - Text summary report

## 🔧 Project Structure

```
EDA-Python/
│
├── main.py              # Main EDA class and execution
├── README.md            # Project documentation
├── requirements.txt     # Python dependencies
├── .gitignore          # Git ignore file
│
└── Generated Files/     # Created after running analysis
    ├── eda_report.html
    ├── eda_report.txt
    ├── distributions.png
    ├── correlation_heatmap.png
    └── target_analysis.png
```

## 🤝 Contributing

Contributions are welcome! Here are some ways you can contribute:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add amazing feature'`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Ideas for Contributions
- Add machine learning models for price prediction
- Implement interactive visualizations with Plotly
- Create a web interface with Streamlit
- Add more advanced statistical tests
- Enhance report styling and interactivity

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

**Jaytech9** - [GitHub Profile](https://github.com/Jaytech9)

Project Link: [https://github.com/Jaytech9/EDA-Python](https://github.com/Jaytech9/EDA-Python)

## 🙏 Acknowledgments

- Boston Housing dataset from UCI Machine Learning Repository
- Inspiration from various EDA projects in the data science community
- Thanks to the Python data science ecosystem (pandas, matplotlib, seaborn, scipy)

## 📚 Learning Resources

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/)
- [EDA Best Practices](https://towardsdatascience.com/exploratory-data-analysis-8fc1cb20fd15)

---

⭐ If you found this project helpful, please give it a star!

**🎯 Perfect for:** Data Analysis portfolios, learning EDA techniques, understanding housing market analysis, practicing Python data analysis skills.
