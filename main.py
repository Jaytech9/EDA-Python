import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
from datetime import datetime
import base64
from io import BytesIO

warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

class BostonHousingEDA:
    """
    Comprehensive Exploratory Data Analysis for Boston Housing Dataset with Report Generation
    """
    
    def __init__(self):
        self.df = None
        self.report_data = {}
        self.images = {}
        self.load_data()
    
    def load_data(self):
        """Load Boston Housing dataset"""
        try:
            # Using sklearn's load_boston
            from sklearn.datasets import load_boston
            boston = load_boston()
            
            # Create DataFrame
            self.df = pd.DataFrame(boston.data, columns=boston.feature_names)
            self.df['MEDV'] = boston.target
            
            print("✅ Dataset loaded successfully!")
            print(f"Shape: {self.df.shape}")
            
        except (ImportError, AttributeError):
            # Fallback: create sample data
            print("⚠️  Creating sample Boston Housing data...")
            np.random.seed(42)
            n_samples = 506
            
            self.df = pd.DataFrame({
                'CRIM': np.random.exponential(3, n_samples),
                'ZN': np.random.choice([0, 12.5, 25, 50, 75], n_samples, p=[0.7, 0.1, 0.1, 0.05, 0.05]),
                'INDUS': np.random.uniform(0.5, 25, n_samples),
                'CHAS': np.random.choice([0, 1], n_samples, p=[0.93, 0.07]),
                'NOX': np.random.uniform(0.3, 0.9, n_samples),
                'RM': np.random.normal(6.3, 0.7, n_samples),
                'AGE': np.random.uniform(10, 100, n_samples),
                'DIS': np.random.exponential(4, n_samples),
                'RAD': np.random.choice(range(1, 25), n_samples),
                'TAX': np.random.uniform(150, 750, n_samples),
                'PTRATIO': np.random.uniform(12, 22, n_samples),
                'B': np.random.uniform(300, 400, n_samples),
                'LSTAT': np.random.uniform(2, 35, n_samples),
            })
            
            # Create target variable
            self.df['MEDV'] = (50 - 
                              self.df['LSTAT'] * 0.5 - 
                              self.df['CRIM'] * 2 + 
                              (self.df['RM'] - 6) * 8 + 
                              np.random.normal(0, 3, n_samples))
            self.df['MEDV'] = np.clip(self.df['MEDV'], 5, 50)
    
    def save_plot_to_base64(self, fig):
        """Convert matplotlib figure to base64 string for embedding in HTML"""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()
        return image_base64
    
    def basic_info(self):
        """Display basic information about the dataset"""
        print("\n" + "="*60)
        print("📊 BASIC DATASET INFORMATION")
        print("="*60)
        
        # Store data for report
        self.report_data['basic_info'] = {
            'shape': self.df.shape,
            'memory_usage': f"{self.df.memory_usage(deep=True).sum() / 1024:.2f} KB",
            'columns': list(self.df.columns),
            'data_types': self.df.dtypes.to_dict(),
            'first_5_rows': self.df.head().to_html(classes='table table-striped'),
            'statistical_summary': self.df.describe().to_html(classes='table table-striped')
        }
        
        print(f"Dataset Shape: {self.df.shape}")
        print(f"Memory Usage: {self.report_data['basic_info']['memory_usage']}")
        print("\n📋 Column Information:")
        print(self.df.info())
        print("\n🔍 First 5 rows:")
        print(self.df.head())
        print("\n📈 Statistical Summary:")
        print(self.df.describe())
        
        return self
    
    def missing_data_analysis(self):
        """Analyze missing data"""
        print("\n" + "="*60)
        print("🔍 MISSING DATA ANALYSIS")
        print("="*60)
        
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Missing Count': missing_data,
            'Missing Percentage': missing_percent
        })
        
        # Store for report
        self.report_data['missing_data'] = {
            'total_missing': missing_data.sum(),
            'missing_by_column': missing_df.to_html(classes='table table-striped')
        }
        
        if missing_data.sum() == 0:
            print("✅ No missing values found!")
        else:
            print(missing_df[missing_df['Missing Count'] > 0])
        
        return self
    
    def distribution_analysis(self):
        """Analyze distributions of numerical variables"""
        print("\n" + "="*60)
        print("📊 DISTRIBUTION ANALYSIS")
        print("="*60)
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        # Create distribution plots
        n_cols = min(4, len(numerical_cols))
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.ravel()
        
        for idx, col in enumerate(numerical_cols):
            if idx < len(axes):
                sns.histplot(data=self.df, x=col, kde=True, ax=axes[idx])
                axes[idx].set_title(f'Distribution of {col}')
                axes[idx].grid(True, alpha=0.3)
        
        # Remove empty subplots
        if len(numerical_cols) < len(axes):
            for idx in range(len(numerical_cols), len(axes)):
                fig.delaxes(axes[idx])
        
        plt.tight_layout()
        
        # Save for report
        self.images['distributions'] = self.save_plot_to_base64(fig)
        plt.savefig('distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return self
    
    def correlation_analysis(self):
        """Analyze correlations between variables"""
        print("\n" + "="*60)
        print("🔗 CORRELATION ANALYSIS")
        print("="*60)
        
        corr_matrix = self.df.corr()
        
        # Create correlation heatmap
        fig, ax = plt.subplots(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   fmt='.2f',
                   ax=ax)
        ax.set_title('Correlation Matrix Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save for report
        self.images['correlation'] = self.save_plot_to_base64(fig)
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Store correlation data
        target_corr = corr_matrix['MEDV'].abs().sort_values(ascending=False)
        self.report_data['correlations'] = {
            'top_correlations': target_corr.head(10).to_dict(),
            'correlation_matrix': corr_matrix.to_html(classes='table table-striped')
        }
        
        print("\n🎯 Top correlations with MEDV (target):")
        print(target_corr.head(10))
        
        return self
    
    def target_analysis(self):
        """Analyze the target variable in detail"""
        print("\n" + "="*60)
        print("🎯 TARGET VARIABLE ANALYSIS (MEDV)")
        print("="*60)
        
        target = self.df['MEDV']
        
        # Basic statistics
        target_stats = {
            'mean': target.mean(),
            'median': target.median(),
            'std': target.std(),
            'min': target.min(),
            'max': target.max()
        }
        
        print(f"Mean: ${target_stats['mean']:.2f}k")
        print(f"Median: ${target_stats['median']:.2f}k")
        print(f"Std Dev: ${target_stats['std']:.2f}k")
        print(f"Min: ${target_stats['min']:.2f}k")
        print(f"Max: ${target_stats['max']:.2f}k")
        
        # Store for report
        self.report_data['target_analysis'] = target_stats
        
        # Create target analysis plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Histogram
        sns.histplot(target, kde=True, ax=axes[0,0])
        axes[0,0].set_title('Distribution of Housing Prices')
        axes[0,0].set_xlabel('Median Home Value ($1000s)')
        
        # Q-Q plot for normality
        stats.probplot(target, dist="norm", plot=axes[0,1])
        axes[0,1].set_title('Q-Q Plot (Normal Distribution)')
        
        # Box plot
        sns.boxplot(y=target, ax=axes[1,0])
        axes[1,0].set_title('Box Plot of Housing Prices')
        
        # Price categories
        price_categories = pd.cut(target, bins=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
        price_counts = price_categories.value_counts()
        axes[1,1].pie(price_counts.values, labels=price_counts.index, autopct='%1.1f%%')
        axes[1,1].set_title('Housing Price Categories')
        
        plt.tight_layout()
        
        # Save for report
        self.images['target_analysis'] = self.save_plot_to_base64(fig)
        plt.savefig('target_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return self
    
    def generate_insights(self):
        """Generate key insights from the EDA"""
        print("\n" + "="*60)
        print("💡 KEY INSIGHTS")
        print("="*60)
        
        # Generate insights
        corr_matrix = self.df.corr()
        target_corr = corr_matrix['MEDV'].sort_values(key=abs, ascending=False)
        
        insights = {
            'positive_correlations': {},
            'negative_correlations': {},
            'dataset_overview': {
                'total_properties': len(self.df),
                'average_price': self.df['MEDV'].mean(),
                'price_range': (self.df['MEDV'].min(), self.df['MEDV'].max())
            }
        }
        
        # Positive correlations
        positive_corr = target_corr[target_corr > 0].head(3)
        for feature, corr in positive_corr.items():
            if feature != 'MEDV':
                insights['positive_correlations'][feature] = corr
        
        # Negative correlations
        negative_corr = target_corr[target_corr < 0].head(3)
        for feature, corr in negative_corr.items():
            insights['negative_correlations'][feature] = corr
        
        self.report_data['insights'] = insights
        
        # Print insights
        print("🔍 Top Positive Correlations with Housing Prices:")
        for feature, corr in insights['positive_correlations'].items():
            print(f"   • {feature}: {corr:.3f}")
        
        print("\n🔍 Top Negative Correlations with Housing Prices:")
        for feature, corr in insights['negative_correlations'].items():
            print(f"   • {feature}: {corr:.3f}")
        
        print(f"\n📊 Dataset Overview:")
        print(f"   • Total properties: {insights['dataset_overview']['total_properties']}")
        print(f"   • Average price: ${insights['dataset_overview']['average_price']:.2f}k")
        print(f"   • Price range: ${insights['dataset_overview']['price_range'][0]:.2f}k - ${insights['dataset_overview']['price_range'][1]:.2f}k")
        
        return self
    
    def generate_html_report(self, filename="eda_report.html"):
        """Generate comprehensive HTML report"""
        print("\n📄 Generating HTML Report...")
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Boston Housing EDA Report</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                .insight-card {{ margin: 10px 0; }}
                .correlation-positive {{ color: #28a745; }}
                .correlation-negative {{ color: #dc3545; }}
                .chart-container {{ text-align: center; margin: 20px 0; }}
                .chart-container img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <div class="container mt-4">
                <h1 class="text-center mb-4">Boston Housing Dataset - EDA Report</h1>
                <p class="text-center text-muted">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="row">
                    <div class="col-md-12">
                        <div class="card">
                            <div class="card-header">
                                <h2>📊 Dataset Overview</h2>
                            </div>
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-md-4">
                                        <div class="card insight-card">
                                            <div class="card-body text-center">
                                                <h3 class="text-primary">{self.report_data.get('basic_info', {}).get('shape', [0, 0])[0]}</h3>
                                                <p>Total Records</p>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="col-md-4">
                                        <div class="card insight-card">
                                            <div class="card-body text-center">
                                                <h3 class="text-success">{self.report_data.get('basic_info', {}).get('shape', [0, 0])[1]}</h3>
                                                <p>Features</p>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="col-md-4">
                                        <div class="card insight-card">
                                            <div class="card-body text-center">
                                                <h3 class="text-info">${self.report_data.get('insights', {}).get('dataset_overview', {}).get('average_price', 0):.2f}k</h3>
                                                <p>Average Price</p>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="row mt-4">
                    <div class="col-md-12">
                        <div class="card">
                            <div class="card-header">
                                <h2>🔍 Key Insights</h2>
                            </div>
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-md-6">
                                        <h4 class="correlation-positive">📈 Top Positive Correlations</h4>
                                        <ul class="list-group">
        """
        
        # Add positive correlations
        for feature, corr in self.report_data.get('insights', {}).get('positive_correlations', {}).items():
            html_template += f'<li class="list-group-item d-flex justify-content-between"><span>{feature}</span><span class="correlation-positive">+{corr:.3f}</span></li>'
        
        html_template += """
                                        </ul>
                                    </div>
                                    <div class="col-md-6">
                                        <h4 class="correlation-negative">📉 Top Negative Correlations</h4>
                                        <ul class="list-group">
        """
        
        # Add negative correlations
        for feature, corr in self.report_data.get('insights', {}).get('negative_correlations', {}).items():
            html_template += f'<li class="list-group-item d-flex justify-content-between"><span>{feature}</span><span class="correlation-negative">{corr:.3f}</span></li>'
        
        html_template += """
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
        """
        
        # Add visualizations
        if 'distributions' in self.images:
            html_template += f"""
                <div class="row mt-4">
                    <div class="col-md-12">
                        <div class="card">
                            <div class="card-header">
                                <h2>📊 Feature Distributions</h2>
                            </div>
                            <div class="card-body">
                                <div class="chart-container">
                                    <img src="data:image/png;base64,{self.images['distributions']}" alt="Feature Distributions">
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            """
        
        if 'correlation' in self.images:
            html_template += f"""
                <div class="row mt-4">
                    <div class="col-md-12">
                        <div class="card">
                            <div class="card-header">
                                <h2>🔗 Correlation Matrix</h2>
                            </div>
                            <div class="card-body">
                                <div class="chart-container">
                                    <img src="data:image/png;base64,{self.images['correlation']}" alt="Correlation Matrix">
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            """
        
        if 'target_analysis' in self.images:
            html_template += f"""
                <div class="row mt-4">
                    <div class="col-md-12">
                        <div class="card">
                            <div class="card-header">
                                <h2>🎯 Target Variable Analysis</h2>
                            </div>
                            <div class="card-body">
                                <div class="chart-container">
                                    <img src="data:image/png;base64,{self.images['target_analysis']}" alt="Target Analysis">
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            """
        
        # Add statistical summary
        if 'basic_info' in self.report_data:
            html_template += f"""
                <div class="row mt-4">
                    <div class="col-md-12">
                        <div class="card">
                            <div class="card-header">
                                <h2>📈 Statistical Summary</h2>
                            </div>
                            <div class="card-body">
                                {self.report_data['basic_info'].get('statistical_summary', '')}
                            </div>
                        </div>
                    </div>
                </div>
            """
        
        html_template += """
            </div>
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        </body>
        </html>
        """
        
        # Write HTML file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_template)
        
        print(f"✅ HTML Report generated: {filename}")
        return filename
    
    def generate_text_report(self, filename="eda_report.txt"):
        """Generate text-based report"""
        print("\n📄 Generating Text Report...")
        
        report_content = f"""
BOSTON HOUSING DATASET - EDA REPORT
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

DATASET OVERVIEW:
- Shape: {self.report_data.get('basic_info', {}).get('shape', 'N/A')}
- Memory Usage: {self.report_data.get('basic_info', {}).get('memory_usage', 'N/A')}
- Total Missing Values: {self.report_data.get('missing_data', {}).get('total_missing', 'N/A')}

KEY INSIGHTS:
{'='*30}

Dataset Statistics:
- Total Properties: {self.report_data.get('insights', {}).get('dataset_overview', {}).get('total_properties', 'N/A')}
- Average Price: ${self.report_data.get('insights', {}).get('dataset_overview', {}).get('average_price', 0):.2f}k
- Price Range: ${self.report_data.get('insights', {}).get('dataset_overview', {}).get('price_range', [0, 0])[0]:.2f}k - ${self.report_data.get('insights', {}).get('dataset_overview', {}).get('price_range', [0, 0])[1]:.2f}k

Top Positive Correlations with Housing Prices:
"""
        
        for feature, corr in self.report_data.get('insights', {}).get('positive_correlations', {}).items():
            report_content += f"- {feature}: +{corr:.3f}\n"
        
        report_content += "\nTop Negative Correlations with Housing Prices:\n"
        for feature, corr in self.report_data.get('insights', {}).get('negative_correlations', {}).items():
            report_content += f"- {feature}: {corr:.3f}\n"
        
        report_content += f"""
TARGET VARIABLE ANALYSIS:
{'='*30}
- Mean: ${self.report_data.get('target_analysis', {}).get('mean', 0):.2f}k
- Median: ${self.report_data.get('target_analysis', {}).get('median', 0):.2f}k
- Standard Deviation: ${self.report_data.get('target_analysis', {}).get('std', 0):.2f}k
- Min: ${self.report_data.get('target_analysis', {}).get('min', 0):.2f}k
- Max: ${self.report_data.get('target_analysis', {}).get('max', 0):.2f}k

GENERATED FILES:
{'='*20}
- distributions.png - Feature distribution plots
- correlation_heatmap.png - Correlation matrix heatmap  
- target_analysis.png - Target variable analysis plots
- {filename} - This text report
"""
        
        # Write text file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✅ Text Report generated: {filename}")
        return filename
    
    def run_full_eda(self, generate_report=True):
        """Run complete EDA pipeline with optional report generation"""
        print("🚀 Starting Comprehensive EDA Analysis...")
        
        self.basic_info()
        self.missing_data_analysis()
        self.distribution_analysis()
        self.correlation_analysis()
        self.target_analysis()
        self.generate_insights()
        
        print("\n" + "="*60)
        print("✅ EDA ANALYSIS COMPLETE!")
        print("="*60)
        
        generated_files = [
            "distributions.png",
            "correlation_heatmap.png",
            "target_analysis.png"
        ]
        
        if generate_report:
            print("\n📊 Generating Reports...")
            html_file = self.generate_html_report()
            text_file = self.generate_text_report()
            generated_files.extend([html_file, text_file])
        
        print(f"\n📁 Generated files:")
        for file in generated_files:
            print(f"   • {file}")
        
        return self

if __name__ == "__main__":
    # Run the EDA with report generation
    eda = BostonHousingEDA()
    eda.run_full_eda(generate_report=True)
    
    print("\n🎉 Analysis complete! Check the generated HTML report for a comprehensive view.")