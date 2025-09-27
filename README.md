# 🔒 Fraud Detection System

A comprehensive machine learning-based fraud detection system built with **Streamlit**, **Scikit-learn**, and **XGBoost**. This application provides real-time fraud detection capabilities with an intuitive web interface.

## 📋 Table of Contents
- [Features](#features)
- [Dataset](#dataset)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Screenshots](#screenshots)
- [Contributing](#contributing)

## ✨ Features

### 🔍 Single Transaction Prediction
- **Interactive Input Form**: Easy-to-use interface for entering transaction details
- **Dual Model Predictions**: Compare results from Logistic Regression and XGBoost
- **Real-time Results**: Instant fraud probability calculations
- **Visual Comparisons**: Bar charts showing model confidence levels

### 📊 Dataset Analytics
- **Comprehensive Statistics**: Key metrics including fraud rate and transaction patterns
- **Interactive Visualizations**: 
  - Fraud vs Legitimate transaction distribution
  - Transaction amounts by fraud status
  - Merchant category analysis
  - Location-based insights

### 📈 Model Performance Analysis
- **Accuracy Metrics**: Performance comparison between models
- **Probability Distributions**: Fraud score distributions for both models
- **Feature Importance**: XGBoost feature importance visualization
- **Confusion Matrices**: Model performance evaluation

## 📊 Dataset

The system uses a fraud detection dataset with the following features:

| Feature | Description | Type |
|---------|-------------|------|
| `TransactionID` | Unique transaction identifier | Numeric |
| `Amount` | Transaction amount in USD | Numeric |
| `Time` | Time since first transaction (seconds) | Numeric |
| `Location` | Transaction location | Categorical |
| `MerchantCategory` | Type of merchant | Categorical |
| `CardHolderAge` | Age of cardholder | Numeric |
| `IsFraud` | Target variable (0=Legitimate, 1=Fraud) | Binary |

### Data Preprocessing
- **Categorical Encoding**: OrdinalEncoder for location and merchant category
- **Missing Value Handling**: Mean imputation for numeric features
- **Feature Scaling**: StandardScaler for numeric features
- **Data Validation**: Robust handling of unknown categories

## 🤖 Models

### 1. Logistic Regression
- **Purpose**: Baseline model for interpretable predictions
- **Accuracy**: ~94.4%
- **Advantages**: Fast, interpretable, good baseline performance
- **Use Case**: Quick fraud screening and interpretable results

### 2. XGBoost Classifier
- **Purpose**: Advanced ensemble model for complex pattern recognition
- **Accuracy**: ~94.4%
- **Advantages**: Handles non-linear relationships, feature importance, robust performance
- **Use Case**: Production-ready fraud detection with high accuracy

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start

1. **Clone or Download the Project**
   ```bash
   git clone https://github.com/Amanbig/fraud_detection
   cd assign_fraud
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Models (First Time Only)**
   ```bash
   python retrain_models.py
   ```

4. **Run the Application**
   ```bash
   streamlit run fraud_detection_app.py
   ```

5. **Open Your Browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, navigate to the URL shown in your terminal

### Manual Installation
```bash
pip install streamlit pandas numpy scikit-learn xgboost plotly seaborn matplotlib
```

## 📖 Usage

### 🔍 Making Predictions

1. **Navigate to Single Prediction Page**
2. **Enter Transaction Details**:
   - Transaction ID (numeric)
   - Amount in USD
   - Time (seconds since epoch)
   - Location (dropdown selection)
   - Merchant Category (dropdown selection)
   - Cardholder Age

3. **Click "Predict Fraud"**
4. **View Results**:
   - Both models' predictions
   - Fraud probability scores
   - Visual comparison chart

### 📊 Exploring Data

1. **Go to Dataset Overview Page**
2. **Review Key Metrics**:
   - Total transactions
   - Fraud cases count
   - Overall fraud rate
   - Average transaction amount

3. **Analyze Visualizations**:
   - Transaction distributions
   - Location patterns
   - Merchant category insights

### 📈 Model Analysis

1. **Visit Model Analytics Page**
2. **Compare Model Performance**:
   - Accuracy scores
   - Probability distributions
   - Feature importance (XGBoost)
   - Confusion matrices

## 📁 Project Structure

```
assign_fraud/
├── 📄 README.md                    # Project documentation
├── 🐍 fraud_detection_app.py       # Main Streamlit application
├── 🐍 retrain_models.py           # Model retraining script
├── 📊 model-deployment.ipynb      # Original development notebook
├── 📈 fraud-detection.csv         # Dataset file
├── 📋 requirements.txt            # Python dependencies
├── 🤖 logistic.pkl               # Trained Logistic Regression model
├── 🤖 xgboost.pkl                # Trained XGBoost model
├── ⚙️ ordinal_encoder.pkl        # Saved categorical encoder
├── ⚙️ scaler.pkl                 # Saved feature scaler
└── ⚙️ feature_names.pkl          # Saved feature names
```

## 📈 Model Performance

### Training Results
- **Dataset Size**: 500 transactions
- **Fraud Rate**: ~14% (realistic imbalanced dataset)
- **Train/Test Split**: 75/25

### Performance Metrics
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 94.4% | High | Good | Good |
| XGBoost | 94.4% | High | Good | Good |

### Key Insights
- Both models achieve similar high accuracy
- XGBoost provides better feature importance insights
- Logistic Regression offers faster predictions and interpretability
- System handles class imbalance effectively

## 🖼️ Screenshots

### Main Dashboard
- Clean, professional interface
- Easy navigation between features
- Real-time prediction results

### Prediction Interface
- User-friendly input forms
- Dropdown selections for categorical features
- Clear fraud/legitimate indicators

### Analytics Dashboard
- Interactive charts and visualizations
- Model comparison tools
- Performance metrics display

## ⚙️ Configuration

### Model Retraining
To retrain models with your own data:

1. **Update Dataset**: Replace `fraud-detection.csv` with your data
2. **Run Retraining**: `python retrain_models.py`
3. **Restart App**: `streamlit run fraud_detection_app.py`

### Customization Options
- **Styling**: Modify CSS in the Streamlit app
- **Features**: Add/remove input fields as needed
- **Models**: Integrate additional ML algorithms
- **Visualizations**: Customize charts using Plotly

## 🔧 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   ```bash
   python retrain_models.py  # Regenerate models
   ```

2. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt --upgrade
   ```

3. **Dataset Not Found**
   - Ensure `fraud-detection.csv` is in the project directory
   - Check file encoding (use UTF-8)

4. **Port Already in Use**
   ```bash
   streamlit run fraud_detection_app.py --server.port 8502
   ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Scikit-learn**: Machine learning library
- **XGBoost**: Gradient boosting framework
- **Streamlit**: Web app framework
- **Plotly**: Interactive visualizations
- **Seaborn & Matplotlib**: Statistical plotting

## 📞 Support

For questions or support, please:
1. Check the troubleshooting section
2. Review existing issues
3. Create a new issue with detailed description

---

**Built with ❤️ for fraud detection and financial security**

### Quick Commands Reference
```bash
# Install dependencies
pip install -r requirements.txt

# Retrain models (first time or after data changes)
python retrain_models.py

# Run the application
streamlit run fraud_detection_app.py

# Run on different port
streamlit run fraud_detection_app.py --server.port 8502
```

🔒 **Stay Safe, Detect Fraud!**