# Restaurant Review Feature Engineering Tool

A comprehensive Python tool for processing and analyzing restaurant reviews with advanced feature engineering and machine learning capabilities.

## 🚀 Features

- **Advanced Text Preprocessing**: Handles URLs, emails, special characters, and noise
- **Comprehensive Feature Extraction**: 15+ text-based features including sentiment indicators
- **Robust TF-IDF Processing**: Progressive fallback strategy to handle "empty vocabulary" errors
- **Restaurant-Specific Analysis**: Food, service, atmosphere, and price mention detection
- **Batch Processing**: Efficient handling of large JSON datasets
- **Excel Integration**: Clean output with illegal character handling
- **Machine Learning Ready**: Feature importance analysis with Random Forest

## 📋 Requirements

```bash
pip install pandas numpy scikit-learn openpyxl nltk
```

## 🛠️ Installation

1. Clone or download the script
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn openpyxl nltk
   ```
3. Update the `JSON_DIR` path in the script to point to your data directory

## ⚙️ Configuration

Update these variables in the script:

```python
JSON_DIR = r"path/to/your/json/files"  # Directory containing JSON review files
OUTPUT_FILE = 'processed_reviews_2.xlsx'  # Output Excel file name
BATCH_SIZE = 4000  # Number of records per batch
```

## 🎯 Usage

### Basic Usage
```bash
python feature_engineering.py
```

### Advanced Usage
```python
from feature_engineering import RestaurantReviewFeatureEngineering

# Initialize the analyzer
analyzer = RestaurantReviewFeatureEngineering()

# Load your data
df = pd.read_excel('your_reviews.xlsx')

# Process features
X, y, processed_df = analyzer.fit_transform(df)

# Get feature importance
results = analyzer.get_feature_importance(X, y)
print(f"Model accuracy: {results['accuracy']:.3f}")
print(results['feature_importance'].head(10))
```

## 📊 Features Extracted

### Text Statistics
- Text length, word count, sentence count
- Average word length
- Punctuation counts (!, ?, .)
- Uppercase ratio

### Sentiment Features
- Positive word count (18 predefined words)
- Negative word count (16 predefined words)
- Sentiment ratio calculation

### Restaurant-Specific Features
- Food mentions (taste, flavor, delicious, etc.)
- Service mentions (staff, waiter, service, etc.)
- Atmosphere mentions (ambiance, environment, etc.)
- Price mentions (expensive, affordable, etc.)

### TF-IDF Features
- Advanced text vectorization with multiple fallback strategies
- Handles empty vocabulary errors automatically
- Configurable n-grams and feature limits

## 🔧 Key Functions

### `preprocess_text(text)`
Comprehensive text cleaning including:
- URL and email removal
- Whitespace normalization
- Punctuation handling
- Number filtering

### `safe_tfidf_transform(texts)`
Robust TF-IDF with progressive fallback:
1. Primary strategy: Full features with bigrams
2. Fallback 1: Reduced features, unigrams only
3. Fallback 2: Minimal features, basic tokenization
4. Final fallback: Count Vectorizer

### `extract_text_features(df)`
Extracts 15+ engineered features from review text

### `get_feature_importance(X, y)`
Trains Random Forest model and returns feature importance rankings

## 📈 Output

The script generates:
- **Excel file**: Processed reviews with all features
- **Feature matrix**: Ready for ML training
- **Feature importance**: Rankings of most predictive features
- **Model accuracy**: Performance metrics

## 🛡️ Error Handling

- **Empty vocabulary**: Progressive TF-IDF fallback strategies
- **Illegal Excel characters**: Automatic cleaning and removal
- **Missing data**: Graceful handling of NaN values
- **File processing**: Batch processing to handle memory limitations

## 📊 Example Output

```
=== Starting Feature Engineering Pipeline ===
1. Preprocessing text...
   Kept 3847 samples after removing empty texts

2. Extracting text features...

3. Applying TF-IDF transformation...
Trying strategy 1: max_features=1000, min_df=2, max_df=0.95
✓ TF-IDF successful! Shape: (3847, 956)

4. Encoding categorical features...
Encoded business_name_grouped: 14 features

5. Combining all features...
   Added 15 numerical features
   Added 956 TF-IDF features
   Added 14 categorical features

=== Feature Engineering Complete ===
Final feature matrix shape: (3847, 985)
Model accuracy: 0.847
```

## 🔍 Troubleshooting

### Common Issues:

1. **"Empty vocabulary" error**
   - The script automatically handles this with fallback strategies
   - Ensure your text data isn't completely empty

2. **Memory issues**
   - Reduce `BATCH_SIZE` 
   - Decrease `max_features` in TF-IDF

3. **File path errors**
   - Use raw strings: `r"C:\path\to\files"`
   - Ensure directory exists and contains JSON files

4. **Excel character errors**
   - The script automatically cleans illegal characters

## 📝 Data Format

Expected JSON structure:
```json
{
  "user_id": "unique_identifier",
  "text": "review text content",
  "rating": 4.5,
  "business_name": "Restaurant Name"
}
```

## 🎯 Perfect For

- Restaurant review analysis
- Sentiment classification
- Spam/fake review detection
- Quality assessment
- Feature engineering for ML models
- Large-scale text processing

## 📄 License

Open source - free to use and modify
