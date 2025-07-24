# Fake News Detection Using Machine Learning

A machine learning project that classifies news articles as real or fake using text vectorization and the Passive Aggressive Classifier algorithm, achieving 92% accuracy.

## **Table of Contents**
- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Details](#technical-details)
- [Future Improvements](#future-improvements)

## **Overview**

This project implements a binary classification system to detect fake news articles using natural language processing and machine learning techniques. The model analyzes textual patterns and linguistic features to distinguish between authentic and fabricated news content.

**Key Features:**
- Text preprocessing and vectorization using Bag of Words approach
- Linear classification with Passive Aggressive Classifier
- High accuracy performance (92%) on test data
- Generalizable to external fake news articles outside the training dataset

## **Dataset**

The project uses a CSV dataset containing news articles with the following structure:

| Column | Description |
|--------|-------------|
| `title` | News article headline |
| `text` | Full article content |
| `label` | Binary classification ('REAL' or 'FAKE') |

**Dataset Characteristics:**
- Format: CSV file
- Features: Text-based (title + content)
- Target: Binary classification
- Split: Training/Testing sets for model validation

## **Methodology**

### **1. Data Preprocessing**
- Text cleaning and normalization
- Removal of special characters and noise
- Lowercasing and tokenization
- Stop word handling

### **2. Feature Engineering**
- **Vectorization Technique**: Bag of Words (BoW)
- **Text Representation**: Term Frequency-Inverse Document Frequency (TF-IDF)
- **Feature Space**: High-dimensional sparse vectors
- **N-gram Analysis**: Unigrams and bigrams for capturing word patterns

### **3. Model Selection**
- **Algorithm**: Passive Aggressive Classifier
- **Type**: Linear classifier optimized for online learning
- **Advantages**: 
  - Effective for high-dimensional text data
  - Robust to outliers and noise
  - Maintains large margins between classes

## **Model Architecture**

```
Raw Text Data (CSV)
        ↓
Text Preprocessing
        ↓
TF-IDF Vectorization
        ↓
Feature Matrix (Sparse)
        ↓
Passive Aggressive Classifier
        ↓
Binary Classification Output
```

### **Key Components:**

1. **Text Vectorizer**: Converts text to numerical features
   - TF-IDF weighting scheme
   - Vocabulary building from training corpus
   - Sparse matrix representation for memory efficiency

2. **Passive Aggressive Classifier**: Linear model with specific properties
   - Updates weights aggressively for misclassified examples
   - Maintains passivity for correctly classified instances
   - Suitable for large-scale text classification

## **Results**

### **Performance Metrics**
- **Accuracy**: 92%
- **Model Type**: Linear Classifier
- **Generalization**: Successfully classifies external fake news articles
- **Training Efficiency**: Fast convergence due to linear nature

### **Model Insights**
The high accuracy indicates the model successfully learned distinguishing patterns such as:
- Linguistic style differences between fake and real news
- Vocabulary patterns and word usage frequencies
- Structural characteristics of news writing
- Emotional language indicators

## **Installation**

```bash
# Clone the repository
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection

# Install required dependencies
pip install -r requirements.txt
```

### **Required Libraries**
```
pandas
numpy
scikit-learn
matplotlib
seaborn
nltk
```

## **Usage**

### **Training the Model**
```python
# Load and preprocess data
python preprocess_data.py

# Train the model
python train_model.py

# Evaluate performance
python evaluate_model.py
```

### **Making Predictions**
```python
from model import FakeNewsDetector

# Initialize detector
detector = FakeNewsDetector()

# Load trained model
detector.load_model('trained_model.pkl')

# Predict on new article
article_text = "Your news article text here..."
prediction = detector.predict(article_text)
confidence = detector.predict_proba(article_text)

print(f"Prediction: {'Fake' if prediction else 'Real'}")
print(f"Confidence: {confidence:.2f}")
```

## **Technical Details**

### **Vectorization Process**
1. **Text Tokenization**: Split articles into individual words/tokens
2. **Vocabulary Creation**: Build dictionary of unique terms
3. **TF-IDF Calculation**: 
   - Term Frequency: Word occurrence in document
   - Inverse Document Frequency: Rarity across corpus
4. **Sparse Representation**: Memory-efficient storage of feature vectors

### **Passive Aggressive Algorithm**
- **Update Rule**: Aggressive updates for misclassified examples
- **Margin Maximization**: Maintains separation between classes
- **Online Learning**: Can adapt to new data incrementally
- **Regularization**: Built-in protection against overfitting

### **Model Advantages**
- **Computational Efficiency**: Linear time complexity
- **Scalability**: Handles large datasets effectively
- **Interpretability**: Feature weights provide insights
- **Robustness**: Stable performance across different text domains

## **Future Improvements**

- Implement deep learning approaches (LSTM, BERT)
- Add ensemble methods for improved accuracy
- Include metadata features (publication date, source)
- Develop real-time prediction API
- Create web interface for easy interaction
- Expand dataset with more diverse sources
- Add multilingual support
- Implement active learning for model updates
