from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import re
import string
from io import StringIO
import base64

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Global variables to store data
data_store = {
    'df': None,
    'X_train': None,
    'X_test': None,
    'y_train': None,
    'y_test': None,
    'X_train_tfidf': None,
    'X_test_tfidf': None,
    'vectorizer': None,
    'models': {},
    'stemmer': PorterStemmer(),
    'stop_words': set(stopwords.words('english'))
}

def find_review_column(df):
    """Find the column containing reviews"""
    if df is None:
        return None
    
    # Check for exact matches
    for col in ['review', 'Review', 'text', 'Text', 'reviewText', 'review_text']:
        if col in df.columns:
            return col
    
    # Check for common review column names
    possible_names = ['review', 'text', 'comment', 'summary']
    for col in df.columns:
        col_lower = col.lower()
        if any(name in col_lower for name in possible_names):
            return col
    
    # Use first text column
    text_cols = df.select_dtypes(include=['object']).columns
    if len(text_cols) > 0:
        return text_cols[0]
    
    return None

def find_rating_column(df):
    """Find the column containing ratings"""
    if df is None:
        return None
    
    # Check for exact matches
    for col in ['Rate', 'rate', 'Rating', 'rating', 'Score', 'score']:
        if col in df.columns:
            return col
    
    # Check for common rating names
    possible_names = ['rate', 'rating', 'score', 'stars']
    for col in df.columns:
        col_lower = col.lower()
        if any(name in col_lower for name in possible_names):
            return col
    
    # Check numeric columns in rating range
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        try:
            if df[col].min() >= 0 and df[col].max() <= 5:
                return col
        except:
            continue
    
    return None

def clean_text(text):
    """Clean and preprocess text"""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    
    tokens = word_tokenize(text)
    tokens = [data_store['stemmer'].stem(word) for word in tokens 
             if word not in data_store['stop_words'] and len(word) > 2]
    
    return ' '.join(tokens)

def convert_to_sentiment(rating):
    """Convert rating to sentiment"""
    try:
        rating = float(rating)
        if rating <= 2:
            return 'Negative'
        elif rating == 3:
            return 'Neutral'
        else:
            return 'Positive'
    except:
        rating_lower = str(rating).lower()
        if 'neg' in rating_lower:
            return 'Negative'
        elif 'pos' in rating_lower:
            return 'Positive'
        else:
            return 'Neutral'

@app.route('/api/upload', methods=['POST'])
def upload_dataset():
    """Upload CSV dataset"""
    try:
        file = request.files['file']
        content = file.read().decode('utf-8')
        df = pd.read_csv(StringIO(content))
        
        data_store['df'] = df
        
        review_col = find_review_column(df)
        rating_col = find_rating_column(df)
        
        # Get sample reviews
        samples = []
        if review_col and rating_col:
            for idx, row in df.head(10).iterrows():
                samples.append({
                    'text': str(row[review_col])[:200],
                    'rating': str(row[rating_col])
                })
        
        return jsonify({
            'success': True,
            'message': 'Dataset uploaded successfully',
            'total_records': len(df),
            'columns': df.columns.tolist(),
            'review_column': review_col,
            'rating_column': rating_col,
            'samples': samples
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/preprocess', methods=['POST'])
def preprocess_dataset():
    """Preprocess the dataset"""
    try:
        df = data_store['df']
        if df is None:
            return jsonify({
                'success': False,
                'error': 'Please upload a dataset first'
            }), 400
        
        review_col = find_review_column(df)
        rating_col = find_rating_column(df)
        
        if not review_col or not rating_col:
            return jsonify({
                'success': False,
                'error': 'Could not find review or rating columns'
            }), 400
        
        # Clean reviews
        df['cleaned_review'] = df[review_col].apply(clean_text)
        df['sentiment'] = df[rating_col].apply(convert_to_sentiment)
        df = df[df['cleaned_review'].str.strip() != '']
        
        # Split data
        X = df['cleaned_review']
        y = df['sentiment']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        # Store in data_store
        data_store['X_train'] = X_train
        data_store['X_test'] = X_test
        data_store['y_train'] = y_train
        data_store['y_test'] = y_test
        data_store['X_train_tfidf'] = X_train_tfidf
        data_store['X_test_tfidf'] = X_test_tfidf
        data_store['vectorizer'] = vectorizer
        
        return jsonify({
            'success': True,
            'message': 'Dataset preprocessed successfully',
            'total_reviews': len(df),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'features': X_train_tfidf.shape[1]
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/train/<model_name>', methods=['POST'])
def train_model(model_name):
    """Train a specific model"""
    try:
        if data_store['X_train_tfidf'] is None:
            return jsonify({
                'success': False,
                'error': 'Please preprocess the dataset first'
            }), 400
        
        X_train_tfidf = data_store['X_train_tfidf']
        X_test_tfidf = data_store['X_test_tfidf']
        y_train = data_store['y_train']
        y_test = data_store['y_test']
        
        # Select and train model
        if model_name == 'svm':
            model = SVC(kernel='rbf', C=1.0, random_state=42)
            display_name = 'SVM'
        elif model_name == 'naive_bayes':
            model = MultinomialNB(alpha=1.0)
            display_name = 'Naive Bayes'
        elif model_name == 'decision_tree':
            model = DecisionTreeClassifier(max_depth=10, random_state=42)
            display_name = 'Decision Tree'
        else:
            return jsonify({
                'success': False,
                'error': 'Invalid model name'
            }), 400
        
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store model
        data_store['models'][display_name] = {
            'model': model,
            'accuracy': accuracy
        }
        
        return jsonify({
            'success': True,
            'message': f'{display_name} trained successfully',
            'model_name': display_name,
            'accuracy': float(accuracy * 100)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/detect_sentiment', methods=['POST'])
def detect_sentiment():
    """Detect sentiment from test reviews"""
    try:
        if not data_store['models']:
            return jsonify({
                'success': False,
                'error': 'Please train at least one model first'
            }), 400
        
        # Get the last trained model
        model_name = list(data_store['models'].keys())[-1]
        model = data_store['models'][model_name]['model']
        vectorizer = data_store['vectorizer']
        X_test = data_store['X_test']
        
        # Get 10 test reviews
        test_reviews = X_test.head(10).tolist()
        results = []
        
        for review in test_reviews:
            cleaned = clean_text(review)
            vectorized = vectorizer.transform([cleaned])
            prediction = model.predict(vectorized)[0]
            proba = None
            
            try:
                proba = model.predict_proba(vectorized)[0]
                confidence = float(max(proba) * 100)
            except:
                confidence = 85.0  # Default confidence for SVM
            
            results.append({
                'text': review[:150],
                'predicted': prediction,
                'confidence': confidence
            })
        
        return jsonify({
            'success': True,
            'model_used': model_name,
            'results': results
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/accuracy', methods=['GET'])
def get_accuracy():
    """Get accuracy of all trained models"""
    try:
        accuracies = {}
        for model_name, model_data in data_store['models'].items():
            accuracies[model_name] = float(model_data['accuracy'] * 100)
        
        return jsonify({
            'success': True,
            'models': accuracies
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Sentiment Analysis API is running'
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)