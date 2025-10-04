#!/usr/bin/env python3
"""
Test script to verify backend setup and dependencies
Run this before starting the Flask server
"""

import sys

print("=" * 60)
print("Testing Sentiment Analysis Backend Setup")
print("=" * 60)

# Test Python version
print("\n1. Checking Python version...")
if sys.version_info >= (3, 8):
    print(f"   ✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
else:
    print(f"   ✗ Python version too old. Need 3.8+, have {sys.version_info.major}.{sys.version_info.minor}")
    sys.exit(1)

# Test imports
print("\n2. Testing imports...")
required_packages = {
    'flask': 'Flask',
    'flask_cors': 'Flask-CORS',
    'pandas': 'Pandas',
    'numpy': 'NumPy',
    'sklearn': 'Scikit-learn',
    'nltk': 'NLTK',
}

missing_packages = []

for package, name in required_packages.items():
    try:
        __import__(package)
        print(f"   ✓ {name}")
    except ImportError:
        print(f"   ✗ {name} - NOT INSTALLED")
        missing_packages.append(package)

if missing_packages:
    print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
    print("\nInstall them with:")
    print("pip install flask flask-cors pandas numpy scikit-learn nltk")
    sys.exit(1)

# Test NLTK data
print("\n3. Checking NLTK data...")
import nltk

nltk_data = ['punkt', 'stopwords', 'wordnet']
missing_data = []

for data in nltk_data:
    try:
        nltk.data.find(f'tokenizers/{data}' if data == 'punkt' else f'corpora/{data}')
        print(f"   ✓ {data}")
    except LookupError:
        print(f"   ✗ {data} - NOT DOWNLOADED")
        missing_data.append(data)

if missing_data:
    print(f"\n⚠️  Missing NLTK data: {', '.join(missing_data)}")
    print("Downloading now...")
    for data in missing_data:
        nltk.download(data)
    print("✓ Download complete")

# Test scikit-learn components
print("\n4. Testing ML components...")
try:
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import SVC
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.tree import DecisionTreeClassifier
    print("   ✓ All ML components available")
except ImportError as e:
    print(f"   ✗ Error importing ML components: {e}")
    sys.exit(1)

# Test sample processing
print("\n5. Testing text processing...")
try:
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import PorterStemmer
    
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    
    sample_text = "This is a great product! I love it."
    tokens = word_tokenize(sample_text.lower())
    processed = [stemmer.stem(word) for word in tokens if word.isalpha() and word not in stop_words]
    
    print(f"   ✓ Original: {sample_text}")
    print(f"   ✓ Processed: {' '.join(processed)}")
except Exception as e:
    print(f"   ✗ Error in text processing: {e}")
    sys.exit(1)

# Test Flask
print("\n6. Testing Flask setup...")
try:
    from flask import Flask
    from flask_cors import CORS
    
    test_app = Flask(__name__)
    CORS(test_app)
    print("   ✓ Flask and CORS configured")
except Exception as e:
    print(f"   ✗ Error setting up Flask: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print("\nYour backend is ready to run. Start it with:")
print("   python app.py")
print("\nThe API will be available at: http://localhost:5000")
print("=" * 60)