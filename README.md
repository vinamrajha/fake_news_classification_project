# Fake News Classification using LSTM

## 🎯 Project Overview
Built a deep learning NLP model to detect fake news articles using LSTM neural networks, achieving reliable classification of misinformation from genuine news sources.

## 📊 Dataset: WELFake
- **Size**: 72,134 articles (35,028 real + 37,106 fake)
- **Features**: Article title, text, and binary labels
- **Source**: Combined from Reuters, Politifact, Wikipedia, Kaggle datasets
- **Challenge**: Handling class imbalance and nuanced linguistic patterns

## 🛠 Technical Implementation

### Preprocessing Pipeline
- Text cleaning (lowercasing, punctuation/digit removal)
- Tokenization with TensorFlow Keras Tokenizer (vocab size: 10,000)
- Sequence padding (max length: 200 tokens)
- Train/test split: 80/20

### Model Architecture
```
Embedding Layer (128 dimensions)
    ↓
LSTM Layer (64 units)
    ↓
Dense Layer (24 units, ReLU)
    ↓
Output Layer (sigmoid activation)
```

### Training
- **Optimizer**: Adam
- **Loss Function**: Binary crossentropy
- **Metrics**: Accuracy
- **Epochs**: 10 with validation monitoring

## 📈 Results
The model demonstrates strong performance in distinguishing fake from real news, with comprehensive evaluation including:
- Classification report (precision, recall, F1-score)
- Confusion matrix visualization
- Training/validation accuracy and loss curves

## 💻 Tech Stack
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat&logo=python&logoColor=white)

## 🔑 Key Skills Demonstrated
- Deep Learning (LSTM, Sequential Models)
- Natural Language Processing (NLP)
- Text Preprocessing & Tokenization
- TensorFlow/Keras API
- Model Evaluation & Visualization
- Binary Classification
- Data Handling (Pandas, NumPy)

## 📁 Project Structure
```
fake_news_classification_project/
├── fake_news_classification.py    # Main implementation
└── WELFake_Dataset.csv            # Dataset file
```

## 🚀 Quick Start
```bash
# Clone the repository
git clone https://github.com/vinamrajha/fake_news_classification_project.git

# Install dependencies
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn

# Run the model
python fake_news_classification.py
```

## 📝 Use Cases
- Social media content moderation
- News platform verification systems
- Misinformation detection tools
- Media literacy applications

---
*This project demonstrates end-to-end ML pipeline development from data preprocessing to model deployment and evaluation.*
