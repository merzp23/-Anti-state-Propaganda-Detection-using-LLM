# Vietnamese Political Content Analysis and Classification

This repository contains a comprehensive pipeline for analyzing, processing, and classifying Vietnamese political content to detect and categorize subversive or politically sensitive comments.

## Project Structure

```bash
Paper_ChongPha/GithubUpload/
│
├── thuyetminhdetai.pdf      # Project proposal/description document
│
├── Data/                    # Data processing scripts
│   ├── 1_first_clean.py     # Initial data cleaning
│   ├── 2_summarize_and_prepare.py  # Text summarization and preparation
│   ├── 3_gemini_label.py    # Labeling using Google's Gemini API
│   ├── 4_check_dataset.py   # GUI tool for dataset verification
│   ├── Dataset_ChongPha.csv # The main dataset
│   └── preprocessing(1).ipynb # Jupyter notebook for preprocessing steps
│
└── Models/                  # Machine learning models implementations
    ├── CafeBERT.ipynb       # Fine-tuning and evaluation of CafeBERT
    ├── CNN-LSTM.ipynb       # Fine-tuning and evaluation of CNN and LSTM
    ├── PhoBERT.ipynb        # Fine-tuning and evaluation of PhoBERT
    ├── Qwen3-32B.ipynb      # Fine-tuning and evaluation of Qwen3-32B
    ├── randomforest-naivebayes-logisticregression.ipynb # Fine-tuning and evaluation of Random Forest, Multinomial Naive Bayes, Multinomial Logistic Regression
    ├── SeaLLM.ipynb         # Fine-tuning and evaluation of SeaLLM
    ├── vistral7b.ipynb      # Fine-tuning and evaluation of Vistral-7B
    └── XLM_ROBERTA.ipynb    # Fine-tuning and evaluation of XLM-RoBERTa
```

## Dataset

The dataset (`Paper_ChongPha/GithubUpload/Data/Dataset_ChongPha.csv`) contains Vietnamese social media comments that have been labeled into three categories:

- `PHAN_DONG`: Comments considered politically subversive or anti-government
- `KHONG_PHAN_DONG`: Comments that are not politically subversive
- `KHONG_LIEN_QUAN`: Comments that are not related to politics

Each entry in the dataset contains:
- Comment text
- Summary of context
- Label classification

## Data Processing Pipeline

### 1. Data Cleaning (`Paper_ChongPha/GithubUpload/Data/1_first_clean.py`)

- Removes URLs, HTML tags, emojis, and other non-textual content
- Filters out comments that are too long or too short
- Normalizes text by removing excessive punctuation and whitespace
- Balances comments per post to avoid bias from over-represented posts

### 2. Text Summarization (`Paper_ChongPha/GithubUpload/Data/2_summarize_and_prepare.py`)

- Summarizes each post into three sections:
  - Content overview
  - Issues identified
  - Assessment of whether it contains subversive content
- Uses Google's Generative AI API for summarization

### 3. Labeling (`Paper_ChongPha/GithubUpload/Data/3_gemini_label.py`)

- Utilizes Google's Gemini API to classify comments
- Handles rate limiting and API key management
- Outputs structured JSON with classifications

### 4. Dataset Verification (`Paper_ChongPha/GithubUpload/Data/4_check_dataset.py`)

- Provides a GUI interface for manually reviewing and validating labels
- Allows for correcting misclassified entries
- Supports batch operations and data export

## Models

The project experiments with several state-of-the-art language models for Vietnamese:

- **BERT-based Models**
  - PhoBERT: Specialized for Vietnamese language
  - CafeBERT: Fine-tuned for Vietnamese social media content
  - XLM-RoBERTa: Multilingual transformer model

- **Large Language Models**
  - Qwen3-32B: Advanced generative language model
  - SeaLLM: Southeast Asian language model
  - Vistral-7B: Vietnamese-optimized model

- **Traditional Models**
  - CNN-LSTM: Combining convolutional and recurrent neural networks
  - Random Forest, Naive Bayes, Logistic Regression: Traditional machine learning approaches

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- pandas
- Google GenerativeAI API access
- TensorFlow (for specific models)
- tkinter (for GUI components)
- CUDA-capable GPU (recommended for model training)

## Usage

### Data Processing

```bash
# Step 1: Clean raw data
python Data/1_first_clean.py --version v1 --source output --target output

# Step 2: Summarize and prepare data
python Data/2_summarize_and_prepare.py --version v1

# Step 3: Label using Gemini API
python Data/3_gemini_label.py --version v1 --model gemini-pro

# Step 4: Verify dataset (launches GUI)
python Data/4_check_dataset.py --version v1 --file Dataset_ChongPha.csv
```

### Model Training and Evaluation

The Jupyter notebooks in the `Models/` directory contain comprehensive code for:
- Data loading and preprocessing
- Model configuration and training
- Evaluation and performance metrics
- Prediction on test data

Each notebook can be run in a GPU-enabled environment such as Google Colab or a local machine with CUDA support.

## Classification System

The classification system identifies content along three dimensions:

1. **PHAN_DONG** (Subversive): Content that contains anti-government rhetoric, historical revisionism, or attempts to incite political opposition

2. **KHONG_PHAN_DONG** (Non-subversive): Content that may discuss politics but does not contain subversive elements

3. **KHONG_LIEN_QUAN** (Irrelevant): Content unrelated to political topics

## Training Results

Final test-set metrics from the notebooks:
- Random Forest (TF-IDF): accuracy 0.70; F1 (weighted) 0.67; F1 (macro) 0.58
- Multinomial Naive Bayes (TF-IDF): accuracy 0.66; F1 (weighted) 0.62; F1 (macro) 0.47
- Logistic Regression (TF-IDF): accuracy 0.72

- CNN-LSTM: accuracy 0.71; F1 (macro) 0.62; F1 (weighted) 0.70

- XLM-RoBERTa: accuracy 0.7585; balanced accuracy 0.6837; F1 (weighted) 0.7555; F1 (macro) 0.6986; per-class F1s: PHAN_DONG 0.5508; KHONG_PHAN_DONG 0.7240; KHONG_LIEN_QUAN 0.8208
- CafeBERT: accuracy 0.7495; balanced accuracy 0.6940; F1 (weighted) 0.7484; F1 (macro) 0.6997; per-class F1s: 0.5811 / 0.7026 / 0.8155
- PhoBERT: accuracy 0.7406; F1 (weighted) 0.7321; F1 (macro) 0.6791

- Qwen3-32B: accuracy 0.6802; F1 (macro) 0.64; F1 (weighted) 0.68
- SeaLLM: accuracy 0.7283; F1 (macro) 0.68; F1 (weighted) 0.72
- Vistral-7B: accuracy 0.76; F1 (macro) 0.74; F1 (weighted) 0.76


Notes:
- Three-class setting: PHAN_DONG, KHONG_PHAN_DONG, KHONG_LIEN_QUAN
- Metrics are copied from the corresponding notebook outputs (best runs shown when multiple are available)

## License
Tran Huynh Anh Phuc - UIT - 22521141
Tran Tan Tai - UIT - 22521287

```
