# Anti-state Propaganda Detection using LLM

This repository contains models and code for detecting anti-state propaganda in Vietnamese social media content using various Large Language Models (LLMs).

## Setup Instructions

### 1. Hugging Face Token Setup

To use the models in this repository, you need to set up your Hugging Face token securely:

1. **Get your Hugging Face token:**
   - Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
   - Create a new token with appropriate permissions

2. **Set up environment variables:**
   ```bash
   # Option 1: Export in your shell
   export HF_TOKEN="your_token_here"
   
   # Option 2: Create a .env file (make sure it's in .gitignore)
   echo "HF_TOKEN=your_token_here" > .env
   ```

3. **Update the notebooks:**
   - Replace `login(token='YOUR_HF_TOKEN_HERE')` with `login(token=os.getenv('HF_TOKEN'))`
   - Add `import os` at the top of each notebook if not already present

### 2. Data Setup

The dataset should be placed in the `Data/` directory. The main dataset file is `Dataset_ChongPha.csv`.

### 3. Model Files

The repository contains several model implementations:
- **CafeBERT**: Vietnamese BERT model
- **PhoBERT**: Vietnamese RoBERTa model  
- **XLM-RoBERTa**: Multilingual model
- **Qwen3-32B**: Large language model
- **SeaLLM**: Southeast Asian language model
- **Vistral7B**: Vietnamese Mistral model
- **CNN-LSTM**: Traditional deep learning approach
- **Random Forest, Naive Bayes, Logistic Regression**: Classical ML approaches

## Security Notes

- **Never commit tokens or API keys** to version control
- **Use environment variables** for sensitive credentials
- **The .gitignore file** is configured to prevent accidental commits of sensitive files
- **If you accidentally commit a token**, revoke it immediately and generate a new one

## Usage

Each model is implemented in its own Jupyter notebook in the `Models/` directory. Follow the setup instructions above before running any notebooks.

## License

[Add your license information here]
