# Hindi NLP Ambiguity Resolution using mBERT

This project focuses on understanding and classifying ambiguity in Hindi sentences using a Transformer-based model. Instead of relying on manual rules, the model learns contextual patterns using Multilingual BERT (mBERT).

---

## Motivation

Language is naturally ambiguous, and Hindi is no exception. A single word or sentence can carry multiple meanings depending on context.

Examples:
- “आम” → mango or common  
- Sentence structure can change meaning entirely  
- Scope of words like “हर”, “कोई”, “नहीं” can alter interpretation  

This project attempts to model these variations using deep learning.

---

## Task Description

The model classifies Hindi sentences into three categories:

- **Lexical Ambiguity (0)** → Word has multiple meanings  
- **Syntactic Ambiguity (1)** → Sentence structure is unclear  
- **Semantic Ambiguity (2)** → Meaning depends on scope or interpretation  

---

## Model Details

- Model: bert-base-multilingual-cased  
- Architecture: Transformer (BERT)  
- Framework: PyTorch  
- Task: Multi-class classification (3 classes)  

---

## Dataset

- ~50+ manually curated Hindi sentences  
- Balanced across all ambiguity types  
- Split into train, validation, and test sets  

Label mapping:

0 → Lexical  
1 → Syntactic  
2 → Semantic  

Example:
- “मुझे आम खाना है।” → Lexical  
- “उसने कहा कि वह जाएगा।” → Syntactic  
- “हर छात्र ने कोई किताब नहीं पढ़ी।” → Semantic  

---

## Pipeline

1. Tokenization using mBERT tokenizer  
2. Input encoding (input_ids, attention_mask)  
3. Fine-tuning mBERT for classification  
4. Evaluation using test data  

Training includes:
- Cross-entropy loss  
- AdamW optimizer  
- Learning rate scheduler  
- Gradient clipping  

---

## How to Run

Install dependencies:

pip install transformers datasets torch scikit-learn seaborn matplotlib

Run the script:

python hindi_nlp_ambiguity_transformer.py

Note: First run will download mBERT (~700MB)

---

## Outputs

- Training & validation loss curves  
- Accuracy plots  
- Confusion matrix  
- Classification report  
- Attention visualization  

Generated files:
- training_curves.png  
- confusion_matrix.png  
- attention_map.png  

---

## Sample Predictions

Sentence: "वह आम आदमी है।"  
Prediction: Lexical  

Sentence: "उसने कहा कि वह जाएगा।"  
Prediction: Syntactic  

Sentence: "हर छात्र कोई भाषा जानता है।"  
Prediction: Semantic  

---

## Inference

Use the function:

predict_ambiguity(sentences, tokenizer, model, device)

Returns:
- Predicted label  
- Confidence score  
- Probability distribution  

---

## Project Structure

hindi_nlp_ambiguity_transformer.py  
best_model.pt  
training_curves.png  
confusion_matrix.png  
attention_map.png  
README.md  

---

## Limitations

- Small dataset  
- Limited generalization  
- Not optimized for production  

---

## Future Work

- Larger dataset  
- Try IndicBERT / MuRIL  
- Better hyperparameter tuning  
- Deployment as API  

---

## Note

This project is built for academic and learning purposes.
Feel free to modify and extend it.
