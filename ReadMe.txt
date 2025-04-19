# 🧠 Child Stress Level Detector

This is a simple Machine Learning app with GUI that detects emotional stress in children based on what they write (in any language like Hindi, English, Gujarati, etc.).

It uses:
- SentenceTransformer for multilingual sentence embeddings
- Logistic Regression for classification
- Tkinter for GUI

## 💻 How to Run

1. Clone the repo:
https://github.com/your-username/child-stress-detector.git

2. Install requirements:

3. Run the app:


## 📊 Dataset

The dataset should be in `dataset.xlsx` with two columns:
- `text`: the child's message
- `label`: 0 = not stressed, 1 = stressed

## 🔍 Example Inputs

- **"I had fun today!"** → 😊 No significant stress
- **"I'm scared to go to school."** → 🚨 High stress

## 🧠 Future Ideas

- Add voice input
- Save results to history
- Alert parents automatically

---

