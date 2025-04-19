import pandas as pd
import tkinter as tk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer
import numpy as np

# ---------------------------
# STEP 1: Load and Prepare Dataset
# ---------------------------
file_path = "dataset.xlsx"  # <-- Make sure this file exists
df = pd.read_excel(file_path)

# Drop empty rows
df_clean = df.dropna(subset=['text', 'label']).copy()
df_clean['label'] = df_clean['label'].astype(int)

# ---------------------------
# STEP 2: Load Multilingual Embedding Model
# ---------------------------
print("🔄 Loading multilingual sentence transformer model...")
embedder = SentenceTransformer('distiluse-base-multilingual-cased-v2')

# Encode all messages into embeddings
print("🔄 Encoding sentences...")
X = embedder.encode(df_clean['text'].tolist(), show_progress_bar=True)
y = df_clean['label'].values

# ---------------------------
# STEP 3: Train Model
# ---------------------------
print("🔧 Training Logistic Regression classifier...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

classifier = LogisticRegression(class_weight='balanced', max_iter=1000)
classifier.fit(X_train, y_train)

# Optional: Show training accuracy
train_acc = accuracy_score(y_train, classifier.predict(X_train))
test_acc = accuracy_score(y_test, classifier.predict(X_test))
print(f"✅ Model trained. Train Accuracy: {train_acc:.2f}, Test Accuracy: {test_acc:.2f}")

# ---------------------------
# STEP 4: Prediction Function
# ---------------------------
def get_stress_percentage(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return "❗ Please enter a message."
    if len(text.strip().split()) < 3:
        return "⚠️ Please enter a more complete sentence to analyze."

    vector = embedder.encode([text])
    stress_proba = classifier.predict_proba(vector)[0][1]  # Probability of class 1 (stress)
    stress_pct = round(stress_proba * 100, 2)

    if stress_pct >= 80:
        msg = f"🚨 High stress detected ({stress_pct}%). Please talk to your child."
    elif stress_pct >= 50:
        msg = f"😟 Moderate stress detected ({stress_pct}%). Keep an eye on them."
    elif stress_pct >= 20:
        msg = f"🙂 Low stress detected ({stress_pct}%). Nothing to worry much."
    else:
        msg = f"😊 No significant stress detected ({stress_pct}%). All seems well."

    return msg

# ---------------------------
# STEP 5: GUI Application
# ---------------------------
def run_gui():
    def on_check():
        user_input = entry.get("1.0", tk.END).strip()
        result = get_stress_percentage(user_input)
        result_label.config(text=result)

    root = tk.Tk()
    root.title("🧠 Child Stress Level Detector")
    root.geometry("600x360")
    root.resizable(False, False)
    root.configure(bg="#f7f7f7")

    label = tk.Label(root, text="Enter a message from your child (any language):",
                     font=("Arial", 14), bg="#f7f7f7")
    label.pack(pady=10)

    entry = tk.Text(root, height=5, width=65, font=("Arial", 12))
    entry.pack(pady=5)

    button = tk.Button(root, text="Check Stress Level", command=on_check,
                       font=("Arial", 12), bg="#4CAF50", fg="white", padx=10, pady=5)
    button.pack(pady=10)

    result_label = tk.Label(root, text="", font=("Arial", 13), fg="blue",
                            wraplength=550, justify="left", bg="#f7f7f7")
    result_label.pack(pady=10)

    root.mainloop()

# ---------------------------
# STEP 6: Launch GUI
# ---------------------------
run_gui()
