import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('hateSpeech_dataset.csv')
X = data['tweet']
y = data['class']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the trained model and vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train_vectors, y_train)

def predict_hate_speech(text):
    text_vector = vectorizer.transform([text])
    prediction = classifier.predict(text_vector)
    if prediction[0] == 1:
        return "Hate Speech"
    else:
        return "Not Hate Speech"

def analyze_text():
    analyzed_text = text_box.get("1.0", "end-1c")
    if analyzed_text.strip() == "":
        messagebox.showwarning("Warning", "Please enter some text to analyze.")
    else:
        prediction = predict_hate_speech(analyzed_text)
        analysis_window = tk.Toplevel(root)
        analysis_window.title("Hate Speech Analyzer Result")

        analysis_label = tk.Label(analysis_window, text="Hate Speech Analyzer Result:", font=("Helvetica", 14))
        analysis_label.pack(pady=10)

        analysis_text = tk.Text(analysis_window, height=10, width=40)
        analysis_text.insert(tk.END, f"Input Text:\n{analyzed_text}\n\nPrediction:\n{prediction}")
        analysis_text.config(state="disabled")
        analysis_text.pack(padx=20, pady=5)

    # Calculate and display accuracy
    y_pred = classifier.predict(X_test_vectors)
    print(accuracy_score(y_test, y_pred))

root = tk.Tk()
root.title("Hate Speech Analyzer")
root.geometry("400x350")

canvas = tk.Canvas(root, bg="#263D42")
canvas.place(relwidth=1, relheight=1)

frame = tk.Frame(root, bg="white", highlightbackground="black", highlightcolor="black", highlightthickness=2)
frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)

text_label = tk.Label(frame, text="Enter Text:", font=("Helvetica", 14), bg="white")
text_label.pack(pady=10)

text_box = tk.Text(frame, height=8, width=30, bg="gray", fg="black", font=("Helvetica", 12), highlightbackground="black", highlightcolor="black", highlightthickness=2)
text_box.pack(pady=5)

analyze_button = tk.Button(frame, text="Analyze", command=analyze_text, font=("Helvetica", 12), bg="#263D42",
                           fg="Black", relief="raised", bd=3, padx=10, pady=5, activebackground="#6495ED", highlightbackground="black", highlightcolor="black", highlightthickness=2)
analyze_button.pack(pady=10)

root.mainloop()
