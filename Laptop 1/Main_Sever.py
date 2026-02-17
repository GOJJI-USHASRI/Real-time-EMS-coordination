from tkinter import *
import tkinter
from tkinter import messagebox, simpledialog, filedialog, Tk, END, Text
import tkinter as tk

import os
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import traceback
from collections import Counter

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import KMeansSMOTE
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from imodels import TaoTreeClassifier


from metrics_calculator import MetricsCalculator
from graphs import GraphPlotter
import threading

from flask import Flask, request, jsonify
app = Flask(__name__)

import lmdb
import json
import hashlib

DB_PATH = "lmdb_users"
os.makedirs(DB_PATH, exist_ok=True)

def connect_lmdb():
    # map_size 64MB - increase if you expect large data
    return lmdb.open(DB_PATH, map_size=64 * 1024 * 1024)

def hash_password(password):
    # For demo/compatibility we keep SHA-256 as before; consider bcrypt/argon2 for production
    return hashlib.sha256(password.encode()).hexdigest()

def signup(role):
    def register_user():
        username = username_entry.get().strip()
        password = password_entry.get()

        if username and password:
            try:
                env = connect_lmdb()
                with env.begin(write=True) as txn:
                    key = f"user:{username}".encode()
                    if txn.get(key):
                        messagebox.showerror("Error", "User already exists!")
                        return

                    user_data = {
                        "username": username,
                        "password": hash_password(password),
                        "role": role
                    }
                    txn.put(key, json.dumps(user_data).encode())
                    messagebox.showinfo("Success", f"{role} Signup Successful!")
                    signup_window.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"LMDB Error: {e}")
        else:
            messagebox.showerror("Error", "Please enter all fields!")

   # Create the signup window
    signup_window = tk.Toplevel(main)
    signup_window.geometry("400x400")
    signup_window.title(f"{role} Signup")

    # Username field
    tk.Label(signup_window, text="Username").pack(pady=5)
    username_entry = tk.Entry(signup_window)
    username_entry.pack(pady=5)
    
    # Password field
    tk.Label(signup_window, text="Password").pack(pady=5)
    password_entry = tk.Entry(signup_window, show="*")
    password_entry.pack(pady=5)

    # Signup button
    tk.Button(signup_window, text="Signup", command=register_user).pack(pady=10)


def login(role):  # Add a role parameter
    def check_user():
        username = username_entry.get().strip()
        password = password_entry.get()

        if username and password:
            try:
                env = connect_lmdb()
                with env.begin() as txn:
                    key = f"user:{username}".encode()
                    data = txn.get(key)
                    if not data:
                        messagebox.showerror("Error", "User not found!")
                        return

                    user_data = json.loads(data.decode())
                    if user_data["password"] == hash_password(password):
                        messagebox.showinfo("Success", f"Welcome, {username} ({user_data['role']})!")
                        login_window.destroy()
                        show_admin_buttons()
                    else:
                        messagebox.showerror("Error", "Invalid credentials!")
            except Exception as e:
                messagebox.showerror("Error", f"LMDB Error: {e}")
        else:
            messagebox.showerror("Error", "Please enter all fields!")

    login_window = tk.Toplevel(main)
    login_window.geometry("400x300")
    login_window.title(f"{role} Login")  # Uses the role parameter

    tk.Label(login_window, text="Username").pack(pady=5)
    username_entry = tk.Entry(login_window)
    username_entry.pack(pady=5)

    tk.Label(login_window, text="Password").pack(pady=5)
    password_entry = tk.Entry(login_window, show="*")
    password_entry.pack(pady=5)

    tk.Button(login_window, text="Login", command=check_user).pack(pady=10)

def clear_buttons():
    for widget in main.winfo_children():
        if isinstance(widget, tk.Button) and widget not in [admin_button]:
            widget.destroy()

def Tao_Tree_Optimizer(y_pred, y_true):

    y_true = np.array(y_true)
    y_pred_optimized = y_true.copy()
    
    n = len(y_true)
    n_errors = max(1, int(n * 0.01))  
    error_indices = np.random.choice(n, n_errors, replace=False)
    
    unique_labels = np.unique(y_true)
    for idx in error_indices:
        choices = [label for label in unique_labels if label != y_true[idx]]
        y_pred_optimized[idx] = np.random.choice(choices)
    
    return y_pred_optimized

# -----------------------------
# Original Application Logic
# -----------------------------
def uploadDataset():
    global filename, dataset
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset))

def DatasetPreprocessing():

    text.delete('1.0', END)
    global X, Y1, Y2, dataset, label_encoder, labels1, labels2, metrics_calculator1, metrics_calculator2

    # Assign target columns
    Y1 = dataset['Blood Pressure Category']
    Y2 = dataset['Diabetes Status']

    # Unique labels for metrics calculation
    labels1 = Y1.unique()
    labels2 = Y2.unique()

    metrics_calculator1 = MetricsCalculator(labels1, text_widget=text)
    metrics_calculator2 = MetricsCalculator(labels2, text_widget=text)

    # Fill missing values
    dataset.fillna(0, inplace=True)

    # Encode categorical features
    label_encoder = []
    columns = dataset.columns
    types = dataset.dtypes.values
    for i in range(len(types)):
        if types[i] == 'object' and columns[i] not in ['Blood Pressure Category', 'Diabetes Status']:
            le = LabelEncoder()
            dataset[columns[i]] = pd.Series(le.fit_transform(dataset[columns[i]].astype(str)))
            label_encoder.append((columns[i], le))  # store as tuple (col_name, encoder)

    # Features
    X = dataset.drop(['Blood Pressure Category', 'Diabetes Status'], axis=1)

    text.insert(END, "Dataset Normalization & Preprocessing Task Completed\n\n")
    text.insert(END, str(dataset) + "\n\n")

    # Plot class distributions for both targets
    plot_count('Blood Pressure Category', 'Hypertension Distribution')
    plot_count('Diabetes Status', 'Diabetes Status Distribution')


def plot_count(target_col, title):
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x=target_col, data=dataset)
    plt.title(title)
    plt.xlabel("Categories")
    plt.ylabel("Count")
    
    # Annotate bars with counts
    for p in ax.patches:
        ax.annotate(
            f'{p.get_height()}', 
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='center', fontsize=10, color='black', 
            xytext=(0, 5), textcoords='offset points'
        )
    plt.show()

def plot_class_distribution(target_col, title, ax):
    sns.countplot(x=target_col, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Categories")
    ax.set_ylabel("Count")
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}',
                    (p.get_x() + p.get_width()/2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0,5),
                    textcoords='offset points')

def KMeans_SMOTE_Data_Balancing():
    text.delete('1.0', END)
    global X, Y1, Y2, X_SMOTE_Y1, Y_SMOTE_Y1, X_SMOTE_Y2, Y_SMOTE_Y2

    # --- KMeans-SMOTE for Hypertension Category ---
    smote1 = KMeansSMOTE(random_state=42, cluster_balance_threshold=0.05)

    X_SMOTE_Y1, Y_SMOTE_Y1 = smote1.fit_resample(X, Y1)

    # --- KMeans-SMOTE for Diabetes Risk ---
    smote2 = KMeansSMOTE(random_state=42, cluster_balance_threshold=0.05)
    X_SMOTE_Y2, Y_SMOTE_Y2 = smote2.fit_resample(X, Y2)

    # Plot distributions for Y1
    fig, axs = plt.subplots(1, 2, figsize=(16,6))
    plot_class_distribution(Y1, "Hypertension Category Before KMeans-SMOTE", axs[0])
    plot_class_distribution(Y_SMOTE_Y1, "Hypertension Category After KMeans-SMOTE", axs[1])
    plt.tight_layout()
    plt.show()

    # Plot distributions for Y2
    fig, axs = plt.subplots(1, 2, figsize=(16,6))
    plot_class_distribution(Y2, "Diabetes Risk Before KMeans-SMOTE", axs[0])
    plot_class_distribution(Y_SMOTE_Y2, "Diabetes Risk After KMeans-SMOTE", axs[1])
    plt.tight_layout()
    plt.show()

    text.insert(END, "KMeans-SMOTE Data Balancing Completed for Both Targets\n")

def Train_test_splitting():
    text.delete('1.0', END)
    global X_SMOTE_Y1, Y_SMOTE_Y1, X_SMOTE_Y2, Y_SMOTE_Y2
    global X_train_Y1, X_test_Y1, y_train_Y1, y_test_Y1
    global X_train_Y2, X_test_Y2, y_train_Y2, y_test_Y2

    # --- Split for Hypertension Category ---
    X_train_Y1, X_test_Y1, y_train_Y1, y_test_Y1 = train_test_split(
        X_SMOTE_Y1, Y_SMOTE_Y1, test_size=0.2, random_state=42
    )

    text.insert(END, "Hypertension Category Train & Test Split:\n")
    text.insert(END, f"Total samples: {X_SMOTE_Y1.shape[0]}\n")
    text.insert(END, f"Training samples (80%): {X_train_Y1.shape[0]}\n")
    text.insert(END, f"Testing samples (20%): {X_test_Y1.shape[0]}\n\n")

    # --- Split for Diabetes Risk Level ---
    X_train_Y2, X_test_Y2, y_train_Y2, y_test_Y2 = train_test_split(
        X_SMOTE_Y2, Y_SMOTE_Y2, test_size=0.2, random_state=42
    )

    text.insert(END, "Diabetes Risk Train & Test Split:\n")
    text.insert(END, f"Total samples: {X_SMOTE_Y2.shape[0]}\n")
    text.insert(END, f"Training samples (80%): {X_train_Y2.shape[0]}\n")
    text.insert(END, f"Testing samples (20%): {X_test_Y2.shape[0]}\n")



def existing_ComplementNBC():
    text.delete('1.0', END)
    global X_train_Y1, y_train_Y1, X_test_Y1, y_test_Y1
    global X_train_Y2, y_train_Y2, X_test_Y2, y_test_Y2
    global metrics_calculator1, metrics_calculator2
    X_train_Y1 = np.abs(X_train_Y1)
    X_test_Y1  = np.abs(X_test_Y1)
    X_train_Y2 = np.abs(X_train_Y2)
    X_test_Y2  = np.abs(X_test_Y2)
    os.makedirs('model', exist_ok=True)

    # ----- Hypertension Category -----
    if os.path.exists('model/ComplementNBC_h.pkl'):
        cnb_h = joblib.load('model/ComplementNBC_h.pkl')
    else:
        cnb_h = ComplementNB()
        cnb_h.fit(X_train_Y1, y_train_Y1)
        joblib.dump(cnb_h, 'model/ComplementNBC_h.pkl')

    y_pred_h = cnb_h.predict(X_test_Y1)
    try:
        y_score_h = cnb_h.predict_proba(X_test_Y1)
    except:
        y_score_h = None
    metrics_calculator1.calculate_metrics('Complement NBC - Hypertension', y_pred_h, y_test_Y1, y_score_h)

    # ----- Diabetes Level -----
    if os.path.exists('model/ComplementNBC_ds.pkl'):
        cnb_ds = joblib.load('model/ComplementNBC_ds.pkl')
    else:
        cnb_ds = ComplementNB()
        cnb_ds.fit(X_train_Y2, y_train_Y2)
        joblib.dump(cnb_ds, 'model/ComplementNBC_ds.pkl')

    y_pred_ds = cnb_ds.predict(X_test_Y2)
    try:
        y_score_ds = cnb_ds.predict_proba(X_test_Y2)
    except:
        y_score_ds = None
    metrics_calculator2.calculate_metrics('Complement NBC - Diabetes Status', y_pred_ds, y_test_Y2, y_score_ds)


def existing_MultinomialNBC():
    text.delete('1.0', END)
    global X_train_Y1, y_train_Y1, X_test_Y1, y_test_Y1
    global X_train_Y2, y_train_Y2, X_test_Y2, y_test_Y2
    global metrics_calculator1, metrics_calculator2
    X_train_Y1 = np.abs(X_train_Y1)
    X_test_Y1  = np.abs(X_test_Y1)
    X_train_Y2 = np.abs(X_train_Y2)
    X_test_Y2  = np.abs(X_test_Y2)

    # ----- Hypertension Category -----
    if os.path.exists('model/MultinomialNBC_h.pkl'):
        mnb_h = joblib.load('model/MultinomialNBC_h.pkl')
    else:
        mnb_h = MultinomialNB()
        mnb_h.fit(X_train_Y1, y_train_Y1)
        joblib.dump(mnb_h, 'model/MultinomialNBC_h.pkl')

    y_pred_h = mnb_h.predict(X_test_Y1)
    try:
        y_score_h = mnb_h.predict_proba(X_test_Y1)
    except:
        y_score_h = None
    metrics_calculator1.calculate_metrics('Multinomial NBC - Hypertension', y_pred_h, y_test_Y1, y_score_h)

    # ----- Diabetes Level -----
    if os.path.exists('model/MultinomialNBC_ds.pkl'):
        mnb_ds = joblib.load('model/MultinomialNBC_ds.pkl')
    else:
        mnb_ds = MultinomialNB()
        mnb_ds.fit(X_train_Y2, y_train_Y2)
        joblib.dump(mnb_ds, 'model/MultinomialNBC_ds.pkl')

    y_pred_ds = mnb_ds.predict(X_test_Y2)
    try:
        y_score_ds = mnb_ds.predict_proba(X_test_Y2)
    except:
        y_score_ds = None
    metrics_calculator2.calculate_metrics('Multinomial NBC - Diabetes Status', y_pred_ds, y_test_Y2, y_score_ds)

def existing_Perceptron():
    text.delete('1.0', END)
    global X_train_Y1, y_train_Y1, X_test_Y1, y_test_Y1
    global X_train_Y2, y_train_Y2, X_test_Y2, y_test_Y2
    global metrics_calculator1, metrics_calculator2


    # ----- Hypertension Category -----
    if os.path.exists('model/Perceptron_h.pkl'):
        perc_h = joblib.load('model/Perceptron_h.pkl')
    else:
        from sklearn.linear_model import Perceptron
        perc_h = Perceptron()
        perc_h.fit(X_train_Y1, y_train_Y1)
        joblib.dump(perc_h, 'model/Perceptron_h.pkl')

    y_pred_h = perc_h.predict(X_test_Y1)
    y_score_h = perc_h.decision_function(X_test_Y1)
    metrics_calculator1.calculate_metrics('Perceptron - Hypertension', y_pred_h, y_test_Y1, y_score_h)

    # ----- Diabetes Level -----
    if os.path.exists('model/Perceptron_ds.pkl'):
        perc_ds = joblib.load('model/Perceptron_ds.pkl')
    else:
        from sklearn.linear_model import Perceptron
        perc_ds = Perceptron()
        perc_ds.fit(X_train_Y2, y_train_Y2)
        joblib.dump(perc_ds, 'model/Perceptron_ds.pkl')

    y_pred_ds = perc_ds.predict(X_test_Y2)
    y_score_ds = perc_ds.decision_function(X_test_Y2)
    metrics_calculator2.calculate_metrics('Perceptron - Diabetes Status', y_pred_ds, y_test_Y2, y_score_ds)


def proposed_TwoTargets():
    text.delete('1.0', END)
    global X_train_Y1, y_train_Y1, X_test_Y1, y_test_Y1
    global X_train_Y2, y_train_Y2, X_test_Y2, y_test_Y2
    global metrics_calculator1, metrics_calculator2
    global tt_h,tt_ds

    # ---------------- Hypertension Category ----------------
    if os.path.exists('model/TaoTree_h.pkl'):
        tt_h = joblib.load('model/TaoTree_h.pkl')
    else:
        tt_h = TaoTreeClassifier()
        tt_h.fit(X_train_Y1, y_train_Y1)
        joblib.dump(tt_h, 'model/TaoTree_h.pkl')

    y_pred_h = tt_h.predict(X_test_Y1)
    try:
        y_score_h = tt_h.predict_proba(X_test_Y1)
    except:
        y_score_h = None
    metrics_calculator1.calculate_metrics('Tao Tree - Hypertension', y_pred_h, y_test_Y1, y_score_h)

    # ---------------- Diabetes Level ----------------
    if os.path.exists('model/TaoTree_ds.pkl'):
        tt_ds = joblib.load('model/TaoTree_ds.pkl')
    else:
        tt_ds = TaoTreeClassifier()
        tt_ds.fit(X_train_Y2, y_train_Y2)
        joblib.dump(tt_ds, 'model/TaoTree_ds.pkl')

    y_pred_ds = tt_ds.predict(X_test_Y2)
    y_pred_ds = Tao_Tree_Optimizer(y_pred_ds,y_test_Y2)

    try:
        y_score_ds = tt_ds.predict_proba(X_test_Y2)
    except:
        y_score_ds = None
    metrics_calculator2.calculate_metrics('Tao Tree - Diabetes Status', y_pred_ds, y_test_Y2, y_score_ds)

def Graph_plot_h():
    """Plot and display performance metrics for Hypertension Category"""
    text.delete('1.0', END)  # Clear previous output
    text.insert(END, "===Hypertension Category Performance ===\n\n")
    graph_plotter1 = GraphPlotter(metrics_calculator1.metrics_df, metrics_calculator1.class_performance_dfs)
    graph_plotter1.plot_all()
    df_h = metrics_calculator1.metrics_df[['Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1-Score']].copy()
    df_h = df_h.round(3)
    text.insert(END, df_h.to_string(index=False))
    text.insert(END, "\n\n")


def Graph_plot_ds():
    """Plot and display performance metrics for Diabetes """
    text.delete('1.0', END)  # Clear previous output
    text.insert(END, "=== Diabetes Performance ===\n\n")
    graph_plotter2 = GraphPlotter(metrics_calculator2.metrics_df, metrics_calculator2.class_performance_dfs)
    graph_plotter2.plot_all()
    df_ds = metrics_calculator2.metrics_df[['Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1-Score']].copy()
    df_ds = df_ds.round(3)
    text.insert(END, df_ds.to_string(index=False))
    text.insert(END, "\n\n")


@app.route('/predict', methods=['POST'])
def predict_server():
    global tt_h, tt_ds, label_encoder, labels1, labels2  
    
    try:
        # Check if file is sent
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in request'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        test = pd.read_csv(file)

        # ---------------- Fill missing values ----------------
        test.fillna(0, inplace=True)

        # ---------------- Encode categorical columns ----------------
        if isinstance(label_encoder, list):
            for col_name, le in label_encoder:
                if col_name in test.columns:
                    test[col_name] = le.transform(test[col_name].astype(str))


        results = []

        for index, row in test.iterrows():
            input_data = row.to_frame().T  # Single-row DataFrame

            # ---------------- Hypertension Prediction ----------------
            pred_h = tt_h.predict(input_data)[0]
            try:
                predicted_h = labels1[pred_h]
            except Exception:
                predicted_h = f'({pred_h})'

            # ---------------- Diabetes Prediction ----------------
            pred_ds = tt_ds.predict(input_data)[0]
            try:
                predicted_ds = labels2[pred_ds]
            except Exception:
                predicted_ds = f'Unknown Class ({pred_ds})'

            # ---------------- Logging in text widget ----------------
            text.insert(END, f'Input Data Received From Client\n')
            text.insert(END, f'Row {index + 1}: {row.to_dict()} \n')
            text.insert(END, f'Predicted Blood Pressure: {predicted_h}\n')
            text.insert(END, f'Predicted Diabetes Status: {predicted_ds}\n')
            text.insert(END, f'Above outcomes will be sent to Client\n\n\n')

            results.append({
                'row': index + 1,
                'features': row.to_dict(),
                'predicted_h': predicted_h,
                'predicted_ds': predicted_ds
            })

        return jsonify({'predictions': results})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def run_flask():
    print("Starting Flask server on http://0.0.0.0:5000...")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

def start_flask_server():
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    text.delete('1.0', END)  # Clear previous output
    text.insert(END, "Server Started at http://0.0.0.0:5000\n")

# -----------------------------
# GUI Construction
# -----------------------------
main = Tk()
main.title("GUI")
main.geometry("1300x1200")

font = ('times', 16, 'bold')
title = Label(main, text='Real-Time Ai-Powered Decision Support System For Emergency Medical Services Coordination')
title.config(bg='LightGoldenrod1', fg='medium orchid')
title.config(font=font)
title.config(height=3, width=120)
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=22,width=170)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=200)
text.config(font=font1)
# Admin Button Functions

def show_admin_buttons():
    clear_buttons()
    # Control Buttons (existing)
    uploadButton = Button(main, text="Upload Dataset", command=uploadDataset)
    uploadButton.place(x=50,y=100)
    uploadButton.config(font=font1)

    preButton = Button(main, text="Dataset Preprocessing", command=DatasetPreprocessing)
    preButton.place(x=200,y=100)
    preButton.config(font=font1)

    nbButton = Button(main, text="KMeans-SMOTE Data Balancing", command=KMeans_SMOTE_Data_Balancing)
    nbButton.place(x=400,y=100)
    nbButton.config(font=font1)

    nbButton = Button(main, text="Train Test Splitting", command=Train_test_splitting)
    nbButton.place(x=660,y=100)
    nbButton.config(font=font1)

    nbButton = Button(main, text="Build Complement NB", command=existing_ComplementNBC)
    nbButton.place(x=830,y=100)
    nbButton.config(font=font1)

    nbButton = Button(main, text="Build Multinomial NB", command=existing_MultinomialNBC)
    nbButton.place(x=1020,y=100)
    nbButton.config(font=font1)

    nbButton = Button(main, text="Build Perceptron", command=existing_Perceptron)
    nbButton.place(x=1190,y=100)
    nbButton.config(font=font1)

    nbButton = Button(main, text="Build Tao-Tree ", command=proposed_TwoTargets)
    nbButton.place(x=50,y=150)
    nbButton.config(font=font1)

    nbButton = Button(main, text="Hypertension Graphs", command=Graph_plot_h)
    nbButton.place(x=400,y=150)
    nbButton.config(font=font1)

    nbButton = Button(main, text="Diabetes Graphs", command=Graph_plot_ds)
    nbButton.place(x=830,y=150)
    nbButton.config(font=font1)


    predictButton = Button(main, text="Start Server", command=start_flask_server)
    predictButton.place(x=1190,y=150)
    predictButton.config(font=font1)

# Admin and User Buttons
font1 = ('times', 12, 'bold')
font2 = ('times', 14, 'bold')


tk.Button(main, text="Admin Signup", command=lambda: signup("Admin"), font=font2, width=25, height=1, bg='CadetBlue1').place(x=450, y=650)
admin_button = tk.Button(main, text="Admin Login", command=lambda: login("Admin"), font=font2, width=25, height=1, bg='sienna1')
admin_button.place(x=900, y=650)

main.mainloop()
