import lmdb
import hashlib
import pickle
import pandas as pd
import requests
from tkinter import *
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk

# Replace with your server IP
SERVER_URL = 'http://192.168.0.137:5000/predict'

# LMDB database path
LMDB_PATH = './user_db'

# ------------------- LMDB Functions -------------------
def connect_lmdb():
    env = lmdb.open(LMDB_PATH, map_size=10**7)  # 10MB for user DB
    return env

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def store_user(username, password, role):
    env = connect_lmdb()
    hashed_password = hash_password(password)
    with env.begin(write=True) as txn:
        key = f"user:{username}".encode()
        if txn.get(key):
            return False  # User exists
        value = pickle.dumps({"username": username, "password": hashed_password, "role": role})
        txn.put(key, value)
    env.close()
    return True

def get_user(username):
    env = connect_lmdb()
    with env.begin(write=False) as txn:
        key = f"user:{username}".encode()
        value = txn.get(key)
        if value:
            user_data = pickle.loads(value)
            return user_data
    env.close()
    return None

# ------------------- Signup -------------------
def signup(role):
    def register_user():
        username = username_entry.get()
        password = password_entry.get()
        if username and password:
            if store_user(username, password, role):
                messagebox.showinfo("Success", f"{role} Signup Successful!")
                signup_window.destroy()
            else:
                messagebox.showerror("Error", "User already exists!")
        else:
            messagebox.showerror("Error", "Please enter all fields!")

    signup_window = Toplevel(main)
    signup_window.geometry("400x400")
    signup_window.title(f"{role} Signup")

    Label(signup_window, text="Username").pack(pady=5)
    username_entry = Entry(signup_window)
    username_entry.pack(pady=5)

    Label(signup_window, text="Password").pack(pady=5)
    password_entry = Entry(signup_window, show="*")
    password_entry.pack(pady=5)

    Button(signup_window, text="Signup", command=register_user).pack(pady=10)

# ------------------- Login -------------------
def login(role):
    def verify_user():
        username = username_entry.get()
        password = password_entry.get()
        if username and password:
            user_data = get_user(username)
            if user_data and user_data["role"] == role and user_data["password"] == hash_password(password):
                messagebox.showinfo("Success", f"{role} Login Successful!")
                login_window.destroy()
                if role == "Admin":
                    show_admin_buttons()
                elif role == "User":
                    show_user_buttons()
            else:
                messagebox.showerror("Error", "Invalid Credentials!")
        else:
            messagebox.showerror("Error", "Please enter all fields!")

    login_window = Toplevel(main)
    login_window.geometry("400x300")
    login_window.title(f"{role} Login")

    Label(login_window, text="Username").pack(pady=5)
    username_entry = Entry(login_window)
    username_entry.pack(pady=5)

    Label(login_window, text="Password").pack(pady=5)
    password_entry = Entry(login_window, show="*")
    password_entry.pack(pady=5)

    Button(login_window, text="Login", command=verify_user).pack(pady=10)

# ------------------- Predict from Server -------------------
def predict_from_server():
    file_path = filedialog.askopenfilename(initialdir=".", filetypes=[("CSV files", "*.csv")])
    if not file_path:
        return
    text.delete('1.0', END)
    text.insert(END, f"Sending file: {file_path} to server...\n")

    try:
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(SERVER_URL, files=files)

        if response.ok:
            predictions = response.json()
            df = pd.read_csv(file_path)
            text.insert(END, "Predictions received:\n\n")
            for index, row in df.iterrows():
                item = predictions['predictions'][index]
                text.insert(END, f'Input Data Sent to Client\n')
                text.insert(END, f'Row {index + 1}: {row.to_dict()} \n')
                text.insert(END, f"Predicted Blood Pressure: {item['predicted_h']}\n")
                text.insert(END, f"Predicted Diabetes Status: {item['predicted_ds']}\n\n")
        else:
            text.insert(END, f"Error: {response.text}\n")
    except Exception as e:
        text.insert(END, f"Exception occurred: {str(e)}\n")

# ------------------- UI Buttons -------------------
def show_user_buttons():
    clear_buttons()
    Button(main, text="Predict from Test CSV", command=predict_from_server, font=font1).place(x=350, y=150)

def clear_buttons():
    for widget in main.winfo_children():
        if isinstance(widget, Button) and widget not in [user_button]:
            widget.destroy()

# ------------------- Tkinter Main -------------------
main = Tk()
main.title("Remote Prediction Client")
screen_width = main.winfo_screenwidth()
screen_height = main.winfo_screenheight()
main.geometry(f"{screen_width}x{screen_height}")

bg_image = Image.open("background.png")
bg_image = bg_image.resize((screen_width, screen_height))
bg_photo = ImageTk.PhotoImage(bg_image)
background_label = Label(main, image=bg_photo)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

font = ('times', 17, 'bold')
title = Label(main, text='LMDB Integrated User-Cloud Framework with Hypertension and Diabetes Prediction For Emergency Response')
title.config(bg='LightGoldenrod1', fg='medium orchid', font=font, height=3)
title.pack(fill='x', anchor='center')  # Center horizontally


font1 = ('times', 12, 'bold')
font2 = ('times', 14, 'bold')

text = Text(main, height=22, width=170)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50, y=200)
text.config(font=font1)

Button(main, text="User Signup", command=lambda: signup("User"), font=font2, width=25, height=1, bg='CadetBlue1').place(x=450, y=650)

user_button = Button(main, text="User Login", command=lambda: login("User"), font=font2, width=25, height=1, bg='sienna1')
user_button.place(x=900, y=650)

main.mainloop()
