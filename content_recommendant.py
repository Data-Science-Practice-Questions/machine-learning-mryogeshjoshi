import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

data = pd.read_csv("content_recommendation_data.csv")

def recommend():
    try:
        sd = int(scroll_input.get())
        ts = float(time_input.get())
        user_input = pd.DataFrame([[sd, ts]], columns=['scroll_depth', 'time_spent'])

        features = data[['scroll_depth', 'time_spent']]
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)
        user_scaled = scaler.transform(user_input)

        cluster = AgglomerativeClustering(n_clusters=3)
        labels = cluster.fit_predict(scaled)
        data['cluster'] = labels
        user_cluster = cluster.fit_predict(user_scaled)[0]

        recommendations = data[data['cluster'] == user_cluster].sample(3)
        result = "\n".join(recommendations['article_title'].tolist())
        messagebox.showinfo("Recommendations", result)
    except Exception as e:
        messagebox.showerror("Error", str(e))

# GUI
root = tk.Tk()
root.geometry("400x300")
root.title("Reading Pattern Recommender")

tk.Label(root, text="Scroll Depth (%):").pack(padx=10, pady=10)
scroll_input = tk.Entry(root)
scroll_input.pack()

tk.Label(root, text="Time Spent (s):").pack(padx=10, pady=10)
time_input = tk.Entry(root)
time_input.pack()

tk.Button(root, text="Recommend", command=recommend).pack(padx=10, pady=10)
root.mainloop()
