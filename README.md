# Top-10 Item Recommendation System

## 📌 Project Overview

This project builds a **Top-10 Item Recommendation System** that suggests relevant items to users based on their preferences and interaction data.

The goal of this project is to demonstrate core concepts in recommendation systems, including:

* Similarity-based recommendation
* Ranking and filtering items
* Building an interactive data application

---

## Live Demo

👉 Try the app here:
https://pongaaa-top10-item-recommendation-system-app-na6umc.streamlit.app/

⚠️ **Note:**

* The app is deployed using Streamlit Community Cloud (free tier).
* If the app is inactive, it may go to sleep.
* Please click **"Wake up"** and wait about **15–20 seconds** for it to load.

---

## Features

* Recommend **Top 10 items** based on user input
* Interactive UI built with Streamlit
* Fast response using preprocessed data
* Simple and interpretable recommendation logic
* Suitable for demo and learning purposes

---

## How It Works

The recommendation system follows these steps:

1. **Data Preprocessing**

   * Clean and structure input data
   * Extract relevant features

2. **Similarity Calculation**

   * Compute similarity between items/users
   * Example: Cosine Similarity

3. **Ranking**

   * Score items based on similarity
   * Sort items in descending order

4. **Recommendation**

   * Return Top-N items (Top 10)

---

## Business Value

* Increase user engagement through personalized recommendations
* Improve conversion rate in platforms like:

  * E-commerce (Shopee, Lazada)
  * Content platforms (TikTok, YouTube)
* Help users discover relevant items faster

---

## 🛠️ Tech Stack

* **Python**
* **Pandas / NumPy**
* **Scikit-learn**
* **Streamlit**

---

## ▶️ Run Locally

```bash
git clone <your-repo-link>
cd <your-project-folder>
pip install -r requirements.txt
streamlit run app.py
```

---

## Future Improvements

* Collaborative Filtering (User-based / Item-based)
* Matrix Factorization (ALS, SVD)
* Deep Learning-based recommendation
* Real-time user behavior tracking
* Deploy with scalable backend (Docker, Cloud)

---

## 📎 Notes

* This project is built for **learning and demonstration purposes**
* The dataset and model can be extended for real-world applications
* Performance can be improved with larger datasets and advanced models

---

## 👤 Author

* Name: Duy
* Major: Computer Science
* Focus: Data Analysis / Machine Learning / Recommendation Systems
