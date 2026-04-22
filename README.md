# 🏦 Smart Loan Approval Predictor

An end-to-end **Machine Learning based Loan Approval System** that predicts whether a loan applicant is eligible and also suggests the **maximum loan amount that can be granted** if the requested amount is rejected.

---

## 🚀 Features

* ✅ Predict loan eligibility (Eligible / Not Eligible)
* 📊 Displays prediction probability
* 💳 Uses **CIBIL Score** for realistic credit assessment
* 🔄 Suggests **maximum eligible loan amount** if rejected
* 🌐 FastAPI backend for real-time predictions
* 🖥️ Streamlit frontend for user interaction

---

## 🧠 Problem Statement

Traditional loan approval systems:

* Are manual and time-consuming
* Provide binary results (approve/reject)
* Do not guide customers on eligible loan limits

👉 This project solves it by:

* Automating decision-making using ML
* Providing **data-driven eligibility prediction**
* Suggesting a **lower loan amount instead of rejection**

---

## 🛠️ Tech Stack

### 👨‍💻 Languages

* Python

### 📊 Machine Learning

* Scikit-learn (SVM)
* NumPy, Pandas

### ⚙️ Backend

* FastAPI
* Uvicorn

### 🖥️ Frontend

* Streamlit

### 📦 Others

* Joblib (model persistence)
* Requests (API communication)

---

## 🧩 Project Architecture

```
User Input (Streamlit UI)
        ↓
FastAPI Backend (/predict API)
        ↓
Data Preprocessing + Feature Encoding
        ↓
SVM Model Prediction
        ↓
Eligibility Result + Probability
        ↓
(If Not Eligible)
Iterative Loan Adjustment Logic
        ↓
Max Eligible Loan Amount
```

---

## ⚙️ How It Works

1. User enters applicant details including:

   * Income, Loan Amount, Credit History, etc.
   * CIBIL Score

2. Backend:

   * Converts input into DataFrame
   * Encodes categorical features
   * Aligns columns with training data

3. Model:

   * Predicts eligibility using **SVM (RBF kernel)**
   * Returns probability

4. 💡 If applicant is NOT eligible:

   * System reduces loan amount step-by-step
   * Re-checks eligibility
   * Returns **maximum eligible amount**

---

## 📊 CIBIL Score Integration

Since real CIBIL data is confidential, a **synthetic CIBIL score** is generated using:

* Credit history
* Applicant & co-applicant income
* Loan amount
* Number of dependents

👉 Range: **300 – 900**

---

## 🔄 Max Loan Eligibility Logic

If the applicant is not eligible:

```python
while loan_amount > 0:
    reduce loan_amount
    check eligibility
```

✔ First eligible amount = **maximum grantable loan**

---

## ▶️ How to Run the Project

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/loan-predictor.git
cd loan-predictor
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Train Model

```bash
python train_model.py
```

### 5️⃣ Run Backend

```bash
uvicorn main:app --reload
```

### 6️⃣ Run Frontend (New Terminal)

```bash
streamlit run app.py
```

---

## 📸 Sample Output

* Prediction: **Eligible / Not Eligible**
* Probability:

  ```
  Eligible: 0.82  
  Not Eligible: 0.18  
  ```
* If rejected:

  ```
  Max Eligible Loan Amount: ₹120 (in thousands)
  ```

---

## ⚠️ Limitations

* Uses **synthetic CIBIL score (for demo purpose)**
* No real-time banking or fraud detection integration
* Dataset is limited in size

---

## 🚀 Future Enhancements

* Integrate real **CIBIL API**
* Add **fraud detection model**
* Use **SHAP for explainability**
* Deploy on **AWS / Docker**
* Add **user authentication system**

---

## 🏆 Key Learnings

* End-to-end ML pipeline development
* Feature engineering (CIBIL score)
* Model deployment using FastAPI
* Frontend integration with Streamlit
* Real-world system design thinking

---

## 📌 Author

Venkatesh Deshpande
📧 [venkateshd3096@gmail.com](mailto:venkateshd3096@gmail.com)
🔗 GitHub: [https://github.com/venkatesh616](https://github.com/venkatesh616)
