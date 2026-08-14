# 🚗 Car Price Prediction Machine Learning Project

A complete, production-ready Machine Learning regression system that predicts the **selling price of used cars** based on historical vehicle characteristics using scikit-learn and Streamlit.

---
App Link :-  https://car-price-prediction-dxtlquent4ohktvhxd7wan.streamlit.app/
## 📌 Project Objective

The goal of this project is to build an end-to-end regression pipeline to predict car resale values (`Selling_Price` in **₹ Lakhs**). The project follows strict machine learning best practices:

$$\text{Data Collection} \rightarrow \text{Data Cleaning} \rightarrow \text{Feature Engineering} \rightarrow \text{EDA} \rightarrow \text{Pipeline Building} \rightarrow \text{Model Evaluation} \rightarrow \text{Streamlit Deployment}$$

---

## 📊 Dataset Information

* **Source File:** `data/car data.csv` (Cardekho Vehicle Dataset)
* **Total Records:** 301 cars (299 unique records after dropping duplicates)
* **Target Variable:** `Selling_Price` (Resale price in **₹ Lakhs**)

### Features Used:

| Feature Column | Data Type | Description |
| :--- | :--- | :--- |
| `Car_Name` | Categorical | Brand / Model name of the vehicle |
| `Year` | Numerical | Year of purchase (used to derive `Car_Age`) |
| `Present_Price` | Numerical | Current ex-showroom price (in **₹ Lakhs**) |
| `Driven_kms` | Numerical | Total distance driven in kilometers |
| `Fuel_Type` | Categorical | `Petrol`, `Diesel`, or `CNG` |
| `Selling_type` | Categorical | `Dealer` or `Individual` |
| `Transmission` | Categorical | `Manual` or `Automatic` |
| `Owner` | Numerical | Number of previous owners (0, 1, 3) |
| `Car_Age` | Derived | Calculated as $\text{Current\_Year} - \text{Year}$ |

---

## 📈 Model Performance & Evaluation Summary

Four machine learning regression models were trained and evaluated on an 80/20 train-test split using reproducible random states (`random_state=42`):

| Model Name | MAE (₹ Lakhs) | MSE | RMSE (₹ Lakhs) | $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | **1.3632** | **5.7122** | **2.3900** | **0.7784** |
| **Gradient Boosting** | 1.1313 | 6.5728 | 2.5638 | 0.7450 |
| **Decision Tree** | 1.3748 | 9.0061 | 3.0010 | 0.6506 |
| **Random Forest** | 1.3837 | 11.0122 | 3.3185 | 0.5727 |

* **Best Model Selected:** **Linear Regression** (or Gradient Boosting depending on feature grouping configuration) saved cleanly to `models/car_price_model.pkl`.

---

## 📁 Project Folder Structure

```text
car-price-prediction/
│
├── data/
│   └── car data.csv                 # Raw dataset file
│
├── notebooks/
│   └── car_price_prediction.ipynb   # Complete step-by-step Jupyter Notebook
│
├── src/
│   ├── preprocessing.py              # Data cleaning & ColumnTransformer pipeline
│   ├── train_model.py                # Model training, evaluation & plot saving
│   └── predict.py                    # Standalone prediction module
│
├── models/
│   └── car_price_model.pkl           # Saved Scikit-learn Pipeline (Preprocessor + Estimator)
│
├── plots/                            # Generated EDA & Evaluation plots
│   ├── selling_price_dist.png
│   ├── selling_vs_present.png
│   ├── selling_vs_driven.png
│   ├── selling_vs_age.png
│   ├── selling_by_fuel.png
│   ├── selling_by_transmission.png
│   ├── correlation_heatmap.png
│   ├── actual_vs_predicted.png
│   └── residual_plot.png
│
├── app.py                            # Interactive Streamlit Web Application
├── requirements.txt                  # Required Python packages
├── README.md                         # Comprehensive documentation
└── .gitignore                        # Git exclusion rules
```

---

## 💻 How to Execute this Program in VS Code (Visual Studio Code)

Follow these easy step-by-step instructions to run the project inside **VS Code**:

### Step 1: Open Project in VS Code
1. Launch **Visual Studio Code**.
2. Go to **File** $\rightarrow$ **Open Folder...** (or press `Ctrl + K, Ctrl + O`).
3. Select the project directory:
   `C:\Users\patup\.gemini\antigravity\scratch\car-price-prediction`

### Step 2: Open Integrated Terminal
1. In VS Code, open the terminal by pressing ``Ctrl + ` `` (or select **Terminal** $\rightarrow$ **New Terminal** from top menu).

### Step 3: Create & Activate Virtual Environment

**On Windows Command Prompt (cmd.exe):**
```cmd
cd /d C:\Users\patup\.gemini\antigravity\scratch\car-price-prediction
python -m venv venv
venv\Scripts\activate.bat
```

**On Windows (PowerShell):**
```powershell
cd C:\Users\patup\.gemini\antigravity\scratch\car-price-prediction
python -m venv venv
.\venv\Scripts\Activate.ps1
```

*(If PowerShell blocks activation, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` first)*

**On Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running Project Modules

### 1. Train the ML Models & Generate Visualizations
Execute `train_model.py` to process data, train regressors, generate plots in `plots/`, and save `models/car_price_model.pkl`:

```bash
python src/train_model.py
```

### 2. Test Prediction via Command Line
Run `predict.py` to execute sample inference:

```bash
python src/predict.py
```

### 3. Launch the Streamlit Web Application
Run the Streamlit app to launch the interactive UI in your browser:

```bash
streamlit run app.py
```
*(The web app will automatically open in your default browser at `http://localhost:8501`)*

### 4. Running the Jupyter Notebook in VS Code
1. Install the **Jupyter** extension in VS Code if prompted.
2. In VS Code File Explorer, open `notebooks/car_price_prediction.ipynb`.
3. Select your `venv` Python kernel at the top right.
4. Click **Run All** to execute all cells sequentially!

---

## 💡 Example Prediction Output

```text
--- Car Price Prediction Test ---
Sample Input: {
    'Car_Name': 'ciaz', 
    'Year': 2017, 
    'Present_Price': 9.85, 
    'Driven_kms': 6900, 
    'Fuel_Type': 'Petrol', 
    'Selling_type': 'Dealer', 
    'Transmission': 'Manual', 
    'Owner': 0
}

Estimated Selling Price: ₹ 7.59 Lakhs (₹ 7,58,689.42)
```

---

## 🔮 Future Improvements

1. Add hyperparameter tuning (GridSearchCV / RandomizedSearchCV) for Gradient Boosting & Random Forest.
2. Integrate real-time used-car market pricing APIs.
3. Deploy Streamlit app to Streamlit Community Cloud or HuggingFace Spaces.
