💡 Household Energy Bill Prediction Using SVM

This project is a complete end-to-end machine learning pipeline to predict household electricity bills based on household features using Support Vector Machine (SVM). It is organized with clean, class-based Python code, and presented in Jupyter Notebook format.




📁 Project Structure

project/
├── notebook.ipynb
├── graphs/
│   ├── histograms.png
│   └── heatmap.png
├── models/
│   └── svm_model.pkl
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── data_analyzer.py
│   ├── data_splitter.py
│   └── svm_model.py
└── README.md





🔗 Dataset

File: Household energy bill data.csv

Target Variable: amount_paid

Features: no_of_people, no_of_rooms, no_of_appliances, monthly_income, etc.






🧠 Project Workflow


📥 Phase 1: Data Collection

The dataset was provided in CSV format.



🔍 Data Understanding

Loaded the data using a custom DataLoader class.

Displayed shape, data types, null values, and duplicates.

Descriptive statistics were generated.




📊 Univariate & 📈 Bivariate Analysis

Histograms for distribution.

Heatmap to show feature correlation.




🧹 Data Preprocessing

Missing values handled using median.

Removed duplicates.

Done using Preprocessor class.




✂️ Data Splitting

Used DataSplitter class to split data into training and test sets.

Target column: amount_paid

Split ratio: 80% train, 20% test




🤖 Model Training (SVM)

Created and trained an SVR model using SVMModel class.

Model evaluated using Mean Squared Error and R-squared.



💾 Model Storage

Saved the model to models/svm_model.pkl using joblib.



📷 Visualizations

Histograms of each numeric column

Correlation heatmap between features

(Visuals stored in graphs/ folder)




📊 Results

Metric

Value

MSE

10891.23 (example)

R²

0.76 (example)

(Values should be replaced with your actual output from notebook)




🛠 Technologies Used

Python 3.x

Jupyter Notebook

Pandas, NumPy

Seaborn, Matplotlib

Scikit-learn

joblib





🧑‍🎓 Author

Name: Your NameUniversity: Virtual UniversityDepartment: Computer ScienceSupervisor: [Your Professor's Name]





📎 How to Run This Project

Clone the repository

Install dependencies: pip install -r requirements.txt

Open notebook.ipynb

Run each cell sequentially

View results and saved model in models/





✅ Future Improvements

Try different regression models

Perform hyperparameter tuning

Add more advanced visualizations





📌 License

This project is for academic purposes only. Do not use commercially without permission.