# Iris Flower Classification

A complete beginner-friendly data science project that trains a Logistic Regression classifier to predict an Iris flower species from four measurements.

## Objective

Build a reproducible baseline classification workflow: inspect the data, split it into training and test sets, scale features without leaking test information, train a model, evaluate its predictions, and save the fitted pipeline.

## Dataset

`iris_dataset.csv` contains 150 rows: 50 samples for each of three species. The four numeric measurements are sepal length, sepal width, petal length, and petal width, all in centimeters. The target is `species`.

The data is from the [UCI Machine Learning Repository Iris dataset](https://archive.ics.uci.edu/dataset/53/iris) (Fisher, 1936; DOI: [10.24432/C56C76](https://doi.org/10.24432/C56C76)), licensed CC BY 4.0. Column names in the included CSV are normalized for Python use.

## Project files

- `iris_dataset.csv` — included dataset; no download or internet access is needed to run the project.
- `iris_modeling.py` — data checks, exploratory summaries, training, evaluation, prediction, and model export.
- `requirements.txt` — Python package requirements.
- `iris_logistic_regression.joblib` — created when the script runs; contains the fitted scaling and classification pipeline.
- `confusion_matrix.png` — created when the script runs.

## Methodology

1. Load the local CSV and inspect its shape, sample rows, data types, summary statistics, class counts, missing values, and exact duplicates.
2. Use the four measurement columns as features and `species` as the target.
3. Make a stratified 80/20 train-test split with `random_state=42`.
4. Fit `StandardScaler` and Logistic Regression in a scikit-learn pipeline. The scaler learns its parameters from training data only.
5. Evaluate once on the held-out test split and display a confusion matrix.
6. Demonstrate a prediction for a sample flower and save the fitted pipeline with joblib.

## Metrics

- **Accuracy:** fraction of test examples classified correctly.
- **Precision (macro):** unweighted average of class-specific precision; indicates how often predictions for each class are correct.
- **Recall (macro):** unweighted average of class-specific recall; indicates how many examples of each class are found.
- **F1 (macro):** unweighted average of class-specific precision/recall balance.
- **Classification report:** per-class precision, recall, F1, and support.
- **Confusion matrix:** counts actual versus predicted species.

Macro averages give each species equal weight. The exact values are printed when you run the script.

## How to run

### Windows PowerShell / Command Prompt

1. Extract the ZIP and open a terminal in the extracted folder.
2. Create and activate a virtual environment (recommended):

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If `py` is unavailable, use `python` in its place.
3. Install dependencies and run the project:

   ```powershell
   python -m pip install -r requirements.txt
   python iris_modeling.py
   ```

The script writes the model file and confusion-matrix image beside itself. The pip “new release available” notice is informational and does not mean the project failed.

### Google Colab

Upload and extract the ZIP in Colab, then run:

```python
%cd /content/iris_classification_project
!pip install -r requirements.txt
!python iris_modeling.py
```

If Colab extracts the folder under a different name, update the `%cd` path accordingly.

## Limitations

- Iris is a small, clean, classic teaching dataset and does not represent the messiness or variety of a real production dataset.
- One 80/20 split gives a noisy estimate; cross-validation and an untouched external test set would give stronger evidence.
- Logistic Regression is a simple linear baseline. Its decision boundaries may not capture every relationship between measurements and species.
- Strong results on this dataset do not establish that the model will work on flowers measured elsewhere or with different measurement procedures.
- The sample prediction is only an illustration, not a scientific identification tool.

## Conclusion

This project demonstrates an end-to-end classification baseline using a public dataset. It reports multiple complementary metrics, visualizes errors, and saves a reusable preprocessing-plus-model pipeline. Treat its score as a teaching result; assess more diverse data and use cross-validation before drawing broader conclusions.
