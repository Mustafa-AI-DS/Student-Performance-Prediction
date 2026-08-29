import os
import zipfile
import requests
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")

print("Libraries imported successfully.")





OUTPUT_DIR = "/kaggle/working/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Output folder is ready:", OUTPUT_DIR)





import os
import zipfile
import requests
import pandas as pd

def find_file(filename, search_path):
    for root, dirs, files in os.walk(search_path):
        if filename in files:
            return os.path.join(root, filename)
    return None


def extract_all_zips(folder):
    """
    Extract any nested zip files inside the folder.
    """
    extracted_any = True

    while extracted_any:
        extracted_any = False

        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith(".zip"):
                    zip_file_path = os.path.join(root, file)
                    extract_to = os.path.join(root, file.replace(".zip", ""))

                    if not os.path.exists(extract_to):
                        os.makedirs(extract_to, exist_ok=True)

                        try:
                            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                                zip_ref.extractall(extract_to)

                            print("Extracted nested zip:", zip_file_path)
                            extracted_any = True

                        except zipfile.BadZipFile:
                            print("Bad zip file skipped:", zip_file_path)


# 1. First search inside Kaggle input
student_por_path = find_file("student-por.csv", "/kaggle/input")

if student_por_path:
    print("Dataset found in Kaggle input:")
    print(student_por_path)

else:
    print("Dataset not found in /kaggle/input. Trying to download from UCI...")

    DATA_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"

    zip_path = "/kaggle/working/student_performance.zip"
    extract_folder = "/kaggle/working/student_performance_data"

    os.makedirs(extract_folder, exist_ok=True)

    response = requests.get(DATA_URL)

    print("Download status code:", response.status_code)
    print("Downloaded file size:", len(response.content), "bytes")

    with open(zip_path, "wb") as file:
        file.write(response.content)

    # Extract main zip
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_folder)

    print("Main zip extracted successfully.")

    # Extract nested zip files if they exist
    extract_all_zips(extract_folder)

    # Find CSV after extraction
    student_por_path = find_file("student-por.csv", extract_folder)

    print("\nFiles found after extraction:")
    for root, dirs, files in os.walk(extract_folder):
        for file in files:
            print(os.path.join(root, file))

    if student_por_path is None:
        raise FileNotFoundError("student-por.csv was not found after extraction.")

    print("\nDataset path:")
    print(student_por_path)


# 2. Load dataset
df = pd.read_csv(student_por_path, sep=";")

print("\nDataset loaded successfully.")
print("Shape:", df.shape)

df.head()





print("Rows and Columns:", df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Info:")
df.info()

print("\nDescriptive Statistics:")
display(df.describe())





print("Total missing values:", df.isnull().sum().sum())

print("\nMissing values by column:")
display(df.isnull().sum())

print("\nNumber of duplicate rows:", df.duplicated().sum())

df = df.drop_duplicates()

print("\nShape after removing duplicate rows:", df.shape)





df["pass_final"] = np.where(df["G3"] >= 10, 1, 0)

display(df[["G1", "G2", "G3", "pass_final"]].head())

print("Target distribution:")
display(df["pass_final"].value_counts())





plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="pass_final")
plt.title("Final Result Distribution: Fail vs Pass")
plt.xlabel("Final Result: 0 = Fail, 1 = Pass")
plt.ylabel("Number of Students")
plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/target_distribution.png", dpi=300)
plt.show()





plt.figure(figsize=(8, 5))
sns.histplot(df["G3"], bins=20, kde=True)
plt.title("Distribution of Final Grade G3")
plt.xlabel("Final Grade G3")
plt.ylabel("Number of Students")
plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/final_grade_distribution.png", dpi=300)
plt.show()





plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="studytime", hue="pass_final")
plt.title("Study Time vs Final Result")
plt.xlabel("Study Time Level")
plt.ylabel("Number of Students")
plt.legend(title="Final Result", labels=["Fail", "Pass"])
plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/studytime_vs_result.png", dpi=300)
plt.show()





plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="failures", hue="pass_final")
plt.title("Previous Failures vs Final Result")
plt.xlabel("Number of Previous Class Failures")
plt.ylabel("Number of Students")
plt.legend(title="Final Result", labels=["Fail", "Pass"])
plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/failures_vs_result.png", dpi=300)
plt.show()





numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

corr_with_g3 = df[numeric_cols].corr()["G3"].sort_values(ascending=False)

print("Correlation with Final Grade G3:")
display(corr_with_g3)

plt.figure(figsize=(8, 7))
corr_with_g3.drop("G3").sort_values().plot(kind="barh")
plt.title("Correlation of Numerical Features with Final Grade G3")
plt.xlabel("Correlation")
plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/correlation_with_g3.png", dpi=300)
plt.show()





def create_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(X):
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_features = X.select_dtypes(exclude=["object"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", create_one_hot_encoder())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor





def run_experiment(experiment_name, drop_previous_grades=False):
    data = df.copy()

    y = data["pass_final"]

    drop_columns = ["G3", "pass_final"]

    if drop_previous_grades:
        drop_columns += ["G1", "G2"]

    X = data.drop(columns=drop_columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42)
    }

    results = []
    trained_models = {}

    for model_name, model in models.items():
        preprocessor = build_preprocessor(X_train)

        pipeline = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)

        results.append({
            "Experiment": experiment_name,
            "Model": model_name,
            "Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4),
            "ROC_AUC": round(roc_auc, 4)
        })

        trained_models[model_name] = {
            "pipeline": pipeline,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred
        }

    results_df = pd.DataFrame(results)

    return results_df, trained_models





results_with_grades, models_with_grades = run_experiment(
    experiment_name="With Previous Grades G1 and G2",
    drop_previous_grades=False
)

display(results_with_grades)





results_without_grades, models_without_grades = run_experiment(
    experiment_name="Without Previous Grades G1 and G2",
    drop_previous_grades=True
)

display(results_without_grades)





all_results = pd.concat(
    [results_with_grades, results_without_grades],
    ignore_index=True
)

all_results = all_results.sort_values(by="F1", ascending=False)

print("Final Model Comparison:")
display(all_results)

all_results.to_csv(f"{OUTPUT_DIR}/model_results.csv", index=False)

print("Results saved to:")
print(f"{OUTPUT_DIR}/model_results.csv")





plt.figure(figsize=(12, 6))
sns.barplot(data=all_results, x="Model", y="F1", hue="Experiment")
plt.title("Model Comparison Based on F1 Score")
plt.xlabel("Machine Learning Model")
plt.ylabel("F1 Score")
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/model_comparison_f1.png", dpi=300)
plt.show()





best_row = all_results.iloc[0]

best_experiment = best_row["Experiment"]
best_model_name = best_row["Model"]

print("Best Experiment:", best_experiment)
print("Best Model:", best_model_name)

if best_experiment == "With Previous Grades G1 and G2":
    best_model_data = models_with_grades[best_model_name]
else:
    best_model_data = models_without_grades[best_model_name]

cm = confusion_matrix(
    best_model_data["y_test"],
    best_model_data["y_pred"]
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Fail", "Pass"]
)

disp.plot()
plt.title(f"Confusion Matrix - {best_model_name}")
plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/confusion_matrix_best_model.png", dpi=300)
plt.show()





print("Classification Report for Best Model:")

print(
    classification_report(
        best_model_data["y_test"],
        best_model_data["y_pred"],
        target_names=["Fail", "Pass"]
    )
)





def plot_random_forest_importance(model_dictionary, title, output_filename):
    rf_pipeline = model_dictionary["Random Forest"]["pipeline"]

    preprocess = rf_pipeline.named_steps["preprocess"]
    model = rf_pipeline.named_steps["model"]

    feature_names = preprocess.get_feature_names_out()
    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(15)

    print(title)
    display(importance_df)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x="Importance", y="Feature")
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()

    plt.savefig(f"{OUTPUT_DIR}/{output_filename}", dpi=300)
    plt.show()


plot_random_forest_importance(
    models_with_grades,
    "Top 15 Important Features - Random Forest With Previous Grades",
    "feature_importance_with_grades.png"
)

plot_random_forest_importance(
    models_without_grades,
    "Top 15 Important Features - Random Forest Without Previous Grades",
    "feature_importance_without_grades.png"
)





df.to_csv(f"{OUTPUT_DIR}/cleaned_student_performance.csv", index=False)

print("Cleaned dataset saved to:")
print(f"{OUTPUT_DIR}/cleaned_student_performance.csv")





print("Saved output files:")

for file in os.listdir(OUTPUT_DIR):
    print(file)
