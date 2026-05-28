from __future__ import annotations
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
import matplotlib.pyplot as plt
import pandas as pd
FIG_DIR = Path("/content/drive/MyDrive/breast-cancer/figures")

def get_results(model, name, X_train, X_test, y_train, y_test):
  y_train_pred = model.predict(X_train)
  y_train_score = model.predict_proba(X_train)[:,1]
  y_pred = model.predict(X_test)
  y_score = model.predict_proba(X_test)[:,1]

  # Report the train/test accuracy, roc_auc, and average precision.
  return {
      "model": name,
      "train_acc": accuracy_score(y_train, y_train_pred),
      "test_acc": accuracy_score(y_test, y_pred),
      "train_roc_auc": roc_auc_score(y_train, y_train_score),
      "test_roc_auc": roc_auc_score(y_test, y_score),
      "train_ap": average_precision_score(y_train, y_train_score),
      "test_ap": average_precision_score(y_test, y_score)
  }

def plot_coef(name, features, coefficents):
  coef_df=pd.DataFrame({
      "Feature": features,
      "Coefficent": coefficents
  })
  coef_df = coef_df.set_index("Feature")
  coef_df = coef_df.sort_values(by="Coefficent",key=abs, ascending=False)

  coef_df.plot(kind="bar",figsize=(16,5))
  plt.title(f"{name} Coefficients")
  plt.ylabel("Coefficient Value")
  plt.xlabel("Feature")
  plt.xticks(rotation=45, ha="right")
  plt.tight_layout()
  plt.savefig(FIG_DIR / f"{name}_coef.png", dpi=300, bbox_inches="tight")
  plt.show()

def evaluate_model(model_list, X_train, X_test, y_train, y_test, features, RESULTS_DIR):
  results = []
  for name, model in model_list:
    model.fit(X_train, y_train)
    if isinstance(model, GridSearchCV):
      best_params = model.best_params_
      model = model.best_estimator_
      results.append(get_results(model, f"{name}_{best_params}", X_train, X_test, y_train, y_test))
    else:
      results.append(get_results(model, name, X_train, X_test, y_train, y_test))

    results_df = pd.DataFrame(results)
    if RESULTS_DIR is not None:
      results_df.to_csv(RESULTS_DIR, index=False)

    # confusion matrix, ROC curve and precision-recall curves.
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, normalize="true")
    plt.title(f"{name} Confusion Matrix")
    plt.savefig(FIG_DIR / f"{name}_cm.png")
    plt.show()

    RocCurveDisplay.from_estimator(model, X_test, y_test, name=name)
    plt.title(f"{name} ROC Curve")
    plt.savefig(FIG_DIR / f"{name}_roc.png")
    plt.show()

    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, name=name)
    plt.title(f"{name} Precision-Recall Curve")
    plt.savefig(FIG_DIR / f"{name}_pr.png")
    plt.show()

    # Visualize feature coefficients as vertical bar chart
    if hasattr(model, "coef_"):
      plot_coef(name, features, model.coef_[0])
  
  return results_df
