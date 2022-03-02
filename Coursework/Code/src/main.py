"""
Downloads the MovieLens dataset and saves it as an artifact
"""
import os

import mlflow
import numpy as np
import pandas as pd
from pandas._config.config import reset_option
from sklearn.base import BaseEstimator
from sklearn.utils import estimator_html_repr

from src.cross_validation import custom_cross_validate


def run_mlflow_pipeline(
    input_filepath: str, pipeline: BaseEstimator, pipeline_viz_path: str = None, run_test: bool = False
):
    mlflow.set_tracking_uri("http://svm-hjg1g17-gdp39.ecs.soton.ac.uk")
    mlflow.set_experiment("/")
    mlflow.sklearn.autolog()

    with mlflow.start_run():

        train = pd.read_csv(os.path.join(input_filepath, "train_users_2.csv"))
        test = pd.read_csv(os.path.join(input_filepath, "test_users.csv"))

        train.replace('-unknown-',np.nan, inplace=True)
        train.replace('NaN',np.nan, inplace=True)
        test.replace('-unknown-',np.nan, inplace=True)
        test.replace('NaN',np.nan, inplace=True)

        train['empty_vals'] = train.isna().sum(axis=1).astype(int)
        test['empty_vals'] = test.isna().sum(axis=1).astype(int)

        y_label = "country_destination"

        X, y = train.loc[:, train.columns != y_label], train[y_label]

        if pipeline_viz_path:
            with open(pipeline_viz_path, "w") as f:
                f.write(estimator_html_repr(pipeline))

        if not run_test:
            # print(cross_val_score(pipeline, X, y, cv=5, scoring="ndcg"))
            ndcg_scores, f1_scores = custom_cross_validate(pipeline, X, y)
            print("nDCG Scores: {}".format(ndcg_scores))
            print("Average nDCG: {}".format(np.mean(ndcg_scores)))

            print("f1 Scores: {}".format(f1_scores))
            print("Average f1: {}".format(np.mean(f1_scores)))

            mlflow.log_metric("cv_ndcg_Score", np.mean(ndcg_scores))
            mlflow.log_metric("cv_f1_Score", np.mean(f1_scores))

        else:
            model = pipeline.fit(X, y)
            preds = model.predict_proba(test)

            
            test_results = []
            for pred, id in zip(preds, test["id"]):
                for label, __ in sorted(list(zip(model.classes_, pred)), key=lambda x: x[1], reverse=True)[:5]:
                    test_results.append((id, label))

            results_df = pd.DataFrame(test_results)
            results_df.rename(columns={0: "id", 1: "country"}, inplace=True)

            results_df.to_csv("data/results/submission.csv", index=False)
