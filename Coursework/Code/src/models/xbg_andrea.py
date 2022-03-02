from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import estimator_html_repr
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost.sklearn import XGBClassifier

from src.preprocessing import BrowserMapper, DateEncoder, DeviceMapper

import src.main as main

import pandas as pd
import numpy as np

pipeline = make_pipeline(
    ColumnTransformer(transformers=[
        (
            "categorical",
            make_pipeline(
                ColumnTransformer(transformers=[
                    (
                        "date_encoder",
                        DateEncoder(),
                        ["date_account_created"]
                    ),
                    (
                        "date_encoder_2",
                        DateEncoder(date_format='%Y%m%d%H%M%S'),
                        ["timestamp_first_active"]
                    ),
                    (
                        "device_mapper",
                        DeviceMapper(),
                        ["first_device_type"]
                    ),
                    (
                        "browser_mapper",
                        BrowserMapper(),
                        ["first_browser"]
                    )
                ], remainder="passthrough"
                ),
                SimpleImputer(strategy="most_frequent"),
                OneHotEncoder(handle_unknown="ignore")
            ),
            [
                "signup_method",
                "gender",
                "language",
                "affiliate_channel",
                "affiliate_provider",
                "first_affiliate_tracked",
                "signup_app",
                "signup_flow",
                "date_account_created",
                "timestamp_first_active",
                "first_device_type",
                "first_browser",
            ]
        ),
        (
            "continuous",
            make_pipeline(
                SimpleImputer(strategy="mean"),
                StandardScaler()
            ), ["age","empty_vals"]
        )
    ]),
    XGBClassifier(
        objective='multi:softprob',
        n_estimators=150,
        verbosity=1,
        max_depth=10
    )
)

main.run_mlflow_pipeline("data/raw", pipeline, run_test=False)
