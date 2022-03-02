from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
from collections import defaultdict


class DateEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, date_format="%Y-%m-%d"):
        self.date_format = date_format

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        sub_dates = []
        for c in X:
            X[c] = pd.to_datetime(X[c], format=self.date_format)
            dt = X[c].dt
            sub_dates.append(dt.year.rename(dt.name + "_year"))
            sub_dates.append(dt.month.rename(dt.name + "_month"))
            sub_dates.append(dt.day.rename(dt.name + "_day"))
            sub_dates.append(dt.dayofweek.rename(dt.name + "_dayofweek"))

            if "%H" in self.date_format or "%I" in self.date_format:
                sub_dates.append(dt.hour.rename(dt.name + "_hour"))

        df = pd.concat(sub_dates, axis=1)
        self.feature_names_ = list(df.columns.values)
        return df

    def get_feature_names(self):
        return self.feature_names_


class DeviceMapper(BaseEstimator, TransformerMixin):
    def __init__(self, software=True, hardware=True):
        self.software = software
        self.hardware = hardware

    def fit(self, X, y=None):
        if self.hardware:
            self.hardware_map = self.fit_hardware_map(X)
        if self.software:
            self.software_map = self.fit_software_map(X)
        return self

    def transform(self, X):
        if self.software:
            X["first_device_type_software"] = X["first_device_type"].map(
                self.software_map
            )
        if self.hardware:
            X["first_device_type_hardware"] = X["first_device_type"].map(
                self.hardware_map
            )

        X = X.drop(columns={"first_device_type"})
        self.feature_names_ = list(X.columns.values)

        return X

    def get_feature_names(self):
        return self.feature_names_

    @staticmethod
    def fit_hardware_map(df):
        hardware_map = defaultdict(key="Other")

        unique_devices = df["first_device_type"].unique().tolist()

        for device in unique_devices:
            words = [x.lower() for x in device.split(" ")]
            phones = ["phone", "iphone", "smartphone"]

            if "desktop" in words:
                hardware_map[device] = "Desktop"

            elif any(d in phones for d in words):
                hardware_map[device] = "Phone"

            elif "tablet" in words or "iPad" in words:
                hardware_map[device] = "Tablet"

            else:
                hardware_map[device] = "Other"

        return hardware_map

    @staticmethod
    def fit_software_map(df):

        software_map = defaultdict(key="Other")

        unique_devices = df["first_device_type"].unique().tolist()

        for device in unique_devices:
            words = [x.lower() for x in device.split(" ")]
            apple = ["mac", "iphone", "ipad"]

            if "windows" in words:
                software_map[device] = "Windows"
            elif any(d in apple for d in words):
                software_map[device] = "Apple"
            elif "android" in words:
                software_map[device] = "Android"
            else:
                software_map[device] = "Other"

        return software_map


class BrowserMapper(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.browser_map = self.fit_browser_map(X)
        return self

    def transform(self, X):
        is_known_browser = X["first_browser"].isin(self.browser_map)

        X.loc[is_known_browser, "first_browser"] = X.loc[
            is_known_browser, "first_browser"
        ].map(self.browser_map)
        X.loc[~is_known_browser, "first_browser"] = "Other"

        self.feature_names_ = list(X.columns.values)
        return X

    def get_feature_names(self):
        return self.feature_names_

    @staticmethod
    def fit_browser_map(df):
        browser_map = defaultdict(key="Other")
        browser_percents = (
            df["first_browser"].value_counts() / df["first_browser"].count()
        ) * 100
        for browser, percent in browser_percents.items():
            split = browser.split(" ")
            if "Mobile" in split:
                split.remove("Mobile")
                browser_map[browser] = "".join(split)
            elif percent > 0.5:
                browser_map[browser] = browser
            else:
                browser_map[browser] = "Other"

        browser_map["Mozilla"] = "Firefox"
        browser_map["Opera Mobile"] = "Other"

        return browser_map

