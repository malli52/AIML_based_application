import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class RainfallRiskModel:

    FEATURES = [
        "temperature_c",
        "humidity_pct",
        "pressure_hpa",
        "wind_speed_kmh",
        "rainfall_1h_mm",
        "cloud_cover_pct",
        "radar_reflectivity_dbz",
        "satellite_rain_index",
        "storm_cell_density",
        "cloud_top_temp_c"
    ]

    TARGET = "risk_label"

    def __init__(self, csv_path):
        self.csv_path = Path(csv_path)

        self.model = None
        self.validation_accuracy = 0.0
        self.training_rows = 0
        self.model_ready = False

    def load_data(self):

        df = pd.read_csv(self.csv_path)

        required_columns = self.FEATURES + [self.TARGET]

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing columns in training.csv: {missing_columns}"
            )

        # Convert feature columns to numeric
        for column in self.FEATURES:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        # Remove rows where target is missing
        df = df.dropna(subset=[self.TARGET])

        # Fill missing feature values using median
        for column in self.FEATURES:
            df[column] = df[column].fillna(
                df[column].median()
            )

        X = df[self.FEATURES]

        y = df[self.TARGET].astype(str).str.strip().str.upper()

        self.training_rows = len(df)

        return X, y

    def train(self):

        X, y = self.load_data()

        if y.nunique() < 2:
            raise ValueError(
                "risk_label must contain at least two different classes."
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        self.model = RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            random_state=42,
            class_weight="balanced"
        )

        self.model.fit(X_train, y_train)

        predictions = self.model.predict(X_test)

        self.validation_accuracy = accuracy_score(
            y_test,
            predictions
        )

        self.model_ready = True

        return self.validation_accuracy

    def predict(self, weather_data):

        if not self.model_ready:
            raise RuntimeError(
                "Model has not been trained yet."
            )

        input_row = {}

        for feature in self.FEATURES:

            value = weather_data.get(feature, 0)

            try:
                value = float(value)
            except (TypeError, ValueError):
                value = 0.0

            input_row[feature] = value

        X = pd.DataFrame(
            [input_row],
            columns=self.FEATURES
        )

        predicted_label = self.model.predict(X)[0]

        probabilities = self.model.predict_proba(X)[0]

        classes = self.model.classes_

        probability_dict = {
            str(label): round(float(prob) * 100, 2)
            for label, prob in zip(classes, probabilities)
        }

        confidence = max(probabilities) * 100

        return {
            "risk_label": str(predicted_label),
            "confidence": round(float(confidence), 2),
            "probabilities": probability_dict
        }