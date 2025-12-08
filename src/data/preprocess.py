"""
Minimal 'basic' Titanic preprocessor using sklearn ColumnTransformer.

Implements the following behaviour (only 'basic'):
- Drop uninformative columns (PassengerId, Ticket, Cabin, Embarked)
- Optionally drop `Name` (controlled by `keep_name`)
- Numeric features: impute missing values with the mean
- Ordinal features (Pclass): impute with the mode, then encode as ordinal
- Categorical features (Sex): impute with the mode, then one-hot encode

The class exposes `fit`, `transform`, and `fit_transform`, and returns a
pandas.DataFrame from `transform` so downstream code can access column names.
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import warnings

warnings.filterwarnings('ignore')


class TitanicPreprocessor(BaseEstimator, TransformerMixin):
    """Basic Titanic preprocessor.

    Only the 'basic' method is implemented. Other methods are deliberately
    unsupported here so you can add them later.
    """

    DEFAULT_DROP = ['PassengerId', 'Ticket', 'Cabin', 'Embarked']

    def __init__(
        self,
        method='basic',
        keep_name=False,
        numeric_features=None,
        ordinal_features=None,
        categorical_features=None,
    ):
        # Accept 'keep_name' for backwards compatibility with existing scripts
        self.method = method
        if self.method != 'basic':
            raise ValueError("This implementation supports only method='basic'.")

        self.keep_name = keep_name
        self.numeric_features = numeric_features or ['Age', 'SibSp', 'Parch', 'Fare']
        self.ordinal_features = ordinal_features or ['Pclass']
        self.categorical_features = categorical_features or ['Sex']

        # columns we will remove before preprocessing
        self.columns_to_drop = [c for c in self.DEFAULT_DROP]
        if not self.keep_name:
            # user asked not to keep the Name column -> drop it
            self.columns_to_drop.append('Name')

        self.pipeline = None
        self.feature_names_out_ = None

    def _build_pipeline(self):
        # Numeric: mean imputation
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
        ])

        # Ordinal: most frequent (mode) imputation, then OrdinalEncoder
        ordinal_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
        ])

        # Categorical: most frequent imputation, then one-hot encode
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
        ])

        transformers = []
        if self.numeric_features:
            transformers.append(('num', numeric_transformer, self.numeric_features))
        if self.ordinal_features:
            transformers.append(('ord', ordinal_transformer, self.ordinal_features))
        if self.categorical_features:
            transformers.append(('cat', categorical_transformer, self.categorical_features))

        preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
        return preprocessor

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the basic preprocessor on training data.

        This will compute the required statistics (means/modes) on the provided
        training data. Always call this on training data only.
        """
        Xc = X.copy()

        # Drop unneeded columns if present
        Xc = Xc.drop(columns=[c for c in self.columns_to_drop if c in Xc.columns])

        # Build and fit pipeline
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(Xc)

        # build output feature names
        self._build_feature_names_out()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform a dataframe and return a new dataframe with column names.

        Raises:
            ValueError: if fit() was not called first.
        """
        if self.pipeline is None:
            raise ValueError('Preprocessor not fitted. Call fit() first.')

        Xc = X.copy()
        Xc = Xc.drop(columns=[c for c in self.columns_to_drop if c in Xc.columns])

        arr = self.pipeline.transform(Xc)

        # convert to DataFrame with proper feature names
        cols = self.get_feature_names_out()
        return pd.DataFrame(arr, columns=cols, index=Xc.index)

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)

    def _build_feature_names_out(self):
        """Construct feature names after transformation (informational)."""
        feature_names = []

        # numeric features remain with their names
        feature_names.extend(self.numeric_features)

        # ordinal features: suffix with _ordinal
        feature_names.extend([f"{c}_ordinal" for c in self.ordinal_features])

        # categorical one-hot names
        if self.categorical_features:
            onehot = self.pipeline.named_transformers_['cat'].named_steps['onehot']
            cat_names = list(onehot.get_feature_names_out(self.categorical_features))
            feature_names.extend(cat_names)

        self.feature_names_out_ = feature_names

    def get_feature_names_out(self):
        if self.feature_names_out_ is None:
            raise ValueError('Preprocessor not fitted. Call fit() first.')
        return self.feature_names_out_

    def get_config(self):
        return {
            'method': 'basic',
            'keep_name': self.keep_name,
            'numeric_features': self.numeric_features,
            'ordinal_features': self.ordinal_features,
            'categorical_features': self.categorical_features,
            'columns_to_drop': self.columns_to_drop,
        }
