"""
Family-related feature transformers.

Features derived from SibSp (siblings/spouses) and Parch (parents/children):
- FamilySize: Total family members aboard (SibSp + Parch + 1 for self)
- IsAlone: Binary flag for solo travelers (future addition)
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FamilySizeTransformer(BaseEstimator, TransformerMixin):
    """
    Creates FamilySize feature from SibSp and Parch columns.
    
    FamilySize = SibSp + Parch + 1 (including the passenger themselves)
    
    Parameters
    ----------
    drop_original : bool, default=True
        If True, drops the original SibSp and Parch columns after creating FamilySize.
    include_self : bool, default=True
        If True, adds 1 to count the passenger themselves in family size.
        
    Attributes
    ----------
    feature_names_out_ : list
        Names of output features after transformation.
        
    Example
    -------
    >>> from features import FamilySizeTransformer
    >>> transformer = FamilySizeTransformer(drop_original=True)
    >>> X_transformed = transformer.fit_transform(X)
    """
    
    def __init__(self, drop_original=True, include_self=True):
        self.drop_original = drop_original
        self.include_self = include_self
        self.feature_names_out_ = None
        self._input_columns = None
    
    def fit(self, X, y=None):
        """
        Fit the transformer. 
        
        This transformer is stateless, but fit() records input columns
        for get_feature_names_out().
        
        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe containing SibSp and Parch columns.
        y : ignored
        
        Returns
        -------
        self
        """
        # Validate required columns exist
        required = ["SibSp", "Parch"]
        missing = [col for col in required if col not in X.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Store input columns for feature name generation
        self._input_columns = list(X.columns)
        self._build_feature_names_out()
        
        return self
    
    def transform(self, X):
        """
        Transform the dataframe by adding FamilySize column.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe containing SibSp and Parch columns.
            
        Returns
        -------
        pd.DataFrame
            Transformed dataframe with FamilySize column added.
        """
        if self._input_columns is None:
            raise ValueError("Transformer not fitted. Call fit() first.")
        
        X = X.copy()
        
        # Create FamilySize
        family_size = X["SibSp"] + X["Parch"]
        if self.include_self:
            family_size += 1
        X["FamilySize"] = family_size
        
        # Optionally drop original columns
        if self.drop_original:
            X = X.drop(columns=["SibSp", "Parch"])
        
        return X
    
    def _build_feature_names_out(self):
        """Build output feature names based on configuration."""
        output_cols = self._input_columns.copy()
        
        # Add FamilySize
        output_cols.append("FamilySize")
        
        # Remove originals if configured
        if self.drop_original:
            output_cols = [c for c in output_cols if c not in ["SibSp", "Parch"]]
        
        self.feature_names_out_ = output_cols
    
    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names.
        
        Parameters
        ----------
        input_features : ignored
            For sklearn compatibility.
            
        Returns
        -------
        list
            List of output feature names.
        """
        if self.feature_names_out_ is None:
            raise ValueError("Transformer not fitted. Call fit() first.")
        return self.feature_names_out_
