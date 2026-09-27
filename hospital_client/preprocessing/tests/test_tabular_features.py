import pandas as pd
import numpy as np
from hospital_client.preprocessing.tabular.quality import TabularQualityAssessor
from hospital_client.preprocessing.tabular.missing_values import ImputationFitter
from hospital_client.preprocessing.tabular.encoding import EncoderFitter
from hospital_client.preprocessing.config import TabularConfig, TabularImputeStrategy
from hospital_client.preprocessing.output.materializer import Materializer
from hospital_client.preprocessing.config import StorageSafetyConfig

def test_tabular_quality_empty():
    config = TabularConfig()
    qa = TabularQualityAssessor(config)
    is_valid, reasons, meta = qa.assess_quality(pd.DataFrame())
    assert not is_valid
    
def test_tabular_quality_excluded():
    config = TabularConfig(excluded_columns=["ignore_me"])
    qa = TabularQualityAssessor(config)
    df = pd.DataFrame({"ignore_me": [np.nan, np.nan], "keep_me": [1, 2]})
    is_valid, reasons, meta = qa.assess_quality(df)
    assert "ignore_me" not in meta["column_stats"]

def test_imputation_fitter_strategies():
    df = pd.DataFrame({
        "mean_col": [1.0, 3.0, np.nan, 2.0],
        "median_col": [1.0, 100.0, 2.0, np.nan],
        "const_num": [np.nan, np.nan, np.nan, np.nan],
        "cat_col": ["A", "B", "A", None],
        "cat_const": [None, None, None, None]
    })
    
    # Mean
    fitter = ImputationFitter(TabularConfig(missing_numeric_strategy=TabularImputeStrategy.MEAN, missing_categorical_strategy=TabularImputeStrategy.MODE))
    meta = fitter.fit(df)
    assert meta["mean_col"] == 2.0
    assert meta["cat_col"] == "A"
    
    # Median
    fitter = ImputationFitter(TabularConfig(missing_numeric_strategy=TabularImputeStrategy.MEDIAN))
    meta = fitter.fit(df[["median_col"]])
    assert meta["median_col"] == 2.0
    
    # Mode numeric
    fitter = ImputationFitter(TabularConfig(missing_numeric_strategy=TabularImputeStrategy.MODE))
    meta = fitter.fit(pd.DataFrame({"mode_col": [1, 1, 2, np.nan]}))
    assert meta["mode_col"] == 1.0

    # Constant
    fitter = ImputationFitter(TabularConfig(missing_numeric_strategy=TabularImputeStrategy.CONSTANT, missing_categorical_strategy=TabularImputeStrategy.CONSTANT))
    meta = fitter.fit(pd.DataFrame({"num": [np.nan], "cat": [None]}))
    assert meta["num"] == 0.0
    assert meta["cat"] == "unknown"

def test_imputation_empty():
    fitter = ImputationFitter(TabularConfig())
    assert fitter.fit(pd.DataFrame()) == {}
    assert fitter.transform(pd.DataFrame({"a": [1]})).shape == (1,1)

def test_encoding_empty():
    fitter = EncoderFitter(TabularConfig())
    assert fitter.fit(pd.DataFrame()) == {}
    assert fitter.transform(pd.DataFrame({"a": [1]})).shape == (1,1)

def test_encoder_scaling_edge_cases():
    df = pd.DataFrame({
        "zero_std": [5.0, 5.0, 5.0],
        "nan_col": [np.nan, np.nan, np.nan],
        "minmax_zero_range": [2.0, 2.0, 2.0]
    })
    
    # Standard
    fitter = EncoderFitter(TabularConfig(numerical_scaling="standard"))
    meta = fitter.fit(df)
    assert meta["scaling"]["zero_std"]["std"] == 1.0
    assert meta["scaling"]["nan_col"]["mean"] == 0.0
    
    # Minmax
    fitter = EncoderFitter(TabularConfig(numerical_scaling="minmax"))
    meta = fitter.fit(df)
    assert meta["scaling"]["nan_col"]["min"] == 0.0
    assert meta["scaling"]["minmax_zero_range"]["max"] == 3.0
    
    # Transform
    res = fitter.transform(df)
    assert not res.empty

def test_categorical_encoding():
    df = pd.DataFrame({"cat": ["B", "A", "C", np.nan]})
    fitter = EncoderFitter(TabularConfig(categorical_encoding="label"))
    fitter.fit(df)
    res = fitter.transform(pd.DataFrame({"cat": ["A", "C", "D"]})) # D is unseen -> NaN
    assert res["cat"].iloc[0] == 0.0
    assert res["cat"].iloc[1] == 2.0
    assert np.isnan(res["cat"].iloc[2])

def test_materializer_safety_off():
    m = Materializer("dummy", StorageSafetyConfig(enforce_safety=False))
    safe, meta = m.check_storage_safety(1000000000)
    assert safe
    assert meta["enforced"] == False

