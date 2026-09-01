import pytest
import numpy as np


class TestMorganBits:
    def test_output_shape(self):
        from src.features import _morgan_bits
        fp = _morgan_bits("CCO")
        assert isinstance(fp, np.ndarray)
        assert fp.ndim == 1

    def test_invalid_smiles(self):
        from src.features import _morgan_bits
        with pytest.raises(ValueError):
            _morgan_bits("INVALID")





class TestFeatureFilter:
    def test_fit_transform(self):
        from src.features import FeatureFilter
        X = np.random.rand(10, 5)
        X[0, 0] = np.nan
        ff = FeatureFilter(nan_threshold=0.5, var_threshold=0.0, corr_threshold=1.0)
        ff.fit(X)
        Xt = ff.transform(X)
        assert Xt.shape[0] == 10
        assert Xt.shape[1] <= 5


class TestAllDescriptors:
    def test_returns_array(self):
        from src.features import _all_descriptors
        desc = _all_descriptors("CCO")
        assert isinstance(desc, np.ndarray)
        assert len(desc) > 10

    def test_invalid_smiles(self):
        from src.features import _all_descriptors
        with pytest.raises(ValueError):
            _all_descriptors("INVALID")
