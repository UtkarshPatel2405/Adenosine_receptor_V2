import pytest
from rdkit import Chem


class TestCanonicalize:
    def test_valid_smiles(self):
        from src.chem_utils import canonicalize
        result = canonicalize("C(O)C")
        assert result == "CCO"

    def test_invalid_smiles(self):
        from src.chem_utils import canonicalize
        assert canonicalize("INVALID") is None

    def test_empty_smiles(self):
        from src.chem_utils import canonicalize
        assert canonicalize("") is None

    def test_roundtrip(self):
        from src.chem_utils import canonicalize
        can1 = canonicalize("c1ccccc1")
        can2 = canonicalize(can1)
        assert can1 == can2


class TestMolFromSmiles:
    def test_valid_smiles(self):
        from src.chem_utils import mol_from_smiles
        mol = mol_from_smiles("CCO")
        assert mol is not None
        assert mol.GetNumAtoms() >= 3

    def test_invalid_smiles(self):
        from src.chem_utils import mol_from_smiles
        assert mol_from_smiles("INVALID") is None


class TestQedProfile:
    def test_returns_dict(self):
        from src.chem_utils import qed_profile
        result = qed_profile("CCO")
        assert isinstance(result, dict)
        assert "QED" in result
        assert "MW" in result
        assert "LogP" in result

    def test_invalid_smiles(self):
        from src.chem_utils import qed_profile
        assert qed_profile("INVALID") is None


class TestCheckPAINS:
    def test_returns_list(self):
        from src.chem_utils import check_pains
        result = check_pains("CCO")
        assert isinstance(result, list)

    def test_invalid_smiles(self):
        from src.chem_utils import check_pains
        assert check_pains("INVALID") == []


class TestGenerate3DConformer:
    def test_valid_smiles(self):
        from src.chem_utils import generate_3d_conformer
        mol_block, min_c, max_c = generate_3d_conformer("CCO")
        assert max_c >= min_c
