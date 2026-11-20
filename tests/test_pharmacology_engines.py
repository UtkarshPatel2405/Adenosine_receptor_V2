"""Unit and pharmacology contract tests for translational engines, ADMET, conformal calibration, Ki, and AD."""
import pytest

import src.applicability_domain as ad_mod
from src.applicability_domain import check_applicability_domain
from src.config import PROCESSED_DATA_DIR
from src.models.adaptive_conformal import calibrate_adaptive_interval
from src.models.admet_engine import _mpo_tpsa_wager2010, evaluate_cns_admet
from src.models.efficacy_engine import predict_functional_efficacy
from src.models.interaction_engine import analyze_pocket_interactions
from src.models.ki_utils import format_ki_display, pchembl_to_ki_nm
from src.models.safety_engine import evaluate_safety_profile
from src.models.selectivity_engine import compute_selectivity_spectrum
from src.presets import PRESETS


class TestSelectivityEngine:
    def test_highly_selective_ranking_and_pairwise_deltas(self):
        preds = {"A1": 5.5, "A2A": 8.5, "A2B": 5.0, "A3": 5.2}
        res = compute_selectivity_spectrum(preds, in_domain=True)
        assert res["best_target"] == "A2A"
        assert res["rank_hierarchy"] == ["A2A", "A1", "A3", "A2B"]
        assert res["selectivity_margin"] == pytest.approx(3.0)
        assert res["selectivity_fold"] == pytest.approx(1000.0)
        assert "Highly Selective" in res["classification"]
        assert res["pairwise_deltas"]["A1_vs_A2A"] == pytest.approx(-3.0)
        assert res["pairwise_deltas"]["A2A_vs_A2B"] == pytest.approx(3.5)

    def test_subtype_selective_preferring_and_pan_classifications(self):
        res_sel = compute_selectivity_spectrum({"A1": 7.4, "A2A": 6.0, "A2B": 5.5, "A3": 5.5}, in_domain=True)
        assert "Subtype-Selective" in res_sel["classification"]

        res_pref = compute_selectivity_spectrum({"A1": 6.6, "A2A": 6.0, "A2B": 5.5, "A3": 5.5}, in_domain=True)
        assert "Preferring" in res_pref["classification"]

        res_pan = compute_selectivity_spectrum({"A1": 6.2, "A2A": 6.0, "A2B": 5.9, "A3": 5.8}, in_domain=True)
        assert "Pan-Adenosine" in res_pan["classification"]

    def test_inactive_and_out_of_domain(self):
        res_inactive = compute_selectivity_spectrum({"A1": 4.8, "A2A": 4.5, "A2B": 4.2, "A3": 4.1}, in_domain=True)
        assert "Inactive" in res_inactive["classification"]
        assert res_inactive["selectivity_margin"] == 0.0
        assert res_inactive["selectivity_fold"] == 1.0

        res_ood = compute_selectivity_spectrum({"A1": 8.5, "A2A": 5.0, "A2B": 5.0, "A3": 5.0}, in_domain=False)
        assert "Outside Applicability Domain" in res_ood["classification"]
        assert res_ood["selectivity_margin"] == 0.0


class TestEfficacyEngine:
    def test_agonists_classified_correctly(self):
        cgs = predict_functional_efficacy(PRESETS["CGS-21680"]["smiles"], "A2A", 8.0, in_domain=True)
        assert cgs["efficacy_class"] == "Agonist"
        assert cgs["activation_probability"] == pytest.approx(0.95)
        assert "Gs-coupled" in cgs["signaling_pathway"]

        ccpa = predict_functional_efficacy(PRESETS["CCPA"]["smiles"], "A1", 8.0, in_domain=True)
        assert ccpa["efficacy_class"] == "Agonist"
        assert ccpa["activation_probability"] == pytest.approx(0.85)
        assert "Gi/o-coupled" in ccpa["signaling_pathway"]

    def test_xanthine_core_antagonist(self):
        theobromine = "Cn1cnc2c1c(=O)[nH]c(=O)n2C"
        res = predict_functional_efficacy(theobromine, "A2A", 7.0, in_domain=True)
        assert res["efficacy_class"] == "Antagonist"
        assert res["activation_probability"] == pytest.approx(0.05)

    def test_reference_antagonist_presets_classified_as_antagonist(self):
        for name in ("Istradefylline", "PSB-603", "ZM241385"):
            subtype = "A2B" if name == "PSB-603" else "A2A"
            res = predict_functional_efficacy(PRESETS[name]["smiles"], subtype, 8.0, in_domain=True)
            assert res["efficacy_class"] == "Antagonist", (
                f"Reference antagonist {name} classified as {res['efficacy_class']!r} "
                f"({res['mode_of_action']})"
            )

    def test_inactive_and_out_of_domain(self):
        res_low = predict_functional_efficacy(PRESETS["CGS-21680"]["smiles"], "A2A", 4.5, in_domain=True)
        assert res_low["efficacy_class"] == "Non-functional"
        assert res_low["activation_probability"] == 0.0

        res_ood = predict_functional_efficacy(PRESETS["CGS-21680"]["smiles"], "A2A", 8.5, in_domain=False)
        assert res_ood["efficacy_class"] == "Non-functional"
        assert res_ood["activation_probability"] == 0.0


class TestInteractionEngine:
    def test_nucleoside_agonist_pocket_motifs(self):
        res = analyze_pocket_interactions(PRESETS["CGS-21680"]["smiles"], "A2A")
        assert res["asn_anchor_hbond"] is True
        assert res["asn_residue"] == "Asn253 (6.55)"
        assert res["trp_toggle_switch"] is True
        assert res["trp_residue"] == "Trp246 (6.48)"
        assert res["phe_pi_stacking"] is True
        assert res["stereocenter_count"] >= 3
        assert "Natural D-enantiomer" in res["chiral_alert"]

    def test_simple_non_ligand_and_invalid_smiles(self):
        ethanol = analyze_pocket_interactions("CCO", "A1")
        assert ethanol["asn_anchor_hbond"] is False
        assert ethanol["trp_toggle_switch"] is False
        assert ethanol["phe_pi_stacking"] is False
        assert ethanol["stereocenter_count"] == 0
        assert "Achiral" in ethanol["chiral_alert"]

        invalid = analyze_pocket_interactions("INVALID", "A1")
        assert invalid["chiral_alert"] == "Invalid structure"


class TestSafetyEngine:
    def test_a1_and_a3_risk_thresholds(self):
        high = evaluate_safety_profile({"A1": 7.5, "A2A": 6.0, "A2B": 5.0, "A3": 7.5}, "c1ccccc1", in_domain=True)
        assert high["a1_bradycardia_risk"].startswith("HIGH")
        assert high["a3_mast_cell_risk"].startswith("HIGH")
        assert high["overall_safety_index"].startswith("UNFAVORABLE")

        mod = evaluate_safety_profile({"A1": 6.0, "A2A": 7.0, "A2B": 5.0, "A3": 7.49}, "c1ccccc1", in_domain=True)
        assert mod["a1_bradycardia_risk"].startswith("MODERATE")
        assert mod["a3_mast_cell_risk"].startswith("MODERATE")
        assert mod["overall_safety_index"].startswith("FAVORABLE")

        low = evaluate_safety_profile({"A1": 5.99, "A2A": 8.0, "A2B": 5.0, "A3": 5.5}, "c1ccccc1", in_domain=True)
        assert low["a1_bradycardia_risk"].startswith("LOW")
        assert low["a3_mast_cell_risk"].startswith("LOW")
        assert low["pde_cross_reactivity"].startswith("LOW")

    def test_xanthine_and_purine_pde_risk(self):
        theo = evaluate_safety_profile({"A1": 5.0, "A3": 5.0}, "Cn1c(=O)c2[nH]cnc2n(C)c1=O", in_domain=True)
        assert theo["pde_cross_reactivity"].startswith("HIGH")
        assert theo["overall_safety_index"].startswith("MONITORED")

        adenine = evaluate_safety_profile({"A1": 5.0, "A3": 5.0}, "Nc1ncnc2[nH]cnc12", in_domain=True)
        assert adenine["pde_cross_reactivity"].startswith("MODERATE"), (
            f"Purine adenine returned {adenine['pde_cross_reactivity']!r} instead of MODERATE PDE risk"
        )

    def test_xanthine_presets_flag_pde_cross_reactivity(self):
        for name in ("Istradefylline", "PSB-603"):
            res = evaluate_safety_profile({"A1": 5.0, "A3": 5.0}, PRESETS[name]["smiles"], in_domain=True)
            assert res["pde_cross_reactivity"].startswith("HIGH"), (
                f"Xanthine preset {name} returned {res['pde_cross_reactivity']!r} instead of HIGH PDE risk"
            )

    def test_out_of_domain_returns_extrapolation_notice(self):
        ood = evaluate_safety_profile({"A1": 8.5, "A3": 8.5}, "c1ccccc1", in_domain=False)
        assert "Out of Domain" in ood["a1_bradycardia_risk"]
        assert "N/A" in ood["overall_safety_index"]


class TestAdmetEngine:
    @pytest.mark.parametrize(
        "tpsa,expected",
        [
            (20.0, 0.0),
            (40.0, 1.0),
            (60.0, 1.0),
            (90.0, 1.0),
            (105.0, 0.5),
            (120.0, 0.0),
            (30.0, 0.5),
            (150.0, 0.0),
        ],
    )
    def test_mpo_tpsa_wager2010_plateau(self, tpsa, expected):
        assert _mpo_tpsa_wager2010(tpsa) == pytest.approx(expected)

    def test_evaluate_cns_admet_valid_and_invalid(self):
        diazepam = evaluate_cns_admet("CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21")
        assert 4.0 <= diazepam["cns_mpo_score"] <= 6.0
        assert diazepam["log_bb"] > -0.3
        assert "CNS-Penetrant" in diazepam["cns_class"]
        assert "High CNS Permeability" in diazepam["bbb_status"]

        caffeine = evaluate_cns_admet("Cn1cnc2c1c(=O)n(C)c(=O)n2C")
        assert caffeine["cns_mpo_score"] == pytest.approx(6.0)
        assert caffeine["log_bb"] < -0.3
        assert "Moderate CNS Distribution" in caffeine["cns_class"]
        assert set(caffeine["mpo_components"].keys()) == {
            "CLogP Score",
            "CLogD Score",
            "MW Score",
            "TPSA Score",
            "HBD Score",
            "pKa Score",
        }

        psb = evaluate_cns_admet(PRESETS["PSB-603"]["smiles"])
        assert 0.0 <= psb["cns_mpo_score"] <= 6.0
        assert "Peripherally-Restricted" in psb["cns_class"]

        invalid = evaluate_cns_admet("INVALID")
        assert invalid["cns_mpo_score"] == 0.0
        assert invalid["cns_class"] == "Invalid Structure"


class TestAdaptiveConformal:
    def test_pathological_negative_inputs_clamped_safely(self):
        res = calibrate_adaptive_interval(
            pred_val=-0.5,
            base_lower=-1.0,
            base_upper=0.2,
            tanimoto_max=0.5,
            in_domain=True,
        )
        assert 3.0 <= res["lower"] <= res["upper"] <= 11.0
        assert res["width"] >= 0.0
        assert res["confidence"] == pytest.approx(0.90)

    def test_inverted_bounds_and_regimes(self):
        inv = calibrate_adaptive_interval(7.5, base_lower=8.4, base_upper=6.6, tanimoto_max=0.80, in_domain=True)
        assert inv["lower"] <= 7.5 <= inv["upper"]
        assert inv["width"] == pytest.approx(1.8)
        assert "High-Density Core" in inv["regime"]

        std = calibrate_adaptive_interval(7.0, 6.2, 7.8, tanimoto_max=0.55, in_domain=True)
        assert "Standard Scaffold Space" in std["regime"]

        sparse = calibrate_adaptive_interval(7.0, 6.0, 8.0, tanimoto_max=0.30, in_domain=True)
        assert "Sparse Space" in sparse["regime"]

        ood = calibrate_adaptive_interval(7.0, 5.5, 8.5, tanimoto_max=0.15, in_domain=False)
        assert "Out of Applicability Domain" in ood["regime"]


class TestKiUtilsAndApplicabilityDomain:
    def test_pchembl_to_ki_nm_and_formatting(self):
        assert pchembl_to_ki_nm(9.0) == pytest.approx(1.0)
        assert pchembl_to_ki_nm(6.0) == pytest.approx(1000.0)
        assert pchembl_to_ki_nm(0.0) == pytest.approx(100000.0)

        assert format_ki_display(0.5) == "500.0 pM"
        assert format_ki_display(12.34) == "12.34 nM"
        assert format_ki_display(2500.0) == "2.50 μM"
        assert format_ki_display(100000.0) == "> 100 μM (Inactive)"

    def test_applicability_domain_uses_train_fps_when_present(self):
        train_fps_path = PROCESSED_DATA_DIR / "train_fps.pkl"
        assert train_fps_path.is_file(), f"Missing {train_fps_path}"

        ad_mod._load_cached_train_fps.cache_clear()
        ad = check_applicability_domain(PRESETS["ZM241385"]["smiles"])
        assert ad["in_domain"] is True
        assert ad["tanimoto_max"] >= 0.35
        assert ad["reference_source"].startswith("train_fps.pkl")
        assert "training compounds" in ad["reference_source"]

    def test_applicability_domain_fallback_and_edge_cases(self, tmp_path, monkeypatch):
        ad_mod._load_cached_train_fps.cache_clear()
        monkeypatch.setattr(ad_mod, "PROCESSED_DATA_DIR", tmp_path)

        ad_fb = check_applicability_domain(PRESETS["ZM241385"]["smiles"])
        assert ad_fb["in_domain"] is True
        assert ad_fb["reference_source"].startswith("fallback_reference_probes")

        ad_inv = check_applicability_domain("INVALID")
        assert ad_inv["in_domain"] is False
        assert ad_inv["reference_source"] == "none"

        ad_huge = check_applicability_domain("C" * 70)
        assert ad_huge["in_domain"] is False
        assert len(ad_huge["property_violations"]) > 0
        ad_mod._load_cached_train_fps.cache_clear()
