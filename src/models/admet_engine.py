"""Pfizer CNS-MPO (Wager et al., 2010) and Blood-Brain Barrier Permeability Engine."""
from typing import Any, Dict

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

_ALIPHATIC_BASIC_AMINE = Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(NC=O);!$(NS=O);!$(n);!$(N=C);!$(NC#N);!$(Nc)]")
_RING_AMINE = Chem.MolFromSmarts("[NX3;R;!$(NC=O);!$(NS=O);!$(n)]")
_PYRIDINE_LIKE = Chem.MolFromSmarts("n1ccccc1")
_ANILINE_LIKE = Chem.MolFromSmarts("c[NH2,NHR]")
_CARBOXYL = Chem.MolFromSmarts("C(=O)[OH]")


def _mpo_monotonic(val: float, low: float, high: float, increasing: bool = False) -> float:
    """Linear desirability ramp between bounds."""
    if increasing:
        if val <= low:
            return 0.0
        if val >= high:
            return 1.0
        return (val - low) / (high - low)
    if val <= low:
        return 1.0
    if val >= high:
        return 0.0
    return (high - val) / (high - low)


def _mpo_tpsa_wager2010(tpsa: float) -> float:
    """Wager et al. (2010) TPSA hump: 1.0 on [40, 90], linear taper to 0.0 at 20 and 120."""
    if tpsa <= 20.0 or tpsa >= 120.0:
        return 0.0
    if tpsa < 40.0:
        return (tpsa - 20.0) / 20.0
    if tpsa <= 90.0:
        return 1.0
    return (120.0 - tpsa) / 30.0


def evaluate_cns_admet(smiles: str) -> Dict[str, Any]:
    """Compute Pfizer 6-parameter CNS-MPO score (0-6) and Clark LogBB estimate."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "cns_mpo_score": 0.0,
            "cns_class": "Invalid Structure",
            "log_bb": 0.0,
            "bbb_status": "Unknown",
            "mpo_components": {},
        }

    c_logp = float(Descriptors.MolLogP(mol))
    mw = float(Descriptors.MolWt(mol))
    tpsa = float(Descriptors.TPSA(mol))
    hbd = int(Lipinski.NumHDonors(mol))

    has_basic_amine = True
    if mol.HasSubstructMatch(_ALIPHATIC_BASIC_AMINE) or mol.HasSubstructMatch(_RING_AMINE):
        est_pka_base = 9.2
    elif mol.HasSubstructMatch(_PYRIDINE_LIKE):
        est_pka_base = 5.2
    elif mol.HasSubstructMatch(_ANILINE_LIKE):
        est_pka_base = 4.6
    else:
        est_pka_base = 2.0
        has_basic_amine = False

    if est_pka_base > 7.4:
        fn = 1.0 / (1.0 + 10.0 ** (est_pka_base - 7.4))
        c_logd = round(c_logp + (fn - 1.0) * 1.5, 2)
    elif mol.HasSubstructMatch(_CARBOXYL):
        c_logd = round(c_logp - 2.0, 2)
    else:
        c_logd = round(c_logp, 2)

    score_logp = _mpo_monotonic(c_logp, 3.0, 5.0)
    score_logd = _mpo_monotonic(c_logd, 2.0, 4.0)
    score_mw = _mpo_monotonic(mw, 360.0, 500.0)
    score_tpsa = _mpo_tpsa_wager2010(tpsa)
    score_hbd = _mpo_monotonic(float(hbd), 0.5, 3.5)
    score_pka = _mpo_monotonic(est_pka_base, 8.0, 10.0)

    cns_mpo = score_logp + score_logd + score_mw + score_tpsa + score_hbd + score_pka

    log_bb = (0.152 * c_logp) - (0.0148 * tpsa) + 0.139
    bbb_permeable = log_bb > -0.3 and tpsa < 90.0

    if cns_mpo >= 4.0 and bbb_permeable:
        cns_class = "CNS-Penetrant (High BBB Permeability - Favorable for Central Targets)"
        bbb_status = "High CNS Permeability (LogBB > -0.3)"
    elif cns_mpo >= 3.0:
        cns_class = "Moderate CNS Distribution (Balanced Central/Peripheral Exposure)"
        bbb_status = "Moderate CNS Permeability"
    else:
        cns_class = "Peripherally-Restricted (Low CNS Penetration - Favorable for Peripheral Targets)"
        bbb_status = "Low CNS Permeability (LogBB < -0.3)"

    return {
        "cns_mpo_score": round(cns_mpo, 2),
        "cns_class": cns_class,
        "log_bb": round(log_bb, 2),
        "bbb_status": bbb_status,
        "est_pka": round(est_pka_base, 1) if has_basic_amine else None,
        "est_pka_display": f"{est_pka_base:.1f}" if has_basic_amine else "None (Acidic/Neutral)",
        "c_logd74": round(c_logd, 2),
        "mpo_components": {
            "CLogP Score": round(score_logp, 2),
            "CLogD Score": round(score_logd, 2),
            "MW Score": round(score_mw, 2),
            "TPSA Score": round(score_tpsa, 2),
            "HBD Score": round(score_hbd, 2),
            "pKa Score": round(score_pka, 2),
        },
    }
