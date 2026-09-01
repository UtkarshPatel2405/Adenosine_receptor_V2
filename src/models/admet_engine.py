"""Pillar 4: Pfizer CNS-MPO & Blood-Brain Barrier (BBB) Permeability Engine."""
from typing import Dict, Any
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski


def _mpo_monotonic(val: float, low: float, high: float, increasing: bool = False) -> float:
    """Standard linear desirability function between bounds."""
    if increasing:
        if val <= low: return 0.0
        if val >= high: return 1.0
        return (val - low) / (high - low)
    else:
        if val <= low: return 1.0
        if val >= high: return 0.0
        return (high - val) / (high - low)


def evaluate_cns_admet(smiles: str) -> Dict[str, Any]:
    """Compute Pfizer 6-parameter CNS-MPO score (0-6) and predicted BBB LogBB."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"cns_mpo_score": 0.0, "cns_class": "Invalid Structure", "log_bb": 0.0, "bbb_status": "Unknown", "mpo_components": {}}

    c_logp = float(Descriptors.MolLogP(mol))
    mw = float(Descriptors.MolWt(mol))
    tpsa = float(Descriptors.TPSA(mol))
    hbd = int(Lipinski.NumHDonors(mol))
    c_logd = c_logp - 0.5  # Neutral-dominant estimation for standard nucleoside/xanthine bases
    pka = 8.0 if mol.HasSubstructMatch(Chem.MolFromSmarts("c[NH2]")) else 6.5

    # Pfizer CNS-MPO desirability components (each 0.0 to 1.0)
    # 1. CLogP (desirable <= 3.0, unacceptable >= 5.0)
    score_logp = _mpo_monotonic(c_logp, 3.0, 5.0)
    # 2. CLogD (desirable <= 2.0, unacceptable >= 4.0)
    score_logd = _mpo_monotonic(c_logd, 2.0, 4.0)
    # 3. MW (desirable <= 360, unacceptable >= 500)
    score_mw = _mpo_monotonic(mw, 360.0, 500.0)
    # 4. TPSA (desirable 40-90, unacceptable > 120 or < 20)
    score_tpsa = _mpo_monotonic(tpsa, 40.0, 90.0) if tpsa >= 40 else _mpo_monotonic(tpsa, 20.0, 40.0, increasing=True)
    # 5. HBD (desirable <= 0.5, unacceptable >= 3.5)
    score_hbd = _mpo_monotonic(float(hbd), 0.5, 3.5)
    # 6. pKa (desirable <= 8.0, unacceptable >= 10.0)
    score_pka = _mpo_monotonic(pka, 8.0, 10.0)

    cns_mpo = score_logp + score_logd + score_mw + score_tpsa + score_hbd + score_pka

    # Clark's empirical Blood-Brain Barrier LogBB equation: LogBB = 0.152*CLogP - 0.0148*TPSA + 0.139
    log_bb = (0.152 * c_logp) - (0.0148 * tpsa) + 0.139
    bbb_permeable = log_bb > -0.3 and tpsa < 90.0

    if cns_mpo >= 4.0 and bbb_permeable:
        cns_class = "CNS-Penetrant (High BBB Permeability - Ideal for Parkinson's / Neuroprotection)"
        bbb_status = "High CNS Permeability (LogBB > -0.3)"
    elif cns_mpo >= 3.0:
        cns_class = "Moderate CNS Distribution (Balanced Central/Peripheral Exposure)"
        bbb_status = "Moderate CNS Permeability"
    else:
        cns_class = "Peripherally-Restricted (Low CNS Penetration - Ideal for Vasodilation & Oncology)"
        bbb_status = "Low CNS Permeability (LogBB < -0.3)"

    return {
        "cns_mpo_score": round(cns_mpo, 2),
        "cns_class": cns_class,
        "log_bb": round(log_bb, 2),
        "bbb_status": bbb_status,
        "mpo_components": {
            "CLogP Score": round(score_logp, 2),
            "CLogD Score": round(score_logd, 2),
            "MW Score": round(score_mw, 2),
            "TPSA Score": round(score_tpsa, 2),
            "HBD Score": round(score_hbd, 2),
            "pKa Score": round(score_pka, 2),
        },
    }
