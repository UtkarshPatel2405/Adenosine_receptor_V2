"""Thermodynamic Ki equilibrium conversions and display formatting for human adenosine receptors."""
from typing import Dict, Any


def pchembl_to_ki_nm(pchembl: float) -> float:
    """Convert logarithmic pChEMBL (-log10 M) to nanomolar equilibrium constant Ki (nM)."""
    if pchembl <= 0:
        return 100000.0  # > 100 uM inactive
    molar = 10 ** (-pchembl)
    return molar * 1e9


def format_ki_display(ki_nm: float) -> str:
    """Format Ki cleanly in pM, nM, or μM."""
    if ki_nm < 1.0:
        return f"{ki_nm * 1000:.1f} pM"
    elif ki_nm < 1000.0:
        return f"{ki_nm:.2f} nM"
    elif ki_nm < 100000.0:
        return f"{ki_nm / 1000.0:.2f} μM"
    else:
        return "> 100 μM (Inactive)"

