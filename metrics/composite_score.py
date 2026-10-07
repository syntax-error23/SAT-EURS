"""
Composite Score Calculator for EURS Track 2
Formula:
    S_composite = 0.60 * Q_seg + 0.30 * Q_rest + 0.10 * Q_NR
Status:
    Segmentation & Restoration: VERIFIED (Local standard)
    Q_NR: PROVISIONAL STUB
    Official Competition Parity: NOT CLAIMED
"""

from typing import Dict, Any


def compute_composite_score(
    q_seg: float,
    q_rest: float,
    q_nr: float = 0.50
) -> Dict[str, Any]:
    """
    Computes overall competition score with 60% seg / 30% rest / 10% NR weighting.
    """
    composite = 0.60 * q_seg + 0.30 * q_rest + 0.10 * q_nr

    return {
        'Composite_Score': round(composite, 4),
        'Q_seg_weighted': round(0.60 * q_seg, 4),
        'Q_rest_weighted': round(0.30 * q_rest, 4),
        'Q_NR_weighted': round(0.10 * q_nr, 4),
        'status': 'PROVISIONAL_PARTIAL',
        'official_parity': False
    }
