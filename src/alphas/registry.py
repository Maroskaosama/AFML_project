"""
Alpha registry: sector map and per-alpha metadata.
"""

import json as _json, os as _os

def _load_sector_map():
    cfg_path = _os.path.join(_os.path.dirname(__file__), '..', '..', 'configs', 'universe.json')
    try:
        with open(cfg_path) as _f:
            cfg = _json.load(_f)
        return cfg['sector_codes']
    except Exception:
        # Fallback: new universe hardcoded
        return {
            'AAPL': 'IT', 'AMZN': 'CD', 'NVDA': 'IT', 'GOOGL': 'CS',
            'JNJ': 'HC', 'JPM': 'FN', 'MSFT': 'IT', 'XOM': 'EN',
            'BAC': 'FN', 'UNH': 'HC',
        }

SECTOR_MAP = _load_sector_map()

# Tier 1: Pure time-series (no rank, no indneutralize, no cap)
# Tier 2: Cross-sectional rank()
# Tier 3: indneutralize (sector-based)
# Tier 4: Excluded (requires market cap)

ALPHA_REGISTRY = {
    'alpha001': {'tier': 2, 'max_lookback': 20,  'uses_indneutralize': False, 'uses_adv': False},
    'alpha002': {'tier': 2, 'max_lookback': 6,   'uses_indneutralize': False, 'uses_adv': False},
    'alpha003': {'tier': 2, 'max_lookback': 10,  'uses_indneutralize': False, 'uses_adv': False},
    'alpha004': {'tier': 2, 'max_lookback': 9,   'uses_indneutralize': False, 'uses_adv': False},
    'alpha005': {'tier': 2, 'max_lookback': 10,  'uses_indneutralize': False, 'uses_adv': False},
    'alpha006': {'tier': 1, 'max_lookback': 10,  'uses_indneutralize': False, 'uses_adv': False},
    'alpha007': {'tier': 2, 'max_lookback': 60,  'uses_indneutralize': False, 'uses_adv': True},
    'alpha008': {'tier': 2, 'max_lookback': 15,  'uses_indneutralize': False, 'uses_adv': False},
    'alpha009': {'tier': 1, 'max_lookback': 5,   'uses_indneutralize': False, 'uses_adv': False},
    'alpha010': {'tier': 2, 'max_lookback': 4,   'uses_indneutralize': False, 'uses_adv': False},
    'alpha011': {'tier': 2, 'max_lookback': 3,   'uses_indneutralize': False, 'uses_adv': False},
    'alpha012': {'tier': 1, 'max_lookback': 1,   'uses_indneutralize': False, 'uses_adv': False},
    'alpha013': {'tier': 2, 'max_lookback': 5,   'uses_indneutralize': False, 'uses_adv': False},
    'alpha014': {'tier': 2, 'max_lookback': 10,  'uses_indneutralize': False, 'uses_adv': False},
    'alpha015': {'tier': 2, 'max_lookback': 3,   'uses_indneutralize': False, 'uses_adv': False},
    'alpha016': {'tier': 2, 'max_lookback': 5,   'uses_indneutralize': False, 'uses_adv': False},
    'alpha017': {'tier': 2, 'max_lookback': 10,  'uses_indneutralize': False, 'uses_adv': True},
    'alpha018': {'tier': 2, 'max_lookback': 10,  'uses_indneutralize': False, 'uses_adv': False},
    'alpha019': {'tier': 2, 'max_lookback': 250, 'uses_indneutralize': False, 'uses_adv': False},
    'alpha020': {'tier': 2, 'max_lookback': 1,   'uses_indneutralize': False, 'uses_adv': False},
    'alpha021': {'tier': 1, 'max_lookback': 8,   'uses_indneutralize': False, 'uses_adv': True},
    'alpha022': {'tier': 2, 'max_lookback': 20,  'uses_indneutralize': False, 'uses_adv': False},
    'alpha023': {'tier': 1, 'max_lookback': 20,  'uses_indneutralize': False, 'uses_adv': False},
    'alpha024': {'tier': 1, 'max_lookback': 100, 'uses_indneutralize': False, 'uses_adv': False},
    'alpha025': {'tier': 2, 'max_lookback': 1,   'uses_indneutralize': False, 'uses_adv': True},
    'alpha026': {'tier': 2, 'max_lookback': 5,   'uses_indneutralize': False, 'uses_adv': False},
    'alpha027': {'tier': 2, 'max_lookback': 6,   'uses_indneutralize': False, 'uses_adv': False},
    'alpha028': {'tier': 2, 'max_lookback': 5,   'uses_indneutralize': False, 'uses_adv': True},
    'alpha029': {'tier': 2, 'max_lookback': 6,   'uses_indneutralize': False, 'uses_adv': False},
    'alpha030': {'tier': 2, 'max_lookback': 20,  'uses_indneutralize': False, 'uses_adv': False},
    'alpha031': {'tier': 2, 'max_lookback': 12,  'uses_indneutralize': False, 'uses_adv': True},
    'alpha032': {'tier': 2, 'max_lookback': 230, 'uses_indneutralize': False, 'uses_adv': False},
    'alpha033': {'tier': 2, 'max_lookback': 1,   'uses_indneutralize': False, 'uses_adv': False},
    'alpha034': {'tier': 2, 'max_lookback': 5,   'uses_indneutralize': False, 'uses_adv': False},
    'alpha035': {'tier': 1, 'max_lookback': 32,  'uses_indneutralize': False, 'uses_adv': False},
    'alpha036': {'tier': 2, 'max_lookback': 200, 'uses_indneutralize': False, 'uses_adv': True},
    'alpha037': {'tier': 2, 'max_lookback': 200, 'uses_indneutralize': False, 'uses_adv': False},
    'alpha038': {'tier': 2, 'max_lookback': 10,  'uses_indneutralize': False, 'uses_adv': False},
    'alpha039': {'tier': 2, 'max_lookback': 250, 'uses_indneutralize': False, 'uses_adv': True},
    'alpha040': {'tier': 2, 'max_lookback': 10,  'uses_indneutralize': False, 'uses_adv': False},
    'alpha041': {'tier': 1, 'max_lookback': 1,   'uses_indneutralize': False, 'uses_adv': False},
    'alpha042': {'tier': 2, 'max_lookback': 1,   'uses_indneutralize': False, 'uses_adv': False},
    'alpha043': {'tier': 1, 'max_lookback': 20,  'uses_indneutralize': False, 'uses_adv': True},
    'alpha044': {'tier': 2, 'max_lookback': 5,   'uses_indneutralize': False, 'uses_adv': False},
    'alpha045': {'tier': 2, 'max_lookback': 20,  'uses_indneutralize': False, 'uses_adv': False},
    'alpha046': {'tier': 1, 'max_lookback': 20,  'uses_indneutralize': False, 'uses_adv': False},
    'alpha047': {'tier': 2, 'max_lookback': 5,   'uses_indneutralize': False, 'uses_adv': True},
    'alpha048': {'tier': 3, 'max_lookback': 250, 'uses_indneutralize': True,  'uses_adv': False},
    'alpha049': {'tier': 1, 'max_lookback': 20,  'uses_indneutralize': False, 'uses_adv': False},
    'alpha050': {'tier': 2, 'max_lookback': 5,   'uses_indneutralize': False, 'uses_adv': False},
    'alpha051': {'tier': 1, 'max_lookback': 20,  'uses_indneutralize': False, 'uses_adv': False},
    'alpha052': {'tier': 2, 'max_lookback': 240, 'uses_indneutralize': False, 'uses_adv': False},
    'alpha053': {'tier': 1, 'max_lookback': 9,   'uses_indneutralize': False, 'uses_adv': False},
    'alpha054': {'tier': 1, 'max_lookback': 1,   'uses_indneutralize': False, 'uses_adv': False},
    'alpha055': {'tier': 2, 'max_lookback': 12,  'uses_indneutralize': False, 'uses_adv': False},
    'alpha056': {'tier': 4, 'max_lookback': 0,   'uses_indneutralize': False, 'uses_adv': False},
    'alpha057': {'tier': 2, 'max_lookback': 30,  'uses_indneutralize': False, 'uses_adv': False},
    'alpha058': {'tier': 3, 'max_lookback': 8,   'uses_indneutralize': True,  'uses_adv': False},
    'alpha059': {'tier': 3, 'max_lookback': 16,  'uses_indneutralize': True,  'uses_adv': False},
    'alpha060': {'tier': 2, 'max_lookback': 10,  'uses_indneutralize': False, 'uses_adv': False},
    'alpha061': {'tier': 2, 'max_lookback': 18,  'uses_indneutralize': False, 'uses_adv': True},
    'alpha062': {'tier': 2, 'max_lookback': 22,  'uses_indneutralize': False, 'uses_adv': True},
    'alpha063': {'tier': 3, 'max_lookback': 37,  'uses_indneutralize': True,  'uses_adv': True},
    'alpha064': {'tier': 2, 'max_lookback': 17,  'uses_indneutralize': False, 'uses_adv': True},
    'alpha065': {'tier': 2, 'max_lookback': 14,  'uses_indneutralize': False, 'uses_adv': True},
    'alpha066': {'tier': 2, 'max_lookback': 11,  'uses_indneutralize': False, 'uses_adv': False},
    'alpha067': {'tier': 3, 'max_lookback': 6,   'uses_indneutralize': True,  'uses_adv': True},
    'alpha068': {'tier': 2, 'max_lookback': 14,  'uses_indneutralize': False, 'uses_adv': True},
    'alpha069': {'tier': 3, 'max_lookback': 9,   'uses_indneutralize': True,  'uses_adv': True},
    'alpha070': {'tier': 3, 'max_lookback': 18,  'uses_indneutralize': True,  'uses_adv': True},
    'alpha071': {'tier': 2, 'max_lookback': 18,  'uses_indneutralize': False, 'uses_adv': True},
    'alpha072': {'tier': 2, 'max_lookback': 19,  'uses_indneutralize': False, 'uses_adv': True},
    'alpha073': {'tier': 2, 'max_lookback': 17,  'uses_indneutralize': False, 'uses_adv': False},
    'alpha074': {'tier': 2, 'max_lookback': 37,  'uses_indneutralize': False, 'uses_adv': True},
    'alpha075': {'tier': 2, 'max_lookback': 12,  'uses_indneutralize': False, 'uses_adv': True},
    'alpha076': {'tier': 3, 'max_lookback': 20,  'uses_indneutralize': True,  'uses_adv': True},
    'alpha077': {'tier': 2, 'max_lookback': 20,  'uses_indneutralize': False, 'uses_adv': True},
    'alpha078': {'tier': 2, 'max_lookback': 20,  'uses_indneutralize': False, 'uses_adv': True},
    'alpha079': {'tier': 3, 'max_lookback': 15,  'uses_indneutralize': True,  'uses_adv': True},
    'alpha080': {'tier': 3, 'max_lookback': 6,   'uses_indneutralize': True,  'uses_adv': True},
    'alpha081': {'tier': 2, 'max_lookback': 50,  'uses_indneutralize': False, 'uses_adv': True},
    'alpha082': {'tier': 3, 'max_lookback': 17,  'uses_indneutralize': True,  'uses_adv': False},
    'alpha083': {'tier': 2, 'max_lookback': 5,   'uses_indneutralize': False, 'uses_adv': False},
    'alpha084': {'tier': 2, 'max_lookback': 21,  'uses_indneutralize': False, 'uses_adv': False},
    'alpha085': {'tier': 2, 'max_lookback': 10,  'uses_indneutralize': False, 'uses_adv': True},
    'alpha086': {'tier': 2, 'max_lookback': 20,  'uses_indneutralize': False, 'uses_adv': True},
    'alpha087': {'tier': 3, 'max_lookback': 14,  'uses_indneutralize': True,  'uses_adv': True},
    'alpha088': {'tier': 2, 'max_lookback': 21,  'uses_indneutralize': False, 'uses_adv': True},
    'alpha089': {'tier': 3, 'max_lookback': 15,  'uses_indneutralize': True,  'uses_adv': True},
    'alpha090': {'tier': 3, 'max_lookback': 6,   'uses_indneutralize': True,  'uses_adv': True},
    'alpha091': {'tier': 3, 'max_lookback': 16,  'uses_indneutralize': True,  'uses_adv': True},
    'alpha092': {'tier': 2, 'max_lookback': 19,  'uses_indneutralize': False, 'uses_adv': True},
    'alpha093': {'tier': 3, 'max_lookback': 20,  'uses_indneutralize': True,  'uses_adv': True},
    'alpha094': {'tier': 2, 'max_lookback': 20,  'uses_indneutralize': False, 'uses_adv': True},
    'alpha095': {'tier': 2, 'max_lookback': 19,  'uses_indneutralize': False, 'uses_adv': True},
    'alpha096': {'tier': 2, 'max_lookback': 8,   'uses_indneutralize': False, 'uses_adv': True},
    'alpha097': {'tier': 3, 'max_lookback': 20,  'uses_indneutralize': True,  'uses_adv': True},
    'alpha098': {'tier': 2, 'max_lookback': 26,  'uses_indneutralize': False, 'uses_adv': True},
    'alpha099': {'tier': 2, 'max_lookback': 20,  'uses_indneutralize': False, 'uses_adv': True},
    'alpha100': {'tier': 3, 'max_lookback': 5,   'uses_indneutralize': True,  'uses_adv': True},
    'alpha101': {'tier': 1, 'max_lookback': 1,   'uses_indneutralize': False, 'uses_adv': False},
}
