#!/usr/bin/env python3
"""Verify the released record of predictions.

The paper cites SHA-256 7655830a... for prereg_r3.json. That digest belongs to
the file released with the paper. Note that r3_all_in_one.py part0 REGENERATES
this file with a fresh UTC timestamp whenever it is absent, so a from-scratch
re-run produces a different digest with identical predicted values. This script
reports both: the file digest, and whether the predicted values match the
released ones.

Usage:  python3 verify_prereg.py [path/to/prereg_r3.json]
"""
import hashlib, json, sys

RELEASED_FILE_SHA256 = ("7655830a90e5bd6c605549f48928ac4881414c7bba3492b597"
                        "ec25686d8df83a")

# The 13 evaluated quantities, exactly as released.
RELEASED_PREDICTIONS = {
    "A_S_L1_mean_across_seeds": 0.80,
    "A_S_L1_min_across_seeds":  0.70,
    "A_S_L0_max_across_seeds":  0.05,
    "A_S_L0tight_mean":         0.15,
    "B_max_S_over_sweep":       0.25,
    "B_spearman_S_vs_Htot":    -0.30,
    "C_S_precondition_only":    0.12,
    "C_S_stopping_only":        0.05,
    "C_S_ordering_only":        0.05,
    "C_S_all_three":            0.15,
    "D_S_Lrand_matched":        0.12,
    "E_S_L1_1p7B":              0.90,
    "E_S_L0_1p7B":              0.10,
}
# part0 renamed one key in a later revision; treat these as the same quantity.
ALIASES = {"B_corr_S_vs_Htot": "B_spearman_S_vs_Htot"}


def main(path="prereg_r3.json"):
    raw = open(path, "rb").read()
    digest = hashlib.sha256(raw).hexdigest()
    rec = json.loads(raw)

    print(f"file            : {path}")
    print(f"file SHA-256    : {digest}")
    print(f"matches released: {digest == RELEASED_FILE_SHA256}")
    print(f"self-recorded utc: {rec.get('utc')}")

    got = {ALIASES.get(k, k): v for k, v in rec["predictions"].items()}
    missing = sorted(set(RELEASED_PREDICTIONS) - set(got))
    extra   = sorted(set(got) - set(RELEASED_PREDICTIONS))
    diffs   = {k: (RELEASED_PREDICTIONS[k], got[k])
               for k in RELEASED_PREDICTIONS if k in got
               and abs(got[k] - RELEASED_PREDICTIONS[k]) > 1e-9}

    print(f"\npredicted values: {len(got)} keys")
    print(f"  missing        : {missing or 'none'}")
    print(f"  unexpected     : {extra or 'none'}")
    print(f"  differing      : {diffs or 'none'}")
    ok = not (missing or extra or diffs)
    print(f"\nVALUES MATCH RELEASED RECORD: {ok}")
    if ok and digest != RELEASED_FILE_SHA256:
        print("(Digest differs only because part0 rewrote the timestamp; the\n"
              " thirteen predicted values are unchanged.)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:2]))
