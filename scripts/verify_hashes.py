#!/usr/bin/env python3
"""Verify integrity of key artifacts via SHA256."""
import hashlib, os

ARTIFACTS = [
    'results/analysis_bundle.csv',
    'results/condition_summary.csv',
    'results/kappa_metrics.json',
]

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

if __name__ == '__main__':
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("Artifact Integrity Check")
    print("=" * 60)
    for rel in ARTIFACTS:
        path = os.path.join(base, rel)
        if os.path.exists(path):
            h = sha256(path)
            print(f"✓ {rel}")
            print(f"  SHA256: {h}")
        else:
            print(f"✗ {rel} — NOT FOUND")
    print()
    print("Compare hashes against REPRODUCIBILITY.md to verify integrity.")
