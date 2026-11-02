import subprocess
import sys

scripts = [
    "verify_structure.py",
    "inspect_53_features.py",
    "audit_project.py",
    "verify_live_feature_generator.py",
    "verify_live_api_integration.py",
    "verify_step7_ui.py",
    "test_comprehensive_suite.py",
    "test_upgraded_platform.py",
]

print("=" * 70)
print("RUNNING COMPLETE REPOSITORY VERIFICATION SUITE")
print("=" * 70)

all_passed = True
for s in scripts:
    proc = subprocess.run([sys.executable, s], capture_output=True, text=True)
    status = "PASS" if proc.returncode == 0 else "FAIL"
    print(f"[{status:<4}] {s:<35} (Exit code: {proc.returncode})")
    if proc.returncode != 0:
        all_passed = False
        print("  Error output:")
        print(proc.stderr[:400] or proc.stdout[:400])

print("=" * 70)
if all_passed:
    print(f"ALL VERIFICATION SUITES PASSED SUCCESSFULLY ({len(scripts)}/{len(scripts)})")
    sys.exit(0)
else:
    print("SOME SUITES FAILED")
    sys.exit(1)
