"""
Automated validation and submission packaging script.
"""
import os
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
CODE_DIR = ROOT / "code" / "business_entity_resolution"
DOC_FILE = ROOT / "Documentation_template.md"
VALIDATOR = ROOT / "student_resource" / "utils" / "validate_submission.py"
TEST_DIR = ROOT / "student_resource" / "dataset" / "test"

ZIP_NAME = "ApexResolvers_submission.zip"
ZIP_PATH = ROOT / ZIP_NAME


def run_validation():
    print("=" * 60)
    print("Running Official Challenge Validator...")
    print("=" * 60)
    cmd = [
        "python3", str(VALIDATOR),
        "--matching", str(OUTPUT_DIR / "matching_results.tsv"),
        "--candidate", str(OUTPUT_DIR / "candidate_pairs.tsv"),
        "--test-dir", str(TEST_DIR)
    ]
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print("\n[ERROR] Validation failed. Fix errors before submitting.")
        return False
    print("\n[SUCCESS] Validation PASSED (exit code 0).")
    return True


def create_submission_zip():
    print("\n" + "=" * 60)
    print(f"Packaging {ZIP_NAME}...")
    print("=" * 60)
    
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        # Add output files
        z.write(OUTPUT_DIR / "matching_results.tsv", "output/matching_results.tsv")
        z.write(OUTPUT_DIR / "candidate_pairs.tsv", "output/candidate_pairs.tsv")
        print("  Added: output/matching_results.tsv")
        print("  Added: output/candidate_pairs.tsv")
        
        # Add code files
        for root, _, files in os.walk(CODE_DIR):
            for file in files:
                if file.endswith((".py", ".txt", ".md")) and "__pycache__" not in root:
                    full_p = Path(root) / file
                    rel_p = full_p.relative_to(ROOT)
                    z.write(full_p, str(rel_p))
                    print(f"  Added: {rel_p}")
                    
        # Add documentation template
        z.write(DOC_FILE, "Documentation_template.md")
        print("  Added: Documentation_template.md")
        
    print(f"\n[DONE] Package created at: {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    if run_validation():
        create_submission_zip()
