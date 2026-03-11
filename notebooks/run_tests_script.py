"""
Standalone script to run the test suite on Databricks.
Run as a spark_python_task (not a notebook).
"""
import subprocess
import sys
import os

# Install the library from workspace
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e",
     "/Workspace/insurance-reserving-neural", "--quiet"],
    capture_output=True, text=True
)
if result.returncode != 0:
    print("pip install stdout:", result.stdout)
    print("pip install stderr:", result.stderr)
    sys.exit(1)

print("Library installed successfully")

# Run pytest
result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "/Workspace/insurance-reserving-neural/tests/",
     "-v", "--tb=short", "--no-header", "-p", "no:cacheprovider",
     "--ignore=/Workspace/insurance-reserving-neural/tests/test_models.py",
     "--ignore=/Workspace/insurance-reserving-neural/tests/test_bootstrap.py"],
    capture_output=True, text=True
)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[-2000:])

if result.returncode != 0:
    print("\nNon-torch tests FAILED")
    # Now try torch tests separately
    result2 = subprocess.run(
        [sys.executable, "-m", "pytest",
         "/Workspace/insurance-reserving-neural/tests/test_models.py",
         "/Workspace/insurance-reserving-neural/tests/test_bootstrap.py",
         "-v", "--tb=short", "--no-header", "-p", "no:cacheprovider"],
        capture_output=True, text=True
    )
    print("\n--- torch tests ---")
    print(result2.stdout)
    if result2.stderr:
        print("STDERR:", result2.stderr[-2000:])
    sys.exit(result.returncode)
else:
    print("\nNon-torch tests PASSED. Running torch tests...")
    result2 = subprocess.run(
        [sys.executable, "-m", "pytest",
         "/Workspace/insurance-reserving-neural/tests/test_models.py",
         "/Workspace/insurance-reserving-neural/tests/test_bootstrap.py",
         "-v", "--tb=short", "--no-header", "-p", "no:cacheprovider"],
        capture_output=True, text=True
    )
    print("\n--- torch tests ---")
    print(result2.stdout)
    if result2.stderr:
        print("STDERR:", result2.stderr[-2000:])

    overall_rc = result.returncode or result2.returncode
    if overall_rc != 0:
        print("\nSome tests FAILED")
        sys.exit(overall_rc)
    else:
        print("\nAll tests PASSED")
