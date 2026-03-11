# Databricks notebook source
# MAGIC %md
# MAGIC # Run insurance-reserving-neural test suite

# COMMAND ----------

# MAGIC %pip install polars numpy scipy torch pytest pytest-cov --quiet

# COMMAND ----------

import subprocess, sys, shutil, os, tempfile

# Install the library from the workspace-uploaded source
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", "/Workspace/insurance-reserving-neural", "--quiet"],
    capture_output=True, text=True
)
if result.returncode != 0:
    raise RuntimeError("pip install failed:\n" + result.stderr[-2000:])
print("Library installed OK")

# COMMAND ----------

local_test_dir = tempfile.mkdtemp(prefix="irn_tests_")
shutil.copytree("/Workspace/insurance-reserving-neural/tests", local_test_dir, dirs_exist_ok=True)
print(f"Tests: {sorted(os.listdir(local_test_dir))}")

result = subprocess.run(
    [sys.executable, "-m", "pytest", local_test_dir,
     "-v", "--tb=short", "--no-header",
     "-p", "no:cacheprovider",
     "--import-mode=importlib"],
    capture_output=True, text=True,
    cwd=local_test_dir,
    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
)

# Put full output in the raised exception so it appears in the run error
output = result.stdout
if result.stderr:
    output += "\n--- STDERR ---\n" + result.stderr

if result.returncode not in (0, 5):
    raise RuntimeError(f"pytest FAILED (rc={result.returncode}):\n{output[-12000:]}")
print(output[:20000])
print(f"\nReturn code: {result.returncode}")
print("All tests PASSED" if result.returncode == 0 else "No tests collected")
