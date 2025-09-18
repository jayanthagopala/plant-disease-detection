#!/usr/bin/env python3
"""Test runner for the Smart Crop Advisory System."""

import sys
import subprocess
from pathlib import Path

def run_test_file(test_file: str) -> bool:
    """Run a single test file and return success status."""
    print(f"\n🧪 Running {test_file}...")
    print("=" * 50)
    
    try:
        result = subprocess.run([
            sys.executable, test_file
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running {test_file}: {e}")
        return False

def main():
    """Run all tests."""
    print("🌱 Smart Crop Advisory System - Test Suite")
    print("=" * 60)
    
    # Define test files
    test_files = [
        "tests/test_model.py",
        "tests/test_new_features.py",
        "tests/test_data.py",
        "tests/test_models.py"
    ]
    
    results = []
    
    for test_file in test_files:
        if Path(test_file).exists():
            success = run_test_file(test_file)
            results.append((test_file, success))
        else:
            print(f"⚠️  Test file not found: {test_file}")
            results.append((test_file, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_file, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status:12} {test_file}")
        if success:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
