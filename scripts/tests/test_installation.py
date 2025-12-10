#!/usr/bin/env python3
"""
Test script to verify installation and basic functionality.

Usage:
    python scripts/tests/test_installation.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

def test_imports():
    """Test that all required packages are importable."""
    print("Testing package imports...")
    
    # Core scientific packages
    try:
        import numpy as np
        print(f"  ✓ NumPy {np.__version__}")
    except ImportError:
        print("  ✗ NumPy not found")
        return False
    
    try:
        import scipy
        print(f"  ✓ SciPy {scipy.__version__}")
    except ImportError:
        print("  ✗ SciPy not found")
        return False
    
    try:
        import matplotlib
        print(f"  ✓ Matplotlib {matplotlib.__version__}")
    except ImportError:
        print("  ✗ Matplotlib not found")
        return False
    
    return True


def test_core_modules():
    """Test that core simulation modules are importable."""
    print("\nTesting core modules...")
    
    try:
        from scripts.core import llg_engine
        print("  ✓ llg_engine importable")
    except ImportError as e:
        print(f"  ✗ llg_engine import failed: {e}")
        return False
    
    try:
        from scripts.core import phase_diagram
        print("  ✓ phase_diagram importable")
    except ImportError as e:
        print(f"  ✗ phase_diagram import failed: {e}")
        return False
    
    try:
        from scripts.core import soliton_dynamics
        print("  ✓ soliton_dynamics importable")
    except ImportError as e:
        print(f"  ✗ soliton_dynamics import failed: {e}")
        return False
    
    return True


def test_basic_functionality():
    """Test basic numerical operations."""
    print("\nTesting basic functionality...")
    
    import numpy as np
    
    # Test spin normalization
    spins = np.random.randn(100, 3)
    norms = np.linalg.norm(spins, axis=1)
    spins = spins / norms[:, np.newaxis]
    
    if np.allclose(np.linalg.norm(spins, axis=1), 1.0, atol=1e-6):
        print("  ✓ Spin normalization works")
    else:
        print("  ✗ Spin normalization failed")
        return False
    
    # Test periodic boundary conditions (roll)
    test_array = np.arange(10)
    rolled = np.roll(test_array, 1)
    if rolled[0] == 9 and rolled[-1] == 8:
        print("  ✓ Periodic boundaries work")
    else:
        print("  ✗ Periodic boundaries failed")
        return False
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("MAGNETIC SOLITONS 1D - Installation Test")
    print("=" * 60)
    print(f"\nPython version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print()
    
    all_passed = True
    
    # Run tests
    all_passed &= test_imports()
    all_passed &= test_core_modules()
    all_passed &= test_basic_functionality()
    
    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED - Installation successful!")
        print("\nYou can now:")
        print("  1. Run phase diagram: python scripts/core/phase_diagram.py")
        print("  2. Simulate soliton: python scripts/core/soliton_dynamics.py")
        print("  3. Check notebooks: jupyter notebook notebooks/")
        return 0
    else:
        print("❌ SOME TESTS FAILED - Please check installation")
        print("\nTry:")
        print("  pip install -r requirements.txt")
        print("  or")
        print("  conda env create -f environment.yml")
        return 1


if __name__ == '__main__':
    sys.exit(main())
