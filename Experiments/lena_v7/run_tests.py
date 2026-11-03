"""Run test_*.py functions without pytest (the cluster env has no pytest). Usage: python run_tests.py test_lena_v7.py"""
import importlib
import sys
import traceback

failed = 0
for path in sys.argv[1:]:
    mod = importlib.import_module(path[:-3] if path.endswith(".py") else path)
    for name in sorted(n for n in dir(mod) if n.startswith("test_")):
        try:
            getattr(mod, name)()
            print("PASS", name)
        except Exception:
            failed += 1
            print("FAIL", name)
            traceback.print_exc()
print(f"{failed} failed")
sys.exit(1 if failed else 0)
