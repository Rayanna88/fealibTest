"""Verify installation of required packages."""
import sys
import importlib


def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {package_name} is installed (version: {version})")
        return True
    except ImportError as e:
        print(f"✗ {package_name} is NOT installed: {e}")
        return False


def main():
    print(f"Python version: {sys.version}")
    print()

    packages = [
        ("pyfealib", "pyfealib"),
        ("pytest", "pytest"),
        ("pytest-xdist", "xdist"),
        ("PyYAML", "yaml"),
    ]

    results = []
    for pkg_name, import_name in packages:
        results.append(check_package(pkg_name, import_name))

    print()
    if all(results):
        print("All packages are installed successfully!")
        return 0
    else:
        print("Some packages are missing.")
        return 1


if __name__ == "__main__":
    sys.exit(main())