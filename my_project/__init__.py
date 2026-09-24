import pathlib

readme_path = pathlib.Path(__file__).parent.parent / "README.md"

if readme_path.exists():
    __doc__ = readme_path.read_text(encoding="utf-8")
else:
    __doc__ = "Package documentation (README.md not found)."
    
__all__ = ['dataset','train','fileManager']
