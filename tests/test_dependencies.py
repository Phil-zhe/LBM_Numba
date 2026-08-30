import ast
from pathlib import Path
import tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


ROOT = Path(__file__).resolve().parents[1]


def test_declared_dependencies_and_import_contract():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    requirements = [Requirement(x) for x in project["dependencies"]]
    names = {canonicalize_name(r.name): r for r in requirements}
    assert "numba-cuda-mlir" in names
    assert str(names["numba-cuda-mlir"].specifier) == "==0.5.0"
    assert not ({"numba", "numba-cuda"} & names.keys())
    assert "3.11" in project["requires-python"]
    for path in (ROOT / "src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(alias.name.split(".")[0] != "numba" for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] != "numba"
    gpu_source = (ROOT / "src/lbm2d/backends/numba_cuda_mlir.py").read_text(encoding="utf-8")
    assert "from numba_cuda_mlir import cuda" in gpu_source

