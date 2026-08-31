import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.registry import LOADERS, make_loader, variant_key
from ingestion.pdfloader import PDFLoader
from ingestion.base_loader import BaseDocumentLoader

# ============================================================
# Registry wiring - fast, no PDF parsing involved. Each loader's actual
# load_documents() is exercised manually via the loader comparison run
# (see evaluation/results/loader_comparison_*.csv) rather than here -
# the slower ones take tens of seconds to minutes per run, not suited
# to a script re-run on every change.
# ============================================================

EXPECTED_LOADERS = {"pypdf", "pymupdf4llm", "pdfplumber", "unstructured"}
assert EXPECTED_LOADERS.issubset(set(LOADERS))
print(f"LOADERS lists all expected loaders {sorted(EXPECTED_LOADERS)}: PASS")

assert issubclass(PDFLoader, BaseDocumentLoader)
print("PDFLoader implements BaseDocumentLoader: PASS")

pypdf_loader = make_loader("pypdf", PROJECT_ROOT / "data" / "pdf")
assert isinstance(pypdf_loader, PDFLoader)
print("make_loader('pypdf', ...) returns a PDFLoader: PASS")

try:
    make_loader("not_a_real_loader", PROJECT_ROOT / "data" / "pdf")
    assert False, "expected ValueError for unknown loader name"
except ValueError:
    print("make_loader rejects unknown loader name: PASS")

# Every non-pypdf loader avoids importing its dependency tree until it's
# actually asked for - importing ingestion.registry itself must not pull
# any of them in (spacy, numba, ...).
LAZY_MODULE_BY_LOADER = {
    "pymupdf4llm": "pymupdf4llm",
    "pdfplumber": "pdfplumber",
    "unstructured": "unstructured",
}
for loader_name, module_name in LAZY_MODULE_BY_LOADER.items():
    assert module_name not in sys.modules, f"{module_name} was imported before make_loader('{loader_name}', ...) was called"
print("importing ingestion.registry does not eagerly import any lazy loader's dependencies: PASS")

for loader_name, module_name in LAZY_MODULE_BY_LOADER.items():
    loader = make_loader(loader_name, PROJECT_ROOT / "data" / "pdf")
    assert isinstance(loader, BaseDocumentLoader)
    print(f"make_loader('{loader_name}', ...) returns a BaseDocumentLoader: PASS")

# ============================================================
# variant_key - loader_name is isolating but backward-compatible
# ============================================================

# Default (no loader_name passed) matches pre-loader-option behavior
# exactly - existing pypdf-based collections/indexes keep their key.
assert variant_key("recursive", 700, 120) == "recursive__cs700_co120"
print("variant_key defaults to pre-existing pypdf key format: PASS")

assert variant_key("recursive", 700, 120, loader_name="pypdf") == "recursive__cs700_co120"
print("variant_key('pypdf') matches the default: PASS")

seen_keys = set()
for loader_name in EXPECTED_LOADERS - {"pypdf"}:
    key = variant_key("recursive", 700, 120, loader_name=loader_name)
    assert key == f"{loader_name}__recursive__cs700_co120"
    assert key not in seen_keys, f"variant_key collision for loader '{loader_name}'"
    seen_keys.add(key)
print("every non-pypdf loader gets its own isolated variant_key: PASS")

assert variant_key("semantic", 700, 120, loader_name="pdfplumber") == "pdfplumber__semantic"
print("variant_key('pdfplumber') isolates the semantic-chunker variant too: PASS")

print("\nAll registry/loader wiring tests passed.")
