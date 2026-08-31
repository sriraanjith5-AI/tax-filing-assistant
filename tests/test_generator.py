import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from llm.generator import OpenAIGenerator
from llm.generation_dataclass import GenerationResult

# ============================================================
# Prompt construction - no live API calls, no OPENAI_API_KEY needed
# ============================================================

messages = OpenAIGenerator.build_messages(
    query="What is the standard deduction for 2026?",
    contexts=["The standard deduction for single filers is $14,600.", "Married filing jointly is $29,200."],
)

assert len(messages) == 2
role0, content0 = messages[0]
role1, content1 = messages[1]

assert role0 == "system"
assert "context" in content0.lower()

assert role1 == "human"
assert "[1] The standard deduction" in content1
assert "[2] Married filing jointly" in content1
assert "What is the standard deduction for 2026?" in content1

print("build_messages produces a well-formed system+human prompt: PASS")

# Empty context list still produces a valid prompt (no crash, explicit
# placeholder so the model doesn't silently hallucinate from nothing).
empty_messages = OpenAIGenerator.build_messages(query="Any question?", contexts=[])
_, empty_content = empty_messages[1]
assert "(no context retrieved)" in empty_content
print("build_messages handles empty context list: PASS")

# GenerationResult is a plain, serializable-shaped dataclass.
result = GenerationResult(answer="The deduction is $14,600.", latency_ms=123.4)
assert result.answer == "The deduction is $14,600."
assert result.latency_ms == 123.4
print("GenerationResult shape: PASS")

# ============================================================
# sources labels - tag each excerpt so the model can cite it
# ============================================================

labeled_messages = OpenAIGenerator.build_messages(
    query="What is the standard deduction for 2026?",
    contexts=["The standard deduction for single filers is $14,600.", "Married filing jointly is $29,200."],
    sources=["Pub15T.pdf p.3", "Pub15T.pdf p.4"],
)
labeled_role0, labeled_system = labeled_messages[0]
_, labeled_content = labeled_messages[1]
assert "[1] (Pub15T.pdf p.3) The standard deduction" in labeled_content
assert "[2] (Pub15T.pdf p.4) Married filing jointly" in labeled_content
print("build_messages tags excerpts with source labels: PASS")

# The citation instruction is only added to the system prompt when sources
# is given - the evaluation harness (no sources) must keep getting the
# exact same system prompt it always has, so golden-dataset generation
# metrics stay comparable across runs.
assert "bracketed page reference" in labeled_system.lower()
assert "bracketed page reference" not in content0.lower()
print("citation instruction only added when sources is given: PASS")

# sources length must match contexts length.
try:
    OpenAIGenerator.build_messages(query="q", contexts=["a", "b"], sources=["only one"])
    assert False, "expected ValueError for mismatched sources length"
except ValueError:
    print("build_messages rejects mismatched sources length: PASS")

# ============================================================
# citation_labels - the bracket marker itself names the source page
# (e.g. "[p.7]") instead of a plain positional index, so a citation is
# traceable straight to the PDF page without indirecting through a
# chunk number.
# ============================================================

page_labeled_messages = OpenAIGenerator.build_messages(
    query="What is the standard deduction for 2026?",
    contexts=["The standard deduction for single filers is $14,600.", "Married filing jointly is $29,200."],
    sources=["Pub15T.pdf p.3", "Pub15T.pdf p.4"],
    citation_labels=["p.3", "p.4"],
)
_, page_labeled_content = page_labeled_messages[1]
assert "[p.3] (Pub15T.pdf p.3) The standard deduction" in page_labeled_content
assert "[p.4] (Pub15T.pdf p.4) Married filing jointly" in page_labeled_content
print("build_messages uses citation_labels as the bracket marker: PASS")

# citation_labels length must match contexts length.
try:
    OpenAIGenerator.build_messages(
        query="q", contexts=["a", "b"], sources=["s1", "s2"], citation_labels=["only one"],
    )
    assert False, "expected ValueError for mismatched citation_labels length"
except ValueError:
    print("build_messages rejects mismatched citation_labels length: PASS")

# Without citation_labels, falls back to the plain positional index
# (already covered above) - confirms the param is purely additive.
assert "[1] (Pub15T.pdf p.3)" in labeled_content
print("omitting citation_labels keeps the positional-index fallback: PASS")

# ============================================================
# _linkify_citations - turns "[p.N]" into a link back to the source card
# for that page
# ============================================================

from retrieval.query_pipeline import _linkify_citations

linked = _linkify_citations("The deduction is $14,600 [p.7].", valid_pages={3, 7})
assert '<a href="#source-p7" class="cite">[p.7]</a>' in linked
print("_linkify_citations links citations for a known page: PASS")

# A page not among this answer's sources (hallucinated reference) is left
# as plain text rather than linking to a source card that doesn't exist.
unlinked = _linkify_citations("See [p.99] for details.", valid_pages={3, 7})
assert "<a" not in unlinked
assert "[p.99]" in unlinked
print("_linkify_citations leaves unknown-page citations unlinked: PASS")

# HTML in the answer itself is escaped, not injected raw.
escaped = _linkify_citations("<script>alert(1)</script> [p.3]", valid_pages={3})
assert "<script>" not in escaped
assert "&lt;script&gt;" in escaped
print("_linkify_citations escapes HTML in the answer: PASS")
