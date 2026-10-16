"""Check the output of `#print axioms` for the Lean formalisation (lean/AggregatingInertia/Axioms.lean):
every theorem listed there must be reported, and may depend only on Lean's three standard axioms."""
import re
import sys

ALLOWED = {"propext", "Classical.choice", "Quot.sound"}
listed = re.findall(r"^#print axioms ([\w.]+)", open(sys.argv[1]).read(), re.M)
text = open(sys.argv[2]).read()
reports = dict(re.findall(r"'([\w.]+)' (?:depends on axioms: \[([^\]]*)\]|does not depend on any axioms)", text))
missing = [name for name in listed if name not in reports]
if not listed or missing:
    sys.exit("theorems without an axiom report: " + ", ".join(missing or ["(none listed)"]))
bad = []
for name in listed:
    used = {a.strip() for a in reports[name].split(",") if a.strip()}
    if not used <= ALLOWED:
        bad.append(f"{name}: {sorted(used - ALLOWED)}")
if bad:
    sys.exit("non-standard axioms:\n" + "\n".join(bad))
print(f"{len(listed)} theorems, standard axioms only")
