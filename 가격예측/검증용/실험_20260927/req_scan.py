import ast, os, re, sys
from importlib import metadata
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name as cn
root = r"C:\kch_Final_prj"
skip = {"쓰래기통", ".venv", ".git", "mlflow_artifacts", "__pycache__", "node_modules"}
imports = {}
for dp, dns, fns in os.walk(root):
    dns[:] = [d for d in dns if d not in skip and not d.startswith(".")]
    for f in fns:
        if not f.endswith(".py"): continue
        p = os.path.join(dp, f)
        try: tree = ast.parse(open(p, encoding="utf-8").read())
        except Exception as e: print("PARSE FAIL", p, e); continue
        for n in ast.walk(tree):
            mods = []
            if isinstance(n, ast.Import): mods = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module: mods = [n.module]
            for m in mods: imports.setdefault(m.split(".")[0], set()).add(os.path.relpath(p, root))
# also .yml/.bat/.sh command-line tools (mlflow server, streamlit run, etc.)
cli = set()
for dp, dns, fns in os.walk(root):
    dns[:] = [d for d in dns if d not in skip and not d.startswith(".") or d == ".github"]
    for f in fns:
        if f.endswith((".bat", ".yml", ".yaml", ".sh", ".ps1", ".toml", ".cfg")):
            t = open(os.path.join(dp, f), encoding="utf-8", errors="ignore").read()
            for tool in ("mlflow", "streamlit", "uvicorn", "gunicorn", "waitress", "pytest", "alembic"):
                if re.search(rf"\b{tool}\b", t): cli.add((tool, os.path.relpath(os.path.join(dp, f), root)))
pkg_map = metadata.packages_distributions()
local = {d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))} | {f[:-3] for f in os.listdir(root) if f.endswith(".py")}
used = {}
for m, files in imports.items():
    if m in local or m in sys.stdlib_module_names: continue
    for d in pkg_map.get(m, []): used.setdefault(cn(d), set()).update(files)
    if m not in pkg_map: print("UNRESOLVED import:", m, sorted(files)[:3])
for tool, f in cli: used.setdefault(cn(tool), set()).add(f + " (CLI)")
# dependency closure
def deps(name):
    try: reqs = metadata.requires(name) or []
    except metadata.PackageNotFoundError: return []
    out = []
    for r in reqs:
        R = Requirement(r)
        if R.marker and not R.marker.evaluate({"extra": ""}): continue
        out.append(cn(R.name))
    return out
closure, stack = set(), list(used)
parent = {}
while stack:
    x = stack.pop()
    if x in closure: continue
    closure.add(x)
    for d in deps(x):
        if d not in closure: parent.setdefault(d, x); stack.append(d)
reqs = []
for line in open(os.path.join(root, "requirements.txt"), encoding="utf-8"):
    line = line.strip()
    if not line or line.startswith("#"): continue
    reqs.append(cn(re.split(r"[=<>@ ;\[]", line)[0]))
print("\n== direct use ==")
for u in sorted(used): print(f"  {u}: {sorted(used[u])[:2]}")
cand = [r for r in reqs if r not in closure]
print(f"\n== requirements: {len(reqs)} total, {len(reqs)-len(cand)} used/needed, {len(cand)} candidates ==")
for r in cand:
    rev = [cn(d.metadata['Name']) for d in metadata.distributions() if r in [cn(Requirement(x).name) for x in (d.requires or [])]]
    print(f"  {r:<28} required_by={sorted(set(rev))}")
