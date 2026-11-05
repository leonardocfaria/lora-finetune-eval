"""Python stdlib docstrings (PSF licence, ship with the interpreter): pretraining corpus + a held-out task."""
import importlib
import inspect
import random

LABELS = ["os", "math", "statistics", "typing", "asyncio", "logging"]  # task modules: never in pretraining
PRETRAIN_MODULES = """builtins inspect socket threading itertools functools shutil pathlib re json collections subprocess
unittest urllib.parse hashlib csv argparse tarfile zipfile sqlite3 email.message textwrap decimal datetime random
heapq bisect difflib fractions glob gzip pickle queue tempfile uuid weakref zlib""".split()
MAX_PER_CLASS = 60
PROMPT_CHARS = 300


ALIASES = {"posix": "os", "posixpath": "os"}  # os's C implementation and os.path report these modules


def home(obj) -> str:
    """The module an object is *defined* in, e.g. 'collections' for statistics.Counter (a mere re-export)."""
    m = getattr(obj, "__module__", None) or ""
    return ALIASES.get(m, m.lstrip("_"))


def docstrings(module: str, methods: bool = False) -> list[str]:
    """Docstrings (>= 8 words) of public functions/classes *defined* in `module` (re-exports such as
    statistics.Counter are skipped: they would be mislabelled and duplicate another module's text),
    optionally also of class methods. Deterministic."""
    mod, out = importlib.import_module(module), []
    for name in sorted(dir(mod)):
        obj = getattr(mod, name)
        if name.startswith("_") or not (inspect.isfunction(obj) or inspect.isbuiltin(obj) or inspect.isclass(obj)):
            continue
        if not (home(obj) == module or home(obj).startswith(module + ".")):
            continue
        objs = [obj] + ([getattr(obj, m, None) for m in sorted(vars(obj)) if not m.startswith("_")] if methods and inspect.isclass(obj) else [])
        out += [d for o in objs if o is not None and (d := inspect.getdoc(o)) and len(d.split()) >= 8]
    return list(dict.fromkeys(out))  # dedupe, keep order


def pretrain_corpus() -> str:
    return "\n\n".join(d for m in PRETRAIN_MODULES for d in docstrings(m, methods=True))


def task_split(seed: int = 0):
    """[(docstring, label)] train/test, stratified 50/50 per label, at most MAX_PER_CLASS per label."""
    rng = random.Random(seed)
    # data-level guarantee on top of the __module__ filter: aliases like typing.Type carry another
    # object's docstring (builtins.type), so drop anything the pretraining corpus contains
    seen, train, test = {d for m in PRETRAIN_MODULES for d in docstrings(m, methods=True)}, [], []
    for label in LABELS:
        docs = [d for d in docstrings(label) if d not in seen]  # os.path re-exports etc.: one label per text
        seen.update(docs)
        rng.shuffle(docs)
        docs = docs[:MAX_PER_CLASS]
        half = len(docs) // 2
        train += [(d, label) for d in docs[:half]]
        test += [(d, label) for d in docs[half:]]
    rng.shuffle(train)
    return train, test


def prompt(doc: str) -> str:
    return f"Docstring: {' '.join(doc.split())[:PROMPT_CHARS]}\nModule:"
