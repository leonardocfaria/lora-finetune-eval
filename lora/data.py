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


def docstrings(module: str, methods: bool = False) -> list[str]:
    """Docstrings (>= 8 words) of public functions/classes, optionally also of class methods. Deterministic."""
    mod, out = importlib.import_module(module), []
    for name in sorted(dir(mod)):
        obj = getattr(mod, name)
        if name.startswith("_") or not (inspect.isfunction(obj) or inspect.isbuiltin(obj) or inspect.isclass(obj)):
            continue
        objs = [obj] + ([getattr(obj, m, None) for m in sorted(vars(obj)) if not m.startswith("_")] if methods and inspect.isclass(obj) else [])
        out += [d for o in objs if o is not None and (d := inspect.getdoc(o)) and len(d.split()) >= 8]
    return list(dict.fromkeys(out))  # dedupe, keep order


def pretrain_corpus() -> str:
    return "\n\n".join(d for m in PRETRAIN_MODULES for d in docstrings(m, methods=True))


def task_split(seed: int = 0):
    """[(docstring, label)] train/test, stratified 50/50 per label, at most MAX_PER_CLASS per label."""
    rng = random.Random(seed)
    seen, train, test = set(), [], []
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
