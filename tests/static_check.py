"""Static consistency checker for the Persian RAG codebase.

Parses every module (syntax gate) and cross-checks that:
- every `from app...` import resolves to a real module and top-level name
- method calls on known component attributes (self.store, chatbot.
  history_manager, ...) match methods that actually exist on the target class
- every `settings.<attr>` access matches a Settings field or property

Runs on any Python >= 3.8 (ast-only, no project imports), so it doubles as a
syntax gate on interpreters that cannot execute the 3.10+ code.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# attribute name -> defining class (project classes only)
COMPONENTS = {
    "history_manager": "QdrantChatHistory",
    "store": "QdrantVectorStore",
    "embeddings": "EmbeddingService",
    "language_model": "LanguageModel",
    "chatbot": "RAGEngine",
    "processor": "DocumentProcessor",
    "bm25": "BM25Encoder",
    "session_locks": "SessionLocks",
}


def collect_definitions(sources):
    class_methods = {}
    module_top_level = {}  # module path -> set of exported names
    for path, tree in sources.items():
        top = set()
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                top.add(node.name)
                names = {
                    item.name
                    for item in node.body
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
                class_methods.setdefault(node.name, set()).update(names)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                top.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        top.add(target.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                top.add(node.target.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    top.add(alias.asname or alias.name.split(".")[0])
        module_top_level[str(path)] = top
    return class_methods, module_top_level


def module_path_for(module: str) -> Path | None:
    base = ROOT / Path(*module.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.exists():
            return candidate
    return None


def check_imports(sources, module_top_level, errors):
    for path, tree in sources.items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level:
                continue
            module = node.module or ""
            if not module.startswith("app"):
                continue
            target = module_path_for(module)
            if target is None:
                errors.append(f"{path.relative_to(ROOT)}: import from missing module {module}")
                continue
            names = module_top_level.get(str(target), set())
            for alias in node.names:
                if alias.name != "*" and alias.name not in names:
                    errors.append(
                        f"{path.relative_to(ROOT)}: '{alias.name}' not exported by {module}"
                    )


def component_of(base: ast.AST) -> str | None:
    """Component class for the object a method is called on, if recognizable.

    Accepts plain names (`chatbot.x()`) and attribute chains rooted at a
    non-self variable (`chatbot.history_manager.x()`), but for `self` only
    direct attributes (`self.embeddings.x()`), so `self.client.embeddings.x()`
    (the OpenAI SDK) is not mistaken for the project component.
    """
    if isinstance(base, ast.Name):
        return COMPONENTS.get(base.id)
    if isinstance(base, ast.Attribute):
        if isinstance(base.value, ast.Name):
            return COMPONENTS.get(base.attr)
        root = base.value
        while isinstance(root, ast.Attribute):
            root = root.value
        if isinstance(root, ast.Name) and root.id != "self":
            return COMPONENTS.get(base.attr)
    return None


def check_calls(sources, class_methods, errors):
    for path, tree in sources.items():
        rel = path.relative_to(ROOT)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            component = component_of(node.func.value)
            if component and node.func.attr not in class_methods.get(component, set()):
                errors.append(
                    f"{rel}: ...{node.func.value.__class__.__name__}.{node.func.attr}() "
                    f"not found on {component}"
                )


def collect_settings_attributes(sources):
    attrs = set()
    for path, tree in sources.items():
        if path.name != "config.py":
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "Settings":
                for item in node.body:
                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        attrs.add(item.target.id)
                    elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        attrs.add(item.name)
    return attrs


def check_settings_access(sources, settings_attrs, errors):
    for path, tree in sources.items():
        if path.name == "config.py":
            continue
        rel = path.relative_to(ROOT)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "settings"
                and node.attr not in settings_attrs
            ):
                errors.append(f"{rel}: settings.{node.attr} does not exist on Settings")


def main() -> int:
    paths = sorted((ROOT / "app").rglob("*.py"))
    paths += sorted((ROOT / "tests").rglob("*.py"))
    paths.append(ROOT / "main.py")
    sources = {}
    errors = []
    for path in paths:
        try:
            sources[path] = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            errors.append(f"{path.relative_to(ROOT)}: SYNTAX ERROR: {exc}")

    class_methods, module_top_level = collect_definitions(sources)
    check_imports(sources, module_top_level, errors)
    check_calls(sources, class_methods, errors)
    check_settings_access(sources, collect_settings_attributes(sources), errors)

    for error in errors:
        print("FAIL:", error)
    if not errors:
        print(f"OK: {len(sources)} files parsed; imports, component calls, and settings accesses consistent")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
