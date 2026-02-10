from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Set, NamedTuple

import javalang


"""
python "./inlining/java_method_ast_inliner_v3.py" routingTest_call_analysis.json "./inlining-progress/routingTest_inline_output_v3.java" "./inlining-progress/routingTest_inline_report_v3.json"
"""


# ----------------------------
# Data models
# ----------------------------

@dataclass
class Callee:
    method_name: str
    arity: int
    class_name: str
    package_name: str
    fqn: str
    file_path: str
    parameters: List[str]  # e.g. ["String provider", "int seconds"]
    body_full_text: str
    location: Tuple[int, int]  # (start_line, end_line)


@dataclass
class Replacement:
    start: int
    end: int
    new_text: str
    info: Dict[str, Any]


class PickResult(NamedTuple):
    status: str  # "ok" | "ambiguous" | "not_found"
    callee: Optional[Callee]
    detail: str


# ----------------------------
# JSON extraction
# ----------------------------

def walk_call_tree(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    stack = [node]
    while stack:
        cur = stack.pop()
        out.append(cur)
        for c in cur.get("calls", []) or []:
            stack.append(c)
    return out


def _callee_identity(m: Dict[str, Any], name: str, arity: int, cls: str) -> str:
    """
    Stable identity string for dedupe.
    Prefer fqn; else fallback to location+file.
    """
    fqn = (m.get("full_qualified_name") or "").strip()
    if fqn:
        return f"fqn::{fqn}"

    fp = (m.get("file_path") or "").strip()
    loc = m.get("location") or {}
    sl = int(loc.get("start_line") or 0)
    el = int(loc.get("end_line") or 0)
    return f"loc::{fp}::{sl}:{el}::{cls}::{name}/{arity}"


def build_callee_indexes(
    data: Dict[str, Any]
) -> Tuple[Dict[Tuple[str, int], List[Callee]], Dict[Tuple[str, str, int], List[Callee]]]:
    """
    Build two indexes, deduped:
      1) by (method_name, arity)
      2) by (class_name, method_name, arity)
    """
    idx_name: Dict[Tuple[str, int], List[Callee]] = {}
    idx_class: Dict[Tuple[str, str, int], List[Callee]] = {}

    seen: Set[str] = set()

    for top in data["test_method"].get("calls", []) or []:
        for m in walk_call_tree(top):
            name = m.get("name")
            params = m.get("parameters") or []
            arity = len(params)

            body = (m.get("body") or {}).get("full_text") or ""
            if not name or not body:
                continue

            cls = (m.get("class_name") or "").strip()
            ident = _callee_identity(m, name, arity, cls)
            if ident in seen:
                continue
            seen.add(ident)

            loc = m.get("location") or {}
            sl = int(loc.get("start_line") or 0)
            el = int(loc.get("end_line") or 0)

            callee = Callee(
                method_name=name,
                arity=arity,
                class_name=cls,
                package_name=m.get("package_name") or "",
                fqn=m.get("full_qualified_name") or "",
                file_path=m.get("file_path") or "",
                parameters=params,
                body_full_text=body,
                location=(sl, el),
            )

            idx_name.setdefault((name, arity), []).append(callee)
            if cls:
                idx_class.setdefault((cls, name, arity), []).append(callee)

    return idx_name, idx_class


def extract_param_names(param_decls: List[str]) -> List[str]:
    names: List[str] = []
    for p in param_decls:
        p = p.strip()
        if not p:
            continue
        toks = p.split()
        names.append(toks[-1])
    return names


# ----------------------------
# Source span helpers
# ----------------------------

def linecol_to_offset(src: str, line: int, col: int) -> int:
    if line < 1 or col < 1:
        return 0
    lines = src.splitlines(keepends=True)
    if line > len(lines):
        return len(src)
    return sum(len(lines[i]) for i in range(line - 1)) + (col - 1)


def expand_left_for_qualifier(src: str, pos: int) -> int:
    i = pos
    while i > 0 and re.match(r"[A-Za-z0-9_\.$]", src[i - 1]):
        i -= 1
    return i


def scan_to_end_of_call(src: str, start: int) -> int:
    i = start
    n = len(src)

    while i < n and src[i] != '(':
        i += 1
    if i >= n:
        return start

    depth = 0
    while i < n:
        ch = src[i]
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                return i + 1
        elif ch in ('"', "'"):
            quote = ch
            i += 1
            while i < n:
                if src[i] == '\\':
                    i += 2
                    continue
                if src[i] == quote:
                    break
                i += 1
        i += 1

    return n


def extend_to_statement_semicolon(src: str, end_call: int) -> int:
    j = end_call
    while j < len(src) and src[j].isspace():
        j += 1
    if j < len(src) and src[j] == ';':
        return j + 1
    return end_call


def extract_args_text(src: str, call_start: int, call_end: int) -> List[str]:
    call_text = src[call_start:call_end]
    lpar = call_text.find('(')
    rpar = call_text.rfind(')')
    if lpar == -1 or rpar == -1 or rpar <= lpar:
        return []
    args_str = call_text[lpar + 1:rpar].strip()
    if not args_str:
        return []

    args: List[str] = []
    cur: List[str] = []
    depth = 0
    i = 0
    while i < len(args_str):
        ch = args_str[i]
        if ch == '(':
            depth += 1
            cur.append(ch)
        elif ch == ')':
            depth -= 1
            cur.append(ch)
        elif ch in ('"', "'"):
            quote = ch
            cur.append(ch)
            i += 1
            while i < len(args_str):
                cur.append(args_str[i])
                if args_str[i] == '\\':
                    i += 2
                    continue
                if args_str[i] == quote:
                    break
                i += 1
        elif ch == ',' and depth == 0:
            args.append(''.join(cur).strip())
            cur = []
        else:
            cur.append(ch)
        i += 1
    if cur:
        args.append(''.join(cur).strip())

    return args


# ----------------------------
# Callee analysis (safe inlining)
# ----------------------------

# allow optional throws clause
WRAPPER_RETURN_RE = re.compile(
    r"^\s*(?:public|protected|private)?\s*(?:static\s+)?[\w<>\[\], ?]+\s+\w+\s*\([^)]*\)\s*(?:throws\s+[\w\s,]+)?\s*\{\s*return\s+(.*?);\s*\}\s*$",
    re.S
)
WRAPPER_VOID_RE = re.compile(
    r"^\s*(?:public|protected|private)?\s*(?:static\s+)?void\s+\w+\s*\([^)]*\)\s*(?:throws\s+[\w\s,]+)?\s*\{\s*(.*?);\s*\}\s*$",
    re.S
)

PRIVATE_BAD_TOKENS = {
    # likely private/static state from other classes
    "INSTANCE",
    # obvious instance access patterns
    "this.", "super.",
    # project-specific sensitive tokens you already had
    "initCounter", "executor", "vertx", "system", "cache"
}


def parse_wrapper_expr(body_full_text: str) -> Tuple[Optional[str], str]:
    txt = body_full_text.strip()

    m = WRAPPER_RETURN_RE.match(txt)
    if m:
        return m.group(1).strip(), "return"

    m = WRAPPER_VOID_RE.match(txt)
    if m:
        return m.group(1).strip(), "void"

    return None, "not_wrapper"


def has_obvious_private_state(expr_or_body: str) -> bool:
    return any(tok in expr_or_body for tok in PRIVATE_BAD_TOKENS)


def substitute_params(expr: str, param_decls: List[str], arg_texts: List[str]) -> str:
    param_names = extract_param_names(param_decls)
    if len(param_names) != len(arg_texts):
        return expr

    out = expr
    for p, a in zip(param_names, arg_texts):
        out = re.sub(rf"\b{re.escape(p)}\b", f"({a})", out)
    return out


def parse_method_decl_via_dummy(method_text: str) -> Optional[javalang.tree.MethodDeclaration]:
    try:
        dummy = f"class __Dummy__ {{\n{method_text}\n}}"
        cu = javalang.parse.parse(dummy)
        for _, node in cu.filter(javalang.tree.MethodDeclaration):
            return node
        return None
    except Exception:
        return None


def method_is_self_contained(md: javalang.tree.MethodDeclaration) -> bool:
    declared: Set[str] = set()

    for p in md.parameters or []:
        declared.add(p.name)

    for _, node in md.filter(javalang.tree.VariableDeclarator):
        declared.add(node.name)

    for _, node in md.filter(javalang.tree.CatchClauseParameter):
        declared.add(node.name)

    for _, node in md.filter(javalang.tree.LambdaExpression):
        if node.parameters:
            for lp in node.parameters:
                if isinstance(lp, str):
                    declared.add(lp)
                else:
                    declared.add(lp.name)

    for _, _node in md.filter(javalang.tree.This):
        return False
    for _, _node in md.filter(javalang.tree.SuperMethodInvocation):
        return False

    returns = list(md.filter(javalang.tree.ReturnStatement))
    if len(returns) > 1:
        return False
    if md.body and returns:
        if not isinstance(md.body[-1], javalang.tree.ReturnStatement):
            return False

    for _, node in md.filter(javalang.tree.MemberReference):
        if node.qualifier is None:
            if node.member not in declared:
                return False

    return True


def extract_body_block_text(method_decl_text: str) -> Optional[str]:
    s = method_decl_text
    i = s.find('{')
    if i == -1:
        return None

    depth = 0
    j = i
    while j < len(s):
        ch = s[j]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return s[i + 1:j].strip("\n\r ")
        elif ch in ('"', "'"):
            quote = ch
            j += 1
            while j < len(s):
                if s[j] == '\\':
                    j += 2
                    continue
                if s[j] == quote:
                    break
                j += 1
        j += 1

    return None


def is_probably_class_name(s: str) -> bool:
    return bool(s) and s[0].isalpha() and s[0].isupper() and s.replace("$", "").replace("_", "").isalnum()


def extract_call_qualifier_prefix(call_text: str, member_name: str) -> str:
    idx = call_text.rfind(member_name)
    if idx <= 0:
        return ""
    prefix = call_text[:idx]
    if prefix.endswith("."):
        return prefix
    p2 = prefix.strip()
    if p2.endswith("."):
        return p2
    return ""


def qualify_wrapper_expr_if_needed(wrapper_expr: str, call_text: str, call_name: str) -> str:
    qprefix = extract_call_qualifier_prefix(call_text, call_name)
    if not qprefix:
        return wrapper_expr

    if re.match(r"^\s*[A-Za-z_]\w*\s*\.\s*[A-Za-z_]\w*\s*\(", wrapper_expr):
        return wrapper_expr

    if re.match(r"^\s*[A-Za-z_]\w*\s*\(", wrapper_expr):
        return qprefix + wrapper_expr.lstrip()

    return wrapper_expr


# ----------------------------
# Inliner engine
# ----------------------------

class CompilableJavaInliner:
    def __init__(
        self,
        idx_name: Dict[Tuple[str, int], List[Callee]],
        idx_class: Dict[Tuple[str, str, int], List[Callee]],
    ) -> None:
        self.idx_name = idx_name
        self.idx_class = idx_class

    def _pick_unique(self, cands: List[Callee]) -> PickResult:
        if not cands:
            return PickResult("not_found", None, "no_candidates")
        if len(cands) == 1:
            return PickResult("ok", cands[0], "unique")
        # ambiguous in-project (likely overloads or incomplete resolution)
        return PickResult("ambiguous", None, f"{len(cands)}_candidates")

    def choose_callee(
        self,
        method_name: str,
        arity: int,
        qualifier: Optional[str],
        var_types: Dict[str, str],
    ) -> PickResult:
        """
        Fix A with improved diagnostics:
        - If qualifier looks like class name -> idx_class
        - If qualifier looks like variable -> infer type -> idx_class
        - Else idx_name fallback
        """
        q = (qualifier or "").strip() if qualifier is not None else ""

        # qualifier is ClassName
        if q and is_probably_class_name(q):
            res = self._pick_unique(self.idx_class.get((q, method_name, arity), []))
            if res.status != "not_found":
                return res

        # qualifier is variable name -> infer type
        if q and q in var_types:
            t = var_types[q]
            res = self._pick_unique(self.idx_class.get((t, method_name, arity), []))
            if res.status != "not_found":
                return res

        # fallback: by name/arity
        res = self._pick_unique(self.idx_name.get((method_name, arity), []))
        return res

    def infer_local_var_types(self, method_decl: javalang.tree.MethodDeclaration) -> Dict[str, str]:
        types: Dict[str, str] = {}
        for _, node in method_decl.filter(javalang.tree.LocalVariableDeclaration):
            tname = getattr(node.type, "name", None)
            if not tname:
                continue
            for decl in node.declarators or []:
                if decl.name:
                    types[decl.name] = tname
        return types

    def inline(self, test_method_src: str) -> Tuple[str, Dict[str, Any]]:
        report: Dict[str, Any] = {"inlined": [], "skipped": []}
        replacements: List[Replacement] = []

        # Parse test method (wrap into dummy class)
        try:
            dummy = f"class __Dummy__ {{\n{test_method_src}\n}}"
            cu = javalang.parse.parse(dummy)
        except Exception as e:
            raise RuntimeError(f"Cannot parse test method with javalang: {e}")

        method_decl: Optional[javalang.tree.MethodDeclaration] = None
        for _, node in cu.filter(javalang.tree.MethodDeclaration):
            method_decl = node
            break
        if method_decl is None or method_decl.position is None:
            raise RuntimeError("No MethodDeclaration found in test method source")

        # Fix C: line shift alignment between dummy and raw method src
        line_shift = method_decl.position.line - 1

        var_types = self.infer_local_var_types(method_decl)
        invocations = [node for _, node in method_decl.filter(javalang.tree.MethodInvocation)]

        for inv in invocations:
            if inv.position is None:
                continue

            call_name = inv.member
            call_arity = len(inv.arguments or [])
            inv_qualifier = getattr(inv, "qualifier", None)

            pick = self.choose_callee(call_name, call_arity, inv_qualifier, var_types)

            if pick.status != "ok" or pick.callee is None:
                # Better skip reasons
                if pick.status == "ambiguous":
                    reason = "ambiguous_project_callee"
                else:
                    # not_found
                    reason = "library_or_unindexed"
                report["skipped"].append({
                    "call": f"{call_name}/{call_arity}",
                    "qualifier": inv_qualifier,
                    "reason": reason,
                    "detail": pick.detail
                })
                continue

            callee = pick.callee

            # Locate call span
            line, col = inv.position
            line = line - line_shift
            rough = linecol_to_offset(test_method_src, line, col)
            start = expand_left_for_qualifier(test_method_src, rough)
            end_call = scan_to_end_of_call(test_method_src, start)

            call_text = test_method_src[start:end_call]
            arg_texts = extract_args_text(test_method_src, start, end_call)

            # 1) wrapper inline if possible
            wrapper_expr, kind = parse_wrapper_expr(callee.body_full_text)
            if wrapper_expr is not None:
                # Safety gate: INSTANCE / this / super etc.
                if has_obvious_private_state(wrapper_expr):
                    # make this very explicit (this is the getInstance case)
                    report["skipped"].append({
                        "callee": callee.fqn,
                        "call": f"{call_name}/{call_arity}",
                        "qualifier": inv_qualifier,
                        "reason": "unsafe_wrapper_private_state",
                        "detail": "wrapper_expr_contains_private_state_tokens",
                        "wrapper_expr": wrapper_expr
                    })
                    continue

                wrapper_expr2 = qualify_wrapper_expr_if_needed(wrapper_expr, call_text, call_name)
                repl = substitute_params(wrapper_expr2, callee.parameters, arg_texts)

                if kind == "void":
                    end_stmt = extend_to_statement_semicolon(test_method_src, end_call)
                    if end_stmt == end_call:
                        report["skipped"].append({
                            "callee": callee.fqn,
                            "reason": "void_wrapper_not_in_statement_context",
                            "call_text": call_text
                        })
                        continue
                    replacements.append(Replacement(
                        start=start,
                        end=end_stmt,
                        new_text=repl + ";",
                        info={
                            "strategy": "wrapper_void",
                            "callee": callee.fqn,
                            "replaced": test_method_src[start:end_stmt],
                            "replacement": repl + ";",
                        }
                    ))
                else:
                    replacements.append(Replacement(
                        start=start,
                        end=end_call,
                        new_text=repl,
                        info={
                            "strategy": "wrapper_return",
                            "callee": callee.fqn,
                            "replaced": call_text,
                            "replacement": repl,
                        }
                    ))
                continue

            # 2) full-body inline (statement-context only) if self-contained
            end_stmt = extend_to_statement_semicolon(test_method_src, end_call)
            if end_stmt == end_call:
                report["skipped"].append({
                    "callee": callee.fqn,
                    "reason": "not_wrapper_and_not_statement_context",
                    "call_text": call_text
                })
                continue

            if has_obvious_private_state(callee.body_full_text):
                report["skipped"].append({
                    "callee": callee.fqn,
                    "reason": "private_state_tokens_in_body"
                })
                continue

            md = parse_method_decl_via_dummy(callee.body_full_text)
            if md is None:
                report["skipped"].append({
                    "callee": callee.fqn,
                    "reason": "callee_parse_failed"
                })
                continue

            if not method_is_self_contained(md):
                report["skipped"].append({
                    "callee": callee.fqn,
                    "reason": "callee_not_self_contained"
                })
                continue

            body_inside = extract_body_block_text(callee.body_full_text)
            if body_inside is None:
                report["skipped"].append({
                    "callee": callee.fqn,
                    "reason": "cannot_extract_body_block"
                })
                continue

            rewritten_body = body_inside
            param_names = extract_param_names(callee.parameters)
            if len(param_names) == len(arg_texts):
                for p, a in zip(param_names, arg_texts):
                    rewritten_body = re.sub(rf"\b{re.escape(p)}\b", f"({a})", rewritten_body)

            block = "{\n" + rewritten_body.rstrip() + "\n}"
            replacements.append(Replacement(
                start=start,
                end=end_stmt,
                new_text=block,
                info={
                    "strategy": "full_body_statement_block",
                    "callee": callee.fqn,
                    "replaced": test_method_src[start:end_stmt],
                    "replacement": block,
                }
            ))

        # Apply replacements from right to left
        replacements.sort(key=lambda r: r.start, reverse=True)
        rewritten = test_method_src
        for r in replacements:
            rewritten = rewritten[:r.start] + r.new_text + rewritten[r.end:]
            report["inlined"].append(r.info)

        report["inlined_count"] = len(replacements)
        report["skipped_count"] = len(report["skipped"])
        return rewritten, report


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_json", help="Path to analyzer JSON")
    ap.add_argument("output_java", help="Path to write rewritten test method source")
    ap.add_argument("output_report", help="Path to write inlining report JSON")
    args = ap.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    test_method_src = data["test_method"]["body"]["full_text"]
    idx_name, idx_class = build_callee_indexes(data)

    engine = CompilableJavaInliner(idx_name, idx_class)
    rewritten, report = engine.inline(test_method_src)

    with open(args.output_java, "w", encoding="utf-8") as f:
        f.write(rewritten)

    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote: {args.output_java}")
    print(f"Wrote: {args.output_report}")
    print(f"Inlined: {report['inlined_count']}, Skipped: {report['skipped_count']}")


if __name__ == "__main__":
    main()
