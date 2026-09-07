from sympy.printing.latex import LatexPrinter
from sympy import Symbol

# Mapping of greek letter names to LaTeX commands
GREEK_LETTERS = {
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "pi",
    "rho",
    "sigma",
    "tau",
    "upsilon",
    "phi",
    "chi",
    "psi",
    "omega",
    "Gamma",
    "Delta",
    "Theta",
    "Lambda",
    "Xi",
    "Pi",
    "Sigma",
    "Upsilon",
    "Phi",
    "Psi",
    "Omega",
}


def _to_latex_name(name: str) -> str:
    """Convert a function name to its LaTeX representation."""
    if name in GREEK_LETTERS:
        return "\\" + name  # e.g. "phi" -> "\phi"
    return name


class CustomLatexPrinter(LatexPrinter):
    def _print_AppliedUndef(self, expr, **kwargs):
        # Return just the function name without (t)
        return self._print(expr.func)  # type: ignore

    def _print_Derivative(self, expr, **kwargs):
        # Handle derivatives like q'(t) -> \dot{q} etc.
        func = expr.args[0]
        order = expr.args[1][1]
        name = _to_latex_name(func.func.__name__)

        if order == 1:
            return r"\dot{" + name + "}"
        elif order == 2:
            return r"\ddot{" + name + "}"
        else:
            return name + "^{(" + str(order) + ")}"


def latex(expr, **kwargs):
    return CustomLatexPrinter(kwargs).doprint(expr)


def print_simple(expr):
    """Return each term on a new line without complex alignment as a string."""
    if not expr.is_Add:
        return f"$${latex(expr)}$$"

    terms = expr.as_ordered_terms()

    result = """$$
\\begin{aligned}"""

    for i, term in enumerate(terms):
        term_latex = latex(term)
        if i == 0:
            result += f"""
    & {term_latex} \\\\"""
        else:
            if term_latex.startswith("-"):
                result += f"""
    & {term_latex} \\\\"""
            else:
                result += f"""
    & + {term_latex} \\\\"""

    result += """
\\end{aligned}
$$"""

    return result
