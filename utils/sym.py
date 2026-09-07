import sympy as sm
import sympy.physics.mechanics as me
from tqdm import tqdm


def get_external_forces():
    """unified source for external forcing symbols: fb1, fb2, fb3,fx, fy, fz, mx, my, mz, torque_gen
    These symbols match exactly the fortran variables used for them in right_hand_side.f90 in dynfode
    """
    # Torques and Forces
    return sm.symbols("Fb1,Fb2,Fb3,Fx,Fy,Fz,Mx,My,Mz,Q_g")


def split_by_symbol(expr, symbol):
    coeff = expr.coeff(symbol)
    return coeff, expr - sm.expand(symbol * coeff)


def smaa(expression, small_angle):
    """Small angle approximation cos and sin"""
    return expression.replace(
        lambda e: e.func == sm.sin and e.args[0] == small_angle, lambda e: e.args[0]
    ).replace(lambda e: e.func == sm.cos and e.args[0] == small_angle, lambda e: 1)


# def smaa(expression, small_angle):
#     """Small angle approximation cos and sin"""
#     return expression.replace(
#         lambda e: e.func == sm.sin and e.args[0] == small_angle, lambda e: e.args[0]
#     ).replace(lambda e: e.func == sm.cos and e.args[0] == small_angle, lambda e: 1)


def dcm_sma(pitch, roll):
    """
    Compute the linearized Direction Cosine Matrix (DCM) for small-angle rotations.
    USES THE RIGHT HAND RULE FOR POSITIVE ANGLES
    Parameters
    ----------
    pitch : sympy.Expr or sympy.Symbol
        Pitch angle (rotation about Y-axis). Small angle assumption: |θ| ≪ 1 rad.
        In wind turbine applications, this often represents fore-aft tower
        deflection or blade flapwise motion.

    roll : sympy.Expr or sympy.Symbol
        Roll angle (rotation about X-axis). Small angle assumption: |φ| ≪ 1 rad.
        In wind turbine applications, this often represents side-to-side tower
        deflection or blade edgewise motion.

    Returns
    -------
    sympy.Matrix
        3x3 linearized direction cosine matrix (DCM) with structure:

            [[1,   0,   -pitch],
             [0,   1,   roll],
             [pitch, -roll, 1]]
    """
    return sm.Matrix([[1, 0, -pitch], [0, 1, roll], [pitch, -roll, 1]])


def dcm_sma_2nd_order(pitch, roll):
    """
    Second-order linearized DCM for small pitch and roll (explicit form).

    This version explicitly writes the second-order terms without matrix
    multiplication, making it more efficient and easier to inspect.

    Parameters
    ----------
    pitch : sympy.Expr
        Pitch angle (rotation about Y-axis)
    roll : sympy.Expr
        Roll angle (rotation about X-axis)

    Returns
    -------
    sympy.Matrix
        3x3 second-order DCM

    Notes
    -----
    The explicit matrix elements are:

        R[0,0] = 1 - θ²/2
        R[0,1] = 0
        R[0,2] = θ

        R[0,1] = θφ
        R[1,1] = 1 - φ²/2
        R[1,2] = -φ

        R[2,0] = -θ
        R[2,1] = φ
        R[2,2] = 1 - (θ² + φ²)/2

    The term R[0,1] = θφ represents the coupling between pitch and roll
    that appears at second order. This coupling is important for:
        - Blade flap-lag coupling in rotating frames
        - Tower side-to-fore-aft coupling
        - Gyroscopic effects
    """
    # Pre-compute common terms
    pitch2 = pitch**2
    roll2 = roll**2
    pitch_roll = pitch * roll

    # Construct DCM explicitly
    R = sm.Matrix(
        [
            [1 - pitch2 / 2, pitch_roll, -pitch],
            [0, 1 - roll2 / 2, roll],
            [pitch, -roll, 1 - (pitch2 + roll2) / 2],
        ]
    )
    return R


def truncate_degree(expr, vars, max_degree):
    """
    Truncate a SymPy expression to a given total degree in specified variables.

    Parameters
    ----------
    expr : sympy.Expr
        Expression to truncate
    vars : list of sympy.Symbol
        Variables to consider for degree (e.g., [nu_x, nu_y])
    max_degree : int
        Maximum total degree to keep

    Returns
    -------
    sympy.Expr
        Truncated expression
    """
    expr = sm.expand(expr)
    terms = expr.as_ordered_terms()

    kept_terms = []

    for term in terms:
        powers = term.as_powers_dict()

        # Compute total degree in specified variables
        degree = sum(powers.get(v, 0) for v in vars)

        if degree <= max_degree:
            kept_terms.append(term)

    return sm.Add(*kept_terms)


def truncate_matrix(M, vars, max_degree):
    """
    Apply degree truncation elementwise to a SymPy matrix (immutable-safe).

    Parameters
    ----------
    M : sympy.MatrixBase (including ImmutableDenseMatrix)
    vars : list of sympy.Symbol
    max_degree : int

    Returns
    -------
    sympy.ImmutableDenseMatrix
        New truncated matrix
    """
    return sm.ImmutableDenseMatrix(
        M.rows,
        M.cols,
        [truncate_degree(M[i, j], vars, max_degree) for i in range(M.rows) for j in range(M.cols)],
    )


def simplify_factor(expr, sym):
    coeff = sm.collect(expr, sym).coeff(sym)
    simplified_coeff = sm.simplify(coeff)
    return expr.subs(sym * coeff, sym * simplified_coeff)


def simplify_factors(expr, syms):
    for s in syms:
        expr = simplify_factor(expr, s)
    return expr


def custom_simplify(expr, syms, trucate_syms=[], deg=2):
    d = {}
    rest = truncate_degree(expr, trucate_syms, deg)
    for s in syms:
        coef, rest = split_by_symbol(rest, s)
        d[s] = sm.simplify(coef)

    # return sm.Add(*(k * v for k, v in d.items())) + sm.simplify(rest)
    return sm.Add(*(k * v for k, v in d.items())) + rest  # type: ignore


def custom_simplify_matrix(m, syms, trucate_syms=[], deg=2):
    total = m.rows * m.cols
    return sm.Matrix(
        m.rows,
        m.cols,
        [
            custom_simplify(m[i, j], syms, trucate_syms, deg)
            for i, j in tqdm(
                [(i, j) for i in range(m.rows) for j in range(m.cols)],
                total=total,
                desc="Simplifying matrix",
            )
        ],
    )


def flat(nested_list):
    """Flattens a list of lists into a single list."""
    return [item for sublist in nested_list for item in sublist]
