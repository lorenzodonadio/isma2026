import numpy as np


def abm4_integrate(mfunc, ffunc, t_span, y0, dt, ext_func=None):
    """
    Adams-Bashforth-Moulton 4th-order predictor-corrector integrator.

    Solves the second-order system:

        M(q, u, ext) * u' = F(q, u, ext)

    where the state is y = [q_1, ..., q_n, u_1, ..., u_n] and n is inferred
    automatically from y0.  Works for any number of generalized coordinates.

    Parameters
    ----------
    mfunc : callable
        Mass matrix function.  Called as  mfunc(*q, *u, *ext) -> array (n, n).
    ffunc : callable
        Generalised force function.  Called as  ffunc(*q, *u, *ext) -> array (n,).
    t_span : (float, float)
        (t_start, t_end).
    y0 : array_like, shape (2n,)
        Initial state [q_1,...,q_n, u_1,...,u_n].
    dt : float
        Fixed time step.
    ext_func : callable or None
        External inputs as a function of time:  ext_func(t) -> scalar or array.
        If None, no external inputs are passed to mfunc / ffunc.
        If provided, the return value is unpacked as *ext in the function calls.

    Returns
    -------
    t : ndarray, shape (N,)
        Time vector.
    Y : ndarray, shape (N, 2n)
        State history.  Columns are [q_1,...,q_n, u_1,...,u_n].

    Notes
    -----
    The startup phase uses RK4 for the first three steps so that the four
    history points required by the ABM4 predictor are available.

    The Adams-Bashforth predictor (explicit, 4th order):
        y_{n+1}^p = y_n + dt/24 * (55 f_n - 59 f_{n-1} + 37 f_{n-2} - 9 f_{n-3})

    The Adams-Moulton corrector (implicit, 4th order):
        y_{n+1}   = y_n + dt/24 * (9 f_{n+1}^p + 19 f_n - 5 f_{n-1} + f_{n-2})

    Both have local truncation error O(dt^5), giving global error O(dt^4).
    """
    y0 = np.asarray(y0, dtype=float)
    n = len(y0) // 2  # number of generalised coordinates
    assert len(y0) == 2 * n, "y0 must have even length: [q_1,...,q_n, u_1,...,u_n]"

    t0, tf = t_span
    t_arr = np.arange(t0, tf + dt, dt)
    N = len(t_arr)

    # ── external input helper ─────────────────────────────────────────────────
    if ext_func is None:

        def get_ext(ti):  # type: ignore
            return ()  # empty tuple — no extra args

    else:

        def get_ext(ti):
            val = ext_func(ti)
            # scalar → one-element tuple; array → unpacked tuple
            if np.ndim(val) == 0:
                return (float(val),)
            return np.asarray(val, dtype=float)

    # ── state derivative  dy/dt = [u, M^{-1} F] ──────────────────────────────
    def derivatives(y, ti):
        q = y[:n]
        u = y[n:]
        ext = get_ext(ti)

        M = np.asarray(mfunc(*q, *u), dtype=float).reshape(n, n)
        F = np.asarray(ffunc(*q, *u, *ext), dtype=float).flatten()

        u_dot = np.linalg.solve(M, F)  # M * u' = F  →  u' = M^{-1} F
        return np.concatenate([u, u_dot])  # dy/dt = [u, u']

    # ── storage ───────────────────────────────────────────────────────────────
    Y = np.zeros((N, 2 * n))
    Y[0] = y0

    # ── RK4 startup (first 3 steps) ───────────────────────────────────────────
    def rk4_step(y, ti):
        k1 = derivatives(y, ti)
        k2 = derivatives(y + dt / 2 * k1, ti + dt / 2)
        k3 = derivatives(y + dt / 2 * k2, ti + dt / 2)
        k4 = derivatives(y + dt * k3, ti + dt)
        return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    for i in range(min(3, N - 1)):
        Y[i + 1] = rk4_step(Y[i], t_arr[i])

    # ── initialise derivative history ring buffer (length 4) ─────────────────
    # D[0] = oldest (n-3), D[3] = newest (n)
    history_len = min(4, N)
    D = [derivatives(Y[i], t_arr[i]) for i in range(history_len)]

    # ── ABM4 main loop ────────────────────────────────────────────────────────
    for i in range(3, N - 1):
        # Adams-Bashforth predictor
        y_pred = Y[i] + dt / 24 * (55 * D[3] - 59 * D[2] + 37 * D[1] - 9 * D[0])
        d_pred = derivatives(y_pred, t_arr[i + 1])

        # Adams-Moulton corrector
        Y[i + 1] = Y[i] + dt / 24 * (9 * d_pred + 19 * D[3] - 5 * D[2] + D[1])

        # Slide derivative window: drop oldest, append corrected derivative
        D = [D[1], D[2], D[3], derivatives(Y[i + 1], t_arr[i + 1])]

    return t_arr, Y


# def abm4_integrate(mfunc, ffunc, t_span, y0, dt, F_a_func=None):
#     """
#     Adams-Bashforth-Moulton 4th order predictor-corrector integration
#     for M(q)*q'' = F(q, q', t)

#     State vector y = [q1, q2, u1, u2]
#     where ui = qi_dot

#     mfunc, ffunc: lambdified with positional args (q1, q2, u1, u2, F_a)
#     y0: initial state [q1, q2, u1, u2]
#     """
#     t0, tf = t_span
#     t = np.arange(t0, tf + dt, dt)
#     N = len(t)

#     # External forcing defaults to zero if not provided
#     if F_a_func is None:
#         F_a_func = lambda t: 0.0

#     def get_externals(ti):
#         return F_a_func(ti)

#     def derivatives(y, ti):
#         q1, q2, u1, u2 = y
#         F_a = get_externals(ti)
#         args = (q1, q2, u1, u2, F_a)

#         M = np.array(mfunc(*args), dtype=float)
#         F = np.array(ffunc(*args), dtype=float).flatten()

#         q_ddot = np.linalg.solve(M, F)

#         # dy/dt = [u1, u2, u3, q1_ddot, q2_ddot, q3_ddot]
#         # return np.array([u1, u2, u3, q_ddot[0], q_ddot[1], q_ddot[2]])
#         return np.array([u1, u2, q_ddot[0], q_ddot[1]])

#     # Storage
#     Y = np.zeros((N, 4))
#     Y[0] = y0

#     # --- Startup: RK4 for first 3 steps ---
#     def rk4_step(y, ti, dt):
#         k1 = derivatives(y, ti)
#         k2 = derivatives(y + dt / 2 * k1, ti + dt / 2)
#         k3 = derivatives(y + dt / 2 * k2, ti + dt / 2)
#         k4 = derivatives(y + dt * k3, ti + dt)
#         return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

#     for i in range(3):
#         Y[i + 1] = rk4_step(Y[i], t[i], dt)

#     # Precompute derivatives for the first 4 points
#     D = [derivatives(Y[i], t[i]) for i in range(4)]

#     # --- ABM4 main loop ---
#     for i in range(3, N - 1):
#         # Adams-Bashforth predictor (4th order)
#         y_pred = Y[i] + dt / 24 * (55 * D[3] - 59 * D[2] + 37 * D[1] - 9 * D[0])

#         # Derivative at predicted state
#         d_pred = derivatives(y_pred, t[i + 1])

#         # Adams-Moulton corrector (4th order)
#         Y[i + 1] = Y[i] + dt / 24 * (9 * d_pred + 19 * D[3] - 5 * D[2] + D[1])

#         # Slide the derivative window
#         D = [D[1], D[2], D[3], derivatives(Y[i + 1], t[i + 1])]

#     return t, Y
