"""
ModalFlexBody.py — Rayleigh-Ritz flexible body for SymPy Mechanics (Kane's method)

Theory
------
The displacement field of a flexible body is approximated as a finite linear
combination of assumed shape (basis) functions φ_j(x) weighted by time-varying
generalized coordinates q_j(t):

    u(x, t) = Σ_j  φ_j(x) · q_j(t)

For a tower bending in the x-direction the rotation at the top (used to orient
the nacelle frame) is:

    θ_x = Σ_j  r_{x,j} · q_j

where r_{x,j} = dφ_j/ds|_{tip} is the participation factor of mode j at the
body tip (or at whatever cross-section the point of interest sits).

For a two-mode expansion u(z,t) = q₁φ₁(z) + q₂φ₂(z) the continuous kinetic
energy is:

    T = ½ ∫₀ᴴ ρ ẋ² dz
      = ½ ∫₀ᴴ ρ (u̇₁φ₁ + u̇₂φ₂)² dz
      = ½ M₁₁ u̇₁² + M₁₂ u̇₁u̇₂ + ½ M₂₂ u̇₂²

where the modal mass matrix entries are:

    M₁₁ = ∫₀ᴴ ρ φ₁² dz          (diagonal — mode 1 alone)
    M₂₂ = ∫₀ᴴ ρ φ₂² dz          (diagonal — mode 2 alone)
    M₁₂ = ∫₀ᴴ ρ φ₁ φ₂ dz        (off-diagonal coupling term)

This generalises to N modes as the symmetric modal mass matrix M ∈ ℝᴺˣᴺ.

Lumped-Mass (Punctual) Representation of the Cross Terms
---------------------------------------------------------

Implementation Summary
----------------------
1. Each mode j is a diagonal Particle (mass M_jj − cross corrections) at
   position x_cm + q_j · ê_dir.
2. Spring forces −k_j q_j and damper forces −c_j u_j are added to self.loads
   for each mode in each active direction.

The body can flex in up to two of the three Cartesian directions. Each
direction is represented by an independent set of Ritz modes; modes in
*different* directions are kinematically decoupled (independent ê vectors
give orthogonal velocities).  Modes within the *same* direction are coupled
through the off-diagonal mass terms described above.

When processed by KanesMethod, the combination of particle kinematics and
the loads in self.loads yields the correct modal equations of motion:

    Σ_k M_{jk} q̈_k  +  c_j q̇_j  +  k_j q_j  =  Q_j

where Q_j is the generalised active force arising from external loads
projected onto mode j.
"""

import sympy as sm
import sympy.physics.mechanics as me


class ModalFlexBody:
    """
    Flexible body using lumped-mass modal representation (Rayleigh-Ritz).

    The continuous kinetic energy

        T = ½ ∫ ρ ẋ² dz = ½ uᵀ M u

    is reproduced exactly by a set of lumped-mass particles:

    * **Diagonal particles** — one per mode j, carrying the diagonal modal mass
      reduced
      Each particle sits at  x_cm + q_j · ê.

    Elastic (spring) and dissipative (damper) loads are added to ``self.loads``
    and are passed to KanesMethod together with any external forces.

    Parameters
    ----------
    ref : me.ReferenceFrame
        The parent (undeformed) reference frame.
    O : me.Point
        Origin point from which all particle positions are measured.
    name : str
        Human-readable name (e.g. "tower").
    xcm, ycm, zcm : scalar or sympy expression
        Nominal (undeflected) position of the body's geometric centroid
        expressed in ``ref``.
    dof_x, dof_y, dof_z : int
        Number of Ritz modes (degrees of freedom) in each direction.
        At most two directions may be nonzero.
    prefix : str or None
        Short label used in symbol names (defaults to first character of name).
    var_start : int
        Starting index for generated q / u symbols, e.g. var_start=3 gives
        q3, q4, … .  Useful when combining several ModalFlexBody instances.
    elastic_forces : bool
        Whether to add −k·q spring forces automatically.
    damp_forces : bool
        Whether to add −c·u viscous damping forces automatically.
    sym_xy : bool
        When True the body is assumed symmetric in x and y, so the same mass,
        stiffness, damping, and participation-factor symbols are shared between
        both bending directions.  Requires dof_x == dof_y.
    rx, ry, rz : sequence of sympy symbols or None
        Tip-rotation participation factors for each mode in x / y / z.
        Auto-generated if not provided.
    mx, my, kx, ky, cx, cy : sequences of sympy symbols or None
        Diagonal modal masses, stiffnesses, and damping coefficients.
        Auto-generated if not provided.

    Key Attributes
    --------------
    q : tuple of dynamicsymbols
        Generalised coordinates ordered as [x-modes, y-modes, z-modes].
    u : tuple of dynamicsymbols
        Corresponding generalised speeds (q̇_j = u_j enforced via kd_eqs).
    theta_x, theta_y, theta_z : sympy expression
        Tip rotation from all modes in that direction:  θ = Σ r_j q_j.
        Only created when the corresponding dof count is > 0.
    loads : list of (Point, Vector) tuples
        Elastic and damping force contributions.  Append external forces here
        before passing to KanesMethod.
    free_symbols : set of sm.Symbol
        All symbolic parameters (m, k, c, r, ) that must be
        substituted before numerical integration.
    P_x, P_y, P_z : tuples of me.Particle
        Diagonal Ritz particles for each direction.
    """

    def __init__(
        self,
        ref: me.ReferenceFrame,
        O: me.Point,
        name: str,
        xcm=0,
        ycm=0,
        zcm=0,
        dof_x=0,
        dof_y=0,
        dof_z=0,
        prefix=None,
        var_start=1,
        elastic_forces=True,
        damp_forces=True,
        sym_xy=False,
        rx=None,
        ry=None,
        rz=None,
        mx=None,
        my=None,
        kx=None,
        ky=None,
        cx=None,
        cy=None,
    ):

        self.name = name
        self.prefix = name[:1].upper() if prefix is None else prefix
        self.dof_x = dof_x
        self.dof_y = dof_y
        self.dof_z = dof_z
        self.var_start = var_start
        self._is_xflex = dof_x > 0
        self._is_yflex = dof_y > 0
        self._is_zflex = dof_z > 0
        self.O = O
        self.ref = ref
        self.loads = []
        # At most two directions may carry DOFs (three would over-constrain the
        # rigid-body translation of the body's reference point).
        assert (
            self._is_xflex + self._is_yflex + self._is_zflex < 3
        ), "Maximum of 2 coordinates can have DOFs"

        # ------------------------------------------------------------------ #
        # Generalised coordinates q_i and speeds u_i (one per active mode).   #
        # ------------------------------------------------------------------ #
        final_q_num = dof_x + dof_y + dof_z + var_start
        self.q = me.dynamicsymbols(f"q{var_start}:{final_q_num}")
        self.u = me.dynamicsymbols(f"u{var_start}:{final_q_num}")

        # ------------------------------------------------------------------ #
        # Modal mass / stiffness / damping / participation symbols.           #
        # When sym_xy=True the same symbols are reused for x and y so that    #
        # the symmetric structure of the body is reflected in the equations.  #
        # ------------------------------------------------------------------ #
        if sym_xy:
            assert dof_x == dof_y, "for sym_xy dofx must equal dofy"
            # Share a single set of symbols across both bending directions.
            _m = sm.symbols(f"m_{self.prefix}_1:{self.dof_x+1}")
            _k = sm.symbols(f"k_{self.prefix}_1:{self.dof_x+1}")
            _c = sm.symbols(f"c_{self.prefix}_1:{self.dof_x+1}")
            _r = rx if rx else sm.symbols(f"r_{self.prefix}_1:{self.dof_x+1}")

            self._mx, self._my = _m, _m
            self._kx, self._ky = _k, _k
            self._cx, self._cy = _c, _c
            self._rx, self._ry = _r, _r
        else:
            # Independent symbols for each direction.
            self._mx = mx if mx else sm.symbols(f"m_x{self.prefix}_1:{self.dof_x+1}")
            self._my = my if my else sm.symbols(f"m_y{self.prefix}_1:{self.dof_y+1}")
            self._kx = kx if kx else sm.symbols(f"k_x{self.prefix}_1:{self.dof_x+1}")
            self._ky = ky if ky else sm.symbols(f"k_y{self.prefix}_1:{self.dof_y+1}")
            self._cx = cx if cx else sm.symbols(f"c_x{self.prefix}_1:{self.dof_x+1}")
            self._cy = cy if cy else sm.symbols(f"c_y{self.prefix}_1:{self.dof_y+1}")

            self._rx = rx if rx else sm.symbols(f"r_x{self.prefix}_1:{self.dof_x+1}")
            self._ry = ry if ry else sm.symbols(f"r_y{self.prefix}_1:{self.dof_y+1}")

        # z-direction always uses independent symbols.
        self._mz = sm.symbols(f"m_z{self.prefix}_1:{self.dof_z+1}")
        self._kz = sm.symbols(f"k_z{self.prefix}_1:{self.dof_z+1}")
        self._cz = sm.symbols(f"c_z{self.prefix}_1:{self.dof_z+1}")
        self._rz = rz if rz else sm.symbols(f"r_z{self.prefix}_1:{self.dof_z+1}")

        # ------------------------------------------------------------------ #
        # Tip-rotation expressions θ = Σ_j r_j q_j (one per active direction).
        # ------------------------------------------------------------------ #
        if self._is_xflex:
            self.theta_x = sm.Matrix(self._qx()).dot(sm.Matrix(self._rx))
        if self._is_yflex:
            self.theta_y = sm.Matrix(self._qy()).dot(sm.Matrix(self._ry))
        if self._is_zflex:
            self.theta_z = sm.Matrix(self._qz()).dot(sm.Matrix(self._rz))

        # ------------------------------------------------------------------ #
        # Off-diagonal modal mass matrices M_{ij} (i < j).                   #
        # These encode the cross-coupling  M_{ij} = ∫ρ φ_i φ_j dz  between  #
        # different modes in the same bending direction.                      #
        # ------------------------------------------------------------------ #

        # Convenience tuples collecting all modal parameters across directions.
        self.m = self._mx + self._my + self._mz
        self.k = self._kx + self._ky + self._kz
        self.c = self._cx + self._cy + self._cz

        # Place all particles (diagonal + cross) at their nominal positions.
        self.locate(xcm, ycm, zcm)

        # Add elastic spring loads −k_j q_j to self.loads.
        if elastic_forces:
            self.add_elastic_loads()
        # Add viscous damper loads −c_j u_j to self.loads.
        if damp_forces:
            self.add_damp_loads()

        self.mass_symbols = set(self.m)
        # Collect every free parameter for downstream substitution.
        self._calc_free_syms()

    def _calc_free_syms(self):
        self.free_symbols = (
            self.mass_symbols | set(self.k) | set(self.c) | set(self._rx + self._ry + self._rz)
        )

    def locate(self, x, y, z):
        """
        Create all particles and place them at their nominal positions.

        Diagonal particles
        ------------------
        Mode j in direction ê is represented by a particle at
            p_j = O + x_cm·ê_x + y_cm·ê_y + z_cm·ê_z + q_j·ê
        with mass
            m_jj_eff = M_jj
        Parameters
        ----------
        x, y, z : scalar or sympy expression
            Nominal centroid offset from O in the ``ref`` frame.
        """

        cm_v = x * self.ref.x + y * self.ref.y + z * self.ref.z
        self.masscenter = me.Point(f"cm_{self.prefix}")
        self.masscenter.set_pos(self.O, cm_v)
        R = self.ref

        # Nominal positions of the diagonal particles (displaced by q_j only).
        _op_x = [
            self.O.locatenew(f"op_x{self.prefix}_{i+1}", cm_v + q * R.x)
            for i, q in enumerate(self._qx())
        ]
        _op_y = [
            self.O.locatenew(f"op_y{self.prefix}_{i+1}", cm_v + q * R.y)
            for i, q in enumerate(self._qy())
        ]
        _op_z = [
            self.O.locatenew(f"op_z{self.prefix}_{i+1}", cm_v + q * R.z)
            for i, q in enumerate(self._qz())
        ]

        pn = lambda n, i: f"P{self.prefix}{n}_{i+1}"

        # ------------------------------------------------------------------ #
        # Diagonal particles.                                                  #
        # Effective mass = M_jj
        # ------------------------------------------------------------------ #

        self.P_x = tuple(
            me.Particle(pn("x", i), op, m) for i, (op, m) in enumerate(zip(_op_x, self._mx))
        )
        self.P_y = tuple(
            me.Particle(pn("y", i), op, m) for i, (op, m) in enumerate(zip(_op_y, self._my))
        )
        self.P_z = tuple(
            me.Particle(pn("z", i), op, m) for i, (op, m) in enumerate(zip(_op_z, self._mz))
        )

    def add_elastic_loads(self):
        """
        Append linear spring (restoring) forces to ``self.loads``.

        For each mode j in direction ê the load is  F_j = −k_j q_j ê,
        applied at the diagonal particle's location.  Together with the
        particle kinematics this produces the stiffness term k_j q_j on the
        right-hand side of the modal EOM.
        """
        for i, P in enumerate(self.P_x):
            q = self._qx()
            self.loads.append((P.masscenter, -self._kx[i] * q[i] * self.ref.x))
        for i, P in enumerate(self.P_y):
            q = self._qy()
            self.loads.append((P.masscenter, -self._ky[i] * q[i] * self.ref.y))
        for i, P in enumerate(self.P_z):
            q = self._qz()
            self.loads.append((P.masscenter, -self._kz[i] * q[i] * self.ref.z))

    def add_damp_loads(self):
        """
        Append linear viscous damping forces to ``self.loads``.

        For each mode j in direction ê the load is  F_j = −c_j u_j ê,
        applied at the diagonal particle's location.  This contributes the
        damping term c_j q̇_j to the modal EOM (non-conservative force in
        Kane's formulation).
        """
        for i, P in enumerate(self.P_x):
            u = self._ux()
            self.loads.append((P.masscenter, -self._cx[i] * u[i] * self.ref.x))
        for i, P in enumerate(self.P_y):
            u = self._uy()
            self.loads.append((P.masscenter, -self._cy[i] * u[i] * self.ref.y))
        for i, P in enumerate(self.P_z):
            u = self._uz()
            self.loads.append((P.masscenter, -self._cz[i] * u[i] * self.ref.z))

    def get_bodies(self):
        """
        Return a flat list of all particles for use in
        KanesMethod.  The list contains P_x, P_y, P_z particles
        """
        return [*self.P_x, *self.P_y, *self.P_z]

    def get_subs_dict(self, value=0.0):
        """
        Return a substitution dictionary mapping every free symbol to
        ``value`` (default 0.0), together with a symbol-name map.

        Useful for quick numerical testing or for zeroing out unused
        parameters.

        Returns
        -------
        subs : dict  {symbol: value}
        sym_map : dict  {str(symbol): symbol}
        """
        return {sym: value for sym in self.free_symbols}, self.get_symbol_map()

    def get_symbol_map(self):
        """
        Return a dict mapping symbol names (strings) to symbol objects.

        Convenient when the caller needs to look up a symbol by its printed
        name, e.g. sym_map["m_xT_1"] to retrieve the first x-direction modal
        mass symbol of a tower named "T".
        """
        self._calc_free_syms()
        return {str(sym): sym for sym in self.free_symbols}

    # ---------------------------------------------------------------------- #
    # Coordinate/speed slice helpers                                           #
    # ---------------------------------------------------------------------- #

    def _qx(self):
        """Generalised coordinates for x-direction modes."""
        return self.q[: self.dof_x]  # type: ignore

    def _qy(self):
        """Generalised coordinates for y-direction modes."""
        return self.q[self.dof_x : self.dof_y + self.dof_x]  # type: ignore

    def _qz(self):
        """Generalised coordinates for z-direction modes."""
        return self.q[self.dof_y + self.dof_x : self.dof_y + self.dof_x + self.dof_z]  # type: ignore

    def _ux(self):
        """Generalised speeds for x-direction modes."""
        return self.u[: self.dof_x]  # type: ignore

    def _uy(self):
        """Generalised speeds for y-direction modes."""
        return self.u[self.dof_x : self.dof_y + self.dof_x]  # type: ignore

    def _uz(self):
        """Generalised speeds for z-direction modes."""
        return self.u[self.dof_y + self.dof_x : self.dof_y + self.dof_x + self.dof_z]  # type: ignore

    def remove_modal_params_from_non_modal_coords(
        self, mass_mat, force_vect, coordinates, is_full=True, extra_qs=[]
    ):
        modalcordidx = []

        assert mass_mat.shape[0] == mass_mat.shape[1], "mass matrix must be squared"
        assert mass_mat.shape[0] == force_vect.shape[0], "incompatible mass and force dimensions"
        if is_full:
            assert mass_mat.shape[0] % 2 == 0, "is_full needs 2xdof x 2xdof matrix"
            o = int(mass_mat.shape[0] / 2)
        else:
            o = 0

        for i, coord in enumerate(coordinates):
            for q in self.q + extra_qs:  # type: ignore
                if q == coord:
                    modalcordidx.append(i + o)
                    continue

        Msimp = sm.Matrix(mass_mat)
        Fsimp = sm.Matrix(force_vect)
        sbs_modal_mass_zero = {m: 0 for m in self.mass_symbols}
        modalkc = self._cx + self._cy + self._cz + self._kx + self._ky + self._kz
        sbsmodalkczero = {kc: 0 for kc in modalkc}
        for i in range(Fsimp.shape[0]):
            if i not in modalcordidx:
                Fsimp[i] = Fsimp[i].subs(sbs_modal_mass_zero)  # type: ignore
                Fsimp[i] = Fsimp[i].subs(sbsmodalkczero)  # type: ignore

        for i in range(mass_mat.shape[0]):
            for j in range(mass_mat.shape[1]):
                if i not in modalcordidx or j not in modalcordidx:
                    Msimp[i, j] = Msimp[i, j].subs(sbs_modal_mass_zero)  # type: ignore

        return Msimp, Fsimp

    def _add_cross_masses(self, mass_mat: sm.Matrix, coords, qs, name="m", is_full=True):

        if len(qs) == 0 or len(coords) == 0:
            return mass_mat
        assert mass_mat.shape[0] == mass_mat.shape[1], "mass matrix must be squared"
        if is_full:
            assert mass_mat.shape[0] % 2 == 0, "is_full needs 2xdof x 2xdof matrix"

        o = int(mass_mat.shape[0] / 2) if is_full else 0  # OFFSET FOR THE full matrix
        added_masses = []
        for imass, c1 in enumerate(coords):
            for jmass, c2 in enumerate(coords):
                if imass == jmass:
                    continue
                if c1 not in qs or c2 not in qs:
                    continue
                i, j = qs.index(c1) + 1, qs.index(c2) + 1
                mname = f"{name}_{i}{j}" if j > i else f"{name}_{j}{i}"
                mcross = sm.symbols(mname)
                added_masses.append(mcross)
                mass_mat[imass + o, jmass + o] += mcross

        self.mass_symbols = self.mass_symbols | set(added_masses)
        self._calc_free_syms()

        return mass_mat

    def add_all_cross_masses(
        self, mass_mat: sm.Matrix, coords, mx=None, my=None, mz=None, is_full=True
    ):
        mx = f"m_x{self.prefix}" if mx is None else mx
        my = f"m_y{self.prefix}" if my is None else my
        mz = f"m_z{self.prefix}" if mz is None else mz

        mass_mat = self._add_cross_masses(mass_mat, coords, self._qx(), mx, is_full)
        mass_mat = self._add_cross_masses(mass_mat, coords, self._qy(), my, is_full)
        mass_mat = self._add_cross_masses(mass_mat, coords, self._qz(), mz, is_full)
        return mass_mat


def add_modal_mass(mass_mat: sm.Matrix, sym, coord1, coord2, coords: list, is_full=True):
    assert mass_mat.shape[0] == mass_mat.shape[1], "mass matrix must be squared"
    if is_full:
        assert mass_mat.shape[0] % 2 == 0, "is_full needs 2xdof x 2xdof matrix"
    o = int(mass_mat.shape[0] / 2) if is_full else 0  # OFFSET FOR THE full matrix

    i = coords.index(coord1) + o
    j = coords.index(coord2) + o
    mass_mat[i, j] += sym
    mass_mat[j, i] += sym
