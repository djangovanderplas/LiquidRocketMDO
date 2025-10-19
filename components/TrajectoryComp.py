import numpy as np
import openmdao.api as om

from components.PropulsionComp import thrust_from_ambient_isp, CEA

# Atmosphere: use Ambiance if available, else ISA fallback
_USE_AMBIANCE = True
try:
    from ambiance import Atmosphere as _AmbAtmos
except Exception:
    _USE_AMBIANCE = False


class TrajectoryComp(om.ExplicitComponent):
    """
    1D vertical ascent with RK4, using a precomputed Isp_amb(h) table
    to avoid per-step RocketCEA calls. Thrust(h) = mdot * g0 * Isp_amb(h).
    """

    # Earth / gas constants
    _R_E = 6371000.0
    _G0 = 9.80665
    _R_air = 287.05287
    _GAM = 1.4

    def initialize(self):
        # Cache for precomputed tables keyed by (Pc, MR, eps, At, n_table, h_table_max, atmosphere_mode)
        self._cea_table_cache = {}

    def setup(self):
        # ---- Inputs from Propulsion/IVC ----
        self.add_input('At', val=0.0, units='m**2', desc='Throat area')
        self.add_input('Ae', val=0.0, units='m**2', desc='Exit area (not used, but available)')
        self.add_input('Pc', val=40.0, units='bar', desc='Chamber pressure')
        self.add_input('MR', val=3.0, desc='Mixture ratio O/F')
        self.add_input('eps', val=5.0, desc='Expansion ratio')

        self.add_input('prop_mass', val=0.0, units='kg', desc='Usable propellant mass')
        self.add_input('burn_time', val=0.0, units='s', desc='Burn duration')

        self.add_input('dry_mass', val=20.0, units='kg')
        self.add_input('Cd', val=0.5)
        self.add_input('area_ref', val=0.01, units='m**2')
        self.add_input('h0', val=0.0, units='m')
        self.add_input('v0', val=0.0, units='m/s')
        self.add_input('dt', val=0.01, units='s')

        # Table controls
        self.add_input('h_table_max', val=50000.0, units='m', desc='Top altitude for Isp table')
        self.add_input('n_table', val=80.0, desc='Number of altitude samples for Isp table')

        # ---- Outputs ----
        self.add_output('apogee', val=0.0, units='m')
        self.add_output('t_apogee', val=0.0, units='s')
        self.add_output('burnout_alt', val=0.0, units='m')
        self.add_output('burnout_vel', val=0.0, units='m/s')
        self.add_output('max_q', val=0.0, units='Pa')
        self.add_output('t_max_q', val=0.0, units='s')

        # Optional: how close apogee got to the table ceiling (0..1). If near 1, increase h_table_max.
        self.add_output('apogee_frac_of_table', val=0.0)

        # FD totals at model level
        self.declare_partials(of=['apogee','max_q','t_apogee'], wrt=['MR','eps','Pc'], method='fd')

    # -------- Atmosphere --------
    @staticmethod
    def _atm(h):
        """Return (rho [kg/m3], p [Pa], a [m/s], T [K])"""
        if _USE_AMBIANCE:
            at = _AmbAtmos(float(max(0.0, h)))
            return float(at.density), float(at.pressure), float(at.speed_of_sound), float(at.temperature)
        # Fallback ISA (0–11 km lapse, 11–20 km isothermal)
        h = max(0.0, float(h))
        g0 = TrajectoryComp._G0
        R = TrajectoryComp._R_air
        T0, p0 = 288.15, 101325.0
        if h <= 11000.0:
            L = -0.0065
            T = T0 + L * h
            p = p0 * (T / T0) ** (-g0 / (L * R))
            rho = p / (R * T)
        else:
            T11 = 216.65
            p11 = p0 * (T11 / T0) ** (-g0 / (-0.0065 * R))
            p = p11 * np.exp(-(g0 / (R * T11)) * (h - 11000.0))
            rho = p / (R * T11)
            T = T11
        a = (TrajectoryComp._GAM * R * T) ** 0.5
        return rho, p, a, T

    @staticmethod
    def _g(h):
        Re = TrajectoryComp._R_E
        return TrajectoryComp._G0 * (Re / (Re + max(0.0, float(h)))) ** 2

    # -------- RK4 kernel --------
    @staticmethod
    def _rk4_step(h, v, m, dt, acc_func):
        a1, md1 = acc_func(h, v, m)
        k1h, k1v, k1m = v, a1, -md1

        a2, md2 = acc_func(h + 0.5*dt*k1h, v + 0.5*dt*k1v, max(1e-9, m + 0.5*dt*k1m))
        k2h, k2v, k2m = v + 0.5*dt*k1v, a2, -md2

        a3, md3 = acc_func(h + 0.5*dt*k2h, v + 0.5*dt*k2v, max(1e-9, m + 0.5*dt*k2m))
        k3h, k3v, k3m = v + 0.5*dt*k2v, a3, -md3

        a4, md4 = acc_func(h + dt*k3h, v + dt*k3v, max(1e-9, m + dt*k3m))
        k4h, k4v, k4m = v + dt*k3v, a4, -md4

        h_next = h + (dt/6.0) * (k1h + 2*k2h + 2*k3h + k4h)
        v_next = v + (dt/6.0) * (k1v + 2*k2v + 2*k3v + k4v)
        m_next = max(1e-9, m + (dt/6.0) * (k1m + 2*k2m + 2*k3m + k4m))
        return h_next, v_next, m_next

    # -------- Table builder & cache --------
    def _get_isp_table(self, Pc_bar, MR, eps, At, n_table, h_table_max):
        """
        Build or fetch from cache an Isp_amb(h) table and return (h_grid, Isp_grid).
        """
        print("Building Isp Table....")
        # Key uses rounded values so tiny numerical changes don't bust the cache
        key = (round(Pc_bar, 6), round(MR, 6), round(eps, 6), round(At, 9),
               int(n_table), round(h_table_max, 3), 'amb' if _USE_AMBIANCE else 'isa')
        if key in self._cea_table_cache:
            return self._cea_table_cache[key]

        # Build altitude grid (linear in h; you could switch to log(p) sampling later)
        n = max(2, int(n_table))
        h_grid = np.linspace(0.0, float(h_table_max), n)
        Isp_grid = np.zeros(n, dtype=float)

        # Precompute once: c* (for mdot) is not needed here, thrust uses mdot outside
        # We only need Isp_amb(h) from CEA at various Pamb.
        for i, h in enumerate(h_grid):
            _, pamb_pa, _, _ = self._atm(h)
            pamb_bar = pamb_pa / 1.0e5
            IspAmb_s, _mode = thrust_from_ambient_isp(Pc_bar, MR, eps, pamb_bar, At)[1:3]  # (Isp, mode)
            Isp_grid[i] = IspAmb_s

        self._cea_table_cache[key] = (h_grid, Isp_grid)
        return h_grid, Isp_grid

    def _build_atm_table(self, h_max, n):
        h = np.linspace(0.0, h_max, int(n))
        rho = np.empty_like(h);
        p = np.empty_like(h)
        for i, hh in enumerate(h):
            if _USE_AMBIANCE:
                at = _AmbAtmos(float(hh))
                rho[i] = float(at.density);
                p[i] = float(at.pressure)
            else:
                rho[i], p[i], _, _ = self._atm(hh)  # your fallback
        return h, rho, p

    def _interp_lin(self, h, xh, x, last_idx):
        # Monotone ascent ⇒ reuse last_idx; O(1) average
        n = len(xh)
        i = last_idx
        if h <= xh[0]: return x[0], 0
        if h >= xh[-1]: return x[-1], n - 2
        # fast forward until bracketed
        while i < n - 2 and h > xh[i + 1]: i += 1
        while i > 0 and h < xh[i]: i -= 1
        t = (h - xh[i]) / (xh[i + 1] - xh[i])
        return x[i] * (1 - t) + x[i + 1] * t, i

    def compute(self, inputs, outputs):
        # Inputs
        At = float(inputs['At'])
        Pc_bar = float(inputs['Pc'])
        MR = float(inputs['MR'])
        eps = float(inputs['eps'])

        prop_mass = max(0.0, float(inputs['prop_mass']))
        burn_time = max(0.0, float(inputs['burn_time']))

        m_dry = max(1e-6, float(inputs['dry_mass']))
        Cd = float(inputs['Cd'])
        Aref = max(0.0, float(inputs['area_ref']))
        h = float(inputs['h0'])
        v = float(inputs['v0'])
        dt = max(1e-5, float(inputs['dt']))

        h_table_max = float(inputs['h_table_max'])
        n_table = float(inputs['n_table'])

        # Initial mass & bookkeeping
        t = 0.0
        m = m_dry + prop_mass
        max_q = 0.0
        t_max_q = 0.0
        burnout_alt = h
        burnout_vel = v

        # Get Isp_amb(h) table
        h_grid, Isp_grid = self._get_isp_table(Pc_bar, MR, eps, At, n_table, h_table_max)

        # Precompute c* and mdot (constant if Pc, At fixed)
        cstar = CEA.get_Cstar(Pc=Pc_bar, MR=MR)
        mdot_nominal = (Pc_bar * 1.0e5 * At) / cstar  # kg/s
        g0 = self._G0

        # Safety guards
        max_steps = 500000
        coast_guard = 2000.0  # s

        # Interpolant for Isp_amb(h)
        def Isp_of_h(h_loc):
            # Clamp outside table: use endpoints
            if h_loc <= h_grid[0]:
                return Isp_grid[0]
            if h_loc >= h_grid[-1]:
                return Isp_grid[-1]
            return float(np.interp(h_loc, h_grid, Isp_grid))

        # ---- Accelerations ----
        def acc_burn(h_loc, v_loc, m_loc):
            rho, pamb_pa, _, _ = self._atm(h_loc)
            IspAmb_s = Isp_of_h(h_loc)
            T_now = mdot_nominal * g0 * IspAmb_s
            D = 0.5 * rho * Cd * Aref * v_loc * abs(v_loc)
            a_loc = (T_now - np.sign(v_loc)*abs(D) - m_loc*self._g(h_loc)) / max(m_loc, 1e-9)
            md = mdot_nominal if (t < burn_time and (m_loc - m_dry) > 1e-9) else 0.0
            return a_loc, md

        def acc_coast(h_loc, v_loc, m_loc):
            rho, _, _, _ = self._atm(h_loc)
            D = 0.5 * rho * Cd * Aref * v_loc * abs(v_loc)
            a_loc = (- np.sign(v_loc)*abs(D) - m_loc*self._g(h_loc)) / max(m_loc, 1e-9)
            return a_loc, 0.0

        print("Starting Trajectory Sim...")

        # ---- Burn phase ----
        steps_burn = int(np.ceil(burn_time / dt)) if (burn_time > 0.0 and mdot_nominal > 0.0) else 0
        for _ in range(min(steps_burn, max_steps)):
            rho, _, _, _ = self._atm(h)
            q = 0.5 * rho * v * v
            if q > max_q:
                max_q = q
                t_max_q = t

            h, v, m = self._rk4_step(h, v, m, dt, acc_burn)
            t += dt
            if m <= m_dry + 1e-9:
                break

        burnout_alt = h
        burnout_vel = v

        # ---- Coast to apogee ----
        coast_time = 0.0
        for _ in range(max_steps - steps_burn):
            rho, _, _, _ = self._atm(h)
            q = 0.5 * rho * v * v
            if q > max_q:
                max_q = q
            h, v, m = self._rk4_step(h, v, m, dt, acc_coast)
            t += dt
            coast_time += dt
            if v <= 0.0 or coast_time > coast_guard:
                break

        outputs['apogee'] = max(h, burnout_alt)
        outputs['t_apogee'] = t
        outputs['burnout_alt'] = burnout_alt
        outputs['burnout_vel'] = burnout_vel
        outputs['max_q'] = max_q
        outputs['t_max_q'] = t_max_q

        # How close we got to table ceiling (to help you size h_table_max)
        outputs['apogee_frac_of_table'] = float(min(1.0, max(h, burnout_alt) / max(1e-6, h_grid[-1])))
