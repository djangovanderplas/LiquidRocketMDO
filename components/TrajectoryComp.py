import numpy as np
import openmdao.api as om


class TrajectoryComp(om.ExplicitComponent):
    """
    Simple 1D vertical trajectory integrator:
      - Thrust (from PropulsionComp) during burn
      - Mass decreases with mdot_total during burn (down to dry mass)
      - Drag with exponential atmosphere
      - Gravity varying with altitude
      - Coasts after burnout until apogee (v crosses <= 0)
    """

    def setup(self):
        # --- Inputs ---
        # From PropulsionComp
        self.add_input('thrust', val=0.0, units='N', desc='Thrust during burn')
        self.add_input('mdot_total', val=0.0, units='kg/s', desc='Total mass flow rate')
        self.add_input('prop_mass', val=0.0, units='kg', desc='Usable propellant mass')
        self.add_input('burn_time', val=0.0, units='s', desc='Burn duration')

        # From IVC
        self.add_input('dry_mass', val=20.0, units='kg', desc='Vehicle dry mass (no prop)')
        self.add_input('Cd', val=0.5, desc='Reference drag coefficient')
        self.add_input('area_ref', val=0.01, units='m**2', desc='Reference frontal area')
        self.add_input('h0', val=0.0, units='m', desc='Launch altitude')
        self.add_input('v0', val=0.0, units='m/s', desc='Initial vertical velocity (up +)')
        # Internal integrator time step (you can expose if you wish)
        self.add_input('dt', val=0.01, units='s', desc='Integrator time step')

        # --- Outputs ---
        self.add_output('apogee', val=0.0, units='m', desc='Peak altitude above MSL')
        self.add_output('t_apogee', val=0.0, units='s', desc='Time to apogee')
        self.add_output('burnout_alt', val=0.0, units='m', desc='Altitude at burnout')
        self.add_output('burnout_vel', val=0.0, units='m/s', desc='Velocity at burnout')
        self.add_output('max_q', val=0.0, units='Pa', desc='Max dynamic pressure')
        self.add_output('t_max_q', val=0.0, units='s', desc='Time of max dynamic pressure')

        # We approximate totals (Rocket-like dynamics are non-analytic here)
        self.declare_partials(of='*', wrt='*', method='fd')

    # --- Constants ---
    _R_E = 6371000.0     # m, Earth radius
    _G0 = 9.80665        # m/s^2
    _RHO0 = 1.225        # kg/m^3, sea-level density
    _H = 8500.0          # m, scale height

    def _rho(self, h):
        # Simple exponential atmosphere; clamp at positive h
        hh = max(0.0, float(h))
        return self._RHO0 * np.exp(-hh / self._H)

    def _g(self, h):
        # Gravity that falls off with altitude
        return self._G0 * (self._R_E / (self._R_E + max(0.0, float(h))))**2

    def compute(self, inputs, outputs):
        # Inputs
        T = float(inputs['thrust'])
        mdot = float(inputs['mdot_total'])
        m_prop = max(0.0, float(inputs['prop_mass']))
        t_burn = max(0.0, float(inputs['burn_time']))

        m_dry = max(1e-6, float(inputs['dry_mass']))  # keep > 0
        Cd = float(inputs['Cd'])
        A = max(0.0, float(inputs['area_ref']))
        h = float(inputs['h0'])
        v = float(inputs['v0'])
        dt = max(1e-5, float(inputs['dt']))

        # State
        t = 0.0
        m = m_dry + m_prop

        # Book-keeping
        max_q = 0.0
        t_max_q = 0.0
        burnout_alt = h
        burnout_vel = v

        # Safety caps
        max_steps = 200000  # ~2000 s with dt=0.01
        coast_guard = 1200.0  # s max coast time

        # --- Burn phase ---
        steps_burn = int(np.ceil(t_burn / dt)) if mdot > 0.0 and T > 0.0 and t_burn > 0.0 else 0
        for _ in range(min(steps_burn, max_steps)):
            # Environment
            rho = self._rho(h)
            g = self._g(h)

            # Forces
            D = 0.5 * rho * Cd * A * v * abs(v)  # opposes v
            W = m * g
            # Acceleration (drag opposes sign(v))
            a = (T - np.sign(v) * abs(D) - W) / m

            # Semi-implicit Euler: update v with a, then h with new v
            v += a * dt
            h += v * dt
            t += dt

            # Propellant usage
            m = max(m_dry, m - mdot * dt)

            # Dynamic pressure tracking
            q = 0.5 * rho * v * v
            if q > max_q:
                max_q = q
                t_max_q = t

        # Store burnout state
        burnout_alt = h
        burnout_vel = v

        # --- Coast to apogee (thrust=0, mass constant) ---
        # coast until v <= 0 or safety caps trip
        coast_time = 0.0
        for i in range(max_steps - steps_burn):
            rho = self._rho(h)
            g = self._g(h)
            D = 0.5 * rho * Cd * A * v * abs(v)
            a = (- np.sign(v) * abs(D) - m * g) / m  # only drag + weight

            v += a * dt
            h += v * dt
            t += dt
            coast_time += dt

            # Dynamic pressure update during coast too
            q = 0.5 * rho * v * v
            if q > max_q:
                max_q = q
                t_max_q = t

            if v <= 0.0:
                break
            if coast_time > coast_guard:
                # Bail out to avoid infinite loop in weird cases
                break

        outputs['apogee'] = max(h, burnout_alt)
        outputs['t_apogee'] = t
        outputs['burnout_alt'] = burnout_alt
        outputs['burnout_vel'] = burnout_vel
        outputs['max_q'] = max_q
        outputs['t_max_q'] = t_max_q
