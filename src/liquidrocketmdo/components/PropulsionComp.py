import numpy as np
import openmdao.api as om
from rocketcea.cea_obj_w_units import CEA_Obj

CEA = CEA_Obj(
    oxName='N2O', fuelName='Ethanol',
    isp_units='sec',
    cstar_units='m/s',
    pressure_units='Bar',
    temperature_units='K',
    sonic_velocity_units='m/s',
    enthalpy_units='J/kg',
    density_units='kg/m^3',
    specific_heat_units='J/kg-K',
    viscosity_units='millipoise',
    thermal_cond_units='mcal/cm-K-s',
)


class PropulsionComp(om.ExplicitComponent):
    def setup(self):
        # Inputs
        self.add_input('Pc', val=40.0, units='bar')  # chamber pressure [bar]
        self.add_input('eps', val=5.0)  # expansion ratio [-]
        self.add_input('MR', val=3.0)  # O/F mass ratio [-]
        self.add_input('throat_diam', val=0.02, units='m')  # throat diameter [m]
        self.add_input('Pamb', val=1.01325, units='bar')  # ambient pressure [bar]
        self.add_input('burn_time', val=10.0, units='s')  # [s]

        # Outputs
        self.add_output('SL_Isp', val=0.0, units='s')  # [s]
        self.add_output('Cf', val=0.0)  # [-]
        self.add_output('cstar', val=0.0, units='m/s')  # [m/s]
        self.add_output('At', val=0.0, units='m**2')  # [m^2]
        self.add_output('Ae', val=0.0, units='m**2')  # [m^2]
        self.add_output('thrust', val=0.0, units='N')  # [N]
        self.add_output('mdot_total', val=0.0, units='kg/s')  # [kg/s]
        self.add_output('mdot_ox', val=0.0, units='kg/s')  # [kg/s]
        self.add_output('mdot_fuel', val=0.0, units='kg/s')  # [kg/s]
        self.add_output('prop_mass', val=0.0, units='kg')  # [kg]
        self.add_output('prop_ox', val=0.0, units='kg')  # [kg]
        self.add_output('prop_fuel', val=0.0, units='kg')
        self.add_output('P_exit', val=0.0, units='bar')
        self.add_output('V_exit', val=0.0, units='m/s')

        self.declare_partials(of='*', wrt='*', method='cs')


    def compute(self, inputs, outputs):

        # Local inputs
        Pc = float(inputs['Pc'])
        eps = float(inputs['eps'])
        MR = float(inputs['MR'])
        d_throat = float(inputs['throat_diam'])
        Pamb = float(inputs['Pamb'])
        burn_time = float(inputs['burn_time'])

        # CEA Outputs
        SL_Isp = CEA.estimate_Ambient_Isp(Pc=Pc, MR=MR, eps=eps, Pamb=Pamb, frozen=0, frozenAtThroat=0)[0]
        Cf = CEA.get_PambCf(Pc=Pc, MR=MR, eps=eps, Pamb=Pamb)[1]
        cstar = CEA.get_Cstar(Pc=Pc, MR=MR)
        Pc_over_Pexit = CEA.get_PcOvPe(Pc=Pc, MR=MR, eps=eps, frozen=0, frozenAtThroat=0)

        P_exit = Pc / Pc_over_Pexit

        # Nozzle Geometry
        At = np.pi * (0.5 * d_throat) ** 2
        Ae = eps * At

        # Pc in pascals for local stuff
        Pc_Pa = Pc * 1.0e5

        # Thrust Performance
        thrust = Cf * Pc_Pa * At  # N
        mdot_total = (Pc_Pa * At) / cstar  # kg/s

        # O/F split
        mdot_fuel = mdot_total / (1.0 + MR)
        mdot_ox = mdot_total - mdot_fuel

        # Propellant Masses
        prop_mass = mdot_total * burn_time
        prop_fuel = mdot_fuel * burn_time
        prop_ox = mdot_ox * burn_time

        # Calculate real exit velocity
        V_eq = SL_Isp * 9.81
        V_exit = V_eq - (P_exit-Pamb) * 1e5 * Ae / mdot_total

        # Set outputs
        outputs['SL_Isp'] = SL_Isp
        outputs['Cf'] = Cf
        outputs['cstar'] = cstar
        outputs['At'] = At
        outputs['Ae'] = Ae
        outputs['thrust'] = thrust
        outputs['mdot_total'] = mdot_total
        outputs['mdot_ox'] = mdot_ox
        outputs['mdot_fuel'] = mdot_fuel
        outputs['prop_mass'] = prop_mass
        outputs['prop_ox'] = prop_ox
        outputs['prop_fuel'] = prop_fuel
        outputs['P_exit'] = P_exit
        outputs['V_exit'] = V_exit


def thrust_from_ambient_isp(Pc_bar: float, MR: float, eps: float, Pamb_bar: float, At_m2: float):
    """
    Compute thrust and ambient Isp at a given ambient pressure using RocketCEA.

    Returns:
        thrust_N (float), IspAmb_s (float), mode (str), cstar_mps (float)
    Notes:
        mdot = Pc*At / c*  (Pc in Pa, At in m^2, c* in m/s)
        T    = mdot * g0 * IspAmb
    """
    # Ambient Isp + operation mode (Under, Over, Separated)
    IspAmb_s, mode = CEA.estimate_Ambient_Isp(
        Pc=Pc_bar, MR=MR, eps=eps, Pamb=Pamb_bar, frozen=0, frozenAtThroat=0
    )

    # c* at these Pc/MR (does not depend on ambient)
    cstar_mps = CEA.get_Cstar(Pc=Pc_bar, MR=MR)

    # mdot from Pc and At
    mdot = (Pc_bar * 1.0e5 * At_m2) / cstar_mps

    thrust_N = IspAmb_s * 9.81 * mdot
    return thrust_N, IspAmb_s, mode, cstar_mps
