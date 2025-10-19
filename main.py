import openmdao.api as om
from components.PropulsionComp import PropulsionComp
from components.TrajectoryComp import TrajectoryComp

def build_problem():
    # ------------------------------
    # 1) Instantiate Problem
    # ------------------------------
    prob = om.Problem()
    model = prob.model = om.Group()

    # ------------------------------
    # 2) IVC: baseline inputs
    # ------------------------------

    ivc = om.IndepVarComp()

    # Propulsion / nozzle
    ivc.add_output('Pc', 40.0, units='bar') # bar
    ivc.add_output('eps', 10.0) # -
    ivc.add_output('MR',  2) # -
    ivc.add_output('throat_diam', 0.02, units='m') # m
    ivc.add_output('Pamb', 1.01325, units='bar') # bar
    ivc.add_output('burn_time', 10.0, units='s') # s

    # Trajectory
    ivc.add_output('dry_mass', 12.0, units='kg')
    ivc.add_output('Cd', 0.6)
    ivc.add_output('area_ref', 0.012, units='m**2')  # e.g., ~0.125 m dia body => A=πr^2
    ivc.add_output('h0', 0.0, units='m')
    ivc.add_output('v0', 0.0, units='m/s')
    ivc.add_output('dt', 0.001, units='s')

    model.add_subsystem('ivc', ivc, promotes=['*'])

    # ------------------------------
    # 3) Subsystems & wiring
    # ------------------------------

    # 3.1 Propulsion
    model.add_subsystem('prop', PropulsionComp(),
                        promotes_inputs=['Pc', 'eps', 'MR', 'throat_diam', 'Pamb', 'burn_time'],
                        promotes_outputs=['*'])

    # 3.2 Trajectory
    model.add_subsystem(
        'traj',
        TrajectoryComp(),
        promotes_inputs=[
            # from PropulsionComp
            'thrust', 'mdot_total', 'prop_mass', 'burn_time',
            # user-specified / vehicle properties
            'dry_mass', 'Cd', 'area_ref', 'h0', 'v0', 'dt'
        ],
        promotes_outputs=['*']  # apogee, burnout_alt, etc., promoted to top level
    )

    # Global derivatives
    model.approx_totals(method='fd')

    return prob

def print_parameters(prob):
    # Printing
    def g(name):
        val = prob.get_val(name)
        try:
            return float(val[0])
        except Exception:
            return val

    print("------ Inputs ------")
    print("Pc [bar]           :", g("Pc"))
    print("eps [-]            :", g("eps"))
    print("MR [-]             :", g("MR"))
    print("throat_diam [m]    :", g("throat_diam"))
    print("Pamb [bar]         :", g("Pamb"))
    print("burn_time [s]      :", g('burn_time'))

    print("------ Propulsion Outputs ------")
    print("SL Isp [s]        :", g('SL_Isp'))
    print("CF [-]            :", g('Cf'))
    print("cstar [m/s]       :", g('cstar'))
    print("At   [m2]         :", g('At'))
    print("Ae [m2]           :", g('Ae'))
    print("Thrust [N]        :", g('thrust'))
    print("mdot_total [kg/s] :", g('mdot_total'))
    print("mdot_ox [kg/s]    :", g('mdot_ox'))
    print("mdot_fuel [kg/s]  :", g('mdot_fuel'))
    print("prop_mass [kg]    :", g('prop_mass'))
    print("prop_ox [kg]      :", g('prop_ox'))
    print("prop_fuel [kg]    :", g('prop_fuel'))

    print("------ Trajectory Outputs ------")
    print("Apogee [m]        :", g('apogee'))
    print("t_apogee [s]      :", g('t_apogee'))
    print("burnout_alt [m]   :", g('burnout_alt'))
    print("burnout_vel [m/s] :", g('burnout_vel'))
    print("max_q [Pa]        :", g('max_q'))
    print("t_max_q [s]       :", g('t_max_q'))

if __name__ == "__main__":
    prob = build_problem()
    model = prob.model

    # Define Objective
    model.add_objective('apogee', scaler=-1.0)
    model.add_design_var('eps', lower=5, upper=30)
    model.add_design_var('MR', lower=1, upper=10)


    # Driver (set BEFORE setup)
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol'] = 1e-9
    prob.driver.options['maxiter'] = 20

    # Setup, then execute
    prob.setup()
    prob.run_model()

    print("Baseline:")
    print_parameters(prob)

    prob.run_driver()

    print("Optimized:")
    print_parameters(prob)
