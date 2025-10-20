import openmdao.api as om
from rocketpy.simulation import flight

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
    ivc.add_output('eps', 7.0) # -
    ivc.add_output('MR',  2) # -
    ivc.add_output('throat_diam', 0.02, units='m') # m
    ivc.add_output('Pamb', 1.01325, units='bar') # bar
    ivc.add_output('burn_time', 10.0, units='s') # s

    # Trajectory
    ivc.add_output('d_rocket', 0.1, units='m')
    ivc.add_output('m_dry', 10, units='kg')

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
        promotes_inputs=['*'],
        promotes_outputs=['*'],
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
    print("v_rail_exit [m/s] :", g('v_rail_exit'))
    print("max_q [Pa]        :", g('max_q'))
    print("Min SM [-]        :", g('min_static_margin'))
    print("Max SM [-]        :", g('max_static_margin'))

if __name__ == "__main__":
    prob = build_problem()
    model = prob.model

    # Define Objective
    model.add_objective('apogee', scaler=-1.0)
    model.add_design_var('eps', lower=5, upper=8)
    model.add_design_var('MR', lower=1, upper=10)


    # Driver (set BEFORE setup)
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol'] = 1e-6
    prob.driver.options['maxiter'] = 20
    prob.driver.options['disp'] = True
    prob.set_solver_print(2)

    model.approx_totals(method='fd', form='central', step_calc='rel', step=1e-2)

    # Setup, then execute
    prob.setup()

    prob.run_model()

    print("Baseline:")
    print_parameters(prob)

    prob.run_driver()

    print("Optimized:")
    print_parameters(prob)
