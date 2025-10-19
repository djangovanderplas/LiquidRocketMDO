import openmdao.api as om
from components.PropulsionComp import PropulsionComp


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
    ivc.add_output('Pc', 40.0) # bar
    ivc.add_output('eps', 5.0) # -
    ivc.add_output('MR',  8.4) # -
    ivc.add_output('throat_diam', 0.08) # m
    ivc.add_output('Pamb', 1.01325) # bar
    ivc.add_output('burn_time', 10.0) # s

    model.add_subsystem('ivc', ivc, promotes=['*'])

    # ------------------------------
    # 3) Subsystems & wiring
    # ------------------------------

    # 3.1 Propulsion
    model.add_subsystem('prop', PropulsionComp(),
                        promotes_inputs=['Pc', 'eps', 'MR', 'throat_diam', 'Pamb', 'burn_time'],
                        promotes_outputs=['*'])

    # Global derivatives
    model.approx_totals(method='fd')

    return prob

if __name__ == "__main__":
    prob = build_problem()
    model = prob.model

    # Define Objective
    model.add_objective('SL_Isp', scaler=-1.0)
    model.add_design_var('MR', lower=0.2, upper=10)


    # Driver (set BEFORE setup)
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol'] = 1e-6
    prob.driver.options['maxiter'] = 10

    # Setup, then execute
    prob.setup()
    prob.run_driver()
    # prob.run_model()

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
    print("burn_time [s]      :", g('burn_time'))  # fixed label

    print("------ Outputs ------")
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
    print("prop_ox [kg]      :", g('prop_ox'))     # fixed label
    print("prop_fuel [kg]    :", g('prop_fuel'))
