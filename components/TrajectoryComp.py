import numpy as np
import openmdao.api as om
from rocketpy import Environment, Rocket, Flight, Fluid, LiquidMotor, CylindricalTank, MassFlowRateBasedTank
from rocketpy.simulation import flight

# Define fluids
oxidizer_liq = Fluid(name="N2O_l", density=1220)
oxidizer_gas = Fluid(name="N2O_g", density=1.9277)
fuel_liq = Fluid(name="ethanol_l", density=789)
fuel_gas = Fluid(name="ethanol_g", density=1.59)

def _define_engine(thrust, burn_time, mdot_ox, mdot_fuel, Ae, d_rocket, V_tank_SF=1.5):
    # Calculate Tank Volume
    ox_vol = mdot_ox * burn_time / oxidizer_liq.density * V_tank_SF
    fuel_vol = mdot_fuel * burn_time / fuel_liq.density * V_tank_SF

    # Calculate Tank Height from Volume
    ox_h = ox_vol / (np.pi * (d_rocket/2) ** 2)
    fuel_h = fuel_vol / (np.pi * (d_rocket/2) ** 2)

    # Define tanks geometry
    ox_tanks_shape = CylindricalTank(radius=d_rocket / 2, height=ox_h, spherical_caps=False)
    fuel_tanks_shape = CylindricalTank(radius=d_rocket / 2, height=fuel_h, spherical_caps=False)

    # Define tanks
    oxidizer_tank = MassFlowRateBasedTank(
        name="Ox Tank",
        geometry=ox_tanks_shape,
        flux_time=burn_time,
        initial_liquid_mass=burn_time * mdot_ox*1.1,
        initial_gas_mass=0,
        liquid_mass_flow_rate_in=0,
        liquid_mass_flow_rate_out=mdot_ox,
        gas_mass_flow_rate_in=0,
        gas_mass_flow_rate_out=0,
        liquid=oxidizer_liq,
        gas=oxidizer_gas,
    )

    fuel_tank = MassFlowRateBasedTank(
        name="fuel tank",
        geometry=fuel_tanks_shape,
        flux_time=burn_time,
        initial_liquid_mass=burn_time * mdot_fuel*1.1,
        initial_gas_mass=0,
        liquid_mass_flow_rate_in=0,
        liquid_mass_flow_rate_out=mdot_fuel,
        gas_mass_flow_rate_in=0,
        gas_mass_flow_rate_out=0,
        liquid=fuel_liq,
        gas=fuel_gas,
    )

    r_e = np.sqrt(Ae / np.pi)

    engine = LiquidMotor(
        thrust_source=thrust,
        dry_mass=0,
        dry_inertia=(0, 0, 0),
        nozzle_radius=r_e,
        center_of_dry_mass_position=0,
        nozzle_position=0,
        burn_time=burn_time,
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )
    engine.add_tank(tank=oxidizer_tank, position=d_rocket*3 + ox_h / 2)
    engine.add_tank(tank=fuel_tank, position=d_rocket*3 + ox_h + d_rocket*3 + fuel_h / 2 )

    engine_length = 0.3 + ox_h + 0.2 + fuel_h

    return engine, engine_length

def _define_rocket(d_rocket, engine_length, engine, m_dry):
    rocket = Rocket(
        radius=d_rocket/2,
        mass=m_dry,
        inertia=(2.321, 2.321, 0.2), #todo: fix this
        power_off_drag=0.6,
        power_on_drag=0.6,
        center_of_mass_without_motor=0.9*engine_length,
        coordinate_system_orientation="nose_to_tail",
    )

    nose_cone = rocket.add_nose(
        length=d_rocket*4, kind="von karman", position=0
    )

    fin_set = rocket.add_trapezoidal_fins(
        n=4,
        root_chord=d_rocket,
        tip_chord=d_rocket*0.4,
        span=d_rocket,
        position=d_rocket*7+engine_length,
        cant_angle=0,
    )

    rocket.add_motor(engine, position=d_rocket*8+engine_length)
    # rocket.draw()

    return rocket

class TrajectoryComp(om.ExplicitComponent):

    def setup(self):
        self.add_input('thrust', val=1000, units='N')
        self.add_input('burn_time', val=10, units='s')
        self.add_input('mdot_ox', val=2, units='kg/s')
        self.add_input('mdot_fuel', val=1, units='kg/s')
        self.add_input('Ae', val=0.04, units='m**2')
        self.add_input('d_rocket', val=0.1, units='m')
        self.add_input('m_dry', val=10, units='kg')

        self.add_output('apogee', val=0.0, units='m')
        self.add_output('t_apogee', val=0.0, units='m')
        self.add_output('v_rail_exit', val=0.0, units='m/s')
        self.add_output('max_q', val=0.0, units='Pa')
        self.add_output('min_static_margin', val=0.0)
        self.add_output('max_static_margin', val=0.0)
        self.add_output('starting_mass', val=0.0, units='kg')

        self.count = 1

        self.declare_partials(of='*', wrt='*', method='fd',
                              form='central', step_calc='rel', step=1e-2)


    def compute(self, inputs, outputs):
        # Define Engine
        engine, engine_length = _define_engine(thrust=inputs['thrust'][0],
                                               burn_time=inputs['burn_time'][0],
                                               mdot_ox=inputs['mdot_ox'][0],
                                               mdot_fuel=inputs['mdot_fuel'][0],
                                               Ae=inputs['Ae'][0],
                                               d_rocket=inputs['d_rocket'][0])
        # engine.all_info()

        # Define Rocket
        rocket = _define_rocket(d_rocket=inputs['d_rocket'][0],
                                engine_length=engine_length,
                                engine=engine,
                                m_dry=inputs['m_dry'][0],)

        # rocket.draw()
        # rocket.all_info()

        # Define Environment
        env = Environment(latitude=32.990254,
                          longitude=-106.974998,
                          elevation=0)

        # Flight Sim
        print(f"Starting Flight Simulation {self.count}...")
        flight = Flight(rocket=rocket,
                        environment=env,
                        rail_length=10.5,
                        inclination=84,
                        heading=0,
                        terminate_on_apogee=True)

        # flight.all_info()

        print(f"thrust: {inputs['thrust'][0]}, apogee: {flight.apogee}")

        self.count += 1

        # Save Outputs
        outputs['apogee'] = flight.apogee
        outputs['t_apogee'] = flight.apogee_time
        outputs['v_rail_exit'] = flight.out_of_rail_velocity
        outputs['max_q'] = flight.max_dynamic_pressure
        outputs['min_static_margin'] = flight.stability_margin.min
        outputs['max_static_margin'] = flight.stability_margin.max
        outputs['starting_mass'] = rocket.total_mass.max
