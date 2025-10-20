# Liquid Rocket MDO


**LiquidRocketMDO** is me playing around with Multidisciplinary Design Optimization (MDO) in the context of a liquid rocket.  
It started as a personal project to practice MDO techniques, but I figured it could also be useful (or at least interesting) for others working on liquids.

---

Under the hood, it uses:
- **[OpenMDAO](https://openmdao.org/)** for building and connecting disciplines  
- **[RocketPy](https://github.com/RocketPy-Team/RocketPy)** for trajectory simulation  
- **[RocketCEA](https://pypi.org/project/rocketcea/)** for engine performance and CEA calculations  
- **[Ambiance](https://pypi.org/project/ambiance/)** for atmospheric modeling  
- **NumPy** for math and general utilities

---

## Main Components

| Component | Purpose |
|------------|----------|
| `PropulsionComp.py` | Models engine performance using RocketCEA |
| `TrajectoryComp.py` | Handles trajectory simulation through RocketPy |
| `components/` | (Future) individual MDO disciplines and analysis components |
| `main_out/` | Stores results, plots, or outputs from runs |
| `main.py` | Entry point — builds and runs the MDO setup |

---

## Installation

Clone and install:

```bash
git clone git@github.com:djangovanderplas/LiquidRocketMDO.git
cd LiquidRocketMDO
pip install -e .
```
I am currently using conda with Python 3.11 (and an x86 emulation through Rosetta on MacOS).

## Roadmap
If I am ever motivated enough to finish this I would like to include
- [ ] Mass Estimation of the Rocket
- [ ] Structural Calculations on the Tank
- [ ] Structural Calculations on the Engine
- [ ] Electronics/Recovery aspects
- [ ] More advanced aero
- [ ] Make propulsion model better

