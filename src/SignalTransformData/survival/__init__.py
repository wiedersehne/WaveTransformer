import logging

from . import loader
try:
    from . import generate
except:
    logging.info("Failed to import generate module. Check dependencies are fulfilled if simulation is required.")
    generate = None

try:
    from . import simulations
except:
    logging.info("Failed to import simulations module. Check dependencies are fulfilled if simulation is required.")
    simulations = None
