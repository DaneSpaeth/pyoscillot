from simulation_controller import SimulationController
from cfg import parse_global_ini, parse_ticket
from datasaver import DataSaver
from animation_pulsation_new import create_high_res_arrays


def exec_sim(ticket):
    """ Only execute a simulation."""
    global_dict = parse_global_ini()
    conf_dict = parse_ticket(ticket)
    name = str(conf_dict["name"])

    saver = DataSaver(name)

    # Run the Simulation
    SimulationController(ticket)
    folder_ticket = saver.copy_ticket_spectrafolder(ticket)

    # Now also save the high res arrays
    create_high_res_arrays(folder_ticket)


if __name__ == "__main__":
    #ticket = "pulsation.ini"
    #exec_sim(ticket)
    ticket = "talk_ticket_2.ini"
    exec_sim(ticket)
