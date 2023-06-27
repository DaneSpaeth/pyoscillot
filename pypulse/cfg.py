import configparser
from pathlib import Path


def parse_global_ini():
    """ Parse the global.ini file.

        Return parameters as dictionary.
    """
    config = configparser.ConfigParser()
    config.read("/home/dspaeth/pypulse/pypulse/global.ini")
    conf_dict = {}
    conf = config["DEFAULT"]
    for key in conf:
        if "path" in key:
            conf_dict[key] = Path(conf[key])
        else:
            conf_dict[key] = conf[key]
            
    return conf_dict


def parse_ticket(ticketpath):
    """ Parse a ticket ini file.

        Return parameters as dictionary.
    """
    # Read in the ini file
    config = configparser.ConfigParser()
    if not Path(ticketpath).is_file():
        ticketpath = "tickets/" + ticketpath
    config.read(ticketpath)
    superkeys = list(config.keys())

    # Read in the GLOBAL part as global parameter in first level of dictionary
    conf_dict = {}
    conf = config["GLOBAL"]
    for key in conf:
        if "path" in key:
            conf_dict[key] = Path(conf[key])
        else:
            try:
                conf_dict[key] = float(conf[key])
            except ValueError:
                conf_dict[key] = conf[key]

    # If no mode is given assume the mode to be the spectrum mode
    if not "mode" in list(conf_dict.keys()):
        conf_dict["mode"] = "spectrum"
    # It the mode is the specific intensity mode only allow the theoretical RVs
    # Since the resolution is too small to be a sensible choice for CARMENES
    # or HARPS
    if conf_dict["mode"] == "spec_intensity":
        conf_dict["instrument"] = "RAW"

    # Save the keys of all simulations as keys
    superkeys.pop(superkeys.index("GLOBAL"))
    superkeys.pop(superkeys.index("DEFAULT"))
    conf_dict["simulations"] = superkeys

    for sim in superkeys:
        conf = config[sim]
        conf_dict[sim] = {}
        for key in conf:
            try:
                conf_dict[sim][key] = float(conf[key])
            except ValueError:
                conf_dict[sim][key] = conf[key]

    return conf_dict

# singleton for debug dir
debug_dir = None


if __name__ == "__main__":
    conf_dict = parse_global_ini()
    print(conf_dict)
