import configparser
from pathlib import Path


def parse_global_ini():
    """ Parse the global.ini file.

        Return parameters as dictionary.
    """
    config = configparser.ConfigParser()
    config.read("global.ini")
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

    # Save the keys of all simulations as keys
    superkeys.pop(superkeys.index("GLOBAL"))
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


if __name__ == "__main__":
    conf_dict = parse_ticket("example_ticket.ini")
    print(conf_dict)
