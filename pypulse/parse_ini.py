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
    config = configparser.ConfigParser()
    config.read(ticketpath)
    conf_dict = {}
    conf = config["DEFAULT"]
    for key in conf:
        if "path" in key:
            conf_dict[key] = Path(conf[key])
        else:
            try:
                conf_dict[key] = float(conf[key])
            except ValueError:
                conf_dict[key] = conf[key]

    return conf_dict


if __name__ == "__main__":
    conf_dict = parse_global_ini()
    print(conf_dict)
