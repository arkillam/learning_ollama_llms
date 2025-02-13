
import configparser

from datetime import datetime


def load_confg(filename='database.ini', section='postgressql'):
    """
    Reads in config data.  Returns config values in a dictionary.  Intended to be used
    with database.ini, which should contain the following stanza.
    [postgressql]
    host=localhost
    database=xxxxx
    user=xxxxx
    password=xxxxx
    port=5432

    :param filename: the config file to load in (default is database.ini)
    :param section: the section of the config file to load in (default is postgressql)
    """
    parser = configparser.ConfigParser()
    parser.read(filename)

    config = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            config[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))
    
    return config


def logger(text):
    """
    Logs text to the command line, with a timestamp
    :param text: the text to log
    """
    print(str(datetime.now()) + ":  " + text)
