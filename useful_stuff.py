
import configparser

from datetime import datetime


def embedding_setups() -> dict:
    """
    For SQL statements to create the tables referenced below, see "postgres_notes.txt".

    Returns a dictionary where the keys are my names for an embedding approach
    (e.g. nomic_small to use nomic and medium chunks - I am calling 4000-character chunks "mediuam").
    Each value is a dictionary with keys embedding_model, chunk_size, overlap_amount and table_name
    to provide what is needed to generate and store embeddings.
    """
    return {
        "arctic_small" : {"table_name" : "andrew.arctic_small_embeddings", "embedding_model" : "snowflake-arctic-embed2:568m", "chunk_size" : 1000, "overlap_amount" : 100},
        "arctic_medium" : {"table_name" : "andrew.arctic_medium_embeddings", "embedding_model" : "snowflake-arctic-embed2:568m", "chunk_size" : 4000, "overlap_amount" : 200},
        "arctic_large" : {"table_name" : "andrew.arctic_large_embeddings", "embedding_model" : "snowflake-arctic-embed2:568m", "chunk_size" : 8000, "overlap_amount" : 400},
        "granite_small" : {"table_name" : "andrew.granite_small_embeddings", "embedding_model" : "granite-embedding:30m", "chunk_size" : 500, "overlap_amount" : 50},
        "nomic_medium" : {"table_name" : "andrew.nomic_medium_embeddings", "embedding_model" : 'nomic-embed-text:latest', "chunk_size" : 4000, "overlap_amount" : 400},
        "nomic_large" : {"table_name" : "andrew.nomic_large_embeddings", "embedding_model" : 'nomic-embed-text:latest', "chunk_size" : 8000, "overlap_amount" : 400}
    }

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

def rag_setups() -> dict:
    """
    Returns a dictionary where the keys are model names and the values are context window sizes
    """
    return {
        "gemma:2b" : 8192,
        "gemma2:2b" : 16000,
        "gemma2:9b" : 16000,
        "granite3-moe:1b" : 128000,
        "granite3-moe:3b" : 128000,
        "granite3.1-dense:2b" : 128000,
        "llama3.1:8b" : 128000,
        "llama3.2:1b" : 128000,
        "llama3.2:3b" : 128000,
        "mistral:7b" : 32768 * 2, # x2 because this is tokens, not characters, trying to push it a bit ...
        "mistral-nemo:12b" : 128000,
        "phi3.5:3.8b" : 128000,
        "qwen2.5:3b" : 128000,
        "qwen2.5:7b" : 128000
    }