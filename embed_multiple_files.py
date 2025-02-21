
import psycopg2
import useful_stuff

from datetime import datetime
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from useful_stuff import logger

# I started with a single embedding script and table.  Then I expanded to store different models' outputs in different tables, to keep different
# examples around to test, using one script per model.  Now I am improving on that by having one script that gets its inputs - model, chunk size,
# database table etc - from a dictionary provided by the useful_stuff library.

# Model Notes:

# snowflake-arctic-embed2
# - https://ollama.com/library/snowflake-arctic-embed2:568m, https://www.snowflake.com/en/engineering-blog/snowflake-arctic-embed-2-multilingual/
# - context length of 8192 tokens
# - produces vectors of size 1024
# - generating embeddings seems much slower than granite or nomic embeddings, makes sense to me given the larger vector size

# granite-embedding
# - https://ollama.com/library/granite-embedding:30m, https://huggingface.co/ibm-granite/granite-embedding-30m-english
# - the 30M model is English only; the 278M model adds support for more languages
# - context length of 512 tokens (called "Max. Sequence Length" in huggingface link above)
# - produces vectors of size 384
# - very fast, damn small max chunk size, errors out if chunks are too big (>500 characters in my testing)
# - I have found excellent results using these embeddings with a variety of generative models

# nomic-embed-text
# - https://ollama.com/library/nomic-embed-text:latest, https://www.nomic.ai/blog/posts/nomic-embed-text-v1
# - context length of 8192
# - produces vectors of 768 dimensions
# - fast and stable, good results when searching for chunks using embeddings
# - I have found excellent results using these embeddings with a variety of generative models

configurations = useful_stuff.embedding_setups()

# TODO item: let the user interactively pick the one they want, from the list of keys in 'configurations'
current_config = "arctic_small"
#current_config = "arctic_medium"
#current_config = "arctic_large"
#current_config = "granite_small"
#current_config = "nomic_medium"
#current_config = "nomic_large"

logger("current_config " + current_config);

embedding_model = configurations[current_config]["embedding_model"]
chunk_size = configurations[current_config]["chunk_size"]
overlap_amount = configurations[current_config]["overlap_amount"]
table_name = configurations[current_config]["table_name"]

logger("embedding_model " + embedding_model);
logger("chunk_size " + str(chunk_size));
logger("overlap_amount " + str(overlap_amount));
logger("table_name " + table_name);

# the separators used to break text up into chunks
#separators = ['\n\n','\n',' ',''] # this one is the default
#separators = ['\nChapter|\nAppendix','\nTable', '\n\n','\n',' ','']

def embedTextFile(filename, conn, embedder, text_splitter):
    """
    Reads in a text file, chunks it up, and embeds it in the Postgres database.
    :param filename: the full path to the file being loaded in
    :param conn: database connection
    :param embedder: an OllamaEmbeddings object
    :param text_splitter: a RecursiveCharacterTextSplitter object
    """
    logger("called for " + filename)

    # returns a list of langchain Document objects (in this case, one)
    # https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.text.TextLoader.html
    # https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html#langchain_core.documents.base.Document
    # each document has a page_content field, and a metadata field
    raw_documents = TextLoader(filename).load()
    logger('file read in')

    # chunk it up
    documents = text_splitter.split_documents(raw_documents)
    howmanyParts = len(documents)
    logger('document split up into {} parts'.format(howmanyParts))

    # open a cursor to perform operatoins
    cur = conn.cursor()

    # delete existing embeddings for this file
    deleteQuery = "delete from " + table_name + " where source = '{}'".format(filename)
    logger("deleteQuery -> " + deleteQuery)
    cur.execute(deleteQuery)
    conn.commit()

    short_name = filename.split('\\')[-1]

    index = 0;
    for doc in documents:

        single_vector = embedder.embed_query(doc.page_content)

        #print(doc.page_content)

        logger("{} -> {} of {} -> text size {} vector size {}".format(short_name, index, howmanyParts, len(doc.page_content), len(single_vector)))

        insertQuery = "insert into " + table_name + " (source, content, embedding) values (%s, %s, %s)"
        data = (filename, doc.page_content, single_vector)
        cur.execute(insertQuery, data)

        index += 1

    cur.close()
    conn.commit() # alternately, could set conn.autocommit = True above
    logger('database changes committed')

# connect to the database
config = useful_stuff.load_confg()
conn = psycopg2.connect(**config)
logger ('connected to database')

embedder = OllamaEmbeddings(model=embedding_model, base_url='http://localhost:11434')

#text_splitter = RecursiveCharacterTextSplitter(separators=separators, is_separator_regex=True, chunk_size = chunk_size, chunk_overlap  = overlap_amount)
text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap  = overlap_amount)

# list of files to embed; I find putting them in an array is a bit easier to read
# (I have a set of D&D rules from the 1990s I bought in text form ... useful because I can ask questions from a complex set of rules and know if I received correct answers)
#
    # 'D:\\Files\\RPG Books\\ADnD 2E RTFs\\po_combat_tactics.txt',
    # 'D:\\Files\\RPG Books\\ADnD 2E RTFs\\po_skills_powers.txt',
    # 'D:\\Files\\RPG Books\\ADnD 2E RTFs\\po_spells_magic.txt',
filesToEmbed = [
    'D:\\Files\\RPG Books\\ADnD 2E RTFs\\complete_bard.txt',
    'D:\\Files\\RPG Books\\ADnD 2E RTFs\\complete_dwarves.txt',
    'D:\\Files\\RPG Books\\ADnD 2E RTFs\\complete_druid.txt',
    'D:\\Files\\RPG Books\\ADnD 2E RTFs\\complete_elves.txt',
    'D:\\Files\\RPG Books\\ADnD 2E RTFs\\complete_fighter.txt',
    'D:\\Files\\RPG Books\\ADnD 2E RTFs\\complete_gnomes_halflings.txt',
    'D:\\Files\\RPG Books\\ADnD 2E RTFs\\complete_paladin.txt',
    'D:\\Files\\RPG Books\\ADnD 2E RTFs\\complete_thief.txt',
    'D:\\Files\\RPG Books\\ADnD 2E RTFs\\complete_wizard.txt',
    'D:\\Files\\RPG Books\\ADnD 2E RTFs\\dmg.txt',
    'D:\\Files\\RPG Books\\ADnD 2E RTFs\\monster_manual.txt',
    'D:\\Files\\RPG Books\\ADnD 2E RTFs\\phb.txt',    
    'D:\\Files\\RPG Books\\ADnD 2E RTFs\\tom.txt'
]

for f in filesToEmbed:
    embedTextFile(filename=f,conn=conn,embedder=embedder,text_splitter=text_splitter)

conn.close()
logger ('closed database connection')
