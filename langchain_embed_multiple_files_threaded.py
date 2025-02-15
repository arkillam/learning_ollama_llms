
import psycopg2
import threading
import useful_stuff

from datetime import datetime
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from useful_stuff import logger

# note: not sure how well this works; I tried it, but it was taking so long to run I killed the processes and tried something else

# using nomic-embed-text to generate embeddings; it has a 8192 context-length (https://www.nomic.ai/blog/posts/nomic-embed-text-v1)
# produces vectors of 768 dimensions
#embeddingModel = 'nomic-embed-text'
#chunkSize = 500
#overlapAmount = 100
#chunkSize = 4000

# using deepseek-r1:1.5b to generate embeddings; it has a 32k context-length (https://ollama.com/ishumilin/deepseek-r1-coder-tools:1.5b)
# produces vectors of 1536 dimensions
embeddingModel = 'deepseek-r1:1.5b'
chunkSize = 200
overlapAmount = 20

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
    deleteQuery = "delete from andrew.embeddings where source = '{}'".format(filename)
    logger("deleteQuery -> " + deleteQuery)
    cur.execute(deleteQuery)
    conn.commit()

    short_name = filename.split('\\')[-1]

    index = 0;
    for doc in documents:

        single_vector = embedder.embed_query(doc.page_content)

        #print(doc.page_content)

        logger("{} -> {} of {} -> text size {} vector size {}".format(short_name, index, howmanyParts, len(doc.page_content), len(single_vector)))

        insertQuery = "insert into andrew.embeddings (source, content, embedding) values (%s, %s, %s)"
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

embedder = OllamaEmbeddings(model=embeddingModel, base_url='http://localhost:11434')

#text_splitter = RecursiveCharacterTextSplitter(separators=separators, is_separator_regex=True,chunk_size = chunkSize, chunk_overlap  = overlapAmount)
text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunkSize, chunk_overlap  = overlapAmount)

t1 = threading.Thread(target=embedTextFile,args=('D:\\Files\\RPG Books\\ADnD 2E RTFs\\complete_wizard.txt', conn, embedder, text_splitter))
t2 = threading.Thread(target=embedTextFile,args=('D:\\Files\\RPG Books\\ADnD 2E RTFs\\dmg.txt', conn, embedder, text_splitter))
t3 = threading.Thread(target=embedTextFile,args=('D:\\Files\\RPG Books\\ADnD 2E RTFs\\monster_manual.txt', conn, embedder, text_splitter))
t4 = threading.Thread(target=embedTextFile,args=('D:\\Files\\RPG Books\\ADnD 2E RTFs\\phb.txt', conn, embedder, text_splitter))
t5 = threading.Thread(target=embedTextFile,args=('D:\\Files\\RPG Books\\ADnD 2E RTFs\\po_skills_powers.txt', conn, embedder, text_splitter))
t6 = threading.Thread(target=embedTextFile,args=('D:\\Files\\RPG Books\\ADnD 2E RTFs\\po_spells_magic.txt', conn, embedder, text_splitter))
t7 = threading.Thread(target=embedTextFile,args=('D:\\Files\\RPG Books\\ADnD 2E RTFs\\tom.txt', conn, embedder, text_splitter))

t1.start()
t2.start()
t3.start()
t4.start()
t5.start()
t6.start()
t7.start()

t1.join()
t2.join()
t3.join()
t4.join()
t5.join()
t6.join()
t7.join()

logger('all threads complete')

conn.close()
logger ('closed database connection')
