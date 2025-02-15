
import psycopg2
import useful_stuff

from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from useful_stuff import logger

# I tried two PDFs, and for both I found the text extracted was messy, with lots of added spaces around words etc.  I am not going to try more PDF extractions for a while.

# using nomic-embed-text to generate embeddings; it has a 8192 context-length (https://www.nomic.ai/blog/posts/nomic-embed-text-v1)
# produces vectors of 768 dimensions
embedding_model = 'nomic-embed-text'
table_name = 'andrew.pdf_embeddings' # for now I am putting PDF embeddings in a different table, until I have tested RAG with them
chunkSize = 4000
overlapAmount = 400

# this is a free RPG PDF you can download from DrivethruRPG if you are interested
filename = 'c:\\temp\\dark_dungeons_no_art.pdf'

logger("starting on " + filename)

# returns a list of langchain Document objects (in this case, one)
# https://python.langchain.com/docs/how_to/document_loader_pdf/
# https://python.langchain.com/docs/integrations/document_loaders/pypdfloader/
# each document has a page_content field, and a metadata field
loader = PyPDFLoader(filename)
pages = []
raw_documents = loader.load()
logger('file read in')

# chunk it up
text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunkSize, chunk_overlap  = overlapAmount)
documents = text_splitter.split_documents(raw_documents)
howmanyParts = len(documents)
logger('document split up into {} parts'.format(howmanyParts))

# use Ollama to generate embeddings
embedder = OllamaEmbeddings(model=embedding_model, base_url='http://localhost:11434')

# connect to the database
config = useful_stuff.load_confg()
conn = psycopg2.connect(**config)
logger ('connected to database')

# open a cursor to perform operatoins
cur = conn.cursor()

# delete existing embeddings for this file
deleteQuery = "delete from {} where source = '{}'".format(table_name, filename)
logger("deleteQuery -> " + deleteQuery)
cur.execute(deleteQuery)
conn.commit()

index = 0;
for doc in documents:

    single_vector = embedder.embed_query(doc.page_content)

    logger("{} of {} -> text size {} vector size {}".format(index, howmanyParts, len(doc.page_content), len(single_vector)))

    insertQuery = "insert into " + table_name + " (source, content, embedding) values (%s, %s, %s)"
    data = (filename, doc.page_content, single_vector)
    cur.execute(insertQuery, data)

    index += 1

cur.close()
conn.commit() # alternately, could set conn.autocommit = True above
conn.close()
logger ('closed database connection')