
import psycopg2
import useful_stuff

from datetime import datetime
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from useful_stuff import logger

# using nomic-embed-text to generate embeddings; it has a 8192 context-length (https://www.nomic.ai/blog/posts/nomic-embed-text-v1)
embeddingModel = 'nomic-embed-text'
chunkSize = 500
overlapAmount = 100
#chunkSize = 4000

# the separators used to break text up into chunks
separators = ['\n\n','\n',' ',''] # this one is the default
#separators = ['\nChapter|\nAppendix','\nTable', '\n\n','\n',' ','']

filename = 'D:\\Files\\RPG Books\\ADnD 2E RTFs\\phb.txt'

logger("starting on " + filename)

# returns a list of langchain Document objects (in this case, one)
# https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.text.TextLoader.html
# https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html#langchain_core.documents.base.Document
# each document has a page_content field, and a metadata field
raw_documents = TextLoader(filename).load()
logger('file read in')

# chunk it up
text_splitter = RecursiveCharacterTextSplitter(separators=separators, is_separator_regex=True,chunk_size = chunkSize, chunk_overlap  = overlapAmount)
documents = text_splitter.split_documents(raw_documents)
howmanyParts = len(documents)
logger('document split up into {} parts'.format(howmanyParts))

# use Ollama to generate embeddings
embedder = OllamaEmbeddings(model=embeddingModel, base_url='http://localhost:11434')

# connect to the database
config = useful_stuff.load_confg()
conn = psycopg2.connect(**config)
logger ('connected to database')

# open a cursor to perform operatoins
cur = conn.cursor()

# delete existing embeddings for this file
deleteQuery = "delete from andrew.embeddings where source = '{}'".format(filename)
logger("deleteQuery -> " + deleteQuery)
cur.execute(deleteQuery)
conn.commit()

index = 0;
for doc in documents:

    single_vector = embedder.embed_query(doc.page_content)

    logger("{} of {} -> text size {} vector size {}".format(index, howmanyParts, len(doc.page_content), len(single_vector)))

    insertQuery = "insert into andrew.embeddings (source, content, embedding) values (%s, %s, %s)"
    data = (filename, doc.page_content, single_vector)
    cur.execute(insertQuery, data)

    index += 1

cur.close()
conn.commit() # alternately, could set conn.autocommit = True above
conn.close()
logger ('closed database connection')