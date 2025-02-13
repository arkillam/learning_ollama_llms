
import psycopg2
import useful_stuff

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from useful_stuff import logger

# trying to match character limits to context windows ... without yet having a way to calculate token sizes

#modelName = "llama3.1:latest"
#characterLimit = 120000

# used to match the question to the available data
embeddingModel = 'nomic-embed-text'

modelName = "mistral-nemo:latest"
characterLimit = 120000

question = input("What is your question? ")
print (question)

# use Ollama to generate embeddings
embedder = OllamaEmbeddings(model=embeddingModel, base_url='http://localhost:11434')

question_vector = embedder.embed_query(question)
logger('generated embedding for the question')

# connect to the database
config = useful_stuff.load_confg()
conn = psycopg2.connect(**config)
logger ('connected to database')

# open a cursor to perform operatoins
cur = conn.cursor()

# retrieve up to the 20 closest matches ("closest" does not mean "close")
#findClosestQuery = "select id, source, content, 1.0 - (embedding <=> '{}') as similarity from andrew.embeddings order by similarity desc limit 20".format(question_vector)
#findClosestQuery = "select id, source, content, 1.0 - (embedding <=> '{}') as similarity from andrew.embeddings order by source, id".format(question_vector)
findClosestQuery = "select id, source, content from andrew.embeddings where (1.0 - (embedding <=> '{}')) > 0.50 order by source, id".format(question_vector)
#data = (question_vector)
cur.execute(findClosestQuery)
logger("{} rows returned".format(cur.rowcount))

# for matches > 0.60 we capture their text and note their source
text = ""
sources = {''} # this is a set, should ensure sources are not duplicated
for record in cur:
    #print(str(record[1]) + " " + str(record[0]))
    text = text + "\n\n" + record[2]
    sources.add(record[1])
    if (len(text) > 120000):
        logger("maxed out content at {} characters".format(len(text)))
        break

cur.close()
conn.commit() # alternately, could set conn.autocommit = True above
conn.close()
logger ('closed database connection')

sources.remove('')
if (len(sources) < 1):
    logger("unfortunately we could not find any relevant information in the database")
    exit()

logger("source content size:  {} characters".format(len(text)))

# create the prompt template, setting it up for RAG (answer the question based on the info found in the database)
prompt_template = ChatPromptTemplate([
    ("user", """Answer the question based only on the following context:

{context}

Question: {question}

"""
)])

# create the prompt, injecting the question and content to base the answer on
prompt_value = prompt_template.invoke({"context" : text, "question" : question})
logger ('created prompt')

# prep the model
model = ChatOllama(
    model=modelName,
    temperature=0
)
logger ('created model')

# invoke the model (this can take a long time)
logger ('invoking ...')
message = model.invoke(prompt_value)
logger ('done')

# write out the result
print (message.content)
