
import psycopg2
import useful_stuff

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from useful_stuff import logger

# my second attempt was focused on fixing up the context in the prompt; in the first attempt, answers were clearly coming back based on data outside the RAG input

# trying to match character limits to context windows ... without yet having a way to calculate token sizes

# used to match the question to the available data
#embeddingModel = 'nomic-embed-text'
embeddingModel = 'deepseek-r1:1.5b'

# note - gemma seems to small to do what I ask; it fails to answer questions (or I am using it wrong ...)
#modelName = "gemma:2b" # has 8192 context length
#charLimit = 8000

#modelName = "llama3.1:latest" # has 128k context length
#charLimit = 120000

modelName = "mistral-nemo:latest" # has 128k context length
charLimit = 80000 # using 2/3 of expected context length for context input

#modelName = "phi3.5:latest" # has 128k context length
#charLimit = 80000 # using 2/3 of the context window for context, to leave 1/3 for the generated answer

# note: a few quick tests made me think phi3.5 is superior to llama3.2:3b for my purposes (it was pretty fast to respond though)
# https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md
#modelName = "llama3.2:3b" # has 128k context length
#charLimit = 80000 # using 2/3 of the context window for context, to leave 1/3 for the generated answer

# https://huggingface.co/blog/gemma2
#modelName = "gemma2:9b" # has 8192 token context length, so I am thinking that's 32k of characters roughly, and allowing 2/3 of that as the context
#charLimit = 20000 # using 2/3 of the context window for context, to leave 1/3 for the generated answer

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
findClosestQuery = "select id, source, content from andrew.embeddings where (1.0 - (embedding <=> '{}')) > {} order by source, id".format(question_vector, 0.50)
#data = (question_vector)
cur.execute(findClosestQuery)
logger("{} rows returned".format(cur.rowcount))

# we pull from the matches to bulid the information to pass to the model, until we hit the size limit
text = ""
sources = {''} # this is a set, should ensure sources are not duplicated
logger("character limit is " + str(charLimit))
for record in cur:
    #print(str(record[1]) + " " + str(record[0]))
    text = text + "\n\n" + record[2]
    sources.add(record[1])
    if (len(text) > charLimit):
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
# note: I moved the "only use context" instruction to the system because when it was part of the user,
# the model seemed to ignore it half the time
prompt_template = ChatPromptTemplate([
    # removed "limit the response to 60000 characters or less." because I got massive filler added to the response by the phi3.5 model
    ("system", "only answer the question with data from the context."),
    ("user", """{context}

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
# https://python.langchain.com/api_reference/core/messages/langchain_core.messages.base.BaseMessage.html#langchain_core.messages.base.BaseMessage
message = model.invoke(prompt_value)
logger ('done')

# write out the context that was passed in
print("\n\n\n")
logger("Context Provided:")
print(text)

# write out the result
print("\n\n\n")
logger("Response:")
print (message.content)
print (message.response_metadata)
