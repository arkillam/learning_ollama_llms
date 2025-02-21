
import psycopg2
import useful_stuff

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from useful_stuff import logger

# first we pick an embedding model, and the associated table that holds its embeddings and chunks of text
# TODO item: let the user interactively pick the one they want, from the list of keys in 'configurations'
configurations = useful_stuff.embedding_setups()
#current_embedding_config = "arctic_small"
current_embedding_config = "granite_small"
#current_embedding_config = "nomic_medium"
#current_embedding_config = "nomic_large"

# next we pull a model and its window size from our list of supported ones
#model_name = "gemma:2b" # granite and small chunks seem to be a good match; nomic with larger chunks results in "not enough into" while with granite some questions get answers
#model_name = "gemma2:2b"
#model_name = "gemma2:9b" # combines relatively slow performance (due to model size) with a small context window :(
#model_name = "granite3-moe:1b" # speedy, or "low latency" as IBM phrases it
#model_name = "granite3-moe:3b"
#model_name = "granite3.1-dense:2b"
#model_name = "llama3.1:8b" # takes a while, but does quality work
#model_name = "llama3.2:1b"
model_name = "llama3.2:3b"
#model_name = "mistral:7b"
#model_name = "mistral-nemo:12b"
#model_name = "phi3.5:3.8b"
#model_name = "qwen2.5:3b"
#model_name = "qwen2.5:7b"

# embedding-related information
logger("current_config " + current_embedding_config);
embedding_model = configurations[current_embedding_config]["embedding_model"]
table_name = configurations[current_embedding_config]["table_name"]
logger("embedding_model " + embedding_model);
logger("table_name " + table_name);

# generative model information
# to leave room for response code, I only provide 2/3 of the context window as the character limit for the context I pass in
char_limit = (int)(useful_stuff.rag_setups()[model_name] * 2 / 3)
logger("model_name " + model_name)
logger("context window " + str(useful_stuff.rag_setups()[model_name]))
logger("char_limit " + str(char_limit))

#modelName = "mistral-nemo:latest" # has 128k context length
#charLimit = 80000 # using 2/3 of expected context length for context input

question = input("What is your question? ")
print (question)

# use Ollama to generate embeddings
embedder = OllamaEmbeddings(model=embedding_model, base_url='http://localhost:11434')

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

minimum_value = 1.00 # this is a perfect match - very high starting level that will likely never get matches, so we will immediately reduce it
matches = 0
# we will take one match at a high similarity, or wait until we find five matches
while ((minimum_value >= 0.7 and matches == 0) or (matches < 5)):
    minimum_value = minimum_value - 0.03 # used to be 0.05, but wanted smaller steps to avoid the final leap grabbing a lot of chunks
    findClosestQuery = "select id, source, content from {} where (1.0 - (embedding <=> '{}')) > {} order by source, id".format(table_name, question_vector, minimum_value)
    cur.execute(findClosestQuery)
    matches = cur.rowcount
    logger("{} rows returned for cosign distance {}".format(matches, minimum_value))

# we pull from the matches to bulid the information to pass to the model, until we hit the size limit
text = ""
sources = {''} # this is a set, should ensure sources are not duplicated
logger("character limit is " + str(char_limit))
for record in cur:
    #print(str(record[1]) + " " + str(record[0]))
    text = text + "\n\n" + record[2]
    sources.add(record[1])
    if (len(text) > char_limit):
        logger("maxed out content at {} characters; reducing".format(len(text)))
        text = text[0:char_limit]
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
    #("system", "only answer the question with data from the context."),
    ("system", "Strictly only rely on the sources provided in generating your response. Never rely on external sources."),    
    ("user", """{context}

Question: {question}

"""
)])

# create the prompt, injecting the question and content to base the answer on
prompt_value = prompt_template.invoke({"context" : text, "question" : question})
logger ('created prompt')

# prep the model
model = ChatOllama(
    model=model_name,
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

print ("Sources:")
for source in sources:
    print (" - " + source.split('\\')[-1])

logger("done")