
import psycopg2
import useful_stuff

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from useful_stuff import logger

# this third attempt is generally aimed at getting better results, fiddling with context length, chunk size, which models to use for embedding and generating answers

# started with filling whole context window with chunks, now realize have to leave room for responses, so aiming to fill 2/3 of content length with RAG inputs

# used to match the question to the available data
embedding_model = 'nomic-embed-text'
table_name = 'andrew.nomic_embeddings'
#embedding_model = 'deepseek-r1:1.5b'
#table_name = 'andrew.deepseek_embeddings'
#embedding_model = 'granite3-moe:1b'
#table_name = "andrew.granite_embeddings1024"

# note - gemma seems to small to do what I ask; it fails to answer questions (or I am using it wrong ...)
#modelName = "gemma:2b" # has 8192 context length
#charLimit = 5500

#modelName = "llama3.1:latest" # has 128k context length
#charLimit = 120000

#modelName = "mistral-nemo:latest" # has 128k context length
#charLimit = 80000 # using 2/3 of expected context length for context input

#modelName = "phi3.5:latest" # has 128k context length
#charLimit = 80000 # using 2/3 of the context window for context, to leave 1/3 for the generated answer

# I find granite to be a speedy model to run, a bit terse in term of the amount of text returned
#modelName = "granite3.1-dense:2b" # has 128k context length
#modelName = "granite3.1-dense" # (8b) has 128k context length
#charLimit = 80000 # using 2/3 of the context window for context, to leave 1/3 for the generated answer

# note: a few quick tests made me think phi3.5 is superior to llama3.2:3b for my purposes (it was pretty fast to respond though)
# https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md
#modelName = "llama3.2:3b" # has 128k context length
#charLimit = 80000 # using 2/3 of the context window for context, to leave 1/3 for the generated answer
#modelName = "llama3.2:1b" # has 128k context length
#charLimit = 80000 # using 2/3 of the context window for context, to leave 1/3 for the generated answer

modelName = "qwen2.5:3b" # has 128k context length, 7b was damn slow
charLimit = 80000 # using 2/3 of the context window for context, to leave 1/3 for the generated answer

# https://huggingface.co/blog/gemma2
#modelName = "gemma2:2b" # has 8192 token context length, so I am thinking that's 32k of characters roughly, and allowing 2/3 of that as the context
#charLimit = 20000 # using 2/3 of the context window for context, to leave 1/3 for the generated answer

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

print ("Sources:")
for source in sources:
    print (" - " + source.split('\\')[-1])

logger("done")