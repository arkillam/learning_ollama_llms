# Learning Ollama & LLMs

I am using Ollama to run LLMs on my personal computer, while learning about how to work with LLMs, prompts, agents etc.  Mostly, I am trying to build RAG applications as a learning vehicle.

I set up Postgres with pgvector to store embeggings, and am writing code using Python and Java.  The Python is in the main directory of this project, while the Java is in the Eclipse project sub-directory 'learning'.

I am sharing this in case anyone finds benefit in reading it as part of their own learning.

postgres_notes.txt covers Postgres setup.  I did that first.

python_notes.txt covers what I did with Python.  I used langchain and other Python libraries to make things work.  I probably did not use langchain the normal way, since I have not taken a proper langchain course first.

For Java, I am focusing on using the Ollama REST API.  I have read it is identical to the OpenAI API, so hopefully this will be useful for real work.

Notes:
 - so far, it seems that chunks created just using EOLs and spaces to break up content work better than chunks with attempts at logical divides like chapters
 - querying and then sorting chunks so that chunks that were adjacent in the source material are adjacent in the context yields better results
 - my initial impression of deepseek is that using it to generate embeddings and then use those for semantic search of chunks ... does not work well at all
 - I tried two PDFs, and for both I found the text extracted was messy, with lots of added spaces around words etc.  I am not going to try more PDF extractions for a while.

Questions for another day (my todo list):
a) figure out how to get the token count for a string of text
b) see if embedding results include token count
c) try with Java and REST calls
d) update first two rag attempt scripts to have table_name variable as there is in the third
e) look at granite3-moe, which is for "low latency" use

List of things I want to learn (pulled from the list of what Granite 3.1 does):
 - Summarization
 - Text classification
 - Text extraction
 - Question-answering
 - Retrieval Augmented Generation (RAG)
 - Code related tasks
 - Function-calling tasks
 - Multilingual dialog use cases
 - Long-context tasks including long document/meeting summarization, long document QA, etc.

