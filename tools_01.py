
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool, StructuredTool, tool
from useful_stuff import logger

# The Udemy zero-to-hero Ollama course introduced tools and agents to me

# The following code:
# defines a tool
# binds it to a model
# asks the model to answer a question
# gets back instructions on how to invoke the tool (no generated content comes back with that)
# invokves the tool based on the instructions
# writes out the result to the console

# They use ChatOpenAI() instead of ChatOllama() to create the model; this confused me, as the course is on Ollama and I do not want to
# use ChatGPT calls.  I learned with some searching that ChatOllama() does not support this, but I can use ChatOpenAI() in it splace.
# I wish the course had said this! (Or that I had picked up on it if they did ...)
# https://github.com/langchain-ai/langchain/discussions/21907

# Tools have to be defined, bound and called

# IBM's Granite supports tools, so I am using that:  https://ollama.com/library/granite3.1-dense
# 'ollama pull granite3.1-dense:2b' to download the smaller model ...
# I also tried with llama3.2:1b, but found it ran slower
modelName = "granite3.1-dense:2b" # has 128k context length

# here is what a simple tool definition looks like:
@tool
def name_of_tool (input: str) -> str:
    """
    Tool Description
    """
    return "Result"

logger ("name_of_tool.name " + name_of_tool.name)
logger ("name_of_tool.description " + name_of_tool.description)
# following line prints out
# {'input': {'title': 'Input', 'type': 'string'}}
logger ("name_of_tool.args " + str(name_of_tool.args))

@tool
def validate_user (username : str) -> bool:
    """
    this tool says whether a username is valid or not

    Args:
        username (str): the username to validate
    """
    #logger("called for " + username)
    if (username.lower == "andrew"):
        return True
    return False

# prep the model, bind tool to it
model = ChatOpenAI(
    api_key="ollama",
    model=modelName,
    base_url="http://localhost:11434/v1"
).bind_tools([validate_user])
logger ('created model')

tool_mapping = {
    'validate_user' : validate_user
}

result = model.invoke("Is Jason a valid user?")

# note that result.tool_calls returns how to call the tool, it did not actually run it and come back with a response
print (result)

# get the tool from the tool_mapping dict by the name returned from the results
# tool = tool_mapping[result.tool_calls[0]["name"]]

# # invoke the function using the parameters the model gave us
# tool_output = tool.invoke(result.tool_calls[0]["args"])
# print (tool_output)

# loop through the returned tools, calling each, and for now, just outputting the results
# I realize there is only one tool in the array, but this is much better than the above,
# calling by index, having to check if each index exists etc
for tool_call in result.tool_calls:
    tool = tool_mapping[tool_call["name"]]
    tool_output = tool.invoke(tool_call["args"])
    logger("called {} got back {}".format(tool_call["name"], tool_output))

