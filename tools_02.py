
import json

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import DuckDuckGoSearchResults
from useful_stuff import logger

# using duck duck go search instead of google search because it is free

# search makes LLMs much better for answering current events questions, because LLMs alone do not know anything that happened after their model was built

# https://python.langchain.com/docs/integrations/tools/ddg/

# results class (returns text and links)
# https://python.langchain.com/api_reference/community/tools/langchain_community.tools.ddg_search.tool.DuckDuckGoSearchResults.html

# run class (returns text)
# https://python.langchain.com/api_reference/community/tools/langchain_community.tools.ddg_search.tool.DuckDuckGoSearchRun.html

###########################################################################

# DuckDuckGoSearchRun

searchRun = DuckDuckGoSearchRun()
# logger("searchRun.name " + searchRun.name)
# logger("searchRun.description " + searchRun.description)

results = searchRun.run("what are tools in langchain?")
logger(results)

###########################################################################

# DuckDuckGoSearchResults gives you back JSON (if you ask for it), with the results in an array and each result having a snippet (text from the page), title and link (url)

goodSearchTool = DuckDuckGoSearchResults(output_format="json")
# logger("searchResults.name " + searchResults.name)
# logger("searchResults.description " + searchResults.description)

# get the results and print them out as "pretty" indented JSON
results = goodSearchTool.invoke("what are tools in langchain?")
parse = json.loads(results)
print (json.dumps(parse, indent=4))

