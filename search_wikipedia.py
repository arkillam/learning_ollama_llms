
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from useful_stuff import logger

# another useful langchain tool:  searching wikipedia

# https://python.langchain.com/docs/integrations/tools/wikipedia/

# https://python.langchain.com/api_reference/community/utilities/langchain_community.utilities.wikipedia.WikipediaAPIWrapper.html

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(doc_content_chars_max = 12000, lang = 'en', load_all_available_meta = True, top_k_results = 1))

result = wikipedia.invoke("Tragically Hip")

print(result)