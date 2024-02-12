- Currently, user query is directly used to find the similar laws. However, this method often yields suboptimal 
results when the user’s question is vague. Use an LLM to extract keywords from user input and search that over the law data. 
- Apply summarization over the extracted contexts to reduce context size for the Q&A model. This can potentially enable
bringing more reference from the law data. To generate this summarize, map-reduce logic can be applied. A span of text
will be selected and information reduction will be achieved via summarization. 
- Legal data requires complex reason so bigger models should do better.
- Investigate https://www.juraforum.de as a better data source replacement
