#from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
class GenerateResponse:
    def __init__(self):
        self.llm=ChatGroq(
            groq_api_key="",
            model="qwen/qwen3-32b",
            temperature=0.1,
            max_tokens=1024
        )
    def generate_res(self,query,rag_retriever,llm,top_k=3):
        results=rag_retriever.retrieve(query,top_k)
        context = "\n".join([doc["document"] for doc in results]) if results else ""
        if not context:
            print("Not relevant Information")
        # context + query
        prompt = f""" use given context to generate the answer for the query
                Context: {context}
                Query: {query} """

        response = self.llm.invoke([prompt.format(contet=context,query=query)]) # expecting a string as prompt
        return response.content