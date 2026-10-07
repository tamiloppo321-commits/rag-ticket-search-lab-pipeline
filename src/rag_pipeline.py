import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from common import ticket_documents

PROMPT = ChatPromptTemplate.from_template("""Answer the question using ONLY the ticket context below.
Include inline ticket citations such as [TICK-001] after factual claims.
If the context does not contain enough information, say so.

Context:
{context}

Question:
{question}

Answer:""")

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

class RAGPipeline:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL","text-embedding-3-small"))
        self.llm = ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL","gpt-4o-mini"), temperature=0)
        self.store = Chroma.from_documents(ticket_documents(), self.embeddings,
                                           collection_name="rag_ticket_pipeline")

    def retrieve(self, query, k=3, category=None, priority=None):
        filters = {}
        if category: filters["category"] = category
        if priority: filters["priority"] = priority
        kwargs = {"k":k}
        if filters: kwargs["filter"] = filters
        return self.store.similarity_search(query, **kwargs)

    def ask(self, query, k=3, category=None, priority=None):
        docs = self.retrieve(query,k,category,priority)
        if not docs:
            return {"answer":"I don't have relevant ticket history for this question.","documents":[]}
        response = self.llm.invoke(PROMPT.format(context=format_docs(docs),question=query))
        return {"answer":response.content,"documents":docs}

if __name__ == "__main__":
    rag = RAGPipeline()
    for q in ["How do I fix authentication issues?","How do I fix database timeouts?"]:
        r = rag.ask(q)
        print(f"\nQ: {q}\nA: {r['answer']}")
        print("Sources:",[d.metadata["ticket_id"] for d in r["documents"]])
