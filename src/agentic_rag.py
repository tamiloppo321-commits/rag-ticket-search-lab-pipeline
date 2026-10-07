import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import StructuredTool
from langchain_community.vectorstores import Chroma
from pydantic import BaseModel, Field
from common import load_tickets, ticket_documents

class TicketIDInput(BaseModel):
    ticket_id:str=Field(description="Ticket ID such as TICK-001")
class CategoryInput(BaseModel):
    category:str=Field(description="Ticket category")
class PriorityInput(BaseModel):
    priority:str=Field(description="Critical, High, Medium, or Low")

class SupportTicketTools:
    def __init__(self):
        self.tickets=load_tickets()
        emb=OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL","text-embedding-3-small"))
        self.store=Chroma.from_documents(ticket_documents(),emb,collection_name="agentic_ticket_search")

    def search_similar_tickets(self,query:str)->str:
        """Use for troubleshooting and semantically similar issues."""
        docs=self.store.similarity_search(query,k=3)
        return "\n".join(f"[{d.metadata['ticket_id']}] {d.page_content}" for d in docs) or "No relevant tickets found."

    def get_ticket_by_id(self,ticket_id:str)->str:
        """Retrieve a specific ticket by ID."""
        ticket_id=ticket_id.strip().upper()
        if not ticket_id.startswith("TICK-"): return "Error: Invalid ticket ID format."
        for t in self.tickets:
            if t["ticket_id"]==ticket_id: return str(t)
        return f"Ticket {ticket_id} not found."

    def search_by_category(self,category:str)->str:
        """Find tickets in a category."""
        m=[t for t in self.tickets if t["category"].lower()==category.strip().lower()]
        return "\n".join(f"[{t['ticket_id']}] {t['title']} ({t['priority']})" for t in m) or "No tickets found."

    def search_by_priority(self,priority:str)->str:
        """Find tickets with a priority level."""
        p=priority.strip().lower()
        m=[t for t in self.tickets if t["priority"].lower()==p]
        return "\n".join(f"[{t['ticket_id']}] {t['title']} ({t['category']})" for t in m) or "No tickets found."

    def get_ticket_statistics(self,query:str="")->str:
        """Return total ticket count and category counts."""
        counts={}
        for t in self.tickets: counts[t["category"]]=counts.get(t["category"],0)+1
        return f"Total tickets: {len(self.tickets)} | By category: {counts}"

    def get_tools(self):
        return [
            StructuredTool.from_function(func=self.search_similar_tickets,name="SearchSimilarTickets",
                description="Use for troubleshooting, how-to-fix questions and semantically similar tickets."),
            StructuredTool.from_function(func=self.get_ticket_by_id,name="GetTicketByID",args_schema=TicketIDInput,
                description="Use for an exact ticket ID such as TICK-005."),
            StructuredTool.from_function(func=self.search_by_category,name="SearchByCategory",args_schema=CategoryInput,
                description="Use for questions about tickets in a specific category."),
            StructuredTool.from_function(func=self.search_by_priority,name="SearchByPriority",args_schema=PriorityInput,
                description="Use for urgent, important, or priority-level questions."),
            StructuredTool.from_function(func=self.get_ticket_statistics,name="GetTicketStatistics",
                description="Use for counts, totals and ticket statistics.")
        ]

def run_agent(query):
    manager=SupportTicketTools()
    llm=ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL","gpt-4o-mini"),temperature=0).bind_tools(manager.get_tools())
    return llm.invoke("You are an expert support desk assistant. Select appropriate tool(s) and answer from ticket evidence.\nUser: "+query)

if __name__=="__main__":
    for q in ["How do I fix authentication problems?","Show me ticket TICK-005",
              "How many tickets are in each category?","Show me all high priority tickets"]:
        print("\nQuery:",q,"\n",run_agent(q).content)
