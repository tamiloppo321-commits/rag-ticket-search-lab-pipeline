from agentic_rag import run_agent
print("Support Assistant (type quit to exit)")
while True:
    q=input("\nYou: ").strip()
    if q.lower() in {"quit","exit","q"}: break
    if q: print("\nAssistant:",run_agent(q).content)
