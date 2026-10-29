from outils import ToolManager
from langchain_openai import ChatOpenAI
from langchain_classic.agents import initialize_agent, Tool

print('=== Chargement du tool_manager (peut prendre 5s) ===')
tm = ToolManager()
web = tm.tools.get('web_search')
print('web_search trouve:', web is not None)
print('description:', web.description[:100] if web else 'N/A')

def web_search(query: str) -> str:
    r = web.execute(query, {'query': query})
    if isinstance(r, dict) and r.get('success'):
        return str(r.get('output', ''))[:3000]
    return 'ECHEC: ' + str(r.get('error', 'inconnu') if isinstance(r, dict) else r)

tools = [Tool(name='web_search', func=web_search,
              description='Recherche sur Internet. Entree: la requete en francais.')]

llm = ChatOpenAI(model='qwen3-8b', base_url='http://127.0.0.1:8080/v1',
                 api_key='local-no-key', temperature=0.0, max_tokens=2500, timeout=180)

agent = initialize_agent(tools, llm, agent='zero-shot-react-description',
                         verbose=True, max_iterations=4,
                         handle_parsing_errors=True, return_intermediate_steps=True)

print('=== INVOCATION ===')
result = agent.invoke({'input': 'Cherche sur internet l actualite du ballon d or 2026'})

print()
print('=== STEPS ===', len(result.get('intermediate_steps', [])))
for action, obs in result.get('intermediate_steps', []):
    print('-> outil:', action.tool)
    print('   input:', str(action.tool_input)[:100])
    print('   obs  :', str(obs)[:600])
    print()

print('=== FINAL ===')
print(result.get('output', ''))