import sys
sys.path.insert(0, r'D:\kibali-IA')
from outils.web_search_tool import WebSearchTool

t = WebSearchTool()
r = t.execute('actualite ballon d or 2026', {'query': 'actualite ballon d or 2026'})
print('SUCCESS:', r.get('success'))
print('OUTPUT:')
print(r.get('output', r.get('error'))[:1500])