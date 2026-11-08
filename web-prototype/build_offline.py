"""Build a file:// entry point, embedding data to avoid browser fetch restrictions."""
from pathlib import Path
import json

root = Path(__file__).resolve().parent / 'dist'
data = (root / 'data/project.json').read_text(encoding='utf-8')
json.loads(data)
data = data.replace('<', '\\u003c')
html = (root / 'index.html').read_text(encoding='utf-8')
html = html.replace('静态展示 · 公开访问', '本地离线版 · 无需联网')
html = html.replace('复制分享链接', '本地使用说明')
html = html.replace('<script src="app.js">', '<script id="offline-project-data" type="application/json">' + data + '</script><script src="app.js">')
(root / 'offline.html').write_text(html, encoding='utf-8')
(root / 'macau-streets.html').write_text(html, encoding='utf-8')
print('Built dist/macau-streets.html and compatibility entry offline.html')
