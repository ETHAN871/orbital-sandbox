"""Stage runtime files and version every local module through an import map."""
import argparse
import json
import re
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--commit', required=True)
args = parser.parse_args()
if not re.fullmatch(r'[a-f0-9]{40}', args.commit):
    parser.error('--commit must be a full Git SHA')
root = Path(__file__).resolve().parent.parent
out = root / '_site'
out.mkdir(exist_ok=True)
shutil.copytree(root / 'src', out / 'src', dirs_exist_ok=True)
shutil.copy2(root / 'styles.css', out / 'styles.css')
html = (root / 'index.html').read_text()
match = re.search(r'(<script type="importmap">)(.*?)(</script>)', html, re.S)
config = json.loads(match[2])
imports = config['imports']
for file in (root / 'src').glob('*.js'):
    key = './' + file.relative_to(root).as_posix()
    imports[key] = key + '?v=' + args.commit
    for spec in re.findall(r"['\"](\./[^'\"]+\.js\?[^'\"]+)['\"]", file.read_text()):
        path, query = spec.split('?', 1)
        target = './' + (file.parent / path).resolve().relative_to(root).as_posix()
        imports[target + '?' + query] = target + '?v=' + args.commit
html = html[:match.start(2)] + '\n' + json.dumps(config, indent=2) + '\n  ' + html[match.end(2):]
html = re.sub(r'src="src/main\.js[^\"]*"', 'src="src/main.js?v=' + args.commit + '"', html)
html = html.replace('href="styles.css"', 'href="styles.css?v=' + args.commit + '"')
(out / 'index.html').write_text(html)
(out / 'version.json').write_text(json.dumps({'commit': args.commit}) + '\n')
print(f'Staged release {args.commit} in {out}')
