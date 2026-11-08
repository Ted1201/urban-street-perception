"""Restore bundled display thumbnails and build the offline page (Python stdlib only)."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parent
parts=ROOT/'assets/thumbnails'
target=ROOT/'web-prototype/dist/photos'
target.mkdir(parents=True,exist_ok=True)
manifest=json.loads((parts/'manifest.json').read_text(encoding='utf-8'))
for part in manifest['parts']:
    source=parts/part['file']
    if hashlib.sha256(source.read_bytes()).hexdigest()!=part['sha256']:
        raise SystemExit(f"Archive checksum failed: {source.name}")
    with zipfile.ZipFile(source) as archive:
        for info in archive.infolist():
            if Path(info.filename).name!=info.filename or not info.filename.endswith('.jpg'):
                raise SystemExit(f"Unexpected archive entry: {info.filename}")
            dest=target/info.filename
            if not dest.exists() or dest.stat().st_size!=info.file_size:
                archive.extract(info,target)
if len(list(target.glob('*.jpg')))!=manifest['thumbnail_count']:
    raise SystemExit('Thumbnail count does not match the manifest.')
subprocess.run([sys.executable,str(ROOT/'web-prototype/build_offline.py')],check=True)
print(f"Ready: {ROOT/'web-prototype/dist/macau-streets.html'}")
