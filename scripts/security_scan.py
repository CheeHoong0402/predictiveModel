"""Fail publication on known secret patterns, private paths or unreviewed data.

Prints file names and finding types only. This is a focused guard, not a guarantee
that arbitrary secrets/PII can be detected. Run before every public release.
"""
from pathlib import Path
import json
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    'telegram_token': r'\b\d{8,12}:A[A-Za-z0-9_-]{30,45}\b',
    'github_token': r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b',
    'private_key': r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'cloud_key': r'\bAKIA[A-Z0-9]{16}\b',
    'openai_key': r'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{30,}\b',
    'personal_path': r'(?:[A-Z]:[\\/]+Users[\\/]+[^\\/\s]+|' + '/ho' + r'me/[^/\s]+)',
    'credential_url': r'https?://[^/\s:@]+:[^/\s@]+@',
    'email_address': r'\b[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}\b',
}


def scan():
    tracked = subprocess.run(['git','ls-files','-z'], cwd=ROOT, capture_output=True, check=False)
    if tracked.returncode == 0 and tracked.stdout:
        paths=[ROOT/p for p in tracked.stdout.decode().split('\0') if p]
    else:
        paths=[p for p in ROOT.rglob('*') if p.is_file() and not any(
            part in {'.git','.venv','__pycache__','.pytest_cache'} for part in p.relative_to(ROOT).parts)]
    approved=json.loads((ROOT/'PUBLIC_DATA_MANIFEST.json').read_text('utf-8'))['files']
    findings=[];count=0
    for path in paths:
        rel=path.relative_to(ROOT).as_posix()
        if path.name=='secrets.toml' or (path.name.startswith('.env') and path.name!='.env.example'):
            findings.append((rel,'secret_file'))
        if rel.startswith('data/') and rel not in approved:
            findings.append((rel,'unreviewed_data'))
        if path.suffix=='.parquet':
            import pandas as pd
            df=pd.read_parquet(path)
            text='\n'.join(str(v) for v in df.select_dtypes(include=['object','string']).to_numpy().ravel())
            sensitive={'email','phone','customer_name','customer_id','address','chat_id','token','password'}
            if sensitive.intersection(str(c).lower() for c in df.columns):
                findings.append((rel,'private_data_column'))
        else:
            try: text=path.read_text('utf-8')
            except UnicodeDecodeError:
                findings.append((rel,'unreviewed_binary'));continue
        count+=1
        for kind,pattern in PATTERNS.items():
            if re.search(pattern,text):findings.append((rel,kind))
    result={'files_scanned':count,'findings':[{'file':p,'type':t} for p,t in findings]}
    print(json.dumps(result,indent=2))
    return bool(findings)


if __name__=='__main__':sys.exit(scan())
