#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import base64
import os
import re

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
errors = []

required = [
    'pyproject.toml',
    'nanoresearch/cli.py',
    'nanoresearch/pipeline/orchestrator.py',
    'nanoresearch/router_policy.py',
    'nanoresearch/evolution/memory.py',
    'nanoresearch/evolution/skills.py',
    'tools/export_router_sdpo_offpolicy.py',
    'tools/train_router_sdpo_offpolicy.py',
]
for rel in required:
    if not os.path.exists(os.path.join(root, rel)):
        errors.append('missing required workflow file: ' + rel)

for dirpath, _, files in os.walk(root):
    if '__pycache__' in dirpath.split(os.sep):
        errors.append('generated cache directory should not be packaged: ' + os.path.relpath(dirpath, root))
        continue
    for fn in files:
        rel = os.path.relpath(os.path.join(dirpath, fn), root)
        if fn.endswith(('.pyc', '.pyo')):
            errors.append('compiled Python cache should not be packaged: ' + rel)
        if '3x3' in fn:
            errors.append('3x3 preview artifact should not be packaged: ' + rel)

def b64(s):
    return base64.b64decode(s)

security_patterns = {
    'real_api_key': re.compile(b64('c2st') + b'[A-Za-z0-9_-]{20,}'),
    'private_user_name': re.compile(b64('eHVqaW5oYW5n'), re.I),
    'private_mount_path': re.compile(b64('L21udC8=')),
    'internal_model_endpoint': re.compile(b"(" + b64('MzVcLjIyMFwuMTY0XC4yNTI=') + b"|" + b64('bmV3YXBp') + b"|" + b64('anh0YW5n') + rb"|https?://[^\s'\"]+" + b64('OjM4ODg=') + b")", re.I),
    'unsafe_metric_prompt_zh_1': re.compile(b64('5omT6auY5YiG')),
    'unsafe_metric_prompt_zh_2': re.compile(b64('5Yi35YiG')),
    'unsafe_metric_prompt_zh_3': re.compile(b64('5b+F6aG7.*5YiG')),
    'unsafe_metric_prompt_en_1': re.compile(b64('Z2l2ZVxzKyhuYW5vfG91ciBtZXRob2QpLipoaWdo'), re.I),
    'unsafe_metric_prompt_en_2': re.compile(b64('cHJlZmVyXHMrbmFubw=='), re.I),
    'unsafe_metric_prompt_en_3': re.compile(b64('bmFuby4qYmV0dGVy'), re.I),
    'unsafe_metric_prompt_en_4': re.compile(b64('aW5mbGF0ZS4qc2NvcmU='), re.I),
    'unsafe_metric_prompt_en_5': re.compile(b64('ZmF2b3JhYmxlLipzY29yZQ=='), re.I),
    'unsafe_metric_prompt_en_6': re.compile(b64('c2NvcmUuKjhccypbLX50b10rXHMqOQ=='), re.I),
    'unsafe_metric_prompt_en_7': re.compile(b64('c2NhbGUuKjhccypbLX50b10rXHMqOQ=='), re.I),
    'unsafe_data_claim_zh': re.compile(b64('57yW6YCg|55m75pys|5LiN55So55yf5a6e|5LiN6KaB55yf5a6e|5YGH5pWw5o2u|5YGH5a6e6aqM|6Z2i57uE|6YCg5YGH')),
    'unsafe_data_claim_en_1': re.compile(b64('c3ludGhldGljXHMrZmFsbGJhY2s='), re.I),
    'unsafe_data_claim_en_2': re.compile(b64('cXVpY2tccytzeW50aGV0aWM='), re.I),
    'unsafe_data_claim_en_3': re.compile(b64('bWFrZVxzK3VwXHMrKHJlc3VsdHN8bnVtYmVycyk='), re.I),
    'unsafe_data_claim_en_4': re.compile(b64('cGxhdXNpYmxlXHMrYnV0XHMrbm90XHMrcnVu'), re.I),
    'unsafe_data_claim_en_5': re.compile(b64('dHJlYXQgc3ludGhldGlj'), re.I),
    'unsafe_data_claim_en_6': re.compile(b64('aW5kaXN0aW5ndWlzaGFibGUgZnJvbSByZWFs'), re.I),
    'unsafe_data_claim_en_7': re.compile(b64('YXMgaWYgaXQgd2VyZSByZWFsIG1lYXN1cmVk'), re.I),
}

text_suffixes = ('.py', '.md', '.txt', '.json', '.yaml', '.yml', '.sh', '.toml', '.tex')
for dirpath, _, files in os.walk(root):
    if '__pycache__' in dirpath.split(os.sep):
        continue
    for fn in files:
        if not fn.endswith(text_suffixes):
            continue
        path = os.path.join(dirpath, fn)
        if os.path.abspath(path) == os.path.abspath(__file__):
            continue
        rel = os.path.relpath(path, root)
        try:
            data = open(path, 'rb').read()
        except Exception:
            continue
        for check_name, pattern in security_patterns.items():
            if pattern.search(data):
                errors.append('security hygiene violation %s: %s' % (check_name, rel))

# AI-process residue checks target generated papers only. Source prompts may
# legitimately mention citation-key handling or authoring instructions.
ai_trace_patterns = {
    'ai_trace_prompt_reference': re.compile(rb'provided\s+in\s+the\s+prompt|CITATION\s+KEYS|citation\s+keys?\s+(section|list)', re.I),
    'ai_trace_self_narration': re.compile(rb"(Now\s+I|I\s+will|I\'ll|I\s+must|I\s+cannot)\s+write", re.I),
    'ai_trace_search_narration': re.compile(rb'search\s+results\s+(are\s+not\s+relevant|did\s+not\s+return|do\s+not\s+return|confirm)', re.I),
}
for dirpath, _, files in os.walk(root):
    if '__pycache__' in dirpath.split(os.sep):
        continue
    for fn in files:
        if fn != 'paper.tex':
            continue
        path = os.path.join(dirpath, fn)
        rel = os.path.relpath(path, root)
        try:
            data = open(path, 'rb').read()
        except Exception:
            continue
        for check_name, pattern in ai_trace_patterns.items():
            if pattern.search(data):
                errors.append('AI-process trace %s: %s' % (check_name, rel))

if errors:
    for e in errors:
        print('ERROR:', e)
    raise SystemExit(1)
print('release pack smoke checks passed')
