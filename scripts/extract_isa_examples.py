#!/usr/bin/env python3
"""Extract the example snippets of the PTX ISA HTML document into a JSON list (input of isa_examples.py).

usage: extract_isa_examples.py ptx-isa.html examples.json
Download the document from https://docs.nvidia.com/cuda/parallel-thread-execution/index.html (it is not part of this repository).
"""
import html, json, re, sys

s = open(sys.argv[1], encoding="utf8", errors="ignore").read()
txt = lambda x: html.unescape(re.sub(r"<[^>]+>", "", x))
secs = [(m.start(), m.group(1)) for m in re.finditer(r'<section id="([^"]+)">', s)]
out = []
for m in re.finditer(r'<p class="rubric">Examples?</p>\s*<div class="highlight-[^"]*">\s*<div class="highlight"><pre>(.*?)</pre>', s, flags=re.S):
    sec = None
    for p, i in secs:
        if p < m.start(): sec = i
        else: break
    out.append({"sec": sec, "code": txt(m.group(1))})
json.dump(out, open(sys.argv[2], "w"))
print(len(out), "examples")
