#!/usr/bin/env python3
"""
wn2solr.py – Dump WordNet lemmas to Solr-style synonyms.txt
• One equivalence line per synset
• Lower-cases, ASCII-folds, converts '_' → ' '
• Stops when it hits 20 000 rules (adjust if you split maps)
"""
import unicodedata, itertools, sys
from nltk.corpus import wordnet as wn

def norm(t):
    t = t.replace("_", " ").lower()
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()
    return t

seen = set()
for syn in wn.all_synsets():
    lemmas = {norm(l.name()) for l in syn.lemmas()}
    if len(lemmas) > 1:
        line = ", ".join(sorted(lemmas))
        if line not in seen:
            print(line)
            seen.add(line)
            if len(seen) >= 20_000:       # stay under Azure’s limit
                sys.stderr.write("20k reached – stop\n")
                break
