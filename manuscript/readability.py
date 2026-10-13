"""Crude readability metrics of the running text of main.tex (math, floats, appendix and comments removed).
Usage: python3 readability.py [N_longest] [main|appendix|supp]"""
import re
import sys

which = sys.argv[2] if len(sys.argv) > 2 else "main"
t = open("supplement.tex" if which == "supp" else "main.tex").read()
if which == "main":
    t = t[t.index("\\maketitle"):t.index("\\appendix")]
elif which == "appendix":
    t = t[t.index("\\appendix"):]
else:
    t = t[t.index("\\begin{document}"):]
t = re.sub(r"%.*", "", t)
for env in ("figure\\*?", "table\\*?", "equation\\*?", "align\\*?", "tikzpicture", "ruledtabular", "lemma", "proposition", "theorem", "remark", "proof", "description"):
    t = re.sub(r"\\begin\{(%s)\}.*?\\end\{\1\}" % env, " ", t, flags=re.S) if env not in ("lemma", "proposition", "theorem", "remark", "description") else t
t = re.sub(r"\$[^$]*\$", "X", t)
t = re.sub(r"\\(begin|end)\{[^}]*\}(\[[^\]]*\])?", " ", t)
t = re.sub(r"\\(input|label|cite|ref|eqref|texttt|SM)\{[^}]*\}", "X", t)
t = re.sub(r"\\(emph|textbf|paragraph\*?|section|subsection|item)\*?(\[[^\]]*\])?\{?([^}]*)\}?", r"\3", t)
t = re.sub(r"\\[a-zA-Z]+|[{}~\\]", " ", t)
sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", t) if len(s.split()) > 3]


def syl(w):
    w = re.sub(r"[^a-z]", "", w.lower())
    return max(1, len(re.findall(r"[aeiouy]+", w)) - (w.endswith("e") and not w.endswith("le")))


words = [w for s in sents for w in s.split()]
S, W = len(sents), len(words)
Y = sum(syl(w) for w in words)
fre = 206.835 - 1.015 * W / S - 84.6 * Y / W
lens = sorted(((len(s.split()), s) for s in sents), reverse=True)
print(f"sentences {S}, words {W}, mean {W / S:.1f} words/sentence, >35 words: {sum(1 for n, _ in lens if n > 35)}, "
      f">45 words: {sum(1 for n, _ in lens if n > 45)}, Flesch reading ease {fre:.1f}")
for n, s in lens[:int(sys.argv[1]) if len(sys.argv) > 1 else 0]:
    print(n, s[:200])
