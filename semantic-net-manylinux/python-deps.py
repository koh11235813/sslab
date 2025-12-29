import graphviz
from subprocess import run, DEVNULL, PIPE
ss = run(['pipdeptree'], stdout=DEVNULL, stderr=PIPE,
         universal_newlines=True).stderr.rstrip().split('\n')
ss = [s[2:].lower().split()[:3] for s in ss if s[0] in ' *']
g = graphviz.Digraph(format='png', filename='python-dep', engine='dot')
g.edges([(s[2], s[0][:s[0].index('=')]) for s in ss])
g.attr('graph', rankdir='LR')
g.render()

