def to_latex(str):
   dct={'Delta' : '\Delta', 'alpha' : '\alpha', 'ascii_124' : '|', 'beta':'\beta', 'cos':'\cos', \
     'exists':'\exists','forall':'\forall', 'forward_slash':'/', 'gamma':'\gamma', 'geq' :'\geq',\
     'gt':'>', 'in':'\in','infty':'\infty','lambda':'\lambda', 'ldots':'\lodts', 'leq':'\leq',\
     'lim' : '\lim', 'log':'\log','It':'<','mu':'\mu', 'neq':'\neq', 'phi':'\phi','pi':'\pi',\
     'pm':'\pm','rightarrow':'\rightarrow', 'sigma': '\sigma','sin':'\sin', 'sqrt':'\sqrt',\
     'sum': '\sum', 'tan':'\tan', 'theta':'\theta', 'times': '\cdot'} #attention div et int et lim et prime et sum
    for word, symbol in dct.items():
        str = str.replace(word, symbol)
    return str
