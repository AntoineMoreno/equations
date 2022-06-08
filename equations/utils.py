

def to_latex(str):
    """
    Convert regular text to LaTeX code
    """
    dct={'Delta' : '\Delta', 'alpha' : '\\alpha', 'ascii_124' : '|', 'beta':'\\beta', 'cos':'\cos', \
    'exists':'\exists','forall':'\\forall', 'forward_slash':'/', 'gamma':'\gamma', 'geq' :'\geq',\
    'gt':'>', 'in':'\in','infty':'\infty','lambda':'\lambda', 'ldots':'\lodts', 'leq':'\leq',\
    'lim' : '\lim', 'log':'\log','It':'<','mu':'\mu', 'neq':'\\neq', 'phi':'\phi','pi':'\pi',\
    'pm':'\pm','rightarrow':'\\rightarrow', 'sigma': '\sigma','sin':'\sin', 'sqrt':'\sqrt',\
    'sum': '\sum', 'tan':'\\tan', 'theta':'\\theta', 'times': '\cdot'} #attention div et int et lim et prime et sum
    for word, symbol in dct.items():
        str = str.replace(word, symbol)
    return str

def get_class_name(train_ds):
    return tuple(train_ds.class_names)


def give_classes():
    class_names = ['!', '(', ')', '+', ',', '-', '0', '1',
 '2', '3', '4', '5', '6', '7', '8', '9', '=',
 'A', 'C', 'Delta', 'G', 'H', 'M', 'N', 'R', 'S',
 'T', 'x', '[', ']', 'alpha', 'ascii_124', 'b',
 'beta', 'cos', 'd', 'div', 'e', 'exists', 'f',
 'forall', 'forward_slash', 'gamma', 'geq', 'gt',
 'i', 'in', 'infty', 'int', 'j', 'k', 'l', 'lambda',
 'ldots', 'leq', 'lim', 'log', 'lt', 'mu', 'neq',
 'o', 'p', 'phi', 'pi', 'pm', 'prime', 'q',
 'rightarrow', 'sigma', 'sin', 'sqrt', 'sum', 'tan',
 'theta','times', 'u', 'v','w','y','z','{','}']
    return class_names
