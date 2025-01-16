import inspect


def get_members(module):
    def predicate(x): return inspect.ismodule or inspect.function

    for x, y in inspect.getmembers(module, predicate):
        try:
            md = y.__module__
            if md != "__main__":
                yield (x, y.__name__, y.__module__)
        except:
            if inspect.ismodule(y):
                yield (x, y.__name__)


def get_imports(c):

    source_file = inspect.getsourcefile(c)
    with open(source_file, 'r') as f:
        lines = f.readlines()
    imports = [l for l in lines if l.strip().startswith('import')
               or l.strip().startswith('from')]
    return imports


def get_src(c, imports=True):
    src = []
    source_code = inspect.getsource(c)

    if imports:
        import_lines = get_imports(c)
        src += import_lines
        src.append('\n')

    src += source_code
    return src
