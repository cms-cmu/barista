import ast
import inspect


class ClassAttributeDocstring(ast.NodeVisitor):
    def __init__(self):
        super().__init__()

        self.docstrings: dict[type, dict[str, str]] = {}

        self.__cls: dict[str, str] = {}
        self.__attr: tuple[str, int] = None

    # skip methods
    def visit_FunctionDef(self, _: ast.FunctionDef): ...

    # parse class
    def visit_ClassDef(self, node: ast.ClassDef):
        self.__cls = {}
        self.docstrings[node.name] = self.__cls
        self.generic_visit(node)
        self.__cls = {}

    # parse annotated attributes
    def visit_AnnAssign(self, node: ast.AnnAssign):
        if isinstance(node.target, ast.Name):
            self.__attr = (node.target.id, node.end_lineno)

    # parse unannotated attributes
    def visit_Assign(self, node: ast.Assign):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            self.__attr = (node.targets[0].id, node.end_lineno)

    # add docstring
    def visit_Expr(self, node: ast.Expr):
        if (
            isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            and self.__attr is not None
            and node.lineno == (self.__attr[1] + 1)
        ):
            self.__cls[self.__attr[0]] = node.value.value

        if self.__attr is not None and node.lineno > (self.__attr[1] + 1):
            self.__attr = None


def class_attribute_docstring(cls: type, clean: bool = True) -> dict[str, str]:
    parser = ClassAttributeDocstring()
    parser.visit(ast.parse(inspect.getsource(cls)))
    docs = parser.docstrings[cls.__name__]
    if clean:
        for k in docs:
            docs[k] = inspect.cleandoc(docs[k])
    return docs
