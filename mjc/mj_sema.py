import argparse
import pathlib
import sys
from copy import deepcopy
from typing import Any, Dict, Union

from mjc.mj_ast import *
from mjc.mj_parser import MJParser
from mjc.mj_serror import SE, assert_semantic
import inspect
from mjc.mj_type import (
    BooleanType,
    CharArrayType,
    CharType,
    IntArrayType,
    IntType,
    MJType,
    ObjectType,
    StringType,
    VoidType,
)

class SymbolTable:
    """Class representing a symbol table.

    `add` and `lookup` methods are given, however you still need to find a way to
    deal with scopes.

    ## Attributes
    :data: the content of the SymbolTable
    """

    def __init__(self) -> None:
        """Initializes the SymbolTable."""
        self.__data = dict()
        self.__type = dict()
        self.__scope = list()
        self.__class_method = dict()
        self.__class_attributes = dict()
        self.__stack_old_type = list()

    @property
    def data(self) -> Dict[str, Any]:
        """Returns a copy of the SymbolTable."""
        return deepcopy(self.__data)

    def add(self, name: str, value: Any) -> None:
        """Adds to the SymbolTable.

        :param name: the identifier on the SymbolTable
        :param value: the value to assign to the given `name`
        """
        self.__data[name] = value


    def lookup(self, name: str) -> Union[Any, None]:
        """Searches `name` on the SymbolTable and returns the value
        assigned to it.

        :param name: the identifier that will be searched on the SymbolTable
        :return: the value assigned to `name` on the SymbolTable. If `name` is not found, `None` is returned.
        """
        return self.__data.get(name)



    def type_set(self, name: str, type_str: str):
        while not isinstance(type_str, str):
            type_str =  type_str.name

        if name in self.__type:
            self.__stack_old_type.append((name, self.__type[name]))

        if type_str == "boolean":
            type_ = BooleanType
        elif type_str == "char":
            type_ = CharType
        elif type_str == "int":
            type_ = IntType
        elif type_str == "String":
            type_ = StringType
        elif type_str == "void":
            type_ = VoidType
        elif type_str == "int[]":
            type_ = IntArrayType
        elif type_str == "char[]":
            type_ = CharArrayType
        elif type_str == "void":
            type_ =  VoidType
        else: #pode dar pau, pensar em como melhor isso
            type_ = ObjectType(type_str)
        

        self.__type[name] = type_

    def type_get(self, name: str) -> MJType:
        return self.__type.get(name, None)


    def exist_in_old_scope(self, name):
        #irei trabalhae como pilha, então, percorro de trás para frente
        stack = self.__stack_old_type
        for i in range(len(stack) - 1, -1, -1)  :
            if name == stack[i][0]:
                return stack.pop(i)[1]
        return None

    def rm_type(self, names: list[str]):
        for name in names:
            if name in self.__type:
                type =  self.exist_in_old_scope(name)
                if type is not None:
                    self.__type[name] = type
                else:
                    del self.__type[name]


    def push_scope(self, id_name):
        # print(self.__scope)
        if id_name in self.__scope:
            for id in reversed(self.__scope):
                if id_name == id:
                    return True
                #se ele achar um novo escopo ele permite a escrita
                if id == "new_scope":
                    break

                
        self.__scope.append(id_name)
        return False

    # def print(self, name):
    #     print(self.__class_method["class Program"]["find_lcm"])
    #     print(self.__class_method.keys())

    def pop_scope(self, n=1):
        for _ in range(n):
            if self.__scope:
                self.__scope.pop()

    def old_scope(self, n):
        return self.__scope[n:]

    def len_scope(self):
        return len(self.__scope)
    
    def add_loop(self, type, line, column):
        self.__scope.append(f"{type} @ {line}:{column}")

    def rm_loop(self):
        # Remove the last loop scope ("for @" or "while @") from the stack.
        for i in range(len(self.__scope) - 1, -1, -1):
            if self.__scope[i].startswith("for @") or self.__scope[i].startswith("while @"):
                self.__scope.pop(i)
                break

    def in_loop(self):
        for entry in reversed(self.__scope):
            if entry.startswith("for @") or entry.startswith("while @"):
                return True
        return False
    
    def add_class(self, class_name,):
        if not class_name.startswith("class"):
            class_name = "class " + class_name
        self.__scope.append(f"{class_name}")


    def rm_class(self):
        for i in range(len(self.__scope) - 1, -1, -1):
            if self.__scope[i].startswith("class "):
                self.__scope.pop(i)
                break

    def in_class(self):
        for entry in reversed(self.__scope):
            if entry.startswith("class "):
                return entry
        return False

    def search_class(self, name: str):
        if not name.startswith("class "):
           name = "class " + name
        for scope in self.__scope:
            if scope == name:
                return True
        return False
    
    def last_scope_is_class(self):
        """
        Returns True if the last scope is a class and there is no method scope after it.
        """
        if not self.__scope:
            return False
        # Find the last class scope
        for entry in reversed(self.__scope):
            if entry.startswith("class "):
                # Check if there is any method scope after this class scope
                idx = self.__scope.index(entry)
                for later_entry in self.__scope[idx + 1:]:
                    if later_entry.startswith("method "):
                        return False
                return True
        return False

    def add_main_method(self):
        name_class = self.in_class()
        if name_class not in self.__class_method:
            self.__class_method[name_class] = {}
        self.__class_method[name_class]["method main"] = (["String[]"], 1, "void")

    def add_method(self, name, param_init, return_type):
        if not name.startswith("method "):
           name = "method " + name
        
        params = []
        for param in param_init:
            params.append(param.type.name)
            # params.append((param.type.name, param.name.name))

        name_class = self.in_class()

        if name_class not in self.__class_method:
            self.__class_method[name_class] = {}
        self.__class_method[name_class][name] = (params, len(params), return_type)

    def search_method(self, name: str, class_name: str = None) -> Union[Any, None]:
        # for key in self.__class_method.keys(): 
            # print(key, self.__class_method[key])
        
        
        if not name.startswith("method "):
           name = "method " + name

        if class_name is None:
            class_name = self.in_class()

        if not class_name.startswith("class "):
            class_name = "class " + class_name

        for name_method in self.__class_method[class_name]:
            if name == name_method:
               return self.__class_method[class_name][name]
            
            if not "method" in name:
                name = f"method {name}"
            else:
                name = name.replace("method ", "")
            if  name == name_method:
               return self.__class_method[class_name][name]
            
        return None
    
    def in_method(self):
        for entry in reversed(self.__scope):
            if entry.startswith("method "):
                return entry
        return False

    def copy_class_members(self, source_class: str, target_class: str):
        

        if not source_class.startswith("class "):
            source_class = "class " + source_class
        if not target_class.startswith("class "):
            target_class = "class " + target_class


        if source_class in self.__class_method:
            if target_class not in self.__class_method:
                self.__class_method[target_class] = {}
            for method_name, method_info in self.__class_method[source_class].items():
                self.__class_method[target_class][method_name] = deepcopy(method_info)

        if source_class in self.__class_attributes:
            if target_class not in self.__class_attributes:
                self.__class_attributes[target_class] = {}
            for attr_name, attr_type in self.__class_attributes[source_class].items():
                self.__class_attributes[target_class][attr_name] = deepcopy(attr_type)

    def add_attribute(self, class_name: str, attr_name: str, attr_type: str):
        if not class_name.startswith("class "):
            class_name = "class " + class_name
        if class_name not in self.__class_attributes:
            self.__class_attributes[class_name] = {}
        self.__class_attributes[class_name][attr_name] = attr_type

    def has_attribute(self, class_name: str, attr_name: str) -> bool:
        if not class_name.startswith("class "):
            class_name = "class " + class_name
        return (
            class_name in self.__class_attributes
            and attr_name in self.__class_attributes[class_name]
        )
    
    def get_attribute_type(self, class_name: str, attr_name: str):
        if not class_name.startswith("class "):
            class_name = "class " + class_name
        if class_name in self.__class_attributes and attr_name in self.__class_attributes[class_name]:
            return self.__class_attributes[class_name][attr_name]
        return None


class NodeVisitor:
    """A base NodeVisitor class for visiting uc_ast nodes.
    Subclass it and define your own visit_XXX methods, where
    XXX is the class name you want to visit with these
    methods.
    """

    _method_cache = None

    def visit(self, node):
        """Visit a node."""

        if self._method_cache is None:
            self._method_cache = {}

        visitor = self._method_cache.get(node.__class__.__name__)
        if visitor is None:
            method = "visit_" + node.__class__.__name__
            # print("-->", method)
            visitor = getattr(self, method, self.generic_visit)
            self._method_cache[node.__class__.__name__] = visitor

        # try:
        #     coord = node.coord
        # except:
        #     coord = -1

        # print(f"Usando método: {visitor.__name__} | {node.__class__.__name__} : {coord}")
        return visitor(node)

    def generic_visit(self, node):
        """Called if no explicit visitor function exists for a
        node. Implements preorder visiting of the node.
        """

        for _, child in node.children():
            self.visit(child)


class SymbolTableBuilder(NodeVisitor):
    """Symbol Table Builder class.
    This class build the Symbol table of the program by visiting all the AST nodes
    using the visitor pattern.
    """

    def __init__(self):
        self.global_symtab = SymbolTable()
        self.typemap = {
            "boolean": BooleanType,
            "char": CharType,
            "int": IntType,
            "String": StringType,
            "void": VoidType,
            "int[]": IntArrayType,
            "char[]": CharArrayType,
            "object": ObjectType,
        }

    def visit_Program(self, node: Program):
        """Visit the program node to fill in the global symbol table"""
        # Register all classes in the global symbol table
        for class_decl in node.class_decls:
            class_name = class_decl.name.name
            
            if self.global_symtab.lookup(class_name) is not None:
                assert_semantic(
                    condition=False,
                    error_type=SE.ALREADY_DECLARED_CLASS,
                    coord=class_decl.coord,
                    name=class_name,
                )
            self.global_symtab.add(class_name, class_decl)
        
        # Now, process each class to fill in fields and methods
        for class_decl in node.class_decls:
            self.visit(class_decl)

        return self.global_symtab


    def visit_ClassDecl(self, node: ClassDecl):

        self.current_class = node.name.name

        self.global_symtab.add_class(node.name.name)
        self.current_class = self.global_symtab.lookup(node.name.name) 

        if node.extends is not None:
            parent_name = node.extends.super.name
            if self.global_symtab.lookup(parent_name) is None:
                assert_semantic(
                    condition=False,
                    error_type=SE.UNDECLARED_CLASS,
                    coord=node.coord,
                    name=parent_name,
                )

        for field in node.var_decls:
            self.global_symtab.add_attribute(
                f"class {node.name.name}",
                field.name.name,
                field.type
            )
            self.visit(field)

        for method in node.method_decls:
            self.visit(method)

        self.current_class = None


    def visit_VarDecl(self, node: VarDecl):
        var_name = node.name.name


        if self.global_symtab.lookup(var_name) is not None:
            assert_semantic(
                condition=False,
                error_type=SE.ALREADY_DECLARED_NAME,
                coord=node.coord,
                name=var_name,
            )


        self.global_symtab.type_set(var_name, node.type)

        var_type = self.global_symtab.type_get(var_name)

        if var_type == ObjectType:
            # Ensure we propagate errors and do not avoid them
            var_type = node.type.name.name
            if self.global_symtab.search_class(var_type) == False:
                assert_semantic(
                    condition=False,
                    error_type=SE.UNDECLARED_CLASS,
                    coord=node.coord,
                    name=var_type,
                )

        node.mj_type = var_type
        # print(var_type, node.coord)
        # print(node)

        return self.global_symtab
        

    def visit_MethodDecl(self, node: MethodDecl):

        method_name = node.name.name
        
        params = node.param_list.params if node.param_list is not None else []
        if self.global_symtab.add_method(method_name, params, node.type.name) is not None:
            assert_semantic(
                condition=False,
                error_type=SE.ALREADY_DECLARED_METHOD,
                coord=node.coord,
                name=method_name,
            )
        
        self.global_symtab.add(method_name, node)
        return self.global_symtab

    def visit_MainMethodDecl(self, node: MainMethodDecl):
        method_name = "main"

        self.current_class = method_name

        if self.global_symtab.lookup(method_name) is not None:
            assert_semantic(
                condition=False,
                error_type=SE.ALREADY_DECLARED_METHOD,
                coord=node.coord,
                name=method_name,
            )

        self.global_symtab.type_set(method_name, node)

        for statement in node.body.statements:
            self.visit(statement)
        

class SemanticAnalyzer(NodeVisitor):
    """Semantic Analyzer class.
    This class performs semantic analysis on the AST of a MiniJava program.
    You need to define methods of the form visit_NodeName()
    for each kind of AST node that you want to process.
    """

    def __init__(self, global_symtab: SymbolTable):
        """
        :param global_symtab: Global symbol table with all class declaration metadata.
        """
        self.global_symtab = global_symtab
        self.typemap = {
            "boolean": BooleanType,
            "char": CharType,
            "int": IntType,
            "String": StringType,
            "void": VoidType,
            "int[]": IntArrayType,
            "char[]": CharArrayType,
            "object": ObjectType,
        }

    def visit_Program(self, node: Program):
        for cls in node.class_decls:
            self.visit(cls)

    def visit_ClassDecl(self, node: ClassDecl):

        for method in node.method_decls:
            self.global_symtab.add_method(method.name.name, method.param_list.params, method.type.name)

        if node.extends != None:
            self.visit(node.extends)

        for field in node.var_decls:
            self.visit(field)

        for method in node.method_decls:
            self.visit(method)


        self.global_symtab.rm_class()


    def visit_VarDecl(self, node: VarDecl):

        erro_type = SE.ALREADY_DECLARED_FIELD if self.global_symtab.last_scope_is_class() else SE.ALREADY_DECLARED_NAME


        if self.global_symtab.push_scope(node.name.name): 
            assert_semantic(
                condition=False,
                error_type=erro_type,
                coord=node.coord,
                name=node.name.name,
            )


        #TODO: modifique aqui para evitar tipar duas vezes
        self.global_symtab.type_set(node.name.name, node.type.name)

        node.name.mj_type = self.global_symtab.type_get(node.name.name)

        
        self.visit(node.name)

        if hasattr(node.name.mj_type, "real_type"):
            real_type_name = getattr(node.name.mj_type.real_type, "name", None)
            if real_type_name and not self.global_symtab.search_class(real_type_name):
                assert_semantic(
                    condition=False,
                    error_type=SE.UNDECLARED_CLASS,
                    coord=node.coord,
                    name=real_type_name,
                )

        try:
        # if node != None:
            self.visit(node.init)
        except:
            pass

        if "[]" in node.name.mj_type.typename and isinstance(node.init, InitList):
            for exp in node.init.exprs:
                if not isinstance(exp, Constant):
                    assert_semantic(
                        condition=False,
                        error_type=SE.NOT_A_CONSTANT,
                        coord=exp.coord,
                    )
                if not (exp.type in node.name.mj_type.typename):
                    assert_semantic(
                        condition=False,
                        error_type=SE.ARRAY_ELEMENT_TYPE_MISMATCH,
                        coord=exp.coord,
                        rtype=f"type({exp.type})",
                        ltype=f"type({node.name.mj_type.typename.replace('[]','')})",
                        name=exp.value,
                    )

    def visit_MethodDecl(self, node: MethodDecl):

        params = node.param_list.params if node.param_list is not None else []
        self.global_symtab.add_method(f"method {node.name.name}", params, node.type.name)

        if self.global_symtab.push_scope(f"method {node.name.name}"):

            assert_semantic(
                condition=False,
                error_type=SE.ALREADY_DECLARED_METHOD,
                coord=node.coord,
                name=node.name.name,
            )
        self.global_symtab.type_set(f"method {node.name.name}", node.type.name)
        node.mj_type = self.global_symtab.type_get("method {node.name.name}")
        len_scope = self.global_symtab.len_scope()

        for param in params:
            self.global_symtab.type_set(param.name.name, param.type.name)
            if self.global_symtab.push_scope(param.name.name):
                assert_semantic(
                condition=False,
                error_type=SE.PARAMETER_ALREADY_DECLARED,
                coord=param.coord,
                name=param.name.name,
            )
            self.visit(param)

        for stmt in node.body.statements:
            # print(f">>>>>\n{type(stmt)}:{stmt.coord}\n{stmt}\n<<<<<\n")
            self.visit(stmt)


        old_scope = self.global_symtab.old_scope(len_scope)

        self.global_symtab.rm_type(old_scope)
        self.global_symtab.pop_scope(self.global_symtab.len_scope() - len_scope)



        


    def visit_MainMethodDecl(self, node: MainMethodDecl):

        if self.global_symtab.push_scope("method main"):
            assert_semantic(
                condition=False,
                error_type=SE.ALREADY_DECLARED_METHOD,
                name="main",
                coord=node.coord
            )

        #TODO; verificar
        len_scope = self.global_symtab.len_scope()
        self.global_symtab.add_main_method()
        self.global_symtab.type_set(node.args.name, "String") 



        self.visit(node.args)

        len_scope = self.global_symtab.len_scope()

        for stmt in node.body.statements:

            self.visit(stmt)


        old_scope = self.global_symtab.old_scope(len_scope) #pego as variaveis que foram declaradas nesse escopo
        self.global_symtab.rm_type(old_scope)
        self.global_symtab.pop_scope(self.global_symtab.len_scope() - len_scope)


    def visit_ParamList(self, node: ParamList):
        pass

    def visit_ParamDecl(self, node: ParamDecl):
        pass

    def visit_Compound(self, node: Compound):


        len_scope = self.global_symtab.len_scope()
        self.global_symtab.push_scope("new_scope")

        for stmt in node.statements:
            self.visit(stmt)

        old_scope = self.global_symtab.old_scope(len_scope) #pego as variaveis que foram declaradas nesse escopo
        self.global_symtab.rm_type(old_scope)
        self.global_symtab.pop_scope(self.global_symtab.len_scope() - len_scope)


    def visit_If(self, node: If):

        len_scope = self.global_symtab.len_scope()
        self.global_symtab.push_scope("new_scope")
        self.visit(node.iftrue)

        if hasattr(node, "cond"):
            self.visit(node.cond)
            cond_type = getattr(node.cond, "mj_type", None)
            if cond_type is not BooleanType:
                assert_semantic(
                    condition=False,
                    error_type=SE.CONDITIONAL_EXPRESSION_TYPE_MISMATCH,
                    ltype=cond_type,
                    coord=getattr(node, "coord", None),
                )


        old_scope = self.global_symtab.old_scope(len_scope) #pego as variaveis que foram declaradas nesse escopo
        self.global_symtab.rm_type(old_scope)
        self.global_symtab.pop_scope(self.global_symtab.len_scope() - len_scope)

        len_scope = self.global_symtab.len_scope()

        if node.iffalse != None:
            self.visit(node.iffalse)

        old_scope = self.global_symtab.old_scope(len_scope) #pego as variaveis que foram declaradas nesse escopo
        self.global_symtab.rm_type(old_scope)
        self.global_symtab.pop_scope(self.global_symtab.len_scope() - len_scope)


    def visit_While(self, node: While):
        self.global_symtab.add_loop("while", node.coord.line, node.coord.column)
        len_scope = self.global_symtab.len_scope()
        self.global_symtab.push_scope("new_scope")

        self.visit(node.cond)
        self.visit(node.body)

        old_scope = self.global_symtab.old_scope(len_scope) #pego as variaveis que foram declaradas nesse escopo
        self.global_symtab.rm_type(old_scope)
        self.global_symtab.pop_scope(self.global_symtab.len_scope() - len_scope)
        self.global_symtab.rm_loop()

    def visit_For(self, node: For):
        self.global_symtab.add_loop("for", node.coord.line, node.coord.column)
        len_scope = self.global_symtab.len_scope()
        self.global_symtab.push_scope("new_scope")

        # print(type(node.init))
        self.visit(node.init)
        self.visit(node.cond)
        self.visit(node.next)


        self.visit(node.body)

        old_scope = self.global_symtab.old_scope(len_scope) #pego as variaveis que foram declaradas nesse escopo
        self.global_symtab.rm_type(old_scope)
        self.global_symtab.pop_scope(self.global_symtab.len_scope() - len_scope)
        self.global_symtab.rm_loop()

        

    def visit_DeclList(self, node: DeclList):
        for decl in node.decls:
            # print("\t",type(decl))
            self.visit(decl)
        

    def visit_Print(self, node: Print):
        #não sei se cria scope
        if node.expr != None:
            self.visit(node.expr)
            expr_type = getattr(node.expr, "mj_type", None)

            if expr_type not in (StringType, IntType, CharType, CharArrayType):
                assert_semantic(
                    condition=False,
                    error_type=SE.PRINT_EXPRESSION_TYPE_MISMATCH,
                    coord=getattr(node, "coord", None),
            )

    def visit_Assert(self, node: Assert):
        self.visit(node.expr)
        type = node.expr.mj_type

        if type is not BooleanType:
            assert_semantic(
                condition=False, 
                error_type=SE.ASSERT_EXPRESSION_TYPE_MISMATCH,
                coord=node.coord,
            )


    def visit_Break(self, node: Break):
        if not self.global_symtab.in_loop():
            assert_semantic(
                condition=False,
                error_type=SE.WRONG_BREAK_STATEMENT,
                coord=node.coord,
            )

    def visit_Return(self, node: Return):
        if node.expr is not None:
            self.visit(node.expr)

        type_return = None
        if (type(node.expr) is ID):
            type_return = self.global_symtab.type_get(node.expr.name).typename
        elif (type(node.expr) is BinaryOp): #TODO: solução momentânea, deveria criar algo mais robusto para definir o tipo de uma expressão
            type_return = self.global_symtab.type_get(node.expr.lvalue.name).typename

        type_method = self.global_symtab.search_method(self.global_symtab.in_method())[2]

        # Compare types and raise error if mismatch
        if type_return is not None and type_method is not None and type_return != type_method:
            assert_semantic(
                condition=False,
                error_type=SE.RETURN_TYPE_MISMATCH,
                coord=node.coord,
                rtype=f"type({type_method})",
                ltype=f"type({type_return})",
            )

    def visit_Assignment(self, node: Assignment):
        self.visit(node.rvalue)

        self.visit(node.lvalue)



        if isinstance(node.lvalue, ID):
            assert_semantic(
                condition=(node.lvalue is not None),
                error_type=SE.UNDECLARED_NAME,
                coord=node.coord,
                name=node.lvalue.name,
            )

        ltype = getattr(node.lvalue, "mj_type", None)
        rtype = getattr(node.rvalue, "mj_type", None)


        if ltype is None:
            ltype = self.global_symtab.type_get(getattr(node.lvalue, "name", None))
        if rtype is None:
            rtype = self.global_symtab.type_get(getattr(node.rvalue, "name", None))


        if isinstance(node.lvalue, ArrayRef):
            array_type = getattr(node.lvalue.name, "mj_type", None)
            if array_type is None:
                array_type = self.global_symtab.type_get(getattr(node.lvalue.name, "name", None))
            if array_type is IntArrayType:
                expected_type = IntType
            elif array_type is CharArrayType:
                expected_type = CharType
            else:
                expected_type = None


            if rtype != expected_type:
                assert_semantic(
                    condition=False,
                    error_type=SE.ASSIGN_TYPE_MISMATCH,
                    coord=node.coord,
                    ltype=expected_type,
                    rtype=rtype,
                )

        elif isinstance(node.rvalue, MethodCall):
            self.visit(node.rvalue)
            method_name = node.rvalue.method_name.name
            def_method = self.global_symtab.search_method(method_name)
            node.rvalue.mj_type = MJType(def_method[2])
            node.mj_type = def_method

            rtype = MJType(def_method[2])

            # print(type(rtype), type(ltype))

            if ltype.typename != rtype.typename:
                assert_semantic(
                    condition=False,
                    error_type=SE.ASSIGN_TYPE_MISMATCH,
                    coord=node.coord,
                    ltype=ltype,
                    rtype=rtype,
                )

        else:   
            if ltype != rtype:
                assert_semantic(
                    condition=False,
                    error_type=SE.ASSIGN_TYPE_MISMATCH,
                    coord=node.coord,
                    ltype=ltype,
                    rtype=rtype,
                )

        if getattr(node.lvalue, "mj_type", None) == None:
            node.lvalue.mj_type = rtype
            

    def visit_BinaryOp(self, node: BinaryOp):
        # Visit the left expression
        self.visit(node.lvalue)
        # Visit the right expression
        self.visit(node.rvalue)
        # Check if left and right operands have the same type

        ltype = node.lvalue.mj_type
        rtype = node.rvalue.mj_type        


        ltype = node.lvalue.mj_type
        rtype = node.rvalue.mj_type  


        # print(f"{node.coord}\n{node.lvalue}")
        

        if ltype is None and hasattr(node.lvalue, "name"):
            ltype = self.global_symtab.type_get(node.lvalue.name)
        # else:
        #     print(f"L ({type(node.lvalue)}): {node.lvalue.mj_type}")
        if rtype is None and hasattr(node.lvalue.mj_type.rvalue, "name"):
            rtype = self.global_symtab.type_get(node.rvalue.name)
        # else:
        #    print(f"R ({type(node.rvalue)}): {node.rvalue.mj_type}")

        # print(type(node.lvalue), type(node.rvalue))
        # print(type(ltype), type(rtype))
        # print(ltype, rtype)

        if (ltype != rtype):
            assert_semantic(
                condition=(ltype == rtype),
                error_type=SE.BINARY_EXPRESSION_TYPE_MISMATCH,
                coord=node.coord,
                name=node.op,
                ltype=ltype,
                rtype=rtype,
            )

        # Check if the operator is supported by the type
        if node.op not in ltype.binary_ops and node.op not in ltype.rel_ops:
            assert_semantic(
                condition=False,
                error_type=SE.UNSUPPORTED_BINARY_OPERATION,
                coord=node.coord,
                name=node.op,
                ltype=ltype,
            )
        if node.op in ltype.rel_ops:
            node.mj_type = BooleanType
        else:
            node.mj_type = ltype


    def visit_UnaryOp(self, node: UnaryOp):
        self.visit(node.expr)


        expr_type = getattr(node.expr, "mj_type", None)
        if expr_type is None:
            expr_type = self.global_symtab.type_get(getattr(node.expr, "name", None))

        # Check if the operator is supported by the type
        # Use unary_ops for unary operators
        if not hasattr(expr_type, "unary_ops") or node.op not in expr_type.unary_ops:
            assert_semantic(
                condition=False,
                error_type=SE.UNSUPPORTED_UNARY_OPERATION,
                coord=node.coord,
                name=node.op,
            )

        # Assign the result type of the unary expression
        node.mj_type = expr_type



    def visit_ArrayRef(self, node: ArrayRef):
        self.visit(node.name)
        self.visit(node.subscript)

        array_type = getattr(node.name, "mj_type", None)
        if array_type is None:
            array_type = self.global_symtab.type_get(getattr(node.name, "name", None))



        if isinstance(node.name, ID):
            type_var = self.global_symtab.type_get(node.name.name)
        if isinstance(node.name, FieldAccess):
            type_var = node.name.mj_type

        if not "[]" in type_var.typename:
            assert_semantic(
                condition=False,
                error_type=SE.ARRAY_REF_TYPE_MISMATCH,
                coord=node.coord,
                ltype=type_var,
            )


        subscript_type = self.global_symtab.type_get(node.subscript)
        if subscript_type is None:
            subscript_type = self.global_symtab.type_get(getattr(node.subscript, "name", None))
            subscript_type = node.subscript.mj_type


        if subscript_type is not IntType:
            assert_semantic(
                condition=False,
                error_type=SE.ARRAY_DIMENSION_MISMATCH,
                coord=node.coord,
                ltype=subscript_type,
            )

        if array_type is IntArrayType:
            node.mj_type = IntType
        elif array_type is CharArrayType:
            node.mj_type = CharType
        else:
            node.mj_type = None


    def visit_FieldAccess(self, node: FieldAccess):
        self.visit(node.object)


        objectIsNotClass =  True

        if isinstance(node.object, This):
            name_class = self.global_symtab.in_class()
            objectIsNotClass = False
        else:
            if isinstance(node.object.mj_type, ObjectType):
                name_class =  node.object.mj_type.real_type
                # name_class = name_class.name #mudei agora
                objectIsNotClass = False
            else:
                name_class = node.object.name




        if objectIsNotClass:
            assert_semantic(
                condition=False,
                error_type=SE.OBJECT_TYPE_MUST_BE_A_CLASS,
                name=name_class,
                coord=node.object.coord
            )


        if not self.global_symtab.has_attribute(name_class, node.field_name.name):
            if self.global_symtab.search_method(node.field_name.name) is None:
                assert_semantic(
                    condition=False,
                    error_type=SE.UNDECLARED_FIELD,
                    name=node.field_name.name,
                    coord=node.coord
                )

        self.visit(node.field_name)

        if self.global_symtab.has_attribute(name_class, node.field_name.name):
            type_field = self.global_symtab.get_attribute_type(name_class, node.field_name.name).name
    
            if type_field == "boolean":
                field_type = BooleanType
            elif type_field == "char":
                field_type = CharType
            elif type_field == "int":
                field_type = IntType
            elif type_field == "String":
                field_type = StringType
            elif type_field == "void":
                field_type = VoidType
            elif type_field == "int[]":
                field_type = IntArrayType
            elif type_field == "char[]":
                field_type = CharArrayType
            else: #pode dar pau, pensar em como melhor isso
                field_type = ObjectType

            node.mj_type = field_type

            self.global_symtab.type_set(node.field_name.name, type_field)


    def aux_MethodCall(self, name):
        if name == "boolean":
            return BooleanType
        elif name == "char":
            return CharType
        elif name == "int":
            return IntType
        elif name == "String":
            return StringType
        elif name == "void":
            return VoidType
        elif name == "int[]":
            return IntArrayType
        elif name == "char[]":
            return CharArrayType
        elif name == "void":
            return  VoidType
        else: #pode dar pau, pensar em como melhor isso
            return ObjectType(name)
        

    def visit_MethodCall(self, node: MethodCall):


        self.visit(node.object)

        name_class = None



        if isinstance(node.object, This):
            name_class = self.global_symtab.in_class()
        else:
            if isinstance(node.object.mj_type, ObjectType):
                name_class = self.global_symtab.type_get(node.object.name).real_type


        call = self.global_symtab.search_method(node.method_name.name, name_class)
        node.mj_type = self.aux_MethodCall(call[-1])
        if node.mj_type == None and call != None:
            node.mj_type = call[-1]

        if call is None:
            assert_semantic(
                condition=False,
                error_type=SE.UNDECLARED_METHOD,
                coord=node.coord,
                name=node.method_name.name,
            )
        
        args=[]

        if  isinstance(node.args, ExprList):
            args = node.args.exprs if node.args else []
        else:
            args = [node.args]


        if call[1] != len(args):
            assert_semantic(
                condition=False,
                error_type=SE.ARGUMENT_COUNT_MISMATCH,
                coord=node.coord,
                name=node.method_name.name,
            )
        i = 0

        for arg in args:
            self.visit(arg)
            arg.mj_type = self.global_symtab.type_get(arg.name) if isinstance(arg, ID) else getattr(arg, "mj_type", None)

            if isinstance(arg, Constant):
                arg_type = arg.type
            elif isinstance(arg, ID):
                arg_type = arg.mj_type.typename
            elif isinstance(arg, BinaryOp):
                arg_type = arg.mj_type.typename
            else:
                arg_type = None

            if arg_type != call[0][i]:
                assert_semantic(
                    condition=False,
                    error_type=SE.PARAMETER_TYPE_MISMATCH,
                    coord=node.coord,
                    name=arg.name,
                )

            i += 1
        

    def visit_Length(self, node: Length):
        self.visit(node.expr)
        if not "[]" in self.global_symtab.type_get(node.expr.name).typename:
            assert_semantic(
                condition=False,
                error_type=SE.INVALID_LENGTH_TARGET,
                coord=node.expr.coord
            )
                

    def visit_NewArray(self, node: NewArray):
        pass

    def visit_NewObject(self, node: NewObject):
        pass

    def visit_Constant(self, node: Constant):
        type_name = node.type


        if type_name == "boolean":
            node.mj_type = BooleanType
        elif type_name == "char":
            node.mj_type = CharType
        elif type_name == "int":
            node.mj_type = IntType
        elif type_name == "String":
            node.mj_type = StringType
        elif type_name == "char[]":
            node.mj_type = CharArrayType
        elif type_name == "int[]":
            node.mj_type = IntArrayType
        else:
            node.mj_type = None


    def visit_This(self, node: This):
        pass

    def visit_ID(self, node: ID):

        # print(f"AAAAAAAAAAAAAA {node} ")


        if self.global_symtab.type_get(node.name) is None:
            assert_semantic(
                condition=False,
                error_type=SE.UNDECLARED_NAME,
                coord=node.coord,
                name=node.name,
            )

        node.mj_type = self.global_symtab.type_get(node.name)

            
        
    def visit_Type(self, node: Type):
        pass

    def visit_Extends(self, node: Extends):
        super_ = node.super.name
        class_ = self.global_symtab.in_class()
        

        self.global_symtab.copy_class_members(super_, class_)


    def visit_ExprList(self, node: ExprList):
        for expr in node.exprs:
            self.visit(expr)

        #TODO: talvez errado
        try:
            node.mj_type = node.exprs[0].mj_type
        except:
            pass

    def visit_Expr(self, node: Expr):
        if isinstance(node.expr, list):
            for expr in node.expr:
                self.visit(expr)
            if node.expr:
                node.mj_type = getattr(node.expr[-1], "mj_type", None)
            return

        if isinstance(node.expr, ID):
            self.visit(node.expr)


        if (
            hasattr(node.expr, "op")
            and getattr(node.expr, "op", None) == "="
            and hasattr(node.expr, "lvalue")
            and hasattr(node.expr, "rvalue")
        ):
            self.visit(node.expr.lvalue)
            self.visit(node.expr.rvalue)
            node.mj_type = getattr(node.expr.lvalue, "mj_type", None)
            return


    def visit_InitList(self, node: InitList):
        pass


def main():
    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file", help="Path to file to be semantically checked", type=str
    )
    args = parser.parse_args()

    # get input path
    input_file = args.input_file
    input_path = pathlib.Path(input_file)

    # check if file exists
    if not input_path.exists():
        print("Input", input_path, "not found", file=sys.stderr)
        sys.exit(1)

    p = MJParser()
    # open file and parse it
    with open(input_path) as f:
        # Parse the code to an AST
        ast = p.parse(f.read())

        # First, build the global symtab
        global_symtab_builder = SymbolTableBuilder()
        global_symtab = global_symtab_builder.visit(ast)


        # Then, execute the semantic analysis
        sema = SemanticAnalyzer(global_symtab=global_symtab)
        sema.visit(ast)


if __name__ == "__main__":
    main()