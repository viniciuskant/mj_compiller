import argparse
import pathlib
import sys
from typing import Dict, List, Tuple

from mjc.mj_ast import *
from mjc.mj_block import (
    CFG,
    BasicBlock,
    Block,
    ConditionBlock,
    EmitBlocks,
    format_instruction,
)
from mjc.mj_interpreter import MJIRInterpreter
from mjc.mj_parser import MJParser
from mjc.mj_sema import NodeVisitor, SemanticAnalyzer, SymbolTableBuilder
from mjc.mj_type import CharType, IntType, VoidType

import rich

class CodeGenerator(NodeVisitor):
    """
    Node visitor class that creates 3-address encoded instruction sequences
    with Basic Blocks & Control Flow Graph.
    """

    def __init__(self, viewcfg: bool):
        self.viewcfg: bool = viewcfg
        self.current_block: Block = None
        self.ends_block_stack: List = []
        self.next_block: List = None

        # version dictionary for temporaries. We use the name as a Key
        self.fname: str = "_glob_"
        self.versions: Dict[str, int] = {self.fname: 0}
        self.euDesisto = False


        # The generated code (list of tuples)
        # At the end of visit_program, we call each function definition to emit
        # the instructions inside basic blocks. The global instructions that
        # are stored in self.text are appended at beginning of the code
        self.code: List[Tuple[str]] = []

        # Used for global declarations & constants (list, strings)
        self.text: List[Tuple[str]] = []
        self.counter_field = 1

        self.symbol_table: Dict[str, str] = {}


        self.current_class = None # para saber em qual classe estou

        self.isVarDecl =  False
        self.current_var_name = None # para saber em qual classe estou
        self.current_var_type = None

        self.stack_break = [] #vou usar para salvar os rótulos de "end"
        self.stack_currents_var = [] #guarda as variáveis que foram alocadas naquele escopo
        self.current_args: Dict[str, str] = {} 

        self.aux_name: Dict[str, (str, int)] = {} 

        self.global_text = {}



    def push_end_block(self, next_block):
        self.ends_block_stack.append(self.next_block)
    
    def pop_end_block(self):
        if self.ends_block_stack:
            return self.ends_block_stack.pop()
        else:
            return None

    def show(self):
        _str = ""
        for _code in self.code:
            _str += format_instruction(_code) + "\n"
        rich._str.strip()

    def new_temp(self) -> str:
        """
        Create a new temporary variable of a given scope (function name).
        """
        if self.fname not in self.versions:
            self.versions[self.fname] = 1
        name = "%" + "%d" % (self.versions[self.fname])
        self.versions[self.fname] += 1
        return name
    
    def new_var(self, name: str) -> str:
        """
        Create a new variable location name and register it in the symbol table.
        Locals use "%<name>", globals/fields use "@Class.<name>" when current_class is set.
        """
        name = self.set_current_name(name)

        if self.fname == "_glob_":
            loc = f"{self.current_class}.{name}" if self.current_class else name
        else:
            loc = f"%{name}"

        # register in symbol table
        self.symbol_table[name] = loc
        return loc

    def new_text(self, typename: str, nameVar: str = None) -> str:
        """
        Create a new literal constant on global section (text).
        """
        name = "@." + typename + "." + "%d" % (self.versions["_glob_"])
        self.versions["_glob_"] += 1
        if nameVar:
            self.global_text[nameVar] = name

        return name

    def set_current_name(self, name):
        self.stack_currents_var[-1].append(name)

        #(name, id_current, counter)
        if name in self.aux_name:
            counter = self.aux_name[name][2] + 1
            self.aux_name[name] = (name + "." + str(counter), counter, counter)
        else:
            self.aux_name[name] = (name, 0, 0)
        return self.aux_name[name][0]
    
    def get_current_name(self, name):
        if name in self.aux_name:
            return self.aux_name[name][0]
        else:
            self.set_current_name(name)
            return name
    
    def pop_current_name(self, alloc):
        for name in alloc:
            if name in self.aux_name:
                id = self.aux_name[name][1] -1 #voltando um escopo 
                counter = self.aux_name[name][2]
                if (id != 0):
                    new_name = name + "." + str(id)
                if (id == -1):
                    del self.aux_name[name]
                    continue
                else:
                    new_name = name
                self.aux_name[name] = (new_name, id, counter)


    def visit_Program(self, node: Program):
        # First visit all of the Class Declarations
        for class_decl in node.class_decls:
            self.visit(class_decl)

        # At the end of codegen, first init the self.code with the list
        # of global instructions allocated in self.text
        self.code = self.text.copy()


        # After, visit all the class definitions and emit the
        # code stored inside basic blocks.
        for class_decl in node.class_decls:
            block_visitor = EmitBlocks()
            block_visitor.visit(class_decl.cfg)
            for code in block_visitor.code:
                self.code.append(code)

    def visit_ClassDecl(self, node: ClassDecl):
        self.stack_currents_var.append([])

        class_name = "@" + node.name.name
        self.current_class =class_name

        # Create a cfg to hold the class context
        node.cfg = BasicBlock("test")
        node.cfg.append(("class", class_name, None))

        self.current_block = node.cfg

        for var_decl in node.var_decls:
            self.visit(var_decl)

        for method_decl in node.method_decls:
            self.visit(method_decl)

        for method_decl in node.method_decls:
            block_visitor = EmitBlocks()
            block_visitor.visit(method_decl.cfg)

            for instruction in block_visitor.code:
                node.cfg.append(instruction)

        self.current_block = node.cfg

        alloc = self.stack_currents_var.pop()
        self.pop_current_name(alloc)

        # If -cfg flag is present in command line
        if self.viewcfg:
            for method_decl in node.method_decls:
                method_name = getattr(method_decl, "name", None)
                if method_name is not None:
                    method_name = method_name.name
                else:
                    method_name = "main"

                dot = CFG(f"@{node.name.name}.{method_name}")
                dot.view(method_decl.cfg)

    def visit_MethodDecl(self, node: MethodDecl):
        self.ends_block_stack: List = []
        self.next_block: List = None
        self.stack_currents_var.append([])

        prev_fname = self.fname
        prev_block = self.current_block

        entry_block = BasicBlock("entry" + node.name.name)
        exit_block = BasicBlock("exit." + node.name.name)

        self.fname = node.name.name
        if self.fname not in self.versions:
            self.versions[self.fname] = 1

        self.current_block = entry_block
        node.cfg = entry_block 

        self.push_end_block(exit_block)

        type_method = "define_" + node.type.name
        name_method = self.current_class + "." + node.name.name

        param_list = []
        tmp = []
        for param in node.param_list.params:
            self.visit(param)
            param_type = param.type.name
            param_name = param.name.name

            tmp.append(("alloc_" + param_type, "%"+param_name))
            tmp.append(("store_" + param_type, self.symbol_table[param_name], "%"+param_name))

            param_list.append((param_type, param.gen_loc))

            self.symbol_table[param_name] = param.gen_loc
            self.current_args[param_name] = param.gen_loc

        entry_block.append((type_method, name_method, param_list))
        entry_block.append(("entry:",))
        entry_block.append(("entry:",))

        for inst in tmp:
            entry_block.append(inst)

        self.visit(node.body)


        self.current_block.next_block = exit_block
        exit_block.append((exit_block.label + ":",))

        if not self.current_block.instructions or self.current_block.instructions[-1][0] != "return_void":
            self.current_block.append(("jump", "%" + exit_block.label))
        
        if node.type.name == "void":
            exit_block.append(("return_void",))

        # self.current_block = prev_block
        self.fname = prev_fname
        entry_block.next_block = self.pop_end_block() 

        alloc = self.stack_currents_var.pop()
        self.pop_current_name(alloc)
        self.current_args = {}

    def visit_MainMethodDecl(self, node: MainMethodDecl):
        self.ends_block_stack: List = []
        self.next_block: List = None

        prev_fname = self.fname
        self.stack_currents_var.append([])

        type_method = "define_" + node.type.name
        name_method = self.current_class + "." + node.name.name

        node.cfg = BasicBlock(name_method)
        exit_block = BasicBlock("exit." + name_method)
        exit_block.append(("return_void",))    
    


        self.next_block = exit_block
        self.push_end_block(exit_block)

        # self.current_block.next_block = node.cfg
        self.current_block = node.cfg


        self.fname = "main"
        if self.fname not in self.versions:
            self.versions[self.fname] = 1

        node.cfg.append((type_method, name_method, [('String[]', '%args')]))
        self.current_block.append(("entry:",))

        self.visit(node.body)  


        self.current_block.next_block = exit_block

        # self.current_block.append(("jump", "%exit"))
        # self.current_block.append(("exit:",))
        # self.current_block.append(("return_void",))

        # self.current_block = prev_block
        self.fname = prev_fname

        self.pop_end_block()

        alloc = self.stack_currents_var.pop()
        self.pop_current_name(alloc)

    def visit_VarDecl(self, node: VarDecl):
        self.isVarDecl =  True

        self.new_var(node.name.name)

        var_name = node.name.name
        type_name = node.type.name

        var_name = self.get_current_name(var_name) #a questão de scope
        node.gen_loc = var_name

        self.current_var_name = var_name #usarei isso para inicializar algumas variáveis
        self.current_var_type = type_name #usarei isso para inicializar algumas variáveis


        if node.init is not None:
            self.visit(node.init)
        else:
            self.current_block.append(("alloc_" + type_name, "%"+ var_name))

        self.isVarDecl =  False
       
    def visit_Constant(self, node: Constant):
        var_name = self.current_var_name
        var_type = self.current_var_type

        lit_kind = "literal_" + node.type
        value = node.value
        
        if node.type == "String":
            if self.fname == "_glob_":
                memory = self.new_text(f"str", f"{self.current_class}.{var_name}")
                self.text.append((f"global_{node.type}", memory, node.value[1:-1]))
                node.gen_loc = memory
                
                if self.isVarDecl:
                    size = len(node.value[1:-1])
                    self.current_block.append((f"alloc_char[]_{size}", "%"+ var_name))
                    self.current_block.append((f"store_char[]_{size}", node.gen_loc, "%"+var_name))
            else:
                if self.isVarDecl:
                    memory = self.new_text(f"str", f"{var_name}")
                else:
                    memory = self.new_text(f"str")

                self.text.append((f"global_{node.type}", memory, node.value[1:-1]))
                node.gen_loc = memory

                if self.euDesisto:
                    node.gen_loc = self.new_temp()
                    self.current_block.append((f"load_strinf", memory, node.gen_loc))
                
                if self.isVarDecl:
                    size = len(node.value[1:-1])
                    self.current_block.append((f"alloc_char[]_{size}", "%"+ var_name))
                    self.current_block.append((f"store_char[]_{size}", node.gen_loc, "%"+var_name))
                
            return
        


        dest = self.new_temp()
        node.gen_loc = dest

        if self.fname != "_glob_":
            self.current_block.append((lit_kind, value, dest))

        if self.isVarDecl and not self.current_block is None: 
            if self.fname != "_glob_":
                self.current_block.append(("alloc_" + var_type, "%"+ var_name))
                self.current_block.append(("store_" + var_type, dest, "%"+var_name))
            else:
                self.current_block.append(("field_" + var_type, f"{self.current_class}.{var_name}", value))

    def visit_ParamList(self, node: ParamList):
        for param in node.params:
            self.visit(param)

    def visit_ParamDecl(self, node: ParamDecl):
        node.gen_loc = self.new_temp()
        self.symbol_table[node.name.name] = node.gen_loc

    def visit_Compound(self, node: Compound):
        for statement in node.statements:
            self.visit(statement)

    def visit_If(self, node: If):
        self.visit(node.cond)
        cond_loc = getattr(node.cond, "gen_loc", None)
        if cond_loc is None and hasattr(node.cond, "value"):
            cond_loc = node.cond.value

        # Criar blocos
        cond_block = ConditionBlock("if.cond" + f"@{node.coord.line}.{node.coord.column}")
        true_block = BasicBlock("if.true" + f"@{node.coord.line}.{node.coord.column}")
        false_block = BasicBlock("if.false" + f"@{node.coord.line}.{node.coord.column}") if node.iffalse else None
        end_block = BasicBlock("if.end" + f"@{node.coord.line}.{node.coord.column}")

        # Salvar próximo bloco atual e definir novo próximo
        self.push_end_block(self.next_block)

        # Conectar bloco atual ao cond_block
        self.current_block.next_block = cond_block
        
        # Configurar cond_block
        cond_block.taken = true_block
        cond_block.fall_through = false_block if node.iffalse else end_block


        # COND - processar condição
        if node.iffalse:
            cond_block.append(("cbranch", cond_loc, true_block.label, false_block.label))
        else:
            cond_block.append(("cbranch", cond_loc, true_block.label, end_block.label))


        aux = self.next_block

        # TRUE
        self.end_block = end_block
        self.current_block = true_block
        true_block.append((true_block.label + ":",))
        self.visit(node.iftrue)

        if aux != self.next_block:
            true_block.next_block = self.next_block

            self.next_block.append(("jump", "%" + end_block.label))
            self.next_block.next_block = end_block
        else:
            true_block.append(("jump", "%" + self.end_block.label))
            true_block.next_block = self.end_block


        # FALSE 
        self.end_block = end_block

        if node.iffalse:
            self.current_block = false_block
            false_block.append((false_block.label + ":",))
            self.visit(node.iffalse)

            if aux != self.next_block:
                false_block.append(("jump", "%" + self.next_block.label))
                false_block.next_block = self.next_block

                self.next_block.append(("jump", "%" + end_block.label))
                self.next_block.next_block = end_block
            else:
                false_block.append(("jump", "%" + end_block.label))
                false_block.next_block = self.end_block

        # END
        self.current_block = end_block
        end_block.append((end_block.label + ":",))
        end_block.next_block = aux

        if aux != None:
            end_block.append(("jump", "%" + aux.label))


        self.next_block = cond_block

    def visit_For(self, node: For):
        self.stack_currents_var.append([])

        self.visit(node.init)

        line = node.coord.line
        column = node.coord.column

        cond_block = ConditionBlock("for.cond." + f"{line}.{column}")

        body_block = BasicBlock("for.body." + f"{line}.{column}")
        increment_block = BasicBlock("for.increment." + f"{line}.{column}")
        end_block = BasicBlock("for.end." + f"{line}.{column}")

        end_block.next_block = self.next_block

        # Salvar próximo bloco atual e definir novo próximo
        self.push_end_block(self.current_block)
        self.stack_break.append(end_block)

   

        # Configurar fluxo de controle
        cond_block.taken = body_block
        cond_block.fall_through = end_block

        self.current_block.next_block = cond_block

        # Adicionar jump para condição
        self.current_block.append(("jump", "%" + cond_block.label))

        # COND
        self.current_block = cond_block
        self.current_block.append((cond_block.label + ":",))
        self.visit(node.cond)


        cond_block.append(("cbranch", node.cond.gen_loc, body_block.label, end_block.label))

        # BODY
        self.current_block = body_block
        body_block.append((body_block.label + ":",))

        self.next_block = increment_block
        self.next_block = increment_block
        aux = self.next_block

        self.visit(node.body)
        if aux != self.next_block: # tem novos blocos internos
            self.next_block.append(("jump", "%" + increment_block.label))
            self.next_block.next_block = increment_block
        else:
            body_block.append(("jump", "%" + self.next_block.label))
            body_block.next_block = self.next_block
            body_block.next_block = self.next_block


        # INCREMENT
        self.current_block = increment_block
        increment_block.append((increment_block.label + ":",))
        self.visit(node.next)
        increment_block.append(("jump", "%" + cond_block.label))
        increment_block.next_block = cond_block

        # END
        self.current_block = end_block
        end_block.append((end_block.label + ":",))


        self.next_block = cond_block

        alloc = self.stack_currents_var.pop()
        self.stack_break.pop()
        self.pop_current_name(alloc)

    def visit_While(self, node: While):
        self.stack_currents_var.append([])

        # Criar blocos
        cond_block = ConditionBlock("while.cond." + f"@{node.coord.line}.{node.coord.column}")
        body_block = BasicBlock("while.body." + f"@{node.coord.line}.{node.coord.column}")
        end_block = BasicBlock("while.end." + f"@{node.coord.line}.{node.coord.column}")

        # Salvar próximo bloco atual e definir novo próximo
        self.push_end_block(end_block)
        self.stack_break.append(end_block)

        # Conectar bloco atual ao cond_block
        self.current_block.next_block = cond_block

        # Configurar fluxo de controle
        cond_block.taken = body_block
        cond_block.fall_through = end_block


        cond_block.append((cond_block.label + ":",))

        # Adicionar jump para condição
        self.current_block.append(("jump", "%" + cond_block.label))

        # COND
        self.current_block = cond_block
        self.visit(node.cond)
        
        cond_loc = getattr(node.cond, "gen_loc", None)
        if cond_loc is None and hasattr(node.cond, "value"):
            cond_loc = node.cond.value
            
        cond_block.append(("cbranch", cond_loc, body_block.label, end_block.label))

        # BODY
        self.current_block = body_block
        body_block.append((body_block.label + ":",))
        self.visit(node.body)
        body_block.append(("jump", "%" + cond_block.label))
        body_block.next_block = cond_block

        # END
        self.current_block = end_block
        end_block.append((end_block.label + ":",))

        # Restaurar próximo bloco e limpar pilhas
        end_block.next_block = self.next_block
        self.pop_end_block()

        alloc = self.stack_currents_var.pop()
        self.stack_break.pop()
        self.pop_current_name(alloc)

    def visit_DeclList(self, node: DeclList):
        for decl in node.decls:
            self.visit(decl)

    def aux_print(self, expr):
        # self.euDesisto = True
        self.visit(expr)
        # self.euDesisto = False
        expr_loc = expr.gen_loc
        if expr_loc is None and hasattr(expr, "value"):
            expr_loc = expr.value


        print_inst = ("print_" + expr.mj_type.typename, expr_loc)
        self.current_block.append(print_inst)

    def visit_Print(self, node: Print):
        self.isVarDecl = False
        if isinstance(node.expr, Expr) or isinstance(node.expr, ID):
            self.aux_print(node.expr)
            
        elif isinstance(node.expr, ExprList):
            for expr in node.expr.exprs:
                if isinstance(expr, FieldAccess):
                    var_name = f"{self.current_class}.{expr.field_name.name}"
                    if "this" in var_name:
                        var_name = var_name.replace("this", self.current_class)
                    var_name = self.global_text[var_name]
                    print_inst = ("print_" + expr.mj_type.typename, var_name)
                    self.current_block.append(print_inst)
                elif isinstance(expr, ID) and expr.mj_type.typename == "string":
                    var_name = expr.name
                    var_name = self.global_text[var_name]
                    print_inst = ("print_" + expr.mj_type.typename, var_name)
                    self.current_block.append(print_inst)
                else:
                    self.aux_print(expr)
        else:
            self.current_block.append(("print_void",))

    def visit_Assert(self, node: Assert):
        cond_block = ConditionBlock(label="assert" + f"@{node.coord.line}.{node.coord.column}")
        true_block = BasicBlock(label="assert.true" + f"@{node.coord.line}.{node.coord.column}")
        false_block = BasicBlock(label="assert.false" + f"@{node.coord.line}.{node.coord.column}")
        end_block = BasicBlock(label="assert_end" + f"@{node.coord.line}.{node.coord.column}")

        self.current_block.append(("jump", "%" + cond_block.label))

        self.current_block.next_block = cond_block
        cond_block.append((cond_block.label + ":",))
        
        cond_block.taken = true_block
        cond_block.fall_through = false_block

        # Configura a sequência dos blocos
        true_block.next_block = end_block
        false_block.next_block = end_block

        self.current_block = cond_block
        self.current_block.append((self.current_block.label + ":",))
        self.visit(node.expr)

        cond_block.append(("cbranch", node.expr.gen_loc, true_block.label, false_block.label,))

        # TRUE
        self.current_block = true_block
        true_block.append((true_block.label + ":",))
        true_block.append(("jump", "%" + end_block.label))

        # FALSE
        self.current_block = false_block
        false_block.append((false_block.label + ":",))
        coord = str(node.expr.coord).replace("@ ", "")
        log = f"assertion_fail on {coord}"
        target = self.new_text("str")

        self.text.append(("global_String", target, log))
        false_block.append(("print_string", target))
        false_block.append(("jump", "%" + end_block.label))

        # END
        self.current_block = end_block
        end_block.append((end_block.label + ":",))
        
        # Conecta o end_block ao próximo bloco que será criado
        end_block.next_block = None

    def visit_Break(self, node: Break):
        self.end_block = self.stack_break[-1]

    def visit_Return(self, node: Return):
        if node.expr is not None:
            self.visit(node.expr)
            self.current_block.append(('return_' + node.expr.mj_type.typename, node.expr.gen_loc))
        else:
            self.current_block.append(('jump', self.ends_block_stack[-1].label))

    def visit_Assignment(self, node: Assignment):
        self.visit(node.rvalue)
        right_loc = node.rvalue.gen_loc


        if isinstance(node.lvalue, ID):
            var_name = self.get_current_name(node.lvalue.name)
            self.current_block.append(("store_" + node.lvalue.mj_type.typename, right_loc, f"%{var_name}"))
        elif isinstance(node.lvalue, FieldAccess):
            if isinstance(node.lvalue.object, This):
                var_name = "this" + "." + node.lvalue.field_name.name
            else:
                var_name = "TODO"
            self.current_block.append(("store_" + node.lvalue.mj_type.typename, right_loc, f"%{var_name}"))
        
        elif isinstance(node.lvalue, ArrayRef):
            self.visit(node.lvalue)
            
            subscript = node.lvalue.subscript
            self.visit(subscript)
            alloc = self.new_temp()
            if hasattr(node.lvalue.name, "object"):
                name = node.lvalue.name.object.name
                field = node.lvalue.name.field_name.name
            if hasattr(node.lvalue.name, "name"):
                name = node.lvalue.name.name
                field = None
            name = name if name == "this" else self.global_text[name] if name in self.global_text.keys() else name
            if field == None:
                self.current_block.append((f"elem_{subscript.mj_type.typename}", f"%{name}", subscript.gen_loc,  alloc))
            else:
                self.current_block.append((f"elem_{subscript.mj_type.typename}", f"%{name}.{field}", subscript.gen_loc,  alloc))
                
            self.current_block.append((f"store_{subscript.mj_type.typename}_*", right_loc, alloc))

    def aux_BinaryOp(self, expr):
        self.euDesisto = True
        self.visit(expr)
        self.euDesisto = False
        gen_loc = expr.gen_loc


        if gen_loc[1:].isalpha(): 
            return self.symbol_table[gen_loc[1:]]
        return gen_loc

    def visit_BinaryOp(self, node: BinaryOp):
        op_map = {
            "==": "eq_type", "!=": "ne_type", "<=": "le_int", ">=": "ge_int",
            "&&": "and_type", "||": "or_type", "<": "lt_int", ">": "gt_int", 
            "+": "add_int", "-": "sub_int", "*": "mul_int", "/": "div_int", "%": "mod_int", 
            "!": "not_type", }
        
        opcode = op_map.get(node.op, node.op)

        left_loc = self.aux_BinaryOp(node.lvalue)
        right_loc = self.aux_BinaryOp(node.rvalue)

        node.gen_loc = self.new_temp()
        bin_inst = (opcode, left_loc, right_loc, node.gen_loc)
        self.current_block.append(bin_inst)

    def visit_UnaryOp(self, node: UnaryOp):
        op_map = {
            "+": "pos_int",
            "-": "sub_int",
            "!": "not_boolean",
        }

        opcode = op_map.get(node.op, node.op)

        if opcode == "sub_int":
            zero_temp = self.new_temp()
            self.current_block.append(('literal_int', '0', zero_temp))
            if not self.isVarDecl:
                rvalue_loc = self.aux_BinaryOp(node.expr)
                node.gen_loc = self.new_temp()
                self.current_block.append((opcode, zero_temp, rvalue_loc, node.gen_loc))
            else:
                self.isVarDecl = False
                rvalue_loc = self.aux_BinaryOp(node.expr)
                node.gen_loc = self.new_temp()
                self.current_block.append((opcode, zero_temp, rvalue_loc, node.gen_loc))
                self.current_block.append((f"alloc_{self.current_var_type}", f"%{self.current_var_name}"))
                self.current_block.append((f"store_{self.current_var_type}", node.gen_loc, f"%{self.current_var_name}"))
                self.isVarDecl = True            
        else:
            if not self.isVarDecl:
                rvalue_loc = self.aux_BinaryOp(node.expr)
                node.gen_loc = self.new_temp()
                unary_inst = (opcode, rvalue_loc, node.gen_loc)
                self.current_block.append(unary_inst)
            else:
                self.isVarDecl = False
                rvalue_loc = self.aux_BinaryOp(node.expr)
                node.gen_loc = self.new_temp()
                unary_inst = (opcode, rvalue_loc, node.gen_loc)
                self.current_block.append((f"alloc_{self.current_var_type}", f"%{self.current_var_name}"))
                self.current_block.append((f"store_{self.current_var_type}", node.gen_loc, f"%{self.current_var_name}"))
                self.current_block.append(unary_inst)
                self.isVarDecl = True

    def visit_ArrayRef(self, node: ArrayRef):
        if isinstance(node.subscript, ID):
            name_ID = f"%{node.subscript.name}"
        else:
            self.visit(node.subscript)
            name_ID = node.subscript.gen_loc

        alloc = self.new_temp()
        alloc2 = self.new_temp()

        node.gen_loc = alloc2

        elem_type = f"elem_{node.mj_type.typename}"

        if isinstance(node.name, FieldAccess):
            name = f"%{node.name.object.name}.{node.name.field_name.name}"
        else:
            name = "%" + node.name.name
        self.current_block.append((elem_type, name, name_ID, alloc))
        self.current_block.append((f"load_{node.mj_type.typename}_*", alloc, alloc2))
        

    def visit_FieldAccess(self, node: FieldAccess):
        self.visit(node.object) #é necessário?
        temp = self.new_temp()
        node.gen_loc = temp

        if not isinstance(node, object):
            var_name = f"{node.object.mj_type.real_type}.{node.field_name.name}"
            self.current_block.append(('load_' + node.mj_type.typename, f"%{var_name}", temp))
        else:
            var_name = f"{node.object.name}.{node.field_name.name}"
            if node.mj_type.typename == "string":
                if "this" in var_name:
                    var_name = var_name.replace("this", self.current_class)
                var_name = self.global_text[var_name]
                self.current_block.append(('load_' + node.mj_type.typename, f"{var_name}", temp))
            else:
                self.current_block.append(('load_' + node.mj_type.typename, f"%{var_name}", temp))


    

        if self.isVarDecl:  
            name =  self.current_var_name
            type_ =  self.current_var_type
            self.current_block.append(('alloc_' + type_, f"%{name}"))
            self.current_block.append(('store_' + type_, temp, f"%{name}"))

    def visit_MethodCall(self, node: MethodCall):
        method_name = node.method_name.name
        if (isinstance(node.object, This)):
            full_method = f"this.{method_name}"
        else:
            full_method = f"{node.object.name}.{method_name}"

        arg_values = []
        if isinstance(node.args, ExprList):
            for arg in node.args.exprs:
                self.visit(arg)
                arg_values.append(arg.gen_loc)
        else:
            self.visit(node.args)
            arg_values.append(node.args.gen_loc)


        if node.mj_type.typename != "void":
            for arg in arg_values:
                self.current_block.append(('param_' + node.mj_type.typename, arg))

        node.gen_loc = self.new_temp()
        self.current_block.append(('call_' + node.mj_type.typename, "%" + full_method, node.gen_loc))

    def visit_Length(self, node: Length):
        # Visit the expression to set its gen location
        self.visit(node.expr)
        # Alloc a register to store the length
        node.gen_loc = self.new_temp()
        # gen the length instruction
        length_inst = ("length", node.expr.gen_loc, node.gen_loc)
        # Store the length instruction
        self.current_block.append(length_inst)

    def visit_NewObject(self, node: NewObject):
        self.current_block.append(("new_@" + node.type.name.name, "%" + self.current_var_name))

    def visit_NewArray(self, node: NewArray):
        if self.fname == "_glob_":
            index = self.counter_field
            self.current_block.insert(index, (f"field_{node.type.name}", f"{self.current_class}.{self.current_var_name}", int(node.size.value)))
            self.counter_field += 1
        else:
            self.current_block.append((f"new_{node.type.name}_{node.size.value}", f"%{self.current_var_name}"))

    def visit_This(self, node: This):
        pass

    def visit_ID(self, node: Node):
        # if node.name in self.symbol_table:
        if self.fname != "_glob_":
            var_name = node.name
            type_name = node.mj_type.typename
            type_name = type_name if type_name != "object" else node.mj_type.real_type

            # if type_name != "string":
            temp = self.new_temp()
            var_name = self.get_current_name(var_name)
            node.gen_loc = temp
            if type_name != "string":
                self.current_block.append(('load_' + type_name, f"%{var_name}", temp))
            else:
                node.gen_loc = self.global_text[f"{self.current_class}.{var_name}"]
            if self.isVarDecl:
                self.current_block.append((f"alloc_{self.current_var_type}", f"%{self.current_var_name}"))
                self.current_block.append((f"store_{self.current_var_type}", temp, f"%{self.current_var_name}"))
        else:
            node.gen_loc = self.symbol_table[node.name]

    def visit_Type(self, node: Type):
        pass

    def visit_Extends(self, node: Extends):
        pass

    def visit_ExprList(self, node: ExprList):
        for expr in node.exprs:
            self.visit(expr)

    def visit_InitList(self, node: InitList):
        name = self.current_var_name
        type_ = self.current_var_type

        sizeList = len(node.exprs)
        initList = [expr.value for expr in node.exprs]
        if type_ == "int[]":
            initList = [int(elem) for elem in initList]

        if self.fname == "_glob_":
            index = self.counter_field
            self.current_block.insert(index, (f"field_{self.current_var_type}_{sizeList}", f"{self.current_class}.{self.current_var_name}", initList))
            self.counter_field += 1

        else:
            gen_loc = self.new_text(f"const_{name}", name)
            node.gen_loc = gen_loc
            self.text.append((f"global_{type_}_{sizeList}", gen_loc, initList))

            self.current_block.append((f"alloc_{type_}_{sizeList}", "%"+ name))
            self.current_block.append((f"store_{type_}_{sizeList}", gen_loc, "%"+name))

    # TODO: Complete.

def main():
    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        help="Path to file to be used to generate MJIR. By default, this script only runs the interpreter on the MJIR. \
              Use the other options for printing the MJIR, generating the CFG or for the debug mode.",
        type=str,
    )
    parser.add_argument(
        "--ir",
        help="Print MJIR generated from input_file.",
        action="store_true",
    )
    parser.add_argument(
        "--ir-pp",
        help="Print MJIR generated from input_file. (pretty print)",
        action="store_true",
    )
    parser.add_argument(
        "--cfg",
        help="Show the cfg of the input_file.",
        action="store_true",
    )

    args = parser.parse_args()

    print_ir = args.ir
    print_ir_pp = args.ir_pp
    create_cfg = args.cfg

    # get input path
    input_file = args.input_file
    input_path = pathlib.Path(input_file)

    # check if file exists
    if not input_path.exists():
        print("Input", input_path, "not found", file=sys.stderr)
        sys.exit(1)

    # set error function
    p = MJParser()
    # open file and parse it
    with open(input_path) as f:
        ast = p.parse(f.read())

    global_symtab_builder = SymbolTableBuilder()
    global_symtab = global_symtab_builder.visit(ast)
    sema = SemanticAnalyzer(global_symtab=global_symtab)
    sema.visit(ast)

    gen = CodeGenerator(create_cfg)
    gen.visit(ast)
    gencode = gen.code

    if print_ir:
        print("Generated MJIR: --------")
        rich.print(gencode)
        print("------------------------\n")

    elif print_ir_pp:
        print("Generated MJIR: --------")
        gen.show()
        print("------------------------\n")

    else:
        vm = MJIRInterpreter()
        vm.run(gencode)


if __name__ == "__main__":
    main()
