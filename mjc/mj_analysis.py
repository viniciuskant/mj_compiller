import argparse
import pathlib
import sys
from typing import List, Tuple
from pprint import pprint 
import rich

from mjc.mj_ast import ClassDecl, Program
from mjc.mj_block import (
    CFG,
    BasicBlock,
    ConditionBlock,
    EmitBlocks,
    format_instruction,
)
from mjc.mj_code import CodeGenerator
from mjc.mj_interpreter import MJIRInterpreter
from mjc.mj_parser import MJParser
from mjc.mj_sema import NodeVisitor, SemanticAnalyzer, SymbolTableBuilder

import collections


class DataFlow(NodeVisitor):
    def __init__(self, viewcfg: bool, verbose: bool = False):
        # flag to show the optimized control flow graph
        self.viewcfg: bool = viewcfg
        
        # list of code instructions after optimizations
        self.code: List[Tuple[str]] = []
        self.verbose = verbose

    def show(self):
        _str = ""
        for _code in self.code:
            _str += format_instruction(_code) + "\n"
        rich.print(_str.strip())

    def visit_Program(self, node: Program):
        # First, save the global instructions on code member
        self.code = node.text[:]  # [:] to do a copy


        # Visit all class declaration in the program
        for class_decl in node.class_decls:
            self.visit(class_decl)

    def visit_ClassDecl(self, node: ClassDecl):
        bb = EmitBlocks()
        bb.visit(node.cfg)
        for _code in bb.code:
            if "class" in _code[0] or "field" in _code[0]:
                self.code.append(_code)

        for method_decl in node.method_decls:
            self.build_predecessors(method_decl.cfg)

            self.current_func = method_decl
            # start with Reach Definitions Analysis
            self.buildRD_blocks(method_decl.cfg)
            self.computeRD_gen_kill()
            self.computeRD_in_out()

            self.constant_propagation()
            # self.computeRD_gen_kill()
            # self.computeRD_in_out()
            # exit()

            # # after do live variable analysis
            # self.buildLV_blocks(method_decl.cfg)
            # self.computeLV_use_def()
            # self.computeLV_in_out()
            # # and do dead code elimination
            # self.deadcode_elimination()

            # # after that do cfg simplify (optional)
            # self.short_circuit_jumps(method_decl.cfg)
            self.oTaldoJump()
            self.oTaldoLoad()
            method_decl.cfg = self.merge_blocks(method_decl.cfg)


            # self.discard_unused_allocs_iterative(method_decl.cfg)
            # self.discard_unused_allocs(method_decl.cfg)

            # # finally save optimized instructions in self.code
            self.appendOptimizedCode(method_decl.cfg)

            if self.verbose:
                self._log_optimization_results(method_decl)


        if self.viewcfg:
            for method_decl in node.method_decls:
                method_name = getattr(method_decl, "name", None)
                if method_name is not None:
                    method_name = method_name.name
                else:
                    method_name = "main"

                dot = CFG(f"@{node.name.name}.{method_name}.opt")
                dot.view(method_decl.cfg)

    def oTaldoJump(self):
        for block in self.rd_blocks:
            for i, instruc in enumerate(block.instructions):
                if "jump" in instruc[0] and i + 1 < len(block.instructions):
                    block.instructions = block.instructions[:i+1]
                    break
                    
    def oTaldoLoad(self):
        for block in self.rd_blocks:
            i = 0
            while i < len(block.instructions) - 1:

                instr1 = block.instructions[i]
                instr2 = block.instructions[i + 1]

                if "load" in instr1[0] and "load" in instr2[0] and instr1[1] == instr2[1]:
                    old_reg = instr2[2]  
                    new_reg = instr1[2]  

                    for j in range(i + 2, len(block.instructions)):
                        instr = block.instructions[j]
                        instr = list(instr)  

                        changed = False
                        for k, op in enumerate(instr):
                            if op == old_reg:
                                instr[k] = new_reg
                                changed = True

                        if changed:
                            block.instructions[j] = tuple(instr)

                    del block.instructions[i + 1]

                    continue

                i += 1


    def build_predecessors(self, start_block):
        visited = set()
        queue = collections.deque([start_block])
        all_blocks = []
        
        while queue:
            block = queue.popleft()
            if block in visited:
                continue
            visited.add(block)
            all_blocks.append(block)
            

            if hasattr(block, "taken") and block.taken:
                queue.append(block.taken)
            if hasattr(block, "fall_through") and block.fall_through:
                queue.append(block.fall_through)
            if hasattr(block, "next_block") and block.next_block:
                queue.append(block.next_block)

        for block in all_blocks:
            self._find_predecessors_for_block(block, all_blocks)
        
        return all_blocks

    def _find_predecessors_for_block(self, target_block, all_blocks):
        for block in all_blocks:
            if (hasattr(block, "taken") and block.taken == target_block or
                hasattr(block, "fall_through") and block.fall_through == target_block or
                hasattr(block, "next_block") and block.next_block == target_block):
                target_block.predecessors.append(block)

    def buildRD_blocks(self, start_block):
        visited = set()
        stack = [start_block]
        self.rd_blocks = []

        while stack:
            block = stack.pop()
            if block in visited:
                continue
            visited.add(block)
            self.rd_blocks.append(block)

            if hasattr(block, "taken") and block.taken:
                stack.append(block.taken)
            if hasattr(block, "fall_through") and block.fall_through:
                stack.append(block.fall_through)
            if hasattr(block, "next_block") and block.next_block:
                stack.append(block.next_block)

    def computeRD_gen_kill(self):
        all_defs = set()
        for block in self.rd_blocks:
            for instr in block.instructions:
                if self.is_definition(instr):
                    all_defs.add(tuple(instr))


        for block in self.rd_blocks:
            gen = set()
            killed_defs = set()
            local_defs = {}

            for instr in block.instructions:
                if self.is_definition(instr):
                    var = instr[-1] #se o target está numa definição, eu "apago"
                    if var in local_defs:
                        killed_defs.add(tuple(local_defs[var]))
                    local_defs[var] = tuple(instr)

            gen = set(local_defs.values())
            
            kill = set()
            for var in local_defs.keys():
                kill.update(d for d in all_defs 
                        if d[-1] == var and d not in gen)

            block.rd_gen = gen
            block.rd_kill = kill

    def computeRD_in_out(self):
        for block in self.rd_blocks:
            block.rd_in = set()
            block.rd_out = set()

        changed = True
        while changed:
            changed = False
            for block in self.rd_blocks:
                # IN[B] = U OUT[P] para cada predecessor P
                new_in = set()
                for pred in block.predecessors:
                    new_in |= pred.rd_out

                # OUT[B] = GEN[B] U (IN[B] - KILL[B])
                new_out = block.rd_gen | (new_in - block.rd_kill)

                if new_in != block.rd_in or new_out != block.rd_out:
                    changed = True
                    block.rd_in = new_in
                    block.rd_out = new_out

    def is_definition(self, instruction):
        op = instruction[0]
        defining_ops = {"alloc", "store", "new", "literal","load", "eq", "ne", "le", "ge","and", "or", "lt", "gt", "add", "sub", "mul", "div", "mod", "not", "elem"}

        return op.split("_")[0] in defining_ops
    
    def _propagate_constants_in_instruction(self, instruction, constants):
        opcode = instruction[0]
        new_operands = list(instruction[1:])
        return (opcode,) + tuple(new_operands)
        
        if "load" not in opcode and "store" not in opcode:
            for i in range(len(new_operands)):
                operand = new_operands[i]
                if isinstance(operand, str) and operand.startswith("%") and operand in constants:
                    new_operands[i] = str(constants[operand])

        return (opcode,) + tuple(new_operands)

    def constant_propagation(self):
        for block in self.rd_blocks:
            constants = {}
        
            new_instructions = []
            for instr in block.instructions:
                new_instr = list(self._propagate_constants_in_instruction(instr, constants))
                new_instructions.append(new_instr)
                
                self._update_constants_map(new_instr, constants)
            
            block.instructions = new_instructions

    def _update_constants_map(self, instruction, constants):
        opcode = instruction[0]

        if opcode == "literal_int":
            value = instruction[1]
            dest = instruction[2]
            if not value.startswith("%"):  # constante literal
                constants[dest] = int(value)
                # constants[dest] = int(value)
        
        
        elif opcode == "store_int": #viraria cisc
            pass
            # value = instruction[1]
            # dest = instruction[2]
            # if not value.startswith("%"):  # é constante
            #     constants[dest] = int(value)
            # elif value in constants:  # conheço valor constante
            #     constants[dest] = constants[value]
        
        elif opcode in ["add_int", "sub_int", "mul_int", "div_int", "mod_int"]:
            pass
            # calcula operações com constantes em tempo de compilação
            # self._fold_constant_arithmetic(instruction, constants)
        

        else:
            # calcula comparações com constantes em tempo de compilação
            for op in ["eq", "ne", "lt", "le", "gt", "ge"]:
                if op in opcode:
                    self._fold_constant_comparison(instruction, constants)

    def _fold_constant_arithmetic(self, instruction, constants):
        opcode = instruction[0]
        op1 = instruction[1]
        op2 = instruction[2]
        dest = instruction[3]
        
        # se ambos operandos são constantes
        val1 = self._get_constant_value(op1, constants)
        val2 = self._get_constant_value(op2, constants)
        
        if val1 is not None and val2 is not None:
            try:
                if opcode == "add_int":
                    result = val1 + val2
                elif opcode == "sub_int":
                    result = val1 - val2
                    instruction[0] = "literal_int"
                    instruction[1] = str(result)
                    instruction[2] = dest
                    del instruction[3]
                    return
                elif opcode == "mul_int":
                    result = val1 * val2
                elif opcode == "div_int" and val2 != 0:
                    result = val1 // val2  
                elif opcode == "mod_int":
                    result = val1 % val2  
                else:
                    return
                
                constants[dest] = result
            except (ZeroDivisionError, ArithmeticError):
                # ignora erros aritméticos (divisão por zero, etc.)
                pass

    def _get_constant_value(self, operand, constants):
        if not operand.startswith('%'):
            # é uma constante literal
            try:
                return int(operand)
            except ValueError:
                # não é um inteiro, pode ser booleano ou string
                if operand == 'true':
                    return 1
                elif operand == 'false':
                    return 0
                return None
        elif operand in constants:
            return constants[operand]
        return None

    def _fold_constant_comparison(self, instruction, constants):
        opcode = instruction[0]
        if len(instruction) < 4:
            return
        
        op1 = instruction[1]
        op2 = instruction[2]
        dest = instruction[3]
        
        # Verifica se ambos operandos são constantes
        val1 = self._get_constant_value(op1, constants)
        val2 = self._get_constant_value(op2, constants)
        
        if val1 is not None and val2 is not None:
            if opcode == "eq_int":
                result = 1 if val1 == val2 else 0
            elif opcode == "ne_int":
                result = 1 if val1 != val2 else 0
            elif opcode == "lt_int":
                result = 1 if val1 < val2 else 0
            elif opcode == "le_int":
                result = 1 if val1 <= val2 else 0
            elif opcode == "gt_int":
                result = 1 if val1 > val2 else 0
            elif opcode == "ge_int":
                result = 1 if val1 >= val2 else 0
            else:
                return
            
            constants[dest] = result

    def remove_redundant_instructions(self):
        for block in self.rd_blocks:
            new_instructions = []
            for instr in block.instructions:
                if not self._is_redundant_instruction(instr):
                    new_instructions.append(instr)
            block.instructions = new_instructions

    def _is_redundant_instruction(self, instruction):
        opcode = instruction[0]
        
        if opcode in ['store_int', 'store_bool']:
            if len(instruction) >= 3:
                dest = instruction[2]
                if not self._is_variable_used(dest): #pode dar ruim
                    return True
        
        # alocação onde a variável não é mais usada
        elif opcode in ['alloc_int', 'alloc_bool']:
            if len(instruction) >= 2:
                dest = instruction[1]
                if not self._is_variable_used(dest):
                    return True
        
        return False

    def _is_variable_used(self, variable):
        """Verifica se uma variável é usada em algum lugar (implementação simplificada)."""
        for block in self.rd_blocks:
            for instr in block.instructions:
                for operand in instr[1:]:
                    if operand == variable:
                        return True
        return False

    def discard_unused_allocs(self, start_block):
        used_vars = self._collect_used_variables(start_block)
        
        # Agora removemos as alocações não utilizadas
        blocks = self._get_all_blocks(start_block)
        for block in blocks:
            self._remove_unused_allocations_in_block(block, used_vars)

    def _collect_used_variables(self, start_block):
        used_vars = set()
        blocks = self._get_all_blocks(start_block)
        
        for block in blocks:
            for instr in block.instructions:
                for operand in instr[1:]:  # Pula o opcode
                    if isinstance(operand, str) and operand.startswith('%'):
                        if not self._is_definition_destination(instr, operand):
                            used_vars.add(operand)

        return used_vars

    def _is_definition_destination(self, instruction, operand):
        opcode = instruction[0]
        
        # Instruções onde o último operando é o destino
        definition_ops = {
            "alloc", "store", "load", "literal", "new", "add", "sub", "mul", "div_int",
            "eq", "ne", "lt", "le", "gt", "ge", "and", "or" }
        
        for op in definition_ops:
            if op in opcode:
                # O último operando é tipicamente o destino
                return len(instruction) > 1 and operand == instruction[-1]
        
        return False

    def _remove_unused_allocations_in_block(self, block, used_vars):
        new_instructions = []
        for instr in block.instructions:
            opcode = instr[0]
            if "alloc" in opcode:
                var_name = instr[1]
                if var_name not in used_vars:
                    continue  # pula a instrução
            if "literal" in opcode:
                var_name = instr[2]
                if var_name not in used_vars:
                    continue  # pula a instrução
            new_instructions.append(instr)
        
        block.instructions = new_instructions

    def _get_all_blocks(self, start_block):
        """Retorna todos os blocos do CFG usando BFS."""
        visited = set()
        queue = collections.deque([start_block])
        all_blocks = []
        
        while queue:
            block = queue.popleft()
            if block in visited:
                continue
            visited.add(block)
            all_blocks.append(block)
            
            successors = self._get_successors_ordered(block)
            for succ in successors:
                if succ and succ not in visited:
                    queue.append(succ)
        
        return all_blocks

    def discard_unused_allocs_iterative(self, start_block):
        changed = True
        iteration = 0
        max_iterations = 10  # Prevenção contra loop infinito
        
        while changed and iteration < max_iterations:
            used_vars = self._collect_used_variables(start_block)
            blocks = self._get_all_blocks(start_block)
            
            changed = False
            for block in blocks:
                original_count = len(block.instructions)
                self._remove_unused_allocations_in_block(block, used_vars)
                if len(block.instructions) != original_count:
                    changed = True
            
            iteration += 1

    def appendOptimizedCode(self, start_block):
        optimized_instructions = []
        visited = set()
        queue = collections.deque([start_block])

        
        while queue:
            block = queue.popleft()
            if block in visited:
                continue
            visited.add(block)
            
            optimized_instructions.extend(block.instructions)
            
            successors = self._get_successors_ordered(block)
            for succ in successors:
                if succ and succ not in visited:
                    queue.append(succ)


        optimized_instructions = [list(x) for x in optimized_instructions]

        jumps = []
        rm = []
        for inst in optimized_instructions:
            if "jump" in inst[0]:
                jumps.append(inst[1][1:])
            if "cbranch" in inst[0]:
                jumps.append(inst[2])
                jumps.append(inst[3])


        for i, inst in enumerate(optimized_instructions):
            for op in ["eq", "ne", "lt", "le", "gt", "ge"]:
                if op in inst[0]:
                    if "load" in optimized_instructions[i-1][0] and "this" not in optimized_instructions[i-1][1]:
                        rm.append(optimized_instructions[i-1])
                        inst[2] = optimized_instructions[i-1][1]


                    if "load" in optimized_instructions[i-2][0] and "this" not in optimized_instructions[i-2][1]:
                        rm.append(optimized_instructions[i-2])
                        inst[1] = optimized_instructions[i-2][1]



        for inst in optimized_instructions:
            if ":" in inst[0]:
                if inst[0][:-1] not in jumps:
                    rm.append(inst)

        for inst in optimized_instructions:
            if inst in rm:
                continue
            self.code.append(inst)
        
        self.current_func.optimized_instructions = optimized_instructions
        
        if hasattr(self.current_func, 'instructions'):
            self.current_func.instructions = optimized_instructions

    def _get_successors_ordered(self, block):
        successors = []
        if hasattr(block, 'taken') and block.taken:
            successors.append(block.taken)
        if hasattr(block, 'fall_through') and block.fall_through:
            successors.append(block.fall_through)
        if hasattr(block, 'next_block') and block.next_block:
            successors.append(block.next_block)
        
        return successors

    def merge_blocks(self, start_block):
        changed = True
        while changed:
            changed = False
            blocks = self._get_all_blocks(start_block)

            for i, block in enumerate(blocks):
                if not block.predecessors:
                    continue

                # print("teste", block.label)
                    
                if len(block.predecessors) == 1 or "for.cond" in block.label:
                    pred = block.predecessors[0]
                    # 
                    # print( pred.label, "->", block.label)
                    pred_successors = self._get_successors_ordered(pred)
                    if len(pred_successors) == 1 and pred_successors[0] == block:
                        if "body" in pred.label and ("main" in block.label and "exit" not in block.label):
                            break

                        pred = self._merge_two_blocks(pred, block)

                        if i == 1:
                            start_block = pred

                        changed = True
                        break  # Reinicia o loop pois o CFG mudou

        return start_block

    def _merge_two_blocks(self, pred, block):
        if pred.instructions and self._is_unconditional_jump_to_block(pred.instructions[-1], block):    
            pred.instructions = pred.instructions[:-1] #removo o jump

            
        pred.extend(block.instructions)


        tmp = pred
        if type(pred) != type(block):
            new_pred = ConditionBlock(pred.label)
            new_pred.instructions = list(pred.instructions)
            new_pred.predecessors = list(pred.predecessors)


            for predecessor in block.predecessors:
                if predecessor != pred:
                    if isinstance(predecessor, BasicBlock):
                        predecessor.next_block = new_pred

            new_pred.fall_through = block.fall_through
            new_pred.taken = block.taken
    
            tmp = new_pred



        self._update_block_successors(tmp, block)
        
        block_successors = self._get_successors_ordered(block)

        for successor in block_successors:
            if successor is not None and hasattr(successor, 'predecessors'):
                if block in successor.predecessors:
                    successor.predecessors.remove(block)
                if tmp not in successor.predecessors:
                    successor.predecessors.append(tmp)

        block.predecessors = []
        self._clear_block_successors(block)
        block.instructions = []

        return tmp

    def _is_unconditional_jump_to_block(self, instruction, block):
        if instruction[0] != 'jump':
            return False
        
        # Verifica se o jump é para o label deste bloco
        if len(instruction) >= 2 and hasattr(block, 'label'):
            return block.label in instruction[1]
        
        return False

    def _update_block_successors(self, pred, block):
        if hasattr(block, 'next_block'):
            pred.next_block = block.next_block
        if hasattr(block, 'taken'):
            pred.taken = block.taken
            pred.fall_through = block.fall_through

    def _clear_block_successors(self, block):
        if hasattr(block, 'next_block'):
            block.next_block = None
        if hasattr(block, 'taken'):
            block.taken = None
        if hasattr(block, 'fall_through'):
            block.fall_through = None

    def appendOptimizedCode_original_order(self, start_block):
        """Versão que tenta preservar a ordem original dos blocos."""
        optimized_instructions = []
        
        visited = set()
        stack = [start_block]
        
        while stack:
            block = stack.pop()
            if block in visited:
                continue
            visited.add(block)
            
            if hasattr(block, 'label') and block.label:
                optimized_instructions.append(('label', block.label))
            
            optimized_instructions.extend(block.instructions)
            
            successors = self._get_successors_ordered(block)
            for succ in reversed(successors):
                if succ and succ not in visited:
                    stack.append(succ)
        
        self.current_func.optimized_instructions = optimized_instructions

    # Função para gerar o código final completo
    def generate_final_code(self):
        """Gera o código final otimizado para todo o programa."""
        final_code = []
        
        # Adiciona código global (se houver)
        if hasattr(self, 'code') and self.code:
            final_code.extend(self.code)
        
        # Adiciona código otimizado de todos os métodos
        for class_decl in self.class_decls:
            for method_decl in class_decl.method_decls:
                if hasattr(method_decl, 'optimized_instructions'):
                    # Adiciona cabeçalho do método
                    method_header = self._generate_method_header(method_decl)
                    final_code.append(method_header)
                    
                    # Adiciona instruções otimizadas
                    final_code.extend(method_decl.optimized_instructions)
        
        return final_code

    def _generate_method_header(self, method_decl):
        """Gera o cabeçalho do método para o código final."""
        # Exemplo: ('method', 'main', '[]')
        return ('method', method_decl.name, '[]')



    def _log_optimization_results(self, method_decl):
        """Log dos resultados da otimização para debug."""
        print(f"=== Método Otimizado: {method_decl.name} ===")
        if hasattr(method_decl, 'optimized_instructions'):
            for instr in method_decl.optimized_instructions:
                print(f"  {instr}")
        print("=" * 50)

def main():
    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        help="Path to file to be used to generate MJIR. By default, this script runs the interpreter on the optimized MJIR \
              and shows the speedup obtained from comparing original MJIR with its optimized version.",
        type=str,
    )
    parser.add_argument(
        "--opt",
        help="Print optimized MJIR generated from input_file.",
        action="store_true",
    )
    parser.add_argument(
        "--opt-pp",
        help="Print optimized MJIR generated from input_file.",
        action="store_true",
    )
    parser.add_argument(
        "--speedup",
        help="Show speedup from comparing original MJIR with its optimized version.",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "-c",
        "--cfg",
        help="show the CFG of the optimized MJIR for each function in pdf format",
        action="store_true",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="nem por deus"
    )
    args = parser.parse_args()

    speedup = args.speedup
    print_opt_ir = args.opt
    print_opt_ir_pp = args.opt_pp
    create_cfg = args.cfg
    verbose = args.verbose

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

    gen = CodeGenerator(False)
    gen.visit(ast)
    gencode = gen.code
    ast.text = gen.text

    opt = DataFlow(create_cfg, verbose)
    opt.visit(ast)
    optcode = opt.code
    if print_opt_ir:
        print("Optimized MJIR: --------")
        rich.print(optcode)
        print("------------------------\n")

    elif print_opt_ir_pp:
        print("Optimized MJIR: --------")
        opt.show()
        print("------------------------\n")

    speedup = len(gencode) / len(optcode)
    sys.stderr.write(
        "[SPEEDUP] Default: %d Optimized: %d Speedup: %.2f\n\n"
        % (len(gencode), len(optcode), speedup)
    )

    if not (print_opt_ir or print_opt_ir_pp):
        vm = MJIRInterpreter()
        vm.run(optcode)


if __name__ == "__main__":
    main()
