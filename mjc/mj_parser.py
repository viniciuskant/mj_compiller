import argparse
import pathlib
import sys
from io import StringIO

from sly import *

from mjc.mj_ast import *
from mjc.mj_lexer import MJLexer


class Coord:
    """Coordinates of a syntactic element. Consists of:
    - Line number
    - (optional) column number, for the Lexer
    """

    __slots__ = ("line", "column")

    def __init__(self, line, column=None):
        self.line = line
        self.column = column

    def __str__(self):
        if self.line and self.column is not None:
            coord_str = "@ %s:%s" % (self.line, self.column)
        elif self.line:
            coord_str = "@ %s" % (self.line)
        else:
            coord_str = ""
        return coord_str


class ParserLogger:
    """Logger Class used to log messages about the parser in a text stream.
    NOTE: This class overrides the default SlyLogger class
    """

    def __init__(self):
        self.stream = StringIO()

    @property
    def text(self):
        return self.stream.getvalue()

    def debug(self, msg, *args, **kwargs):
        self.stream.write((msg % args) + "\n")

    info = debug

    def warning(self, msg, *args, **kwargs):
        self.stream.write("WARNING: " + (msg % args) + "\n")

    def error(self, msg, *args, **kwargs):
        self.stream.write("ERROR: " + (msg % args) + "\n")

    critical = debug


class MJParser(Parser):
    tokens = MJLexer.tokens

    start = "program"
    debugfile = "parser.debug"
    log = ParserLogger()

    def __init__(self, debug=True):
        """Create a new MJParser."""
        self.debug = debug
        self.mjlex = MJLexer(self._lexer_error)

        self._last_yielded_token = None

    def parse(self, text):
        self._last_yielded_token = None
        return super().parse(self.mjlex.tokenize(text))

    def _lexer_error(self, msg, line, column):
        print("LexerError: %s at %d:%d" % (msg, line, column), file=sys.stdout)
        sys.exit(1)

    def _parser_error(self, msg, coord=None):
        if coord is None:
            print("ParserError: %s" % (msg), file=sys.stdout)
        else:
            print("ParserError: %s %s" % (msg, coord), file=sys.stdout)
        sys.exit(1)

    def _token_coord(self, p):
        last_cr = self.mjlex.text.rfind("\n", 0, p.index)
        if last_cr < 0:
            last_cr = -1
        column = p.index - (last_cr)
        return Coord(p.lineno, column)

    precedence = (
      ('left', 'OR'),
      ('left', 'AND'),
      ('left', 'EQ', 'NE'),
      ('left', 'LT', 'LE', 'GT', 'GE'),
      ('left', 'PLUS', 'MINUS'),
      ('left', 'TIMES', 'DIVIDE', 'MOD'),
    )

    # ------------------------------------------------------------
    # Parser Rules
    # ------------------------------------------------------------


    @_("class_declaration_list")
    def program(self, p):
        return Program(class_decls=p.class_declaration_list)

    @_("class_declaration")
    def class_declaration_list(self, p):
        return [p.class_declaration]

    @_("class_declaration_list class_declaration")
    def class_declaration_list(self, p):
        p.class_declaration_list.append(p.class_declaration)
        return p.class_declaration_list


    @_("CLASS identifier extends_expression_opt LBRACE compound_declaration_list_opt method_declaration_list_opt RBRACE")
    def class_declaration(self, p):
        return ClassDecl(
            name=p.identifier,
            extends=p.extends_expression_opt,
            var_decls=p.compound_declaration_list_opt or [],
            method_decls=p.method_declaration_list_opt or [],  # Agora pode ser vazio
            coord=self._token_coord(p)
        )

    @_("EXTENDS identifier")
    def extends_expression(self, p):
        return Extends(
            super=p.identifier,
            coord=self._token_coord(p)
        )

    @_("")
    def extends_expression_opt(self, p):
        return None

    @_("extends_expression")
    def extends_expression_opt(self, p):
        return p.extends_expression
    
    @_("regular_method_declaration",
       "main_method_declaration")
    def method_declaration(self, p):
        return p[0]
    
    @_("PUBLIC type_specifier identifier LPAREN parameter_list RPAREN compound_statement")
    def regular_method_declaration(self, p):
        return MethodDecl(
            type=p.type_specifier,
            name=p.identifier,
            param_list=p.parameter_list,
            body=p.compound_statement,
            coord=self._token_coord(p)
        )

    @_("PUBLIC type_specifier identifier LPAREN RPAREN compound_statement")
    def regular_method_declaration(self, p):
        return MethodDecl(
            type=p.type_specifier,
            name=p.identifier,
            param_list=None,
            body=p.compound_statement,
            coord=self._token_coord(p)
        )

    @_("parameter_list COMMA parameter_declaration")
    def parameter_list(self, p):
        if isinstance(p.parameter_list, ParamList):
            p.parameter_list.params.append(p.parameter_declaration)
            return p.parameter_list
        else:
            return ParamList(params=[p.parameter_list, p.parameter_declaration])
        
    @_("parameter_declaration")
    def parameter_list(self, p):
        return ParamList(params=[p.parameter_declaration])
    
    @_("method_declaration")
    def method_declaration_list(self, p):
        return [p.method_declaration]

    @_("method_declaration_list method_declaration")
    def method_declaration_list(self, p):
        p.method_declaration_list.append(p.method_declaration)
        return p.method_declaration_list

    @_("PUBLIC STATIC VOID MAIN LPAREN STRING LBRACKET RBRACKET identifier RPAREN compound_statement")
    def main_method_declaration(self, p):
        return MainMethodDecl(
            args=p.identifier,
            body=p.compound_statement,
            coord=self._token_coord(p)
        )

    @_("type_specifier init_declarator_list SEMI")
    def compound_declaration(self, p):
        return [
            VarDecl(
                type=p.type_specifier,  
                name=init['declarator'],
                init=init['initializer'],
                coord=init['coord']
            )
            for init in p.init_declarator_list
        ]

    @_("LBRACE RBRACE")
    def compound_statement(self, p):
        return Compound(statements=[], coord=self._token_coord(p))

    @_("LBRACE statement_list RBRACE")
    def compound_statement(self, p):
        return Compound(statements=p.statement_list, coord=self._token_coord(p))

    @_("LBRACE compound_declaration_list statement_list_opt RBRACE")
    def compound_statement(self, p):
        decls = p.compound_declaration_list or []
        stmts = p.statement_list_opt or []
        return Compound(statements=decls + stmts, coord=self._token_coord(p))


    @_("")
    def method_declaration_list_opt(self, p):
        return []

    @_("method_declaration_list")
    def method_declaration_list_opt(self, p):
        return p.method_declaration_list

    @_("")
    def compound_declaration_list_opt(self, p):
        return []

    @_("compound_declaration_list")
    def compound_declaration_list_opt(self, p):
        return p.compound_declaration_list


    @_("compound_declaration_list compound_declaration")
    def compound_declaration_list(self, p):
        return p.compound_declaration_list + p.compound_declaration

    @_("compound_declaration")
    def compound_declaration_list(self, p):
        return p.compound_declaration

    @_("")
    def statement_list_opt(self, p):
        return []

    @_("statement_list")
    def statement_list_opt(self, p):
        return p.statement_list

    # @_("")
    # def statement_list(self, p): 
    #     return []
    
    @_("statement")
    def statement_list(self, p):
        return [p.statement]

    @_("statement_list statement")
    def statement_list(self, p):
        if p.statement_list is None:
            return [p.statement]
        return p.statement_list + [p.statement]

    @_("init_declarator_list COMMA init_declarator")
    def init_declarator_list(self, p):
        if p.init_declarator_list is None:
            p.init_declarator_list = [p.init_declarator]
        else:
            p.init_declarator_list.append(p.init_declarator)
        return p.init_declarator_list

    @_("init_declarator")
    def init_declarator_list(self, p):
        return [p.init_declarator]

    @_("declarator EQUALS initializer")
    def init_declarator(self, p):
        return InitDeclarator(declarator=p.declarator, initializer=p.initializer, coord=self._token_coord(p))

    @_("declarator")
    def init_declarator(self, p):
        return InitDeclarator(declarator=p.declarator, initializer=None, coord=self._token_coord(p))

    @_("LBRACE initializer_list COMMA RBRACE")
    def initializer(self, p):
        tmp = self._token_coord(p)
        tmp = Coord(tmp.line, tmp.column + 1)
        return InitList(exprs=p.initializer_list, coord=tmp)

    @_("LBRACE RBRACE")
    def initializer(self, p):
        tmp = self._token_coord(p)
        tmp = Coord(tmp.line, tmp.column + 1)
        return InitList(exprs=[], coord=tmp)

    @_("LBRACE initializer_list RBRACE")
    def initializer(self, p):
        tmp = self._token_coord(p)
        tmp = Coord(tmp.line, tmp.column + 1)
        return InitList(exprs=p.initializer_list, coord=tmp)

    @_("assignment_expression")
    def initializer(self, p):
        return p.assignment_expression
    
    @_("initializer_list COMMA initializer")
    def initializer_list(self, p):
        p.initializer_list.append(p.initializer)
        return p.initializer_list
    
    @_("initializer")
    def initializer_list(self, p):
        return [p.initializer]

    @_("LPAREN declarator RPAREN")
    def declarator(self, p):
        return p.declarator

    @_("identifier")
    def declarator(self, p):
        return p.identifier

    @_("type_specifier declarator")
    def parameter_declaration(self, p):
        return ParamDecl(type=p.type_specifier, name=p.declarator, coord=self._token_coord(p))

    @_("identifier")
    def type_specifier(self, p):
        return Type(name=p.identifier, coord=self._token_coord(p))

    @_("INT LBRACKET RBRACKET")
    def type_specifier(self, p):
        return Type("int[]", coord=self._token_coord(p))

    @_("CHAR LBRACKET RBRACKET")
    def type_specifier(self, p):
        return Type("char[]", coord=self._token_coord(p))

    @_("STRING")
    def type_specifier(self, p):
        return Type(name="String", coord=self._token_coord(p))

    @_("INT")
    def type_specifier(self, p):
        return Type(name="int", coord=self._token_coord(p))

    @_("CHAR")
    def type_specifier(self, p):
        return Type(name="char", coord=self._token_coord(p))
  
    @_("BOOLEAN")
    def type_specifier(self, p):
        return Type(name="boolean", coord=self._token_coord(p))

    @_("VOID")
    def type_specifier(self, p):
        return Type(name="void", coord=self._token_coord(p))

    @_("assignment_expression")
    def expression(self, p):
        return p[0]

    @_("expression COMMA assignment_expression")
    def expression(self, p):
        if isinstance(p[0], ExprList):
            return ExprList(exprs=p[0].exprs + [p[2]], coord=self._token_coord(p))
        else:
            return ExprList(exprs=[p[0], p[2]], coord=self._token_coord(p))
    

    @_("unary_expression EQUALS assignment_expression")
    def assignment_expression(self, p):
        return Assignment(op="=", lvalue=p.unary_expression, rvalue=p.assignment_expression, coord=self._token_coord(p))

    @_("binary_expression")
    def assignment_expression(self, p):
        return p.binary_expression

    @_("unary_expression")
    def binary_expression(self, p):
        return p[0]

    @_("binary_expression TIMES binary_expression",
       "binary_expression DIVIDE binary_expression",
       "binary_expression MOD binary_expression",
       "binary_expression PLUS binary_expression",
       "binary_expression MINUS binary_expression",
       "binary_expression LT binary_expression",
       "binary_expression LE binary_expression",
       "binary_expression GT binary_expression",
       "binary_expression GE binary_expression",
       "binary_expression EQ binary_expression",
       "binary_expression NE binary_expression",
       "binary_expression AND binary_expression",
       "binary_expression OR binary_expression")
    def binary_expression(self, p):
        return BinaryOp(
            op=p[1],
            left=p[0],
            right=p[2],
            coord=self._token_coord(p)
        )
        
    @_("unary_operator unary_expression")
    def unary_expression(self, p):
        return UnaryOp(
            op=p.unary_operator,
            expr=p.unary_expression,
            coord=self._token_coord(p)
        )
    
    @_("postfix_expression")
    def unary_expression(self, p):
        return p.postfix_expression

    @_("PLUS")
    def unary_operator(self, p):
        return "+"

    @_("MINUS")
    def unary_operator(self, p):
        return "-"

    @_("NOT")
    def unary_operator(self, p):
        return "!"

    @_("primary_expression")
    def postfix_expression(self, p):
        return p.primary_expression

    @_("postfix_expression DOT LENGTH")
    def postfix_expression(self, p):
        return Length(p.postfix_expression, coord=self._token_coord(p))
    
    @_("postfix_expression DOT identifier")
    def postfix_expression(self, p):
        return FieldAccess( object=p.postfix_expression, field_name=p.identifier, coord=self._token_coord(p))


    @_("postfix_expression DOT identifier LPAREN argument_expression_list RPAREN")
    def postfix_expression(self, p):
        argument_list = p.argument_expression_list
        if argument_list != None and len(argument_list.exprs) == 1:
            argument_list = argument_list.exprs[0]  
        return MethodCall(
            object=p.postfix_expression,
            method_name=p.identifier,
            args=argument_list, 
            coord=self._token_coord(p)
        )

    @_("postfix_expression LBRACKET expression RBRACKET")
    def postfix_expression(self, p):
        return ArrayRef(name=p.postfix_expression, subscript=p.expression, coord=self._token_coord(p))
        
    @_("identifier")
    def primary_expression(self, p):
        return p.identifier

    @_("constant")
    def primary_expression(self, p):
        return p.constant

    @_("this_expression")
    def primary_expression(self, p):
        return This(coord=self._token_coord(p))

    @_("new_expression")
    def primary_expression(self, p):
        return p.new_expression

    @_("LPAREN expression RPAREN")
    def primary_expression(self, p):
        return p.expression

    @_("")
    def argument_expression_list(self, p):
        return None

    @_("assignment_expression")
    def argument_expression_list(self, p):
        if isinstance(p.assignment_expression, ExprList):
            return p.assignment_expression
        return ExprList(exprs=[p.assignment_expression], coord=self._token_coord(p))

    @_("argument_expression_list COMMA assignment_expression")
    def argument_expression_list(self, p):
        if isinstance(p.argument_expression_list, ExprList):
            return ExprList(exprs=p.argument_expression_list.exprs + [p.assignment_expression], coord=self._token_coord(p))
        else:
            return ExprList(exprs=[p.argument_expression_list, p.assignment_expression], coord=self._token_coord(p))
  
    @_("boolean_literal")
    def constant(self, p):
        return p.boolean_literal
    
    @_("CHAR_LITERAL")
    def constant(self, p):
        return Constant(type="char", value=p.CHAR_LITERAL, coord=self._token_coord(p))
    

    @_("INT_LITERAL")
    def constant(self, p):
        return Constant(type="int", value=p.INT_LITERAL, coord=self._token_coord(p))

    @_("STRING_LITERAL")
    def constant(self, p):
        return Constant(type="String", value=p.STRING_LITERAL, coord=self._token_coord(p))

    @_("THIS")
    def this_expression(self, p):
        return This(coord=self._token_coord(p))
    
    @_("NEW CHAR LBRACKET expression RBRACKET")
    def new_expression(self, p):
        return NewArray(type=Type("char[]"), size=p.expression, coord=self._token_coord(p))

    @_("NEW INT LBRACKET expression RBRACKET")
    def new_expression(self, p):
        return NewArray(type=Type("int[]", coord=None), size=p.expression, coord=self._token_coord(p))
        # return NewArray(type=Type("int[]", coord=self._token_coord(p)), size=p.expression, coord=self._token_coord(p))

    @_("NEW identifier LPAREN RPAREN")
    def new_expression(self, p):
        return NewObject(Type(name=p.identifier, coord=p.identifier.coord), coord=self._token_coord(p))

    @_("TRUE")
    def boolean_literal(self, p):
        return Constant(type="boolean", value="true", coord=self._token_coord(p))

    @_("FALSE")
    def boolean_literal(self, p):
        return Constant(type="boolean", value="false", coord=self._token_coord(p))


    @_("ID")
    def identifier(self, p):
        return ID(p.ID, coord=self._token_coord(p))

    @_("print_statement",
       "jump_statement",
       "assert_statement",
       "for_statement",
       "while_statement",
       "if_statement",
       "compound_statement",
       "expression_statement")
    def statement(self, p):
        return p[0]

    @_("expression SEMI")
    def expression_statement(self, p):
        return p.expression
    
    @_("IF LPAREN expression RPAREN statement")
    def if_statement(self, p):
        return If(
            cond=p.expression,
            iftrue=p.statement,
            iffalse=None,
            coord=self._token_coord(p)
        )

    @_("IF LPAREN expression RPAREN statement ELSE statement")
    def if_statement(self, p):
        return If(
            cond=p.expression,
            iftrue=p.statement0,
            iffalse=p.statement1,
            coord=self._token_coord(p)
        )
    
    @_("WHILE LPAREN expression RPAREN statement")
    def while_statement(self, p):
        return While(
            cond=p.expression,
            body=p.statement,
            coord=self._token_coord(p)
        )



    @_("FOR LPAREN compound_declaration expression_opt SEMI expression_opt RPAREN statement")
    def for_statement(self, p):
        return For(
            init=DeclList(decls=p.compound_declaration, coord=self._token_coord(p)),
            cond=p.expression_opt0,
            next=p.expression_opt1,
            body=p.statement,
            coord=self._token_coord(p)
        )
    
    @_("FOR LPAREN expression_opt SEMI expression_opt SEMI expression_opt RPAREN statement")
    def for_statement(self, p):
        return For(
            init=p.expression_opt0,
            cond=p.expression_opt1,
            next=p.expression_opt2,
            body=p.statement,
            coord=self._token_coord(p)
        )


    @_("ASSERT expression SEMI")
    def assert_statement(self, p):
        return Assert(expr=p.expression, coord=self._token_coord(p))
    
    @_("PRINT LPAREN expression_opt RPAREN SEMI")
    def print_statement(self, p):
        return Print(expr=p.expression_opt, coord=self._token_coord(p))


    @_("BREAK SEMI")
    def jump_statement(self, p):
        return Break(coord=self._token_coord(p))

    @_("RETURN expression_opt SEMI")
    def jump_statement(self, p):
        return Return(expr=p.expression_opt, coord=self._token_coord(p))


    @_("expression")
    def expression_opt(self, p):
        return p.expression

    @_("")
    def expression_opt(self, p):
        return None


    # ------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------
    def error(self, p):
        if p:
            self._parser_error(
                "Before %s" % p.value, Coord(p.lineno, self.mjlex.find_tok_column(p))
            )
        else:
            self._parser_error("At the end of input (%s)" % self.mjlex.filename)


def main():
    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to file to be parsed", type=str)
    args = parser.parse_args()

    # get input path
    input_file = args.input_file
    input_path = pathlib.Path(input_file)

    # check if file exists
    if not input_path.exists():
        print("ERROR: Input", input_path, "not found", file=sys.stderr)
        sys.exit(1)

    parser = MJParser(debug=True)

    # open file and print ast
    with open(input_path) as f:
        ast = parser.parse(f.read())
        print(parser.log.text)
        ast.show(buf=sys.stdout, showcoord=True)


if __name__ == "__main__":
    main()
