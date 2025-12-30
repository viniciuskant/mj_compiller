class MJType:
    """
    Class that represents a type in the MiniJava language.  Basic
    Types are declared as singleton instances of this type.
    """

    def __init__(
        self, name, binary_ops=set(), unary_ops=set(), rel_ops=set(), assign_ops=set()
    ):
        """
        You must implement yourself and figure out what to store.
        """
        self.typename = name
        self.unary_ops = unary_ops
        self.binary_ops = binary_ops
        self.rel_ops = rel_ops
        self.assign_ops = assign_ops

    def __str__(self):
        return f"type({self.typename})"


#TODO: TODO
#Temporário
class ObjectType(MJType):
    def __init__(self, real_type):
        """
        :param real_type: The actual class/type this object represents
        """
        super().__init__(name="object")
        self.real_type = real_type

VoidType = MJType(
    name="void"
)

#BasicTypes

BooleanType = MJType(
    name="boolean",
    binary_ops={"&&", "||"},
    unary_ops={"!"},
    rel_ops={"==", "!="},
    assign_ops={"="},
)

CharType = MJType(
    name="char",
    binary_ops=set(),
    unary_ops=set(),
    rel_ops={"==", "!=", "<", ">", "<=", ">="},
    assign_ops={"="},
)

IntType = MJType(
    name="int",
    binary_ops={"+", "-", "*", "/", "%"},
    unary_ops={"+", "-"},
    rel_ops={"==", "!=", "<", ">", "<=", ">="},
    assign_ops={"="},
)

StringType = MJType(
    name="string",
    binary_ops={"+", "==", "!="},
    unary_ops=set(),
    rel_ops={"==", "!="},
    assign_ops={"="},
)

# Array types
class ArrayType(MJType):
    def __init__(self, element_type, size=None):
        """
        :param element_type: MJType instance for array elements
        :param size: Optional size of the array
        """
        name = f"{element_type.typename}[]"
        self.size = size
        self.element_type = element_type
        super().__init__(
            name=name,
            binary_ops=set(),
            unary_ops=set(),
            rel_ops={"==", "!="},
            assign_ops={"="},
        )

# Example array type instances
CharArrayType = ArrayType(CharType)
IntArrayType = ArrayType(IntType)


# TODO: Complete in your repository
