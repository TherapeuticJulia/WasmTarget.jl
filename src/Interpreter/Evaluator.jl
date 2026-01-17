# Julia Interpreter Evaluator - Written in WasmTarget-compatible Julia
# This evaluator executes AST nodes produced by Parser.jl.
#
# Design:
# - Tree-walking interpreter
# - Environment (symbol table) using parallel arrays for WASM compatibility
# - All values represented as tagged union (Value struct)
# - Uses Int32 for all numeric operations (WASM friendly)
# - Recursive eval function for expression evaluation
#
# Value Types:
# - VAL_NOTHING: nothing/nil
# - VAL_INT: 32-bit integer
# - VAL_FLOAT: 32-bit float
# - VAL_BOOL: boolean (0 or 1)
# - VAL_STRING: string reference
# - VAL_FUNC: function reference (AST node)

export Value, ValueType, Env
export VAL_NOTHING, VAL_INT, VAL_FLOAT, VAL_BOOL, VAL_STRING, VAL_FUNC, VAL_ERROR
export val_nothing, val_int, val_float, val_bool, val_string, val_func, val_error
export env_new, env_get, env_set!, env_push_scope!, env_pop_scope!
export eval_node, eval_program
export ControlFlow, CF_NORMAL, CF_RETURN
export OutputBuffer, output_buffer_get, output_buffer_set!, output_buffer_append!, output_buffer_clear!

# ============================================================================
# Value Type Constants
# ============================================================================
const VAL_NOTHING = Int32(0)   # nothing
const VAL_INT     = Int32(1)   # Int32
const VAL_FLOAT   = Int32(2)   # Float32
const VAL_BOOL    = Int32(3)   # Bool (as Int32: 0 or 1)
const VAL_STRING  = Int32(4)   # String
const VAL_FUNC    = Int32(5)   # Function (reference to AST node)
const VAL_ERROR   = Int32(6)   # Error value

# ============================================================================
# Value Structure
# ============================================================================

"""
Value - A tagged union representing any interpreter value.

All Julia values are represented as Value during interpretation.
Uses Int32 for tag to be WASM-friendly.
"""
mutable struct Value
    tag::Int32            # Type tag (VAL_* constant)
    int_val::Int32        # Integer or bool value
    float_val::Float32    # Float value
    str_val::String       # String value
    # For functions: store AST node reference
    func_node::Union{ASTNode, Nothing}
end

# ============================================================================
# Value Constructors
# ============================================================================

"""Create a nothing value."""
@noinline function val_nothing()::Value
    return Value(VAL_NOTHING, Int32(0), Float32(0.0), "", nothing)
end

"""Create an integer value."""
@noinline function val_int(v::Int32)::Value
    return Value(VAL_INT, v, Float32(0.0), "", nothing)
end

"""Create a float value."""
@noinline function val_float(v::Float32)::Value
    return Value(VAL_FLOAT, Int32(0), v, "", nothing)
end

"""Create a boolean value."""
@noinline function val_bool(v::Int32)::Value
    return Value(VAL_BOOL, v, Float32(0.0), "", nothing)
end

"""Create a string value."""
@noinline function val_string(s::String)::Value
    return Value(VAL_STRING, Int32(0), Float32(0.0), s, nothing)
end

"""Create a function value (stores AST node)."""
@noinline function val_func(node::ASTNode)::Value
    return Value(VAL_FUNC, Int32(0), Float32(0.0), "", node)
end

"""Create an error value."""
@noinline function val_error()::Value
    return Value(VAL_ERROR, Int32(0), Float32(0.0), "", nothing)
end

# ============================================================================
# Value Helpers
# ============================================================================

"""Check if a value is truthy (for conditionals)."""
@noinline function val_is_truthy(v::Value)::Int32
    if v.tag == VAL_BOOL
        return v.int_val
    end
    if v.tag == VAL_INT
        if v.int_val != Int32(0)
            return Int32(1)
        end
        return Int32(0)
    end
    if v.tag == VAL_FLOAT
        if v.float_val != Float32(0.0)
            return Int32(1)
        end
        return Int32(0)
    end
    if v.tag == VAL_NOTHING
        return Int32(0)
    end
    if v.tag == VAL_STRING
        if str_len(v.str_val) > Int32(0)
            return Int32(1)
        end
        return Int32(0)
    end
    # Functions are truthy
    if v.tag == VAL_FUNC
        return Int32(1)
    end
    return Int32(0)
end

# ============================================================================
# Environment (Symbol Table)
# ============================================================================

"""
Env - Environment for variable bindings.

Uses parallel arrays for WASM compatibility (no Dict).
Supports nested scopes via scope_start index.
"""
mutable struct Env
    names::Vector{String}    # Variable names
    values::Vector{Value}    # Variable values
    count::Int32             # Number of bindings
    capacity::Int32          # Array capacity
    scope_starts::Vector{Int32}  # Stack of scope start indices
    scope_depth::Int32       # Current scope depth
end

"""Create a new environment with given capacity."""
@noinline function env_new(capacity::Int32)::Env
    names = Vector{String}(undef, capacity)
    values = Vector{Value}(undef, capacity)
    scope_starts = arr_new(Int32, Int32(32))  # Max 32 nested scopes

    # Initialize names to empty strings
    i = Int32(1)
    while i <= capacity
        names[i] = ""
        values[i] = val_nothing()
        i = i + Int32(1)
    end

    # Initialize first scope start at 1
    arr_set!(scope_starts, Int32(1), Int32(1))

    return Env(names, values, Int32(0), capacity, scope_starts, Int32(1))
end

"""Push a new scope (for function calls, blocks)."""
@noinline function env_push_scope!(env::Env)::Nothing
    if env.scope_depth < Int32(32)
        env.scope_depth = env.scope_depth + Int32(1)
        arr_set!(env.scope_starts, env.scope_depth, env.count + Int32(1))
    end
    return nothing
end

"""Pop current scope and all its bindings."""
@noinline function env_pop_scope!(env::Env)::Nothing
    if env.scope_depth > Int32(1)
        # Restore count to scope start
        env.count = arr_get(env.scope_starts, env.scope_depth) - Int32(1)
        env.scope_depth = env.scope_depth - Int32(1)
    end
    return nothing
end

"""Find variable index by name. Returns 0 if not found."""
@noinline function env_find(env::Env, name::String)::Int32
    # Search backwards (most recent bindings first, allows shadowing)
    i = env.count
    while i >= Int32(1)
        if str_eq(env.names[i], name)
            return i
        end
        i = i - Int32(1)
    end
    return Int32(0)
end

"""Get variable value by name. Returns nothing if not found."""
@noinline function env_get(env::Env, name::String)::Value
    idx = env_find(env, name)
    if idx > Int32(0)
        return env.values[idx]
    end
    return val_nothing()
end

"""Set variable value. Creates new binding if not found in current scope."""
@noinline function env_set!(env::Env, name::String, value::Value)::Nothing
    # Look for existing binding
    idx = env_find(env, name)

    if idx > Int32(0)
        # Update existing binding
        env.values[idx] = value
    else
        # Create new binding if space available
        if env.count < env.capacity
            env.count = env.count + Int32(1)
            env.names[env.count] = name
            env.values[env.count] = value
        end
    end
    return nothing
end

"""Define a new variable in current scope (always creates new binding)."""
@noinline function env_define!(env::Env, name::String, value::Value)::Nothing
    if env.count < env.capacity
        env.count = env.count + Int32(1)
        env.names[env.count] = name
        env.values[env.count] = value
    end
    return nothing
end

# ============================================================================
# Control Flow
# ============================================================================

"""Control flow signal for return statements."""
const CF_NORMAL = Int32(0)
const CF_RETURN = Int32(1)

mutable struct ControlFlow
    signal::Int32    # CF_NORMAL or CF_RETURN
    value::Value     # Return value if CF_RETURN
end

@noinline function cf_normal()::ControlFlow
    return ControlFlow(CF_NORMAL, val_nothing())
end

@noinline function cf_return(v::Value)::ControlFlow
    return ControlFlow(CF_RETURN, v)
end

# ============================================================================
# Built-in Functions
# ============================================================================

"""
Evaluate a built-in function call.
Returns (handled::Int32, result::Value) - handled is 1 if this was a builtin.
"""
@noinline function eval_builtin(name::String, args::Vector{Value}, num_args::Int32, env::Env)::Tuple{Int32, Value}
    # println - output to console
    if str_eq(name, "println")
        result = builtin_println(args, num_args, env)
        return (Int32(1), result)
    end

    # print - output without newline
    if str_eq(name, "print")
        result = builtin_print(args, num_args, env)
        return (Int32(1), result)
    end

    # abs - absolute value
    if str_eq(name, "abs")
        if num_args >= Int32(1)
            result = builtin_abs(args[1])
            return (Int32(1), result)
        end
        return (Int32(1), val_error())
    end

    # min - minimum of two values
    if str_eq(name, "min")
        if num_args >= Int32(2)
            result = builtin_min(args[1], args[2])
            return (Int32(1), result)
        end
        return (Int32(1), val_error())
    end

    # max - maximum of two values
    if str_eq(name, "max")
        if num_args >= Int32(2)
            result = builtin_max(args[1], args[2])
            return (Int32(1), result)
        end
        return (Int32(1), val_error())
    end

    # typeof - get type name as string
    if str_eq(name, "typeof")
        if num_args >= Int32(1)
            result = builtin_typeof(args[1])
            return (Int32(1), result)
        end
        return (Int32(1), val_error())
    end

    # string - convert to string
    if str_eq(name, "string")
        if num_args >= Int32(1)
            result = builtin_string(args[1])
            return (Int32(1), result)
        end
        return (Int32(1), val_string(""))
    end

    # length - get length of string
    if str_eq(name, "length")
        if num_args >= Int32(1)
            result = builtin_length(args[1])
            return (Int32(1), result)
        end
        return (Int32(1), val_error())
    end

    # Not a builtin
    return (Int32(0), val_nothing())
end

"""Output implementation - stores output in env for later retrieval."""
# Output buffer struct - mutable struct is easier to compile than Ref{String}
# Using a mutable struct with a single field allows WasmGC compilation
mutable struct OutputBuffer
    content::String
end

# Global output buffer (module-level mutable struct instance)
const _OUTPUT_BUFFER = OutputBuffer("")

# Helper functions for output buffer access (compile cleanly to struct.get/struct.set)
@noinline function output_buffer_get()::String
    return _OUTPUT_BUFFER.content
end

@noinline function output_buffer_set!(s::String)::Nothing
    _OUTPUT_BUFFER.content = s
    return nothing
end

@noinline function output_buffer_append!(s::String)::Nothing
    _OUTPUT_BUFFER.content = _OUTPUT_BUFFER.content * s
    return nothing
end

@noinline function output_buffer_clear!()::Nothing
    _OUTPUT_BUFFER.content = ""
    return nothing
end

@noinline function builtin_println(args::Vector{Value}, num_args::Int32, env::Env)::Value
    output = ""
    i = Int32(1)
    while i <= num_args
        if i > Int32(1)
            output = output * " "
        end
        output = output * value_to_string(args[i])
        i = i + Int32(1)
    end
    output = output * "\n"
    output_buffer_append!(output)
    return val_nothing()
end

@noinline function builtin_print(args::Vector{Value}, num_args::Int32, env::Env)::Value
    output = ""
    i = Int32(1)
    while i <= num_args
        if i > Int32(1)
            output = output * " "
        end
        output = output * value_to_string(args[i])
        i = i + Int32(1)
    end
    output_buffer_append!(output)
    return val_nothing()
end

@noinline function builtin_abs(v::Value)::Value
    if v.tag == VAL_INT
        x = v.int_val
        if x < Int32(0)
            return val_int(-x)
        end
        return val_int(x)
    end
    if v.tag == VAL_FLOAT
        x = v.float_val
        if x < Float32(0.0)
            return val_float(-x)
        end
        return val_float(x)
    end
    return val_error()
end

@noinline function builtin_min(a::Value, b::Value)::Value
    # Handle int-int
    if a.tag == VAL_INT && b.tag == VAL_INT
        if a.int_val < b.int_val
            return val_int(a.int_val)
        end
        return val_int(b.int_val)
    end
    # Handle float-float or mixed
    if a.tag == VAL_FLOAT || b.tag == VAL_FLOAT
        af = value_to_float(a)
        bf = value_to_float(b)
        if af < bf
            return val_float(af)
        end
        return val_float(bf)
    end
    return val_error()
end

@noinline function builtin_max(a::Value, b::Value)::Value
    if a.tag == VAL_INT && b.tag == VAL_INT
        if a.int_val > b.int_val
            return val_int(a.int_val)
        end
        return val_int(b.int_val)
    end
    if a.tag == VAL_FLOAT || b.tag == VAL_FLOAT
        af = value_to_float(a)
        bf = value_to_float(b)
        if af > bf
            return val_float(af)
        end
        return val_float(bf)
    end
    return val_error()
end

@noinline function builtin_typeof(v::Value)::Value
    if v.tag == VAL_NOTHING
        return val_string("Nothing")
    end
    if v.tag == VAL_INT
        return val_string("Int32")
    end
    if v.tag == VAL_FLOAT
        return val_string("Float32")
    end
    if v.tag == VAL_BOOL
        return val_string("Bool")
    end
    if v.tag == VAL_STRING
        return val_string("String")
    end
    if v.tag == VAL_FUNC
        return val_string("Function")
    end
    return val_string("Unknown")
end

@noinline function builtin_string(v::Value)::Value
    return val_string(value_to_string(v))
end

@noinline function builtin_length(v::Value)::Value
    if v.tag == VAL_STRING
        return val_int(str_len(v.str_val))
    end
    return val_error()
end

"""Convert a value to its float representation."""
@noinline function value_to_float(v::Value)::Float32
    if v.tag == VAL_FLOAT
        return v.float_val
    end
    if v.tag == VAL_INT
        return Float32(v.int_val)
    end
    return Float32(0.0)
end

"""Convert a value to string representation."""
@noinline function value_to_string(v::Value)::String
    if v.tag == VAL_NOTHING
        return "nothing"
    end
    if v.tag == VAL_BOOL
        if v.int_val == Int32(1)
            return "true"
        end
        return "false"
    end
    if v.tag == VAL_INT
        return int_to_string(v.int_val)
    end
    if v.tag == VAL_FLOAT
        return float_to_string(v.float_val)
    end
    if v.tag == VAL_STRING
        return v.str_val
    end
    if v.tag == VAL_FUNC
        return "<function>"
    end
    if v.tag == VAL_ERROR
        return "<error>"
    end
    return "<unknown>"
end

# Note: int_to_string and digit_to_str are now defined in Runtime/StringOps.jl
# They are exported and available as WasmTarget.int_to_string

"""Convert Float32 to string (simplified)."""
@noinline function float_to_string(f::Float32)::String
    # Simple implementation: convert integer part and one decimal place
    negative = f < Float32(0.0)
    if negative
        f = -f
    end

    int_part = Int32(floor(f))
    frac_part = Int32(round((f - Float32(int_part)) * Float32(10.0)))

    int_str = int_to_string(int_part)
    frac_str = int_to_string(frac_part)

    if negative
        return "-" * int_str * "." * frac_str
    end
    return int_str * "." * frac_str
end

# ============================================================================
# Expression Evaluation
# ============================================================================

"""Evaluate a binary operation."""
@noinline function eval_binary(op::Int32, left::Value, right::Value)::Value
    # Arithmetic with int-int
    if left.tag == VAL_INT && right.tag == VAL_INT
        return eval_binary_int_int(op, left.int_val, right.int_val)
    end

    # Arithmetic with floats
    if left.tag == VAL_FLOAT || right.tag == VAL_FLOAT
        lf = value_to_float(left)
        rf = value_to_float(right)
        return eval_binary_float_float(op, lf, rf)
    end

    # String concatenation
    if left.tag == VAL_STRING && right.tag == VAL_STRING
        if op == OP_ADD
            return val_string(left.str_val * right.str_val)
        end
    end

    # Equality for any types
    if op == OP_EQ
        return eval_equality(left, right)
    end
    if op == OP_NE
        eq_result = eval_equality(left, right)
        if eq_result.int_val == Int32(1)
            return val_bool(Int32(0))
        end
        return val_bool(Int32(1))
    end

    return val_error()
end

"""Evaluate binary op with two Int32s."""
@noinline function eval_binary_int_int(op::Int32, l::Int32, r::Int32)::Value
    if op == OP_ADD
        return val_int(l + r)
    end
    if op == OP_SUB
        return val_int(l - r)
    end
    if op == OP_MUL
        return val_int(l * r)
    end
    if op == OP_DIV
        if r != Int32(0)
            return val_int(l ÷ r)
        end
        return val_error()
    end
    if op == OP_MOD
        if r != Int32(0)
            return val_int(l % r)
        end
        return val_error()
    end
    if op == OP_POW
        # Simple integer power
        result = Int32(1)
        i = Int32(0)
        while i < r
            result = result * l
            i = i + Int32(1)
        end
        return val_int(result)
    end
    if op == OP_EQ
        if l == r
            return val_bool(Int32(1))
        end
        return val_bool(Int32(0))
    end
    if op == OP_NE
        if l != r
            return val_bool(Int32(1))
        end
        return val_bool(Int32(0))
    end
    if op == OP_LT
        if l < r
            return val_bool(Int32(1))
        end
        return val_bool(Int32(0))
    end
    if op == OP_LE
        if l <= r
            return val_bool(Int32(1))
        end
        return val_bool(Int32(0))
    end
    if op == OP_GT
        if l > r
            return val_bool(Int32(1))
        end
        return val_bool(Int32(0))
    end
    if op == OP_GE
        if l >= r
            return val_bool(Int32(1))
        end
        return val_bool(Int32(0))
    end
    if op == OP_AND
        if l != Int32(0) && r != Int32(0)
            return val_bool(Int32(1))
        end
        return val_bool(Int32(0))
    end
    if op == OP_OR
        if l != Int32(0) || r != Int32(0)
            return val_bool(Int32(1))
        end
        return val_bool(Int32(0))
    end
    return val_error()
end

"""Evaluate binary op with two Float32s."""
@noinline function eval_binary_float_float(op::Int32, l::Float32, r::Float32)::Value
    if op == OP_ADD
        return val_float(l + r)
    end
    if op == OP_SUB
        return val_float(l - r)
    end
    if op == OP_MUL
        return val_float(l * r)
    end
    if op == OP_DIV
        if r != Float32(0.0)
            return val_float(l / r)
        end
        return val_error()
    end
    if op == OP_POW
        # Float power using log/exp
        return val_float(Float32(Float64(l)^Float64(r)))
    end
    if op == OP_EQ
        if l == r
            return val_bool(Int32(1))
        end
        return val_bool(Int32(0))
    end
    if op == OP_NE
        if l != r
            return val_bool(Int32(1))
        end
        return val_bool(Int32(0))
    end
    if op == OP_LT
        if l < r
            return val_bool(Int32(1))
        end
        return val_bool(Int32(0))
    end
    if op == OP_LE
        if l <= r
            return val_bool(Int32(1))
        end
        return val_bool(Int32(0))
    end
    if op == OP_GT
        if l > r
            return val_bool(Int32(1))
        end
        return val_bool(Int32(0))
    end
    if op == OP_GE
        if l >= r
            return val_bool(Int32(1))
        end
        return val_bool(Int32(0))
    end
    return val_error()
end

"""Evaluate equality between any two values."""
@noinline function eval_equality(left::Value, right::Value)::Value
    # Same type comparisons
    if left.tag == right.tag
        if left.tag == VAL_NOTHING
            return val_bool(Int32(1))
        end
        if left.tag == VAL_BOOL || left.tag == VAL_INT
            if left.int_val == right.int_val
                return val_bool(Int32(1))
            end
            return val_bool(Int32(0))
        end
        if left.tag == VAL_FLOAT
            if left.float_val == right.float_val
                return val_bool(Int32(1))
            end
            return val_bool(Int32(0))
        end
        if left.tag == VAL_STRING
            if str_eq(left.str_val, right.str_val)
                return val_bool(Int32(1))
            end
            return val_bool(Int32(0))
        end
    end
    # Different types are not equal (except numeric promotion)
    if (left.tag == VAL_INT && right.tag == VAL_FLOAT) || (left.tag == VAL_FLOAT && right.tag == VAL_INT)
        lf = value_to_float(left)
        rf = value_to_float(right)
        if lf == rf
            return val_bool(Int32(1))
        end
        return val_bool(Int32(0))
    end
    return val_bool(Int32(0))
end

"""Evaluate a unary operation."""
@noinline function eval_unary(op::Int32, operand::Value)::Value
    if op == OP_NEG
        if operand.tag == VAL_INT
            return val_int(-operand.int_val)
        end
        if operand.tag == VAL_FLOAT
            return val_float(-operand.float_val)
        end
        return val_error()
    end
    if op == OP_NOT
        truthy = val_is_truthy(operand)
        if truthy == Int32(1)
            return val_bool(Int32(0))
        end
        return val_bool(Int32(1))
    end
    return val_error()
end

# ============================================================================
# Main Evaluation
# ============================================================================

"""
Evaluate an AST node and return (Value, ControlFlow).
"""
@noinline function eval_node(node::ASTNode, source::String, env::Env)::Tuple{Value, ControlFlow}
    kind = node.kind

    # Literals
    if kind == AST_INT_LIT
        return (val_int(node.int_value), cf_normal())
    end

    if kind == AST_FLOAT_LIT
        return (val_float(node.float_value), cf_normal())
    end

    if kind == AST_BOOL_LIT
        return (val_bool(node.int_value), cf_normal())
    end

    if kind == AST_STRING_LIT
        s = str_substr(source, node.str_start, node.str_length)
        return (val_string(s), cf_normal())
    end

    if kind == AST_NOTHING_LIT
        return (val_nothing(), cf_normal())
    end

    # Identifier (variable lookup)
    if kind == AST_IDENT
        name = str_substr(source, node.str_start, node.str_length)
        v = env_get(env, name)
        return (v, cf_normal())
    end

    # Binary operation
    if kind == AST_BINARY
        return eval_binary_node(node, source, env)
    end

    # Unary operation
    if kind == AST_UNARY
        if node.left !== nothing
            (operand, cf) = eval_node(node.left, source, env)
            if cf.signal == CF_RETURN
                return (operand, cf)
            end
            result = eval_unary(node.op, operand)
            return (result, cf_normal())
        end
        return (val_error(), cf_normal())
    end

    # Function call
    if kind == AST_CALL
        return eval_call(node, source, env)
    end

    # Assignment
    if kind == AST_ASSIGN
        return eval_assignment(node, source, env)
    end

    # If statement
    if kind == AST_IF
        return eval_if(node, source, env)
    end

    # While loop
    if kind == AST_WHILE
        return eval_while(node, source, env)
    end

    # For loop
    if kind == AST_FOR
        return eval_for(node, source, env)
    end

    # Function definition
    if kind == AST_FUNC
        return eval_func_def(node, source, env)
    end

    # Return statement
    if kind == AST_RETURN
        return eval_return(node, source, env)
    end

    # Block
    if kind == AST_BLOCK
        return eval_block(node, source, env)
    end

    # Program
    if kind == AST_PROGRAM
        return eval_block(node, source, env)
    end

    # Error
    return (val_error(), cf_normal())
end

"""Evaluate binary operation node with short-circuit logic."""
@noinline function eval_binary_node(node::ASTNode, source::String, env::Env)::Tuple{Value, ControlFlow}
    op = node.op

    # Short-circuit AND
    if op == OP_AND
        if node.left !== nothing
            (left, cf) = eval_node(node.left, source, env)
            if cf.signal == CF_RETURN
                return (left, cf)
            end
            if val_is_truthy(left) == Int32(0)
                return (val_bool(Int32(0)), cf_normal())
            end
            if node.right !== nothing
                (right, cf2) = eval_node(node.right, source, env)
                if cf2.signal == CF_RETURN
                    return (right, cf2)
                end
                if val_is_truthy(right) == Int32(1)
                    return (val_bool(Int32(1)), cf_normal())
                end
            end
        end
        return (val_bool(Int32(0)), cf_normal())
    end

    # Short-circuit OR
    if op == OP_OR
        if node.left !== nothing
            (left, cf) = eval_node(node.left, source, env)
            if cf.signal == CF_RETURN
                return (left, cf)
            end
            if val_is_truthy(left) == Int32(1)
                return (val_bool(Int32(1)), cf_normal())
            end
            if node.right !== nothing
                (right, cf2) = eval_node(node.right, source, env)
                if cf2.signal == CF_RETURN
                    return (right, cf2)
                end
                if val_is_truthy(right) == Int32(1)
                    return (val_bool(Int32(1)), cf_normal())
                end
            end
        end
        return (val_bool(Int32(0)), cf_normal())
    end

    # Regular binary operation
    if node.left !== nothing && node.right !== nothing
        (left, cf) = eval_node(node.left, source, env)
        if cf.signal == CF_RETURN
            return (left, cf)
        end
        (right, cf2) = eval_node(node.right, source, env)
        if cf2.signal == CF_RETURN
            return (right, cf2)
        end
        result = eval_binary(op, left, right)
        return (result, cf_normal())
    end
    return (val_error(), cf_normal())
end

"""Evaluate function call."""
@noinline function eval_call(node::ASTNode, source::String, env::Env)::Tuple{Value, ControlFlow}
    # Evaluate arguments first
    args = Vector{Value}(undef, 16)
    num_args = node.num_children

    i = Int32(1)
    while i <= num_args
        (arg_val, cf) = eval_node(node.children[i], source, env)
        if cf.signal == CF_RETURN
            return (arg_val, cf)
        end
        args[i] = arg_val
        i = i + Int32(1)
    end

    # Get function name (callee should be an identifier)
    if node.left !== nothing && node.left.kind == AST_IDENT
        func_name = str_substr(source, node.left.str_start, node.left.str_length)

        # Check for builtin
        (handled, result) = eval_builtin(func_name, args, num_args, env)
        if handled == Int32(1)
            return (result, cf_normal())
        end

        # Look up user-defined function
        func_val = env_get(env, func_name)
        if func_val.tag == VAL_FUNC && func_val.func_node !== nothing
            return eval_user_func(func_val.func_node, args, num_args, source, env)
        end
    end

    return (val_error(), cf_normal())
end

"""Evaluate user-defined function call."""
@noinline function eval_user_func(func_node::ASTNode, args::Vector{Value}, num_args::Int32, source::String, env::Env)::Tuple{Value, ControlFlow}
    # Create new scope for function
    env_push_scope!(env)

    # Bind parameters to arguments
    num_params = func_node.int_value  # Number of parameters stored in int_value

    i = Int32(1)
    while i <= num_params && i <= num_args
        param_node = func_node.children[i]
        if param_node.kind == AST_IDENT
            param_name = str_substr(source, param_node.str_start, param_node.str_length)
            env_define!(env, param_name, args[i])
        end
        i = i + Int32(1)
    end

    # Evaluate body (statements after parameters)
    result = val_nothing()
    j = num_params + Int32(1)
    total_children = func_node.num_children

    while j <= total_children
        (stmt_result, cf) = eval_node(func_node.children[j], source, env)
        if cf.signal == CF_RETURN
            result = cf.value
            break
        end
        result = stmt_result
        j = j + Int32(1)
    end

    # Pop function scope
    env_pop_scope!(env)

    return (result, cf_normal())
end

"""Evaluate assignment."""
@noinline function eval_assignment(node::ASTNode, source::String, env::Env)::Tuple{Value, ControlFlow}
    if node.left !== nothing && node.right !== nothing
        if node.left.kind == AST_IDENT
            name = str_substr(source, node.left.str_start, node.left.str_length)
            (value, cf) = eval_node(node.right, source, env)
            if cf.signal == CF_RETURN
                return (value, cf)
            end
            env_set!(env, name, value)
            return (value, cf_normal())
        end
    end
    return (val_error(), cf_normal())
end

"""Evaluate if statement."""
@noinline function eval_if(node::ASTNode, source::String, env::Env)::Tuple{Value, ControlFlow}
    # Evaluate condition
    if node.left !== nothing
        (cond_val, cf) = eval_node(node.left, source, env)
        if cf.signal == CF_RETURN
            return (cond_val, cf)
        end

        if val_is_truthy(cond_val) == Int32(1)
            # Execute then branch (children)
            result = val_nothing()
            i = Int32(1)
            while i <= node.num_children
                (stmt_result, cf2) = eval_node(node.children[i], source, env)
                if cf2.signal == CF_RETURN
                    return (stmt_result, cf2)
                end
                result = stmt_result
                i = i + Int32(1)
            end
            return (result, cf_normal())
        else
            # Execute else branch (right) if present
            if node.right !== nothing
                return eval_node(node.right, source, env)
            end
        end
    end
    return (val_nothing(), cf_normal())
end

"""Evaluate while loop."""
@noinline function eval_while(node::ASTNode, source::String, env::Env)::Tuple{Value, ControlFlow}
    result = val_nothing()
    max_iterations = Int32(10000)  # Safety limit
    iteration = Int32(0)

    while iteration < max_iterations
        # Check condition
        if node.left !== nothing
            (cond_val, cf) = eval_node(node.left, source, env)
            if cf.signal == CF_RETURN
                return (cond_val, cf)
            end

            if val_is_truthy(cond_val) == Int32(0)
                break
            end

            # Execute body
            i = Int32(1)
            while i <= node.num_children
                (stmt_result, cf2) = eval_node(node.children[i], source, env)
                if cf2.signal == CF_RETURN
                    return (stmt_result, cf2)
                end
                result = stmt_result
                i = i + Int32(1)
            end
        else
            break
        end

        iteration = iteration + Int32(1)
    end

    return (result, cf_normal())
end

"""Evaluate for loop (simple integer range iteration)."""
@noinline function eval_for(node::ASTNode, source::String, env::Env)::Tuple{Value, ControlFlow}
    result = val_nothing()

    # Get iterator variable name
    if node.left !== nothing && node.left.kind == AST_IDENT
        var_name = str_substr(source, node.left.str_start, node.left.str_length)

        # Evaluate iterable (for now, support simple integer range 1:n)
        # We'll interpret a single integer as range 1:n for simplicity
        if node.right !== nothing
            (iter_val, cf) = eval_node(node.right, source, env)
            if cf.signal == CF_RETURN
                return (iter_val, cf)
            end

            if iter_val.tag == VAL_INT
                # Iterate from 1 to iter_val
                env_push_scope!(env)

                i = Int32(1)
                n = iter_val.int_val
                while i <= n
                    env_set!(env, var_name, val_int(i))

                    # Execute body
                    j = Int32(1)
                    while j <= node.num_children
                        (stmt_result, cf2) = eval_node(node.children[j], source, env)
                        if cf2.signal == CF_RETURN
                            env_pop_scope!(env)
                            return (stmt_result, cf2)
                        end
                        result = stmt_result
                        j = j + Int32(1)
                    end

                    i = i + Int32(1)
                end

                env_pop_scope!(env)
            end
        end
    end

    return (result, cf_normal())
end

"""Evaluate function definition."""
@noinline function eval_func_def(node::ASTNode, source::String, env::Env)::Tuple{Value, ControlFlow}
    # Get function name
    func_name = str_substr(source, node.str_start, node.str_length)

    # Store function AST in environment
    func_val = val_func(node)
    env_set!(env, func_name, func_val)

    return (func_val, cf_normal())
end

"""Evaluate return statement."""
@noinline function eval_return(node::ASTNode, source::String, env::Env)::Tuple{Value, ControlFlow}
    if node.left !== nothing
        (value, cf) = eval_node(node.left, source, env)
        if cf.signal == CF_RETURN
            return (value, cf)
        end
        return (value, cf_return(value))
    end
    return (val_nothing(), cf_return(val_nothing()))
end

"""Evaluate a block of statements."""
@noinline function eval_block(node::ASTNode, source::String, env::Env)::Tuple{Value, ControlFlow}
    result = val_nothing()

    i = Int32(1)
    while i <= node.num_children
        (stmt_result, cf) = eval_node(node.children[i], source, env)
        if cf.signal == CF_RETURN
            return (stmt_result, cf)
        end
        result = stmt_result
        i = i + Int32(1)
    end

    return (result, cf_normal())
end

# ============================================================================
# Top-Level Program Evaluation
# ============================================================================

"""
Evaluate a complete program. Returns the output as a string.
"""
@noinline function eval_program(program::ASTNode, source::String)::String
    # Clear output buffer
    output_buffer_clear!()

    # Create environment
    env = env_new(Int32(1024))  # Capacity for 1024 variables

    # Evaluate program
    (result, _) = eval_node(program, source, env)

    # If no output was produced but we have a result, show it
    output = output_buffer_get()
    if str_len(output) == Int32(0) && result.tag != VAL_NOTHING
        output = value_to_string(result)
    end

    return output
end

"""
Get current output buffer contents (for testing).
"""
@noinline function get_output()::String
    return output_buffer_get()
end

"""
Clear output buffer.
"""
@noinline function clear_output()::Nothing
    output_buffer_clear!()
    return nothing
end
