# Code Generation - Julia IR to Wasm instructions
# Maps Julia SSA statements to WebAssembly bytecode

export compile_function

"""
    compile_function(f, arg_types, func_name) -> WasmModule

Compile a Julia function to a WebAssembly module.
"""
function compile_function(f, arg_types::Tuple, func_name::String)::WasmModule
    # Get typed IR
    code_info, return_type = get_typed_ir(f, arg_types)

    # Create module
    mod = WasmModule()

    # Determine Wasm types for parameters and return
    param_types = NumType[julia_to_wasm_type(T) for T in arg_types]
    result_types = return_type === Nothing ? NumType[] : NumType[julia_to_wasm_type(return_type)]

    # Generate function body
    body = generate_body(code_info, arg_types, return_type)

    # Add function to module
    func_idx = add_function!(mod, param_types, result_types, NumType[], body)

    # Export the function
    add_export!(mod, func_name, 0, func_idx)

    return mod
end

"""
Generate Wasm bytecode from Julia CodeInfo.
"""
function generate_body(code_info, arg_types::Tuple, return_type)::Vector{UInt8}
    body = UInt8[]

    # Number of parameters (for local.get indices)
    n_params = length(arg_types)

    # SSA value to local index mapping
    # Parameters are locals 0..n_params-1
    # SSA values start after that
    ssa_to_local = Dict{Int, Int}()

    # Process each statement in the IR
    for (i, stmt) in enumerate(code_info.code)
        append!(body, compile_statement(stmt, i, code_info, n_params, ssa_to_local, arg_types))
    end

    # Add end opcode
    push!(body, Opcode.END)

    return body
end

"""
Compile a single IR statement to Wasm bytecode.
"""
function compile_statement(stmt, idx::Int, code_info, n_params::Int,
                          ssa_to_local::Dict{Int, Int}, arg_types::Tuple)::Vector{UInt8}
    bytes = UInt8[]

    if stmt isa Core.ReturnNode
        # Return statement
        if isdefined(stmt, :val)
            append!(bytes, compile_value(stmt.val, n_params, ssa_to_local))
        end
        push!(bytes, Opcode.RETURN)

    elseif stmt isa Core.GotoNode
        # Unconditional branch - handled separately

    elseif stmt isa Core.GotoIfNot
        # Conditional branch - handled separately

    elseif stmt isa Expr
        if stmt.head === :call
            append!(bytes, compile_call(stmt, idx, n_params, ssa_to_local, arg_types))
        elseif stmt.head === :invoke
            # Method invocation - similar to call
            append!(bytes, compile_invoke(stmt, idx, n_params, ssa_to_local, arg_types))
        end
    end

    return bytes
end

"""
Compile a value reference (SSA, Argument, or Literal).
"""
function compile_value(val, n_params::Int, ssa_to_local::Dict{Int, Int})::Vector{UInt8}
    bytes = UInt8[]

    if val isa Core.SSAValue
        # Reference to previous SSA result
        # For now, we don't use locals for SSA values - they're left on stack
        # This is a simplification that works for simple expressions

    elseif val isa Core.Argument
        # Reference to function argument
        # In Julia IR: Argument(1) is the function, Argument(2) is first param, etc.
        # In Wasm: local 0 is first param, local 1 is second param, etc.
        # So: wasm_idx = julia_arg_n - 2
        local_idx = val.n - 2
        if local_idx >= 0
            push!(bytes, Opcode.LOCAL_GET)
            append!(bytes, encode_leb128_unsigned(local_idx))
        end

    elseif val isa Core.SlotNumber
        # Slot reference (also an argument reference in optimized IR)
        local_idx = val.id - 1
        if local_idx > 0  # Skip the function slot
            push!(bytes, Opcode.LOCAL_GET)
            append!(bytes, encode_leb128_unsigned(local_idx - 1))
        end

    elseif val isa Int32
        push!(bytes, Opcode.I32_CONST)
        append!(bytes, encode_leb128_signed(val))

    elseif val isa Int64 || val isa Int
        push!(bytes, Opcode.I64_CONST)
        append!(bytes, encode_leb128_signed(val))

    elseif val isa Float32
        push!(bytes, Opcode.F32_CONST)
        append!(bytes, reinterpret(UInt8, [val]))

    elseif val isa Float64
        push!(bytes, Opcode.F64_CONST)
        append!(bytes, reinterpret(UInt8, [val]))
    end

    return bytes
end

"""
Normalize a function reference to a comparable form.
Handles GlobalRef, direct references, and intrinsics.
"""
function normalize_func(func)
    if func isa GlobalRef
        return (func.mod, func.name)
    elseif func isa Core.IntrinsicFunction
        return (:Core, :intrinsic)
    else
        return func
    end
end

"""
Check if a function matches a known intrinsic/builtin.
"""
function is_intrinsic(func, mod::Module, name::Symbol)
    if func isa GlobalRef
        return func.mod === mod && func.name === name
    elseif func isa Core.IntrinsicFunction
        # Compare by string representation for intrinsics
        return string(func) == string(name)
    else
        # Direct function reference
        try
            return func === getfield(mod, name)
        catch
            return false
        end
    end
end

"""
Compile a function call expression.
"""
function compile_call(expr::Expr, idx::Int, n_params::Int,
                      ssa_to_local::Dict{Int, Int}, arg_types::Tuple)::Vector{UInt8}
    bytes = UInt8[]

    # expr.args[1] is the function
    # expr.args[2:end] are the arguments
    func = expr.args[1]
    args = expr.args[2:end]

    # Push arguments onto the stack
    for arg in args
        append!(bytes, compile_value(arg, n_params, ssa_to_local))
    end

    # Determine argument type for opcode selection
    arg_type = length(args) > 0 ? determine_arg_type(args[1], arg_types) : Int64
    is_32bit = arg_type === Int32 || arg_type === UInt32

    # Check for intrinsics using GlobalRef pattern matching
    if is_add_int(func)
        push!(bytes, is_32bit ? Opcode.I32_ADD : Opcode.I64_ADD)

    elseif is_sub_int(func)
        push!(bytes, is_32bit ? Opcode.I32_SUB : Opcode.I64_SUB)

    elseif is_mul_int(func)
        push!(bytes, is_32bit ? Opcode.I32_MUL : Opcode.I64_MUL)

    elseif is_add_float(func)
        push!(bytes, arg_type === Float32 ? Opcode.F32_ADD : Opcode.F64_ADD)

    elseif is_sub_float(func)
        push!(bytes, arg_type === Float32 ? Opcode.F32_SUB : Opcode.F64_SUB)

    elseif is_mul_float(func)
        push!(bytes, arg_type === Float32 ? Opcode.F32_MUL : Opcode.F64_MUL)

    elseif is_div_float(func)
        push!(bytes, arg_type === Float32 ? Opcode.F32_DIV : Opcode.F64_DIV)

    elseif is_plus_operator(func)
        # High-level + operator - determine opcode by type
        if arg_type === Float32
            push!(bytes, Opcode.F32_ADD)
        elseif arg_type === Float64
            push!(bytes, Opcode.F64_ADD)
        elseif is_32bit
            push!(bytes, Opcode.I32_ADD)
        else
            push!(bytes, Opcode.I64_ADD)
        end

    else
        error("Unsupported function call: $func (type: $(typeof(func)))")
    end

    return bytes
end

# Helper functions to identify intrinsics
function is_add_int(func)
    (func isa GlobalRef && func.name === :add_int) ||
    is_intrinsic(func, Base, :add_int) ||
    is_intrinsic(func, Core.Intrinsics, :add_int)
end

function is_sub_int(func)
    (func isa GlobalRef && func.name === :sub_int) ||
    is_intrinsic(func, Base, :sub_int) ||
    is_intrinsic(func, Core.Intrinsics, :sub_int)
end

function is_mul_int(func)
    (func isa GlobalRef && func.name === :mul_int) ||
    is_intrinsic(func, Base, :mul_int) ||
    is_intrinsic(func, Core.Intrinsics, :mul_int)
end

function is_add_float(func)
    (func isa GlobalRef && func.name === :add_float) ||
    is_intrinsic(func, Base, :add_float) ||
    is_intrinsic(func, Core.Intrinsics, :add_float)
end

function is_sub_float(func)
    (func isa GlobalRef && func.name === :sub_float) ||
    is_intrinsic(func, Base, :sub_float) ||
    is_intrinsic(func, Core.Intrinsics, :sub_float)
end

function is_mul_float(func)
    (func isa GlobalRef && func.name === :mul_float) ||
    is_intrinsic(func, Base, :mul_float) ||
    is_intrinsic(func, Core.Intrinsics, :mul_float)
end

function is_div_float(func)
    (func isa GlobalRef && func.name === :div_float) ||
    is_intrinsic(func, Base, :div_float) ||
    is_intrinsic(func, Core.Intrinsics, :div_float)
end

function is_plus_operator(func)
    (func isa GlobalRef && func.name === :+) ||
    func === Base.:+
end

"""
Compile an invoke expression (method invocation).
"""
function compile_invoke(expr::Expr, idx::Int, n_params::Int,
                        ssa_to_local::Dict{Int, Int}, arg_types::Tuple)::Vector{UInt8}
    # expr.args[1] is the MethodInstance
    # expr.args[2] is the function
    # expr.args[3:end] are the arguments
    bytes = UInt8[]

    args = expr.args[3:end]

    # Push arguments
    for arg in args
        append!(bytes, compile_value(arg, n_params, ssa_to_local))
    end

    # For now, we only handle simple arithmetic
    # We need to inspect the MethodInstance to determine the operation
    mi = expr.args[1]
    if mi isa Core.MethodInstance
        # Check the method name
        meth = mi.def
        if meth isa Method
            name = meth.name
            if name === :+ || name === :add_int
                arg_type = determine_arg_type(args[1], arg_types)
                if arg_type === Int32 || arg_type === UInt32
                    push!(bytes, Opcode.I32_ADD)
                else
                    push!(bytes, Opcode.I64_ADD)
                end
            elseif name === :- || name === :sub_int
                arg_type = determine_arg_type(args[1], arg_types)
                if arg_type === Int32 || arg_type === UInt32
                    push!(bytes, Opcode.I32_SUB)
                else
                    push!(bytes, Opcode.I64_SUB)
                end
            elseif name === :* || name === :mul_int
                arg_type = determine_arg_type(args[1], arg_types)
                if arg_type === Int32 || arg_type === UInt32
                    push!(bytes, Opcode.I32_MUL)
                else
                    push!(bytes, Opcode.I64_MUL)
                end
            else
                error("Unsupported method: $name")
            end
        end
    end

    return bytes
end

"""
Determine the type of an argument from the IR.
"""
function determine_arg_type(arg, arg_types::Tuple)
    if arg isa Core.Argument
        # Argument(2) is first param -> arg_types[1]
        # Argument(3) is second param -> arg_types[2]
        param_idx = arg.n - 1  # Convert to 1-indexed for arg_types
        if param_idx >= 1 && param_idx <= length(arg_types)
            return arg_types[param_idx]
        end
    elseif arg isa Core.SlotNumber
        if arg.id > 1
            return arg_types[arg.id - 1]
        end
    elseif arg isa Int32
        return Int32
    elseif arg isa Int64 || arg isa Int
        return Int64
    elseif arg isa Float32
        return Float32
    elseif arg isa Float64
        return Float64
    end
    return Int64  # Default
end
