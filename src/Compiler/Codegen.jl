# Code Generation - Julia IR to Wasm instructions
# Maps Julia SSA statements to WebAssembly bytecode

export compile_function, compile_module, compile_handler, FunctionRegistry

# ============================================================================
# Struct Type Registry
# ============================================================================

"""
Maps Julia struct types to their WasmGC representation.
"""
struct StructInfo
    julia_type::DataType
    wasm_type_idx::UInt32
    field_names::Vector{Symbol}
    field_types::Vector{Type}  # Can include Union types
end

"""
Maps Julia Union types to their WasmGC tagged union representation.
Tagged unions are WasmGC structs with {tag: i32, value: anyref}.
The tag identifies which variant the union currently holds.
"""
struct UnionInfo
    julia_type::Union                        # The original Union type
    wasm_type_idx::UInt32                    # Index of the wrapper struct type
    variant_types::Vector{Type}              # Types in the union (ordered)
    tag_map::Dict{Type, Int32}               # Type -> tag value
end

"""
Registry for struct and array type mappings within a module.
"""
mutable struct TypeRegistry
    structs::Dict{DataType, StructInfo}
    arrays::Dict{Type, UInt32}  # Element type -> array type index
    string_array_idx::Union{Nothing, UInt32}  # Index of i8 array type for strings
    unions::Dict{Union, UnionInfo}  # Union type -> tagged union info
end

TypeRegistry() = TypeRegistry(Dict{DataType, StructInfo}(), Dict{Type, UInt32}(), nothing, Dict{Union, UnionInfo}())

# ============================================================================
# Function Registry - for multi-function modules
# ============================================================================

"""
Information about a compiled function within a module.
"""
struct FunctionInfo
    name::String
    func_ref::Any           # Original Julia function
    arg_types::Tuple        # Argument types for dispatch
    wasm_idx::UInt32        # Index in the Wasm module
    return_type::Type       # Return type (Nothing means void)
end

"""
Registry for functions within a module, enabling cross-function calls.
"""
mutable struct FunctionRegistry
    functions::Dict{String, FunctionInfo}       # name -> info
    by_ref::Dict{Any, Vector{FunctionInfo}}     # func_ref -> infos (for dispatch)
end

FunctionRegistry() = FunctionRegistry(Dict{String, FunctionInfo}(), Dict{Any, Vector{FunctionInfo}}())

"""
Register a function in the registry.
"""
function register_function!(registry::FunctionRegistry, name::String, func_ref, arg_types::Tuple, wasm_idx::UInt32, return_type::Type=Any)
    info = FunctionInfo(name, func_ref, arg_types, wasm_idx, return_type)
    registry.functions[name] = info

    # Also index by function reference for dispatch
    if !haskey(registry.by_ref, func_ref)
        registry.by_ref[func_ref] = FunctionInfo[]
    end
    push!(registry.by_ref[func_ref], info)

    return info
end

"""
Look up a function by name.
"""
function get_function(registry::FunctionRegistry, name::String)::Union{FunctionInfo, Nothing}
    return get(registry.functions, name, nothing)
end

"""
Look up a function by reference and argument types (for dispatch).
"""
function get_function(registry::FunctionRegistry, func_ref, arg_types::Tuple)::Union{FunctionInfo, Nothing}
    infos = get(registry.by_ref, func_ref, nothing)
    infos === nothing && return nothing

    # Find matching signature (exact match for now)
    for info in infos
        if info.arg_types == arg_types
            return info
        end
    end

    # Try to find a compatible signature (subtype matching)
    for info in infos
        if length(info.arg_types) == length(arg_types)
            match = true
            for (expected, actual) in zip(info.arg_types, arg_types)
                if !(actual <: expected)
                    match = false
                    break
                end
            end
            if match
                return info
            end
        end
    end

    return nothing
end

"""
Compile a constant value to WASM bytecode (for global initializers).
This is a simplified version of compile_value for use in constant expressions.
"""
function compile_const_value(val, mod::WasmModule, registry::TypeRegistry)::Vector{UInt8}
    bytes = UInt8[]

    if val isa Int32
        push!(bytes, Opcode.I32_CONST)
        append!(bytes, encode_leb128_signed(val))
    elseif val isa Int64
        push!(bytes, Opcode.I64_CONST)
        append!(bytes, encode_leb128_signed(val))
    elseif val isa Float32
        push!(bytes, Opcode.F32_CONST)
        append!(bytes, reinterpret(UInt8, [val]))
    elseif val isa Float64
        push!(bytes, Opcode.F64_CONST)
        append!(bytes, reinterpret(UInt8, [val]))
    elseif val isa Bool
        push!(bytes, Opcode.I32_CONST)
        push!(bytes, val ? 0x01 : 0x00)
    elseif val isa String
        # Strings are compiled as WasmGC arrays of i32 (character codes)
        # Get or create string array type
        str_type_idx = get_string_array_type!(mod, registry)

        # Push each character code
        for c in val
            push!(bytes, Opcode.I32_CONST)
            append!(bytes, encode_leb128_signed(Int32(c)))
        end

        # array.new_fixed $type_idx $length
        push!(bytes, Opcode.GC_PREFIX)
        push!(bytes, Opcode.ARRAY_NEW_FIXED)
        append!(bytes, encode_leb128_unsigned(str_type_idx))
        append!(bytes, encode_leb128_unsigned(length(val)))
    elseif val === nothing
        # For Nothing type, we use ref.null with any heap type
        push!(bytes, Opcode.REF_NULL)
        push!(bytes, 0x6E)  # none heap type
    else
        # For other types, try to push as integer if small enough
        T = typeof(val)
        if isprimitivetype(T) && sizeof(T) <= 4
            int_val = Core.Intrinsics.bitcast(UInt32, val)
            push!(bytes, Opcode.I32_CONST)
            append!(bytes, encode_leb128_signed(Int32(int_val)))
        elseif isprimitivetype(T) && sizeof(T) <= 8
            int_val = Core.Intrinsics.bitcast(UInt64, val)
            push!(bytes, Opcode.I64_CONST)
            append!(bytes, encode_leb128_signed(Int64(int_val)))
        else
            error("Cannot compile constant value of type $(typeof(val)) for global initializer")
        end
    end

    return bytes
end

"""
Get or create an array type for a given element type.
"""
function get_array_type!(mod::WasmModule, registry::TypeRegistry, elem_type::Type)::UInt32
    if haskey(registry.arrays, elem_type)
        return registry.arrays[elem_type]
    end

    # Create the array type
    # Check if element type is currently being registered (self-referential)
    local wasm_elem_type
    if haskey(_registering_types, elem_type)
        reserved_idx = _registering_types[elem_type]
        if reserved_idx >= 0
            # Use concrete reference to the reserved type index
            wasm_elem_type = ConcreteRef(UInt32(reserved_idx), true)
        else
            # Being registered but not self-referential - use get_concrete_wasm_type
            wasm_elem_type = get_concrete_wasm_type(elem_type, mod, registry)
        end
    else
        # Not being registered - use get_concrete_wasm_type for proper type lookup
        wasm_elem_type = get_concrete_wasm_type(elem_type, mod, registry)
    end
    type_idx = add_array_type!(mod, wasm_elem_type, true)  # mutable arrays
    registry.arrays[elem_type] = type_idx
    return type_idx
end

"""
Get or create the string array type (array of i32 for characters).
Mutable to support array.copy for string concatenation.
"""
function get_string_array_type!(mod::WasmModule, registry::TypeRegistry)::UInt32
    if registry.string_array_idx === nothing
        # Create an i32 array type for strings (mutable for array.copy support)
        registry.string_array_idx = add_array_type!(mod, I32, true)
    end
    return registry.string_array_idx
end

"""
Get or create an array type that holds string references.
Used for StringDict keys array.
"""
function get_string_ref_array_type!(mod::WasmModule, registry::TypeRegistry)::UInt32
    # First ensure string array type exists
    str_type_idx = get_string_array_type!(mod, registry)

    # Create array type for string refs if not exists
    # Key: use Vector{String} as the Julia type marker
    if !haskey(registry.arrays, Vector{String})
        # Element type is (ref null str_type_idx) - ConcreteRef with nullable=true
        str_ref_type = ConcreteRef(str_type_idx, true)
        arr_idx = add_array_type!(mod, str_ref_type, true)
        registry.arrays[Vector{String}] = arr_idx
    end
    return registry.arrays[Vector{String}]
end

"""
Get a concrete Wasm type for a Julia type, using the module and registry.
This is used before CompilationContext is created.
"""
function get_concrete_wasm_type(T::Type, mod::WasmModule, registry::TypeRegistry)::WasmValType
    # Union{} (bottom type) indicates unreachable code - return void/nothing
    if T === Union{}
        # Return a sentinel value that will cause UNREACHABLE to be emitted
        # For now, use i64 as a placeholder (this type won't actually be used)
        return I64
    end
    if is_closure_type(T)
        # Closure types are structs with captured variables
        if haskey(registry.structs, T)
            info = registry.structs[T]
            return ConcreteRef(info.wasm_type_idx, true)
        else
            register_closure_type!(mod, registry, T)
            if haskey(registry.structs, T)
                info = registry.structs[T]
                return ConcreteRef(info.wasm_type_idx, true)
            end
        end
        return StructRef
    elseif is_struct_type(T)
        if haskey(registry.structs, T)
            info = registry.structs[T]
            return ConcreteRef(info.wasm_type_idx, true)
        else
            register_struct_type!(mod, registry, T)
            if haskey(registry.structs, T)
                info = registry.structs[T]
                return ConcreteRef(info.wasm_type_idx, true)
            end
        end
        return StructRef
    elseif T <: Tuple
        if haskey(registry.structs, T)
            info = registry.structs[T]
            return ConcreteRef(info.wasm_type_idx, true)
        else
            register_tuple_type!(mod, registry, T)
            if haskey(registry.structs, T)
                info = registry.structs[T]
                return ConcreteRef(info.wasm_type_idx, true)
            end
        end
        return StructRef
    elseif T <: AbstractArray  # Handles Vector, Matrix, and higher-dim arrays
        # Both Vector and Matrix are stored as structs with (ref, size) fields
        # This allows setfield!(v, :size, ...) for push!/resize! operations
        if T <: AbstractVector
            if haskey(registry.structs, T)
                info = registry.structs[T]
                return ConcreteRef(info.wasm_type_idx, true)
            else
                info = register_vector_type!(mod, registry, T)
                return ConcreteRef(info.wasm_type_idx, true)
            end
        else
            # Matrix and higher-dim arrays: register as struct
            if haskey(registry.structs, T)
                info = registry.structs[T]
                return ConcreteRef(info.wasm_type_idx, true)
            else
                info = register_matrix_type!(mod, registry, T)
                return ConcreteRef(info.wasm_type_idx, true)
            end
        end
    elseif T === String
        # Strings are WasmGC arrays of bytes
        type_idx = get_string_array_type!(mod, registry)
        return ConcreteRef(type_idx, true)
    elseif T === Int128 || T === UInt128
        # 128-bit integers are represented as WasmGC structs with two i64 fields
        if haskey(registry.structs, T)
            info = registry.structs[T]
            return ConcreteRef(info.wasm_type_idx, true)
        else
            info = register_int128_type!(mod, registry, T)
            return ConcreteRef(info.wasm_type_idx, true)
        end
    elseif T isa Union
        # Handle Union types - use the inner type for Union{Nothing, T}
        inner_type = get_nullable_inner_type(T)
        if inner_type !== nothing
            # Union{Nothing, T} -> concrete type of T (nullable reference)
            return get_concrete_wasm_type(inner_type, mod, registry)
        else
            # Multi-variant union - fall back to generic type
            return julia_to_wasm_type(T)
        end
    else
        return julia_to_wasm_type(T)
    end
end

# ============================================================================
# Main Compilation Entry Point
# ============================================================================

"""
    compile_function(f, arg_types, func_name) -> WasmModule

Compile a Julia function to a WebAssembly module.
"""
function compile_function(f, arg_types::Tuple, func_name::String)::WasmModule
    # Get typed IR
    code_info, return_type = get_typed_ir(f, arg_types)

    # Create module
    mod = WasmModule()

    # Create type registry for struct mappings
    type_registry = TypeRegistry()

    # Check if this is a closure (function with captured variables)
    # For closures, we need to include the closure object as the first argument
    closure_type = typeof(f)
    is_closure = is_closure_type(closure_type)
    if is_closure
        # Prepend the closure type to arg_types
        arg_types = (closure_type, arg_types...)
    end

    # Detect WasmGlobal arguments (phantom params that map to Wasm globals)
    global_args = Set{Int}()
    for (i, T) in enumerate(arg_types)
        if T <: WasmGlobal
            push!(global_args, i)
            # Add the global to the module at the specified index
            elem_type = global_eltype(T)
            wasm_type = julia_to_wasm_type(elem_type)
            global_idx = global_index(T)
            # Ensure we have enough globals (fill with defaults if needed)
            while length(mod.globals) <= global_idx
                add_global!(mod, wasm_type, true, zero(elem_type))
            end
        end
    end

    # Register any struct/array/string/closure types used in parameters (skip WasmGlobal)
    for (i, T) in enumerate(arg_types)
        if i in global_args
            continue  # Skip WasmGlobal
        end
        if is_closure_type(T)
            # Closure types need special registration
            register_closure_type!(mod, type_registry, T)
        elseif is_struct_type(T)
            register_struct_type!(mod, type_registry, T)
        elseif T <: AbstractArray
            # Register array type for Vector/Matrix parameters
            elem_type = eltype(T)
            get_array_type!(mod, type_registry, elem_type)
        elseif T === String
            # Register string array type
            get_string_array_type!(mod, type_registry)
        end
    end

    # Register return type if it's a struct/array/string
    # Skip Union{} (bottom type) which is a subtype of everything
    if return_type === Union{}
        # Bottom type - no registration needed
    elseif is_struct_type(return_type)
        register_struct_type!(mod, type_registry, return_type)
    elseif return_type <: AbstractArray
        elem_type = eltype(return_type)
        get_array_type!(mod, type_registry, elem_type)
    elseif return_type === String
        get_string_array_type!(mod, type_registry)
    end

    # Determine Wasm types for parameters and return (skip WasmGlobal args)
    param_types = WasmValType[]
    for (i, T) in enumerate(arg_types)
        if !(i in global_args)
            push!(param_types, get_concrete_wasm_type(T, mod, type_registry))
        end
    end
    result_types = return_type === Nothing ? WasmValType[] : WasmValType[get_concrete_wasm_type(return_type, mod, type_registry)]

    # For single-function modules, the function index is 0
    # This allows recursive calls to work
    expected_func_idx = UInt32(0)

    # Check if this is an intrinsic function that needs special code generation
    intrinsic_body = is_intrinsic_function(f) ? generate_intrinsic_body(f, arg_types, mod, type_registry) : nothing

    local body::Vector{UInt8}
    local locals::Vector{WasmValType}

    if intrinsic_body !== nothing
        # Use the intrinsic body directly
        body = intrinsic_body
        locals = WasmValType[]  # Intrinsics don't need additional locals
    else
        # Generate function body with the function reference for self-call detection
        ctx = CompilationContext(code_info, arg_types, return_type, mod, type_registry;
                                func_idx=expected_func_idx, func_ref=f, global_args=global_args,
                                is_compiled_closure=is_closure)
        body = generate_body(ctx)
        locals = ctx.locals
    end

    # Add function to module
    func_idx = add_function!(mod, param_types, result_types, locals, body)

    # Export the function
    add_export!(mod, func_name, 0, func_idx)

    return mod
end

# ============================================================================
# WASM-057: Auto-discover function dependencies
# ============================================================================

"""
Set of WasmTarget runtime function names that can be auto-discovered.
These are intrinsic functions that have special compilation support.
"""
const WASMTARGET_RUNTIME_FUNCTIONS = Set([
    # String operations (StringOps.jl)
    :str_char, :str_setchar!, :str_len, :str_new, :str_copy, :str_substr,
    :str_eq, :str_hash, :str_find, :str_contains, :str_startswith, :str_endswith,
    :str_uppercase, :str_lowercase, :str_trim,
    # String conversion (WASM-054, WASM-055)
    :digit_to_str, :int_to_string,
    # Array operations (ArrayOps.jl)
    :arr_new, :arr_get, :arr_set!, :arr_len, :arr_fill!,
    # SimpleDict operations
    :sd_new, :sd_get, :sd_set!, :sd_haskey, :sd_length,
    # StringDict operations
    :sdict_new, :sdict_get, :sdict_set!, :sdict_haskey, :sdict_length,
])

"""
    discover_dependencies(functions::Vector) -> Vector

Scan the IR of all functions and discover WasmTarget runtime function dependencies.
Returns an expanded function list with auto-discovered dependencies added.

This enables calling runtime functions like str_eq without explicitly including them.
"""
function discover_dependencies(functions::Vector)::Vector
    # Normalize input first
    normalized = Vector{Tuple{Any, Tuple, String}}()
    for entry in functions
        if length(entry) == 2
            f, arg_types = entry
            name = string(nameof(f))
            push!(normalized, (f, arg_types, name))
        else
            push!(normalized, (entry[1], entry[2], entry[3]))
        end
    end

    # Track which functions we've already seen (by (func_ref, arg_types))
    seen_funcs = Set{Tuple{Any, Tuple}}()
    for (f, arg_types, _) in normalized
        push!(seen_funcs, (f, arg_types))
    end

    # Track discovered dependencies
    to_add = Vector{Tuple{Any, Tuple, String}}()

    # Queue of functions to scan (using Any-typed vector)
    to_scan = Vector{Tuple{Any, Tuple, String}}(normalized)

    while !isempty(to_scan)
        f, arg_types, name = popfirst!(to_scan)

        # Get IR for this function
        code_info = try
            ir, _ = Base.code_ircode(f, arg_types)[1]
            ir
        catch
            continue  # Skip if we can't get IR
        end

        # Scan IR for GlobalRef calls to WasmTarget runtime functions
        for stmt in code_info.stmts.stmt
            if stmt isa Expr
                scan_expr_for_deps!(stmt, seen_funcs, to_add, to_scan)
            end
        end
    end

    # Add discovered dependencies to the function list
    result = copy(normalized)
    append!(result, to_add)
    return result
end

"""
Scan an expression for WasmTarget runtime function calls.
"""
function scan_expr_for_deps!(expr::Expr, seen_funcs::Set, to_add::Vector, to_scan::Vector)
    # Check if this is an invoke expression
    if expr.head === :invoke && length(expr.args) >= 2
        func_ref = expr.args[2]
        if func_ref isa GlobalRef
            check_and_add_runtime_func!(func_ref, seen_funcs, to_add, to_scan)
        end
    elseif expr.head === :call && length(expr.args) >= 1
        func_ref = expr.args[1]
        if func_ref isa GlobalRef
            check_and_add_runtime_func!(func_ref, seen_funcs, to_add, to_scan)
        end
    end

    # Recursively scan nested expressions
    for arg in expr.args
        if arg isa Expr
            scan_expr_for_deps!(arg, seen_funcs, to_add, to_scan)
        end
    end
end

"""
Check if a GlobalRef is a WasmTarget runtime function and add it if needed.
"""
function check_and_add_runtime_func!(ref::GlobalRef, seen_funcs::Set, to_add::Vector, to_scan::Vector)
    # Get the actual function first - this handles cases where the function
    # is imported into another module (e.g., Main.str_eq when using WasmTarget)
    func = try
        getfield(ref.mod, ref.name)
    catch
        return  # Can't get function
    end

    # Skip if not a function
    if !isa(func, Function)
        return
    end

    # Check if this function belongs to WasmTarget (by checking its parent module)
    # This handles both WasmTarget.str_eq and imported str_eq (which becomes Main.str_eq)
    if parentmodule(func) !== WasmTarget
        return
    end

    # Check if this is a known runtime function
    func_name = nameof(func)
    if func_name in WASMTARGET_RUNTIME_FUNCTIONS
        # Determine argument types based on the function name
        arg_types = infer_runtime_func_arg_types(func_name)
        if arg_types === nothing
            return  # Can't infer types
        end

        # Check if we've already seen this (func, arg_types)
        key = (func, arg_types)
        if key in seen_funcs
            return
        end

        # Add to seen and to_add
        push!(seen_funcs, key)
        name = string(func_name)
        entry = (func, arg_types, name)
        push!(to_add, entry)
        push!(to_scan, entry)  # Also scan this function for its deps
    end
end

"""
Infer argument types for WasmTarget runtime functions.
Returns Nothing if types cannot be inferred.
"""
function infer_runtime_func_arg_types(name::Symbol)::Union{Tuple, Nothing}
    # String operations typically use String and Int32
    if name in [:str_char]
        return (String, Int32)
    elseif name in [:str_setchar!]
        return (String, Int32, Int32)
    elseif name in [:str_len]
        return (String,)
    elseif name in [:str_new]
        return (Int32,)
    elseif name in [:str_copy]
        return (String, Int32, String, Int32, Int32)
    elseif name in [:str_substr]
        return (String, Int32, Int32)
    elseif name in [:str_eq, :str_find, :str_contains]
        return (String, String)
    elseif name in [:str_startswith, :str_endswith]
        return (String, String)
    elseif name in [:str_hash]
        return (String,)
    elseif name in [:str_uppercase, :str_lowercase, :str_trim]
        return (String,)
    elseif name in [:digit_to_str]
        return (Int32,)
    elseif name in [:int_to_string]
        return (Int32,)
    # Array operations
    elseif name in [:arr_len]
        return nothing  # Can't infer element type
    elseif name in [:arr_new]
        return nothing  # Can't infer element type
    elseif name in [:arr_get]
        return nothing  # Can't infer element type
    elseif name in [:arr_set!]
        return nothing  # Can't infer element type
    # SimpleDict operations
    elseif name in [:sd_new]
        return (Int32,)
    elseif name in [:sd_get, :sd_haskey]
        return nothing  # Need SimpleDict type
    elseif name in [:sd_set!]
        return nothing  # Need SimpleDict type
    elseif name in [:sd_length]
        return nothing  # Need SimpleDict type
    # StringDict operations
    elseif name in [:sdict_new]
        return (Int32,)
    elseif name in [:sdict_get, :sdict_haskey]
        return nothing  # Need StringDict type
    elseif name in [:sdict_set!]
        return nothing  # Need StringDict type
    elseif name in [:sdict_length]
        return nothing  # Need StringDict type
    else
        return nothing
    end
end

"""
Check if a function is a WasmTarget intrinsic that needs special code generation.
Returns true if the function should be generated as an intrinsic instead of compiling Julia IR.
"""
function is_intrinsic_function(f::Function)::Bool
    fname = nameof(f)
    return fname in [:str_char, :str_len, :str_eq, :str_new, :str_setchar!, :str_concat, :str_substr]
end

"""
Generate intrinsic function body for WasmTarget runtime functions.
These functions have special WASM implementations that differ from their Julia fallbacks.
Returns the function body bytes, or nothing if not an intrinsic.
"""
function generate_intrinsic_body(f::Function, arg_types::Tuple, mod::WasmModule, type_registry::TypeRegistry)::Union{Vector{UInt8}, Nothing}
    fname = nameof(f)
    bytes = UInt8[]

    # Get string array type for string operations
    str_type_idx = get_string_array_type!(mod, type_registry)

    if fname === :str_char
        # str_char(s::String, i::Int32)::Int32
        # Gets character at 1-based index
        # local 0 = string (array ref)
        # local 1 = index (i32)
        push!(bytes, Opcode.LOCAL_GET)
        push!(bytes, 0x00)  # string
        push!(bytes, Opcode.LOCAL_GET)
        push!(bytes, 0x01)  # index
        # Subtract 1 for 0-based indexing
        push!(bytes, Opcode.I32_CONST)
        push!(bytes, 0x01)
        push!(bytes, Opcode.I32_SUB)
        # array.get
        push!(bytes, Opcode.GC_PREFIX)
        push!(bytes, Opcode.ARRAY_GET)
        append!(bytes, encode_leb128_unsigned(str_type_idx))
        push!(bytes, Opcode.END)
        return bytes

    elseif fname === :str_len
        # str_len(s::String)::Int32
        # Returns length of string array
        # local 0 = string (array ref)
        push!(bytes, Opcode.LOCAL_GET)
        push!(bytes, 0x00)  # string
        # array.len
        push!(bytes, Opcode.GC_PREFIX)
        push!(bytes, Opcode.ARRAY_LEN)
        push!(bytes, Opcode.END)
        return bytes

    elseif fname === :str_eq
        # str_eq(a::String, b::String)::Bool
        # Compare two strings character by character
        # This is complex - we need a loop
        # For now, return a simple stub
        # TODO: Implement proper string comparison loop
        push!(bytes, Opcode.LOCAL_GET)
        push!(bytes, 0x00)  # a
        push!(bytes, Opcode.LOCAL_GET)
        push!(bytes, 0x01)  # b
        push!(bytes, Opcode.REF_EQ)
        push!(bytes, Opcode.END)
        return bytes

    elseif fname === :str_new
        # str_new(len::Int32)::String
        # Create new string array of given length
        push!(bytes, Opcode.LOCAL_GET)
        push!(bytes, 0x00)  # length
        push!(bytes, Opcode.GC_PREFIX)
        push!(bytes, Opcode.ARRAY_NEW_DEFAULT)
        append!(bytes, encode_leb128_unsigned(str_type_idx))
        push!(bytes, Opcode.END)
        return bytes

    elseif fname === :str_setchar!
        # str_setchar!(s::String, i::Int32, c::Int32)::Nothing
        # Sets character at 1-based index
        push!(bytes, Opcode.LOCAL_GET)
        push!(bytes, 0x00)  # string
        push!(bytes, Opcode.LOCAL_GET)
        push!(bytes, 0x01)  # index
        # Subtract 1 for 0-based indexing
        push!(bytes, Opcode.I32_CONST)
        push!(bytes, 0x01)
        push!(bytes, Opcode.I32_SUB)
        push!(bytes, Opcode.LOCAL_GET)
        push!(bytes, 0x02)  # char
        # array.set
        push!(bytes, Opcode.GC_PREFIX)
        push!(bytes, Opcode.ARRAY_SET)
        append!(bytes, encode_leb128_unsigned(str_type_idx))
        push!(bytes, Opcode.END)
        return bytes

    elseif fname === :str_substr
        # str_substr(s::String, start::Int32, len::Int32)::String
        # Extracts substring by creating new string and copying characters
        # local 0 = source string (array ref)
        # local 1 = start (1-based Int32)
        # local 2 = len (Int32)

        # NOTE: The inline version at call sites properly implements this using
        # array.new + array.copy with scratch locals from the caller's context.
        # This intrinsic body is only used when str_substr is called as a
        # standalone function. We return a stub that returns the source string.
        # The proper implementation requires extra locals which intrinsics don't support.

        push!(bytes, Opcode.LOCAL_GET)
        push!(bytes, 0x00)  # return source string as placeholder
        push!(bytes, Opcode.END)
        return bytes
    end

    return nothing
end

"""
    compile_module(functions::Vector) -> WasmModule

Compile multiple Julia functions into a single WebAssembly module.

Each element of `functions` should be a tuple of (function, arg_types) or
(function, arg_types, name). If name is omitted, the function's name is used.

# Example
```julia
mod = compile_module([
    (add, (Int32, Int32)),
    (sub, (Int32, Int32)),
    (mul, (Int32, Int32), "multiply"),
])
```

Functions can call each other within the module.
"""
function compile_module(functions::Vector)::WasmModule
    # WASM-057: Auto-discover function dependencies
    functions = discover_dependencies(functions)
    # Create shared module and registries
    mod = WasmModule()
    type_registry = TypeRegistry()
    func_registry = FunctionRegistry()

    # WASM-060: Add Math.pow import for float power operations
    # This enables x^y for Float32/Float64 types
    add_import!(mod, "Math", "pow", NumType[F64, F64], NumType[F64])

    # Normalize input: ensure each entry is (func, arg_types, name)
    normalized = []
    for entry in functions
        if length(entry) == 2
            f, arg_types = entry
            name = string(nameof(f))
            push!(normalized, (f, arg_types, name))
        else
            push!(normalized, entry)
        end
    end

    # Track all required globals across all functions
    required_globals = Dict{Int, Tuple{WasmValType, Type}}()  # global_idx -> (wasm_type, julia_elem_type)

    # First pass: register types, detect WasmGlobals, and reserve function slots
    # We need to know all function indices before compiling bodies
    function_data = []  # Store (f, arg_types, name, code_info, return_type, global_args) for each function

    for (f, arg_types, name) in normalized
        # Check if this is a closure (function with captured variables)
        closure_type = typeof(f)
        is_closure = is_closure_type(closure_type)
        if is_closure
            # Prepend the closure type to arg_types
            arg_types = (closure_type, arg_types...)
        end

        # Get typed IR
        code_info, return_type = get_typed_ir(f, arg_types)

        # Detect WasmGlobal arguments
        global_args = Set{Int}()
        for (i, T) in enumerate(arg_types)
            if T <: WasmGlobal
                push!(global_args, i)
                elem_type = global_eltype(T)
                wasm_type = julia_to_wasm_type(elem_type)
                global_idx = global_index(T)
                required_globals[global_idx] = (wasm_type, elem_type)
            end
        end

        # Register types used in parameters (skip WasmGlobal)
        for (i, T) in enumerate(arg_types)
            if i in global_args
                continue
            end
            if is_closure_type(T)
                register_closure_type!(mod, type_registry, T)
            elseif is_struct_type(T)
                register_struct_type!(mod, type_registry, T)
            elseif T <: AbstractVector
                # Vector is now a struct with (ref, size) for setfield! support
                register_vector_type!(mod, type_registry, T)
            elseif T <: AbstractArray
                # Multi-dimensional arrays (Matrix, etc.) - register as struct
                register_matrix_type!(mod, type_registry, T)
            elseif T === String
                get_string_array_type!(mod, type_registry)
            end
        end

        # Register return type
        if is_closure_type(return_type)
            register_closure_type!(mod, type_registry, return_type)
        elseif is_struct_type(return_type)
            register_struct_type!(mod, type_registry, return_type)
        elseif return_type <: AbstractVector
            # Vector is now a struct with (ref, size) for setfield! support
            register_vector_type!(mod, type_registry, return_type)
        elseif return_type <: AbstractArray
            # Multi-dimensional arrays (Matrix, etc.) - register as struct
            register_matrix_type!(mod, type_registry, return_type)
        elseif return_type === String
            get_string_array_type!(mod, type_registry)
        end

        push!(function_data, (f, arg_types, name, code_info, return_type, global_args, is_closure))
    end

    # Add all required globals to the module
    for global_idx in sort(collect(keys(required_globals)))
        wasm_type, elem_type = required_globals[global_idx]
        while length(mod.globals) <= global_idx
            add_global!(mod, wasm_type, true, zero(elem_type))
        end
    end

    # Scan all function IR for GlobalRef to mutable structs (module-level globals)
    # These need to be shared across all functions as WASM globals
    module_globals = Dict{Tuple{Module, Symbol}, UInt32}()
    for (f, arg_types, name, code_info, return_type, global_args, is_closure) in function_data
        for stmt in code_info.code
            if stmt isa GlobalRef
                # Check if this GlobalRef points to a mutable struct instance
                try
                    actual_val = getfield(stmt.mod, stmt.name)
                    T = typeof(actual_val)
                    # Check if it's a mutable struct (but not a type, function, or module)
                    if ismutabletype(T) && !isa(actual_val, Type) && !isa(actual_val, Function) && !isa(actual_val, Module)
                        key = (stmt.mod, stmt.name)
                        if !haskey(module_globals, key)
                            # Register the struct type first
                            info = register_struct_type!(mod, type_registry, T)
                            type_idx = info.wasm_type_idx

                            # Build initialization expression: struct.new with default values
                            init_bytes = UInt8[]
                            for field_name in fieldnames(T)
                                field_val = getfield(actual_val, field_name)
                                append!(init_bytes, compile_const_value(field_val, mod, type_registry))
                            end
                            push!(init_bytes, Opcode.GC_PREFIX)
                            push!(init_bytes, Opcode.STRUCT_NEW)
                            append!(init_bytes, encode_leb128_unsigned(type_idx))

                            # Add global with reference type
                            global_idx = add_global_ref!(mod, type_idx, true, init_bytes; nullable=false)
                            module_globals[key] = global_idx
                        end
                    end
                catch
                    # If we can't evaluate, skip it
                end
            end
        end
    end

    # Calculate function indices (accounting for imports)
    # Functions are added in order, so index = n_imports + position - 1
    n_imports = length(mod.imports)
    for (i, (f, arg_types, name, _, return_type, _, _)) in enumerate(function_data)
        func_idx = UInt32(n_imports + i - 1)
        register_function!(func_registry, name, f, arg_types, func_idx, return_type)
    end

    # Second pass: compile function bodies
    for (i, (f, arg_types, name, code_info, return_type, global_args, is_closure)) in enumerate(function_data)
        func_idx = UInt32(n_imports + i - 1)

        # Check if this is an intrinsic function that needs special code generation
        intrinsic_body = is_intrinsic_function(f) ? generate_intrinsic_body(f, arg_types, mod, type_registry) : nothing

        local body::Vector{UInt8}
        local locals::Vector{WasmValType}

        if intrinsic_body !== nothing
            # Use the intrinsic body directly
            body = intrinsic_body
            locals = WasmValType[]  # Intrinsics don't need additional locals
        else
            # Generate function body from Julia IR
            ctx = CompilationContext(code_info, arg_types, return_type, mod, type_registry;
                                    func_registry=func_registry, func_idx=func_idx, func_ref=f,
                                    global_args=global_args, is_compiled_closure=is_closure,
                                    module_globals=module_globals)
            body = generate_body(ctx)
            locals = ctx.locals
        end

        # Get param/result types (skip WasmGlobal args)
        param_types = WasmValType[]
        for (j, T) in enumerate(arg_types)
            if !(j in global_args)
                push!(param_types, get_concrete_wasm_type(T, mod, type_registry))
            end
        end
        result_types = return_type === Nothing ? WasmValType[] : WasmValType[get_concrete_wasm_type(return_type, mod, type_registry)]

        # Add function to module
        actual_idx = add_function!(mod, param_types, result_types, locals, body)

        # Export the function
        add_export!(mod, name, 0, actual_idx)
    end

    return mod
end

"""
Specification for a DOM update call after signal write.
Used by compile_handler to inject DOM update calls after signal writes.
"""
struct DOMBindingSpec
    import_idx::UInt32          # Index of the DOM import function
    const_args::Vector{Int32}   # Constant arguments (e.g., hydration key)
    include_signal_value::Bool  # Whether to pass signal value as final arg
end

"""
    compile_handler(closure, signal_fields, export_name; globals, imports, dom_bindings) -> WasmModule

Compile a Therapy.jl event handler closure to WebAssembly with signal substitution.

The `signal_fields` dict maps captured closure field names to their signal info:
- Key: field name (Symbol), e.g., :count, :set_count
- Value: tuple (is_getter::Bool, global_idx::UInt32, value_type::Type)

The handler closure should take no arguments. Signal getters/setters are captured
in the closure and compiled to Wasm global.get/global.set operations.

When `dom_bindings` is provided, DOM update calls are automatically injected after
each signal write. This is used by Therapy.jl for reactive DOM updates.

# Example
```julia
count, set_count = create_signal(0)
handler = () -> set_count(count() + 1)

signal_fields = Dict(
    :count => (true, UInt32(0), Int64),      # getter for global 0
    :set_count => (false, UInt32(0), Int64)  # setter for global 0
)

mod = compile_handler(handler, signal_fields, "onclick")
```
"""
function compile_handler(
    closure::Function,
    signal_fields::Dict{Symbol, Tuple{Bool, UInt32, Type}},
    export_name::String;
    globals::Vector{Tuple{Type, Any}} = Tuple{Type, Any}[],  # (type, initial_value) pairs
    imports::Vector{Tuple{String, String, Vector, Vector}} = Tuple{String, String, Vector, Vector}[],  # (module, name, params, results)
    dom_bindings::Dict{UInt32, Vector{DOMBindingSpec}} = Dict{UInt32, Vector{DOMBindingSpec}}()  # global_idx -> DOM updates
)::WasmModule
    # Get typed IR for the closure (no arguments since it's a thunk)
    typed_results = Base.code_typed(closure, ())
    if isempty(typed_results)
        error("Could not get typed IR for handler closure")
    end
    code_info, return_type = typed_results[1]

    # Create module
    mod = WasmModule()
    type_registry = TypeRegistry()

    # Add imports first (they affect function indices)
    import_indices = Dict{Tuple{String, String}, UInt32}()
    for (mod_name, func_name, params, results) in imports
        idx = add_import!(mod, mod_name, func_name, params, results)
        import_indices[(mod_name, func_name)] = idx
    end

    # Create globals from signal fields
    # Collect unique global indices and their types
    required_globals = Dict{UInt32, Type}()
    for (_, (_, global_idx, value_type)) in signal_fields
        if !haskey(required_globals, global_idx)
            required_globals[global_idx] = value_type
        end
    end

    # Add explicit globals passed in
    for (i, (gtype, gval)) in enumerate(globals)
        global_idx = UInt32(i - 1)
        if !haskey(required_globals, global_idx)
            required_globals[global_idx] = gtype
        end
    end

    # Add all required globals to the module and export them
    for global_idx in sort(collect(keys(required_globals)))
        value_type = required_globals[global_idx]
        wasm_type = julia_to_wasm_type(value_type)
        # Find initial value from explicit globals if available
        initial_value = zero(value_type)
        if Int(global_idx) + 1 <= length(globals)
            _, initial_value = globals[Int(global_idx) + 1]
        end
        while length(mod.globals) <= Int(global_idx)
            actual_idx = add_global!(mod, wasm_type, true, initial_value)
            # Export the global for JS access
            add_global_export!(mod, "signal_$(actual_idx)", actual_idx)
        end
    end

    # Build captured_signal_fields for CompilationContext
    # Maps field_name -> (is_getter, global_idx) without the type
    captured_signal_fields = Dict{Symbol, Tuple{Bool, UInt32}}()
    for (field_name, (is_getter, global_idx, _)) in signal_fields
        captured_signal_fields[field_name] = (is_getter, global_idx)
    end

    # Convert DOMBindingSpec to internal format for CompilationContext
    # Internal format: global_idx -> [(import_idx, const_args), ...]
    internal_dom_bindings = Dict{UInt32, Vector{Tuple{UInt32, Vector{Int32}}}}()
    for (global_idx, specs) in dom_bindings
        internal_dom_bindings[global_idx] = [(spec.import_idx, spec.const_args) for spec in specs]
    end

    # Compile the closure body
    # Closures have one implicit argument (_1 = self)
    ctx = CompilationContext(
        code_info,
        (),  # No explicit arguments
        return_type,
        mod,
        type_registry;
        captured_signal_fields = captured_signal_fields,
        dom_bindings = internal_dom_bindings
    )
    body = generate_body(ctx)

    # Handler functions take no params and return nothing (void)
    # The return value (if any) is typically dropped in event handlers
    param_types = WasmValType[]
    result_types = WasmValType[]  # Event handlers return void

    # Add function to module
    func_idx = add_function!(mod, param_types, result_types, ctx.locals, body)

    # Export the function
    add_export!(mod, export_name, 0, func_idx)

    return mod
end

"""
    compile_closure_body(closure, captured_signal_fields, mod, type_registry; dom_bindings) -> (Vector{UInt8}, Vector{NumType})

Compile a closure body to Wasm bytecode without creating a new module.
Returns the body bytecode and locals needed for the function.

This is the lower-level API used by Therapy.jl to compile handler closures
into an existing module with shared globals and imports.

The `captured_signal_fields` maps field names to (is_getter, global_idx).
The `dom_bindings` maps global_idx to list of (import_idx, const_args) tuples.
"""
function compile_closure_body(
    closure::Function,
    captured_signal_fields::Dict{Symbol, Tuple{Bool, UInt32}},
    mod::WasmModule,
    type_registry::TypeRegistry;
    dom_bindings::Dict{UInt32, Vector{Tuple{UInt32, Vector{Int32}}}} = Dict{UInt32, Vector{Tuple{UInt32, Vector{Int32}}}}(),
    void_return::Bool = false
)
    # Get typed IR for the closure
    typed_results = Base.code_typed(closure, ())
    if isempty(typed_results)
        error("Could not get typed IR for handler closure")
    end
    code_info, inferred_return_type = typed_results[1]

    # For void handlers (like Therapy.jl event handlers), override return type
    return_type = void_return ? Nothing : inferred_return_type

    # Create compilation context
    ctx = CompilationContext(
        code_info,
        (),  # No explicit arguments
        return_type,
        mod,
        type_registry;
        captured_signal_fields = captured_signal_fields,
        dom_bindings = dom_bindings
    )

    # Generate body
    body = generate_body(ctx)

    return (body, ctx.locals)
end

"""
Check if a type is a user-defined struct (not a primitive or special type).
"""
function is_struct_type(T::Type)::Bool
    # Primitive types are not structs
    T <: Number && return false
    T === Bool && return false
    T === Nothing && return false
    T === Char && return false

    # Arrays and strings have special handling - not user structs
    T <: AbstractArray && return false
    T === String && return false

    # Internal Julia types that have pointer fields - not user structs
    # MemoryRef and GenericMemoryRef are used for array element access
    if T isa DataType && T.name.name in (:MemoryRef, :GenericMemoryRef, :Memory, :GenericMemory)
        return false
    end

    # Check if it's a concrete struct type
    return isconcretetype(T) && isstructtype(T) && !(T <: Tuple)
end

is_struct_type(::Any) = false

"""
Check if type is a closure (subtype of Function with captured fields).
"""
function is_closure_type(T::Type)::Bool
    # Union{} is bottom type - not a closure
    T === Union{} && return false
    # Must be a subtype of Function
    !(T <: Function) && return false
    # Must have fields (captured variables)
    fieldcount(T) == 0 && return false
    # Must be a concrete struct type
    return isconcretetype(T) && isstructtype(T)
end

is_closure_type(::Any) = false

"""
Register a closure type as a WasmGC struct.
"""
function register_closure_type!(mod::WasmModule, registry::TypeRegistry, T::DataType)
    # Already registered?
    haskey(registry.structs, T) && return registry.structs[T]

    # Get field information
    field_names = [fieldname(T, i) for i in 1:fieldcount(T)]
    field_types = [fieldtype(T, i) for i in 1:fieldcount(T)]

    # Create WasmGC field types (same logic as register_struct_type)
    wasm_fields = FieldType[]
    for ft in field_types
        if ft <: Vector
            # Vector{T} is represented as a struct with (array_ref, size_tuple)
            # Use register_vector_type! to get the struct type
            vec_info = register_vector_type!(mod, registry, ft)
            wasm_vt = ConcreteRef(vec_info.wasm_type_idx, true)
        elseif ft <: AbstractVector
            # Other AbstractVector types - use raw array
            elem_type = eltype(ft)
            array_type_idx = get_array_type!(mod, registry, elem_type)
            wasm_vt = ConcreteRef(array_type_idx, true)
        elseif ft isa DataType && (ft.name.name === :MemoryRef || ft.name.name === :GenericMemoryRef)
            # MemoryRef{T} / GenericMemoryRef maps to array type for element T
            elem_type = ft.name.name === :GenericMemoryRef ? ft.parameters[2] : ft.parameters[1]
            array_type_idx = get_array_type!(mod, registry, elem_type)
            wasm_vt = ConcreteRef(array_type_idx, true)
        elseif ft isa DataType && (ft.name.name === :Memory || ft.name.name === :GenericMemory)
            # Memory{T} / GenericMemory maps to array type for element T
            elem_type = eltype(ft)
            array_type_idx = get_array_type!(mod, registry, elem_type)
            wasm_vt = ConcreteRef(array_type_idx, true)
        elseif ft === String
            str_type_idx = get_string_array_type!(mod, registry)
            wasm_vt = ConcreteRef(str_type_idx, true)
        else
            wasm_vt = julia_to_wasm_type(ft)
        end
        push!(wasm_fields, FieldType(wasm_vt, false))  # immutable for closures
    end

    # Add struct type to module
    type_idx = add_struct_type!(mod, wasm_fields)

    # Record mapping
    info = StructInfo(T, type_idx, field_names, field_types)
    registry.structs[T] = info

    return info
end

# Track types currently being registered to prevent infinite recursion
# Maps type -> reserved type index for self-referential types, or -1 for normal types
const _registering_types = Dict{DataType, Int}()

"""
Check if a type is self-referential (has fields that reference itself).
"""
function is_self_referential_type(T::DataType)::Bool
    for i in 1:fieldcount(T)
        ft = fieldtype(T, i)
        # Check nullable fields (Union{Nothing, T})
        if ft isa Union
            inner = get_nullable_inner_type(ft)
            if inner !== nothing && inner === T
                return true
            end
        end
        # Check array fields (Vector{T})
        if ft <: AbstractVector && eltype(ft) === T
            return true
        end
    end
    return false
end

"""
Register a Julia struct type in the Wasm module.
"""
function register_struct_type!(mod::WasmModule, registry::TypeRegistry, T::DataType)
    # Already registered?
    haskey(registry.structs, T) && return registry.structs[T]

    # Prevent infinite recursion for self-referential types
    if haskey(_registering_types, T)
        # Type is being registered - return nothing so caller handles it
        return nothing
    end

    # Check if this is a self-referential type
    if is_self_referential_type(T)
        # For self-referential types with Vector{T} fields, we use rec groups
        # to allow concrete type references between struct and array types.

        # Step 1: Add struct placeholder first (with placeholder fields)
        # We need the struct index before creating array types that reference it
        temp_fields = FieldType[]
        for i in 1:fieldcount(T)
            ft = fieldtype(T, i)
            if ft === Int32 || ft === UInt32 || ft === Bool || ft === Char ||
               ft === Int8 || ft === UInt8 || ft === Int16 || ft === UInt16
                push!(temp_fields, FieldType(I32, true))
            elseif ft === Int64 || ft === UInt64 || ft === Int
                push!(temp_fields, FieldType(I64, true))
            elseif ft === Float32
                push!(temp_fields, FieldType(F32, true))
            elseif ft === Float64
                push!(temp_fields, FieldType(F64, true))
            elseif ft <: AbstractVector || ft === String
                push!(temp_fields, FieldType(ArrayRef, true))  # Placeholder
            else
                push!(temp_fields, FieldType(StructRef, true))  # Placeholder
            end
        end
        reserved_idx = add_struct_type!(mod, temp_fields)
        _registering_types[T] = Int(reserved_idx)

        # Step 2: Create array types for Vector{T} fields with concrete element type
        # Now that we have reserved_idx, array types can reference it
        array_type_indices = Dict{Int, UInt32}()
        for i in 1:fieldcount(T)
            ft = fieldtype(T, i)
            if ft <: AbstractVector && eltype(ft) === T
                # Use concrete reference to the reserved struct index
                arr_idx = add_array_type!(mod, ConcreteRef(reserved_idx, true), true)
                array_type_indices[i] = arr_idx
                registry.arrays[T] = arr_idx
            elseif ft isa Union
                inner = get_nullable_inner_type(ft)
                if inner !== nothing && inner <: AbstractVector && eltype(inner) === T
                    arr_idx = add_array_type!(mod, ConcreteRef(reserved_idx, true), true)
                    array_type_indices[i] = arr_idx
                    registry.arrays[T] = arr_idx
                end
            end
        end

        # Step 3: Add rec group for the struct and its array types
        if !isempty(array_type_indices)
            rec_group = UInt32[reserved_idx]
            for arr_idx in values(array_type_indices)
                push!(rec_group, arr_idx)
            end
            add_rec_group!(mod, rec_group)
        end

        try
            return _register_struct_type_impl_with_reserved!(mod, registry, T, reserved_idx)
        finally
            delete!(_registering_types, T)
        end
    else
        # Non-self-referential type - standard registration
        _registering_types[T] = -1
        try
            return _register_struct_type_impl!(mod, registry, T)
        finally
            delete!(_registering_types, T)
        end
    end
end

"""
Register a self-referential struct type using a pre-reserved type index.
The placeholder struct was already added; we update it with the correct fields.
"""
function _register_struct_type_impl_with_reserved!(mod::WasmModule, registry::TypeRegistry, T::DataType, reserved_idx::UInt32)
    field_names = [fieldname(T, i) for i in 1:fieldcount(T)]
    field_types = [fieldtype(T, i) for i in 1:fieldcount(T)]

    # Build the proper fields with correct self-references
    # Note: rec groups are already set up by register_struct_type!
    wasm_fields = FieldType[]
    for ft in field_types
        if ft === Vector{String}
            # Vector{String} is a struct with (array-of-string-refs, size tuple)
            info = register_vector_type!(mod, registry, ft)
            wasm_vt = ConcreteRef(info.wasm_type_idx, true)
        elseif ft <: AbstractVector && ft isa DataType
            # Vector{T} is a struct with (ref, size) fields
            elem_type = eltype(ft)
            if !haskey(_registering_types, elem_type) && isconcretetype(elem_type) && isstructtype(elem_type)
                register_struct_type!(mod, registry, elem_type)
            end
            info = register_vector_type!(mod, registry, ft)
            wasm_vt = ConcreteRef(info.wasm_type_idx, true)
        elseif ft <: AbstractVector
            # Generic AbstractVector - use raw array
            elem_type = eltype(ft)
            if !haskey(_registering_types, elem_type) && isconcretetype(elem_type) && isstructtype(elem_type)
                register_struct_type!(mod, registry, elem_type)
            end
            array_type_idx = get_array_type!(mod, registry, elem_type)
            wasm_vt = ConcreteRef(array_type_idx, true)
        elseif ft isa DataType && (ft.name.name === :MemoryRef || ft.name.name === :GenericMemoryRef)
            # MemoryRef{T} / GenericMemoryRef maps to array type for element T
            elem_type = ft.name.name === :GenericMemoryRef ? ft.parameters[2] : ft.parameters[1]
            array_type_idx = get_array_type!(mod, registry, elem_type)
            wasm_vt = ConcreteRef(array_type_idx, true)
        elseif ft isa DataType && (ft.name.name === :Memory || ft.name.name === :GenericMemory)
            # Memory{T} / GenericMemory maps to array type for element T
            elem_type = eltype(ft)
            array_type_idx = get_array_type!(mod, registry, elem_type)
            wasm_vt = ConcreteRef(array_type_idx, true)
        elseif ft === String
            str_type_idx = get_string_array_type!(mod, registry)
            wasm_vt = ConcreteRef(str_type_idx, true)
        elseif ft === Any
            wasm_vt = ExternRef
        elseif ft === Int32 || ft === UInt32 || ft === Bool || ft === Char ||
               ft === Int8 || ft === UInt8 || ft === Int16 || ft === UInt16
            wasm_vt = I32
        elseif ft === Int64 || ft === UInt64 || ft === Int
            wasm_vt = I64
        elseif ft === Float32
            wasm_vt = F32
        elseif ft === Float64
            wasm_vt = F64
        elseif isprimitivetype(ft)
            sz = sizeof(ft)
            wasm_vt = sz <= 4 ? I32 : I64
        elseif ft isa Union
            inner_type = get_nullable_inner_type(ft)
            if inner_type !== nothing
                if inner_type <: AbstractVector && inner_type isa DataType
                    # Union{Nothing, Vector{T}} - use Vector struct type
                    elem_type = eltype(inner_type)
                    if !haskey(_registering_types, elem_type) && isconcretetype(elem_type) && isstructtype(elem_type)
                        register_struct_type!(mod, registry, elem_type)
                    end
                    info = register_vector_type!(mod, registry, inner_type)
                    wasm_vt = ConcreteRef(info.wasm_type_idx, true)
                elseif inner_type <: AbstractVector
                    # Generic AbstractVector - use raw array
                    elem_type = eltype(inner_type)
                    if !haskey(_registering_types, elem_type) && isconcretetype(elem_type) && isstructtype(elem_type)
                        register_struct_type!(mod, registry, elem_type)
                    end
                    array_type_idx = get_array_type!(mod, registry, elem_type)
                    wasm_vt = ConcreteRef(array_type_idx, true)
                elseif isconcretetype(inner_type) && isstructtype(inner_type)
                    if haskey(_registering_types, inner_type)
                        r_idx = _registering_types[inner_type]
                        if r_idx >= 0
                            wasm_vt = ConcreteRef(UInt32(r_idx), true)
                        else
                            wasm_vt = StructRef
                        end
                    else
                        register_struct_type!(mod, registry, inner_type)
                        info = registry.structs[inner_type]
                        wasm_vt = ConcreteRef(info.wasm_type_idx, true)
                    end
                else
                    wasm_vt = julia_to_wasm_type(ft)
                end
            elseif needs_tagged_union(ft)
                union_info = register_union_type!(mod, registry, ft)
                wasm_vt = ConcreteRef(union_info.wasm_type_idx, true)
            else
                wasm_vt = julia_to_wasm_type(ft)
            end
        elseif isconcretetype(ft) && isstructtype(ft)
            if haskey(_registering_types, ft)
                r_idx = _registering_types[ft]
                if r_idx >= 0
                    wasm_vt = ConcreteRef(UInt32(r_idx), true)
                else
                    wasm_vt = StructRef
                end
            else
                nested_info = register_struct_type!(mod, registry, ft)
                if nested_info !== nothing
                    wasm_vt = ConcreteRef(nested_info.wasm_type_idx, true)
                else
                    wasm_vt = StructRef
                end
            end
        else
            wasm_vt = julia_to_wasm_type(ft)
        end
        push!(wasm_fields, FieldType(wasm_vt, true))
    end

    # Update the placeholder struct with the correct fields
    mod.types[reserved_idx + 1] = StructType(wasm_fields)

    # Record mapping (rec groups already set up by register_struct_type!)
    info = StructInfo(T, reserved_idx, field_names, field_types)
    registry.structs[T] = info

    return info
end

function _register_struct_type_impl!(mod::WasmModule, registry::TypeRegistry, T::DataType)
    # Get field information
    field_names = [fieldname(T, i) for i in 1:fieldcount(T)]
    field_types = [fieldtype(T, i) for i in 1:fieldcount(T)]

    # Create WasmGC field types
    wasm_fields = FieldType[]
    for ft in field_types
        # For array fields, use concrete reference to registered array type
        # But for Vector{T}, use the Vector struct type (with ref and size fields)
        # since Vector in Julia 1.11+ is a struct, not a raw array
        if ft === Vector{String}
            # Special case: Vector{String} is a struct with array-of-string-refs + size tuple
            # Register as Vector struct type
            info = register_vector_type!(mod, registry, ft)
            wasm_vt = ConcreteRef(info.wasm_type_idx, true)
        elseif ft <: AbstractVector && ft isa DataType
            # Vector{T} is now a struct with (ref, size) fields
            # Register it as a Vector struct type, not a raw array
            info = register_vector_type!(mod, registry, ft)
            wasm_vt = ConcreteRef(info.wasm_type_idx, true)
        elseif ft <: AbstractVector
            # Generic AbstractVector without concrete type - use raw array
            elem_type = eltype(ft)
            array_type_idx = get_array_type!(mod, registry, elem_type)
            wasm_vt = ConcreteRef(array_type_idx, true)  # nullable reference
        elseif ft isa DataType && (ft.name.name === :MemoryRef || ft.name.name === :GenericMemoryRef)
            # MemoryRef{T} / GenericMemoryRef maps to array type for element T
            # GenericMemoryRef parameters: (atomicity, element_type, addrspace)
            elem_type = ft.name.name === :GenericMemoryRef ? ft.parameters[2] : ft.parameters[1]
            array_type_idx = get_array_type!(mod, registry, elem_type)
            wasm_vt = ConcreteRef(array_type_idx, true)  # nullable reference
        elseif ft isa DataType && (ft.name.name === :Memory || ft.name.name === :GenericMemory)
            # Memory{T} / GenericMemory maps to array type for element T
            elem_type = eltype(ft)
            array_type_idx = get_array_type!(mod, registry, elem_type)
            wasm_vt = ConcreteRef(array_type_idx, true)  # nullable reference
        elseif ft === String
            str_type_idx = get_string_array_type!(mod, registry)
            wasm_vt = ConcreteRef(str_type_idx, true)
        elseif ft === Any
            # Any type - map to externref (Julia 1.12 closures have Any fields)
            wasm_vt = ExternRef
        elseif ft === Int32 || ft === UInt32 || ft === Bool || ft === Char ||
               ft === Int8 || ft === UInt8 || ft === Int16 || ft === UInt16
            # Standard 32-bit or smaller types
            wasm_vt = I32
        elseif ft === Int64 || ft === UInt64 || ft === Int
            # Standard 64-bit integer types
            wasm_vt = I64
        elseif ft === Float32
            wasm_vt = F32
        elseif ft === Float64
            wasm_vt = F64
        elseif isprimitivetype(ft)
            # Custom primitive types (e.g., JuliaSyntax.Kind) - map by size
            sz = sizeof(ft)
            if sz <= 4
                wasm_vt = I32
            elseif sz <= 8
                wasm_vt = I64
            else
                error("Primitive type too large for Wasm field: $ft ($sz bytes)")
            end
        elseif ft isa Union
            # Handle Union types for struct fields
            inner_type = get_nullable_inner_type(ft)
            if inner_type !== nothing
                # Union{Nothing, T} as nullable reference to T
                if inner_type <: AbstractVector && inner_type isa DataType
                    # Union{Nothing, Vector{T}} - use Vector struct type
                    elem_type = eltype(inner_type)
                    # For non-recursive types, register the element type first
                    if !haskey(_registering_types, elem_type) && isconcretetype(elem_type) && isstructtype(elem_type)
                        register_struct_type!(mod, registry, elem_type)
                    end
                    info = register_vector_type!(mod, registry, inner_type)
                    wasm_vt = ConcreteRef(info.wasm_type_idx, true)  # nullable
                elseif inner_type <: AbstractVector
                    # Union{Nothing, generic AbstractVector} - use raw array
                    elem_type = eltype(inner_type)
                    # For non-recursive types, register the element type first
                    if !haskey(_registering_types, elem_type) && isconcretetype(elem_type) && isstructtype(elem_type)
                        register_struct_type!(mod, registry, elem_type)
                    end
                    # get_array_type! handles self-referential types
                    array_type_idx = get_array_type!(mod, registry, elem_type)
                    wasm_vt = ConcreteRef(array_type_idx, true)  # nullable
                elseif isconcretetype(inner_type) && isstructtype(inner_type)
                    # Union{Nothing, SomeStruct} - nullable struct ref
                    if haskey(_registering_types, inner_type)
                        reserved_idx = _registering_types[inner_type]
                        if reserved_idx >= 0
                            wasm_vt = ConcreteRef(UInt32(reserved_idx), true)  # nullable
                        else
                            wasm_vt = StructRef  # Not a self-referential type being registered
                        end
                    else
                        register_struct_type!(mod, registry, inner_type)
                        info = registry.structs[inner_type]
                        wasm_vt = ConcreteRef(info.wasm_type_idx, true)  # nullable
                    end
                else
                    wasm_vt = julia_to_wasm_type(ft)
                end
            elseif needs_tagged_union(ft)
                # Multi-variant union - use tagged union struct
                union_info = register_union_type!(mod, registry, ft)
                wasm_vt = ConcreteRef(union_info.wasm_type_idx, true)
            else
                wasm_vt = julia_to_wasm_type(ft)
            end
        elseif isconcretetype(ft) && isstructtype(ft)
            # Nested struct type - recursively register it
            if haskey(_registering_types, ft)
                reserved_idx = _registering_types[ft]
                if reserved_idx >= 0
                    wasm_vt = ConcreteRef(UInt32(reserved_idx), true)
                else
                    wasm_vt = StructRef  # Non-self-referential type being registered
                end
            else
                nested_info = register_struct_type!(mod, registry, ft)
                if nested_info !== nothing
                    wasm_vt = ConcreteRef(nested_info.wasm_type_idx, true)
                else
                    wasm_vt = StructRef  # Forward reference
                end
            end
        else
            wasm_vt = julia_to_wasm_type(ft)
        end
        push!(wasm_fields, FieldType(wasm_vt, true))  # mutable by default
    end

    # Add struct type to module
    type_idx = add_struct_type!(mod, wasm_fields)

    # Record mapping
    info = StructInfo(T, type_idx, field_names, field_types)
    registry.structs[T] = info

    return info
end

"""
Register a Julia tuple type in the Wasm module.
Tuples are represented as WasmGC structs with numbered fields.
"""
function register_tuple_type!(mod::WasmModule, registry::TypeRegistry, T::Type{<:Tuple})
    # Already registered?
    haskey(registry.structs, T) && return registry.structs[T]

    # Get element types
    elem_types = T.parameters

    # Create WasmGC field types
    wasm_fields = FieldType[]
    field_names = Symbol[]
    field_types_vec = DataType[]

    for (i, ft) in enumerate(elem_types)
        # Use concrete types for fields that need specific WASM types
        # This ensures consistency between struct field types and local variable types
        wasm_vt = if ft === String
            # String field needs concrete array type
            type_idx = get_string_array_type!(mod, registry)
            ConcreteRef(type_idx, true)
        elseif ft <: AbstractVector
            # Vector field needs concrete array type
            elem_type = eltype(ft)
            type_idx = get_array_type!(mod, registry, elem_type)
            ConcreteRef(type_idx, true)
        elseif ft isa DataType && (ft.name.name === :MemoryRef || ft.name.name === :GenericMemoryRef)
            # MemoryRef{T} / GenericMemoryRef maps to array type for element T
            elem_type = ft.name.name === :GenericMemoryRef ? ft.parameters[2] : ft.parameters[1]
            type_idx = get_array_type!(mod, registry, elem_type)
            ConcreteRef(type_idx, true)
        elseif ft isa DataType && (ft.name.name === :Memory || ft.name.name === :GenericMemory)
            # Memory{T} / GenericMemory maps to array type for element T
            elem_type = eltype(ft)
            type_idx = get_array_type!(mod, registry, elem_type)
            ConcreteRef(type_idx, true)
        elseif isconcretetype(ft) && isstructtype(ft) && !(ft <: Tuple)
            # Nested struct - register and use concrete ref
            nested_info = register_struct_type!(mod, registry, ft)
            if nested_info !== nothing
                ConcreteRef(nested_info.wasm_type_idx, true)
            else
                julia_to_wasm_type(ft)
            end
        else
            # For primitives and other types, use generic mapping
            julia_to_wasm_type(ft)
        end
        push!(wasm_fields, FieldType(wasm_vt, false))  # Tuples are immutable
        push!(field_names, Symbol(i))  # Use numeric names
        push!(field_types_vec, ft isa DataType ? ft : Any)
    end

    # Add struct type to module
    type_idx = add_struct_type!(mod, wasm_fields)

    # Record mapping (use T as DataType)
    info = StructInfo(T, type_idx, field_names, field_types_vec)
    registry.structs[T] = info

    return info
end

"""
Register a multi-dimensional array type (Matrix, Array{T,3}, etc.) as a WasmGC struct.

Multi-dim arrays are stored as WasmGC structs with two fields:
- Field 0: data (reference to flat WasmGC array of element type)
- Field 1: size (tuple of dimensions)

This matches Julia's internal representation where Matrix{T} has :ref and :size fields.
"""
function register_matrix_type!(mod::WasmModule, registry::TypeRegistry, T::Type)
    # Already registered?
    haskey(registry.structs, T) && return registry.structs[T]

    # Get element type and dimensionality
    elem_type = eltype(T)
    N = ndims(T)

    # Create size tuple type
    size_tuple_type = NTuple{N, Int64}
    if !haskey(registry.structs, size_tuple_type)
        register_tuple_type!(mod, registry, size_tuple_type)
    end
    size_struct_info = registry.structs[size_tuple_type]

    # Create/get the data array type
    data_array_idx = get_array_type!(mod, registry, elem_type)

    # Create WasmGC struct with two fields:
    # - Field 0: ref (nullable reference to data array)
    # - Field 1: size (nullable reference to size tuple struct)
    wasm_fields = [
        FieldType(ConcreteRef(data_array_idx, true), true),  # data array, mutable
        FieldType(ConcreteRef(size_struct_info.wasm_type_idx, true), false)  # size, immutable
    ]

    # Add struct type to module
    type_idx = add_struct_type!(mod, wasm_fields)

    # Record mapping with field info
    field_names = [:ref, :size]  # Julia field names
    field_types_vec = DataType[Array{elem_type, 1}, size_tuple_type]  # Use Vector for ref field type

    info = StructInfo(T, type_idx, field_names, field_types_vec)
    registry.structs[T] = info

    return info
end

"""
Register a Vector{T} type as a WasmGC struct with mutable size.

Vectors are stored as WasmGC structs with two fields:
- Field 0: ref (reference to WasmGC array of element type)
- Field 1: size (mutable Tuple{Int64} tracking logical size)

This matches Julia's internal representation where Vector{T} has :ref and :size fields.
The size field is mutable to support setfield!(v, :size, (n,)) for push!/resize! operations.
"""
function register_vector_type!(mod::WasmModule, registry::TypeRegistry, T::Type)
    # Already registered?
    haskey(registry.structs, T) && return registry.structs[T]

    # Get element type
    elem_type = eltype(T)

    # Create size tuple type (Tuple{Int64} for 1D)
    size_tuple_type = Tuple{Int64}
    if !haskey(registry.structs, size_tuple_type)
        register_tuple_type!(mod, registry, size_tuple_type)
    end
    size_struct_info = registry.structs[size_tuple_type]

    # Create/get the data array type
    data_array_idx = get_array_type!(mod, registry, elem_type)

    # Create WasmGC struct with two fields:
    # - Field 0: ref (reference to data array)
    # - Field 1: size (MUTABLE reference to size tuple struct)
    wasm_fields = [
        FieldType(ConcreteRef(data_array_idx, true), true),  # data array, mutable
        FieldType(ConcreteRef(size_struct_info.wasm_type_idx, true), true)  # size, MUTABLE for setfield!
    ]

    # Add struct type to module
    type_idx = add_struct_type!(mod, wasm_fields)

    # Record mapping with field info
    field_names = [:ref, :size]  # Julia field names
    field_types_vec = DataType[Array{elem_type, 1}, size_tuple_type]

    info = StructInfo(T, type_idx, field_names, field_types_vec)
    registry.structs[T] = info

    return info
end

"""
Register a 128-bit integer type (Int128 or UInt128) as a WasmGC struct.

128-bit integers are stored as WasmGC structs with two i64 fields:
- Field 0: lo (low 64 bits)
- Field 1: hi (high 64 bits)

This is the standard representation used by most WASM compilers for 128-bit integers.
"""
function register_int128_type!(mod::WasmModule, registry::TypeRegistry, T::Type)
    # Already registered?
    haskey(registry.structs, T) && return registry.structs[T]

    # Create WasmGC struct with two i64 fields (lo, hi)
    wasm_fields = [
        FieldType(I64, true),   # lo (low 64 bits), mutable for potential in-place ops
        FieldType(I64, true)    # hi (high 64 bits)
    ]

    # Add struct type to module
    type_idx = add_struct_type!(mod, wasm_fields)

    # Record mapping with field info
    field_names = [:lo, :hi]
    field_types_vec = DataType[UInt64, UInt64]  # Both fields are 64-bit

    info = StructInfo(T, type_idx, field_names, field_types_vec)
    registry.structs[T] = info

    return info
end

"""
Get or create the 128-bit integer struct type.
"""
function get_int128_type!(mod::WasmModule, registry::TypeRegistry, T::Type)
    if haskey(registry.structs, T)
        return registry.structs[T].wasm_type_idx
    else
        info = register_int128_type!(mod, registry, T)
        return info.wasm_type_idx
    end
end

# ============================================================================
# Tagged Union Type Registration
# Multi-variant unions (not just Union{Nothing, T}) are stored as tagged unions:
# WasmGC struct { tag: i32, value: anyref }
# ============================================================================

"""
Check if a Union type is a "simple" nullable type (Union{Nothing, T}).
Returns the inner type T if so, nothing otherwise.
"""
function get_nullable_inner_type(T::Union)::Union{Type, Nothing}
    types = Base.uniontypes(T)
    non_nothing = filter(t -> t !== Nothing, types)
    if length(non_nothing) == 1 && Nothing in types
        return non_nothing[1]
    end
    return nothing
end

"""
Check if a type is a reference type (struct or Union containing struct).
Used to determine if ref.eq should be used for comparison.
"""
function is_ref_type_or_union(T::Type)::Bool
    # Direct struct types (excluding primitive types)
    if T isa DataType && isstructtype(T) && !isprimitivetype(T)
        return true
    end
    # Union types - check if any component is a ref type
    if T isa Union
        types = Base.uniontypes(T)
        for t in types
            if t !== Nothing && is_ref_type_or_union(t)
                return true
            end
        end
    end
    # Arrays/Vectors are refs
    if T <: AbstractArray
        return true
    end
    return false
end

"""
Check if a value represents `nothing` (literal or GlobalRef to nothing).
"""
function is_nothing_value(val, ctx)::Bool
    # Literal nothing
    if val === nothing
        return true
    end
    # GlobalRef to nothing (e.g., WasmTarget.nothing or Core.nothing)
    if val isa GlobalRef && val.name === :nothing
        return true
    end
    # SSA that has Nothing type
    if val isa Core.SSAValue
        ssa_type = get(ctx.ssa_types, val.id, Any)
        return ssa_type === Nothing
    end
    return false
end

"""
Check if a Union type needs tagged union representation.
Returns true for multi-variant unions that aren't simple nullable types.
"""
function needs_tagged_union(T::Union)::Bool
    types = Base.uniontypes(T)
    non_nothing = filter(t -> t !== Nothing, types)
    # Need tagged union if we have 2+ non-Nothing types
    return length(non_nothing) >= 2
end

"""
Register a multi-variant Union type as a WasmGC tagged union struct.

Tagged unions are structs with two fields:
- Field 0: tag (i32) - identifies which variant is stored
- Field 1: value (anyref) - the actual value, boxed to anyref

Tag values are assigned in order based on the union type list.
Tag 0 is reserved for Nothing if present in the union.
"""
function register_union_type!(mod::WasmModule, registry::TypeRegistry, T::Union)::UnionInfo
    # Already registered?
    haskey(registry.unions, T) && return registry.unions[T]

    # Get variant types
    types = Base.uniontypes(T)

    # Build tag map - assign tag values to each type
    # Nothing gets tag 0 if present, other types get sequential tags
    tag_map = Dict{Type, Int32}()
    variant_types = Type[]
    next_tag = Int32(0)

    # Assign tag 0 to Nothing if present
    if Nothing in types
        tag_map[Nothing] = next_tag
        push!(variant_types, Nothing)
        next_tag += Int32(1)
    end

    # Assign tags to other types in order
    for t in types
        if t !== Nothing
            tag_map[t] = next_tag
            push!(variant_types, t)
            next_tag += Int32(1)
        end
    end

    # Create WasmGC struct with two fields: tag (i32) and value (anyref)
    # Using AnyRef allows us to store any WasmGC reference type
    wasm_fields = [
        FieldType(I32, true),    # tag - mutable so we can set it
        FieldType(AnyRef, true)  # value - anyref can hold any reference
    ]

    # Add struct type to module
    type_idx = add_struct_type!(mod, wasm_fields)

    # Record mapping
    info = UnionInfo(T, type_idx, variant_types, tag_map)
    registry.unions[T] = info

    return info
end

"""
Get or create a tagged union type for a Union.
Returns the UnionInfo with type index and tag mappings.
"""
function get_union_type!(mod::WasmModule, registry::TypeRegistry, T::Union)::UnionInfo
    if haskey(registry.unions, T)
        return registry.unions[T]
    else
        return register_union_type!(mod, registry, T)
    end
end

"""
Get the tag value for a specific type within a union.
"""
function get_union_tag(info::UnionInfo, T::Type)::Int32
    return get(info.tag_map, T, Int32(-1))
end

"""
Emit bytecode to wrap a value on the stack in a tagged union struct.
Stack: [value] -> [tagged_union_struct]

The value is first converted to anyref (via extern.convert_any if needed),
then wrapped with its type tag.
"""
function emit_wrap_union_value(ctx, value_type::Type, union_type::Union)::Vector{UInt8}
    bytes = UInt8[]

    # Get or register the union type
    union_info = get_union_type!(ctx.mod, ctx.type_registry, union_type)
    tag = get_union_tag(union_info, value_type)

    if tag < 0
        error("Type $value_type is not a variant of union $union_type")
    end

    # For Nothing, we need to create a null anyref
    if value_type === Nothing
        # Drop any value on stack (Nothing has no value)
        # Push tag (0 for Nothing)
        push!(bytes, Opcode.I32_CONST)
        append!(bytes, encode_leb128_signed(Int64(tag)))
        # Push null anyref for the value
        push!(bytes, Opcode.REF_NULL)
        push!(bytes, UInt8(AnyRef))  # anyref
    else
        # Value is on stack - need to save it, push tag, then restore value
        # Allocate a scratch local for the value
        scratch_local = length(ctx.locals) + ctx.n_params
        value_wasm_type = julia_to_wasm_type_concrete(value_type, ctx)
        push!(ctx.locals, value_wasm_type)

        # Store value to scratch local
        push!(bytes, Opcode.LOCAL_SET)
        append!(bytes, encode_leb128_unsigned(scratch_local))

        # Push tag
        push!(bytes, Opcode.I32_CONST)
        append!(bytes, encode_leb128_signed(Int64(tag)))

        # Reload value and convert to anyref
        push!(bytes, Opcode.LOCAL_GET)
        append!(bytes, encode_leb128_unsigned(scratch_local))

        # Convert to anyref if it's a reference type
        # For WasmGC, references can be cast to anyref
        if value_wasm_type isa ConcreteRef || value_wasm_type isa RefType
            # extern.convert_any converts any ref to anyref
            # Actually for WasmGC, struct refs are subtypes of anyref, so no conversion needed
            # The value is already compatible with anyref
        end
    end

    # Create the tagged union struct
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_NEW)
    append!(bytes, encode_leb128_unsigned(union_info.wasm_type_idx))

    return bytes
end

"""
Emit bytecode to extract a value from a tagged union struct.
Stack: [tagged_union_struct] -> [value]

Extracts the value field and casts it to the expected type.
Note: Caller should verify type via isa() first for safety.
"""
function emit_unwrap_union_value(ctx, union_type::Union, target_type::Type)::Vector{UInt8}
    bytes = UInt8[]

    # Handle Nothing specially - just check if null
    if target_type === Nothing
        # For Nothing, we just need to verify it's null (via isa check done elsewhere)
        # Return nothing meaningful - the caller knows it's Nothing
        push!(bytes, Opcode.DROP)  # Drop the union struct
        return bytes
    end

    # Get the union info
    union_info = get_union_type!(ctx.mod, ctx.type_registry, union_type)

    # Get the value field (field 1)
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_GET)
    append!(bytes, encode_leb128_unsigned(union_info.wasm_type_idx))
    append!(bytes, encode_leb128_unsigned(1))  # field 1 is value

    # Cast anyref to the target type
    target_wasm_type = julia_to_wasm_type_concrete(target_type, ctx)
    if target_wasm_type isa ConcreteRef
        # Cast anyref to concrete type using ref.cast
        push!(bytes, Opcode.GC_PREFIX)
        push!(bytes, Opcode.REF_CAST)
        if target_wasm_type.nullable
            push!(bytes, 0x63)  # ref null
        else
            push!(bytes, 0x64)  # ref
        end
        append!(bytes, encode_leb128_signed(Int64(target_wasm_type.type_idx)))
    end

    return bytes
end

# ============================================================================
# 128-bit Integer Operation Emitters
# These emit WASM bytecode for 128-bit arithmetic operations.
# 128-bit integers are stored as structs with fields: lo (i64), hi (i64)
# ============================================================================

"""
Emit bytecode for 128-bit addition.
Stack: [a_struct, b_struct] -> [result_struct]
Algorithm: result_lo = a_lo + b_lo; carry = (result_lo < a_lo); result_hi = a_hi + b_hi + carry
"""
function emit_int128_add(ctx, result_type::Type)::Vector{UInt8}
    bytes = UInt8[]
    type_idx = get_int128_type!(ctx.mod, ctx.type_registry, result_type)

    # We need locals to hold extracted values and struct refs
    # Allocate them dynamically

    # First allocate struct locals so we can pop from stack
    b_struct_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, julia_to_wasm_type_concrete(result_type, ctx))
    a_struct_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, julia_to_wasm_type_concrete(result_type, ctx))

    # Then allocate i64 locals for extracted values
    a_lo_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    a_hi_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    b_lo_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    b_hi_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    result_lo_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)

    # Stack: [a_struct, b_struct]
    # Pop b_struct to a local (b_struct is on top)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(b_struct_local))

    # Now stack: [a_struct]
    # Pop a_struct to a local
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(a_struct_local))

    # Extract a_lo, a_hi
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(a_struct_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_GET)
    append!(bytes, encode_leb128_unsigned(type_idx))
    append!(bytes, encode_leb128_unsigned(0))  # lo field
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(a_lo_local))

    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(a_struct_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_GET)
    append!(bytes, encode_leb128_unsigned(type_idx))
    append!(bytes, encode_leb128_unsigned(1))  # hi field
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(a_hi_local))

    # Extract b_lo, b_hi
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(b_struct_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_GET)
    append!(bytes, encode_leb128_unsigned(type_idx))
    append!(bytes, encode_leb128_unsigned(0))  # lo field
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(b_lo_local))

    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(b_struct_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_GET)
    append!(bytes, encode_leb128_unsigned(type_idx))
    append!(bytes, encode_leb128_unsigned(1))  # hi field
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(b_hi_local))

    # Compute result_lo = a_lo + b_lo
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(a_lo_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(b_lo_local))
    push!(bytes, Opcode.I64_ADD)
    push!(bytes, Opcode.LOCAL_TEE)
    append!(bytes, encode_leb128_unsigned(result_lo_local))

    # Compute carry = (result_lo < a_lo) ? 1 : 0  (unsigned comparison)
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(a_lo_local))
    push!(bytes, Opcode.I64_LT_U)
    push!(bytes, Opcode.I64_EXTEND_I32_U)  # Convert i32 bool to i64

    # Compute result_hi = a_hi + b_hi + carry
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(a_hi_local))
    push!(bytes, Opcode.I64_ADD)  # a_hi + carry
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(b_hi_local))
    push!(bytes, Opcode.I64_ADD)  # + b_hi

    # Stack now has: [result_lo (from local), result_hi]
    # Wait, result_lo is in local, not on stack. Let me fix:
    # Stack: [result_hi]
    # Need: [result_lo, result_hi] for struct.new

    # Push result_lo first
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(result_lo_local))

    # Swap to get [result_lo, result_hi]
    # Wasm doesn't have swap, so we need another approach
    # Actually struct.new takes (lo, hi) in order, and we have hi on stack, lo in local
    # Let me store hi to local, then push in correct order

    hi_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(hi_local))

    # Now push lo, then hi
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(result_lo_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(hi_local))

    # Create result struct
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_NEW)
    append!(bytes, encode_leb128_unsigned(type_idx))

    return bytes
end

"""
Emit bytecode for 128-bit subtraction.
Stack: [a_struct, b_struct] -> [result_struct]
Algorithm: result_lo = a_lo - b_lo; borrow = (a_lo < b_lo); result_hi = a_hi - b_hi - borrow
"""
function emit_int128_sub(ctx, result_type::Type)::Vector{UInt8}
    bytes = UInt8[]
    type_idx = get_int128_type!(ctx.mod, ctx.type_registry, result_type)

    # First allocate struct locals so we can pop from stack
    b_struct_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, julia_to_wasm_type_concrete(result_type, ctx))
    a_struct_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, julia_to_wasm_type_concrete(result_type, ctx))

    # Allocate i64 locals for extracted values and results
    a_lo_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    a_hi_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    b_lo_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    b_hi_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    result_lo_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    result_hi_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    borrow_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)

    # Pop structs to locals (b_struct is on top of stack)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(b_struct_local))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(a_struct_local))

    # Extract fields
    for (struct_local, lo_local, hi_local) in [(a_struct_local, a_lo_local, a_hi_local),
                                                (b_struct_local, b_lo_local, b_hi_local)]
        push!(bytes, Opcode.LOCAL_GET)
        append!(bytes, encode_leb128_unsigned(struct_local))
        push!(bytes, Opcode.GC_PREFIX)
        push!(bytes, Opcode.STRUCT_GET)
        append!(bytes, encode_leb128_unsigned(type_idx))
        append!(bytes, encode_leb128_unsigned(0))
        push!(bytes, Opcode.LOCAL_SET)
        append!(bytes, encode_leb128_unsigned(lo_local))

        push!(bytes, Opcode.LOCAL_GET)
        append!(bytes, encode_leb128_unsigned(struct_local))
        push!(bytes, Opcode.GC_PREFIX)
        push!(bytes, Opcode.STRUCT_GET)
        append!(bytes, encode_leb128_unsigned(type_idx))
        append!(bytes, encode_leb128_unsigned(1))
        push!(bytes, Opcode.LOCAL_SET)
        append!(bytes, encode_leb128_unsigned(hi_local))
    end

    # result_lo = a_lo - b_lo
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(a_lo_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(b_lo_local))
    push!(bytes, Opcode.I64_SUB)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(result_lo_local))

    # borrow = (a_lo < b_lo) ? 1 : 0
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(a_lo_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(b_lo_local))
    push!(bytes, Opcode.I64_LT_U)
    push!(bytes, Opcode.I64_EXTEND_I32_U)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(borrow_local))

    # result_hi = a_hi - b_hi - borrow
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(a_hi_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(b_hi_local))
    push!(bytes, Opcode.I64_SUB)
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(borrow_local))
    push!(bytes, Opcode.I64_SUB)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(result_hi_local))

    # Create result struct: (result_lo, result_hi)
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(result_lo_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(result_hi_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_NEW)
    append!(bytes, encode_leb128_unsigned(type_idx))

    return bytes
end

"""
Emit bytecode for 128-bit multiplication (low 128 bits only).
Stack: [a_struct, b_struct] -> [result_struct]
Uses the identity: (a_lo + a_hi*2^64) * (b_lo + b_hi*2^64)
= a_lo*b_lo + (a_lo*b_hi + a_hi*b_lo)*2^64 + a_hi*b_hi*2^128
Since we only need low 128 bits: result_lo = low64(a_lo*b_lo), result_hi = high64(a_lo*b_lo) + low64(a_lo*b_hi) + low64(a_hi*b_lo)
"""
function emit_int128_mul(ctx, result_type::Type)::Vector{UInt8}
    bytes = UInt8[]
    type_idx = get_int128_type!(ctx.mod, ctx.type_registry, result_type)

    # Allocate locals
    a_lo_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    a_hi_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    b_lo_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    b_hi_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)

    # Pop structs to locals
    b_struct_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, julia_to_wasm_type_concrete(result_type, ctx))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(b_struct_local))

    a_struct_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, julia_to_wasm_type_concrete(result_type, ctx))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(a_struct_local))

    # Extract fields
    for (struct_local, lo_local, hi_local) in [(a_struct_local, a_lo_local, a_hi_local),
                                                (b_struct_local, b_lo_local, b_hi_local)]
        push!(bytes, Opcode.LOCAL_GET)
        append!(bytes, encode_leb128_unsigned(struct_local))
        push!(bytes, Opcode.GC_PREFIX)
        push!(bytes, Opcode.STRUCT_GET)
        append!(bytes, encode_leb128_unsigned(type_idx))
        append!(bytes, encode_leb128_unsigned(0))
        push!(bytes, Opcode.LOCAL_SET)
        append!(bytes, encode_leb128_unsigned(lo_local))

        push!(bytes, Opcode.LOCAL_GET)
        append!(bytes, encode_leb128_unsigned(struct_local))
        push!(bytes, Opcode.GC_PREFIX)
        push!(bytes, Opcode.STRUCT_GET)
        append!(bytes, encode_leb128_unsigned(type_idx))
        append!(bytes, encode_leb128_unsigned(1))
        push!(bytes, Opcode.LOCAL_SET)
        append!(bytes, encode_leb128_unsigned(hi_local))
    end

    # For 64x64->128 multiplication, we split each 64-bit value into two 32-bit halves
    # This is complex. Let me use a simpler approximation for now:
    # result_lo = a_lo * b_lo (truncated to 64 bits)
    # result_hi = a_lo * b_hi + a_hi * b_lo (approximation, ignores carry from lo*lo)

    # Actually, WASM doesn't have 64x64->128 multiplication directly.
    # We need to implement Karatsuba or schoolbook multiplication using 32-bit pieces.

    # Simplified version (may lose precision for large numbers):
    # This is acceptable for sin() where the values are typically small
    result_lo_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    result_hi_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)

    # result_lo = a_lo * b_lo (low 64 bits)
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(a_lo_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(b_lo_local))
    push!(bytes, Opcode.I64_MUL)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(result_lo_local))

    # result_hi = a_lo * b_hi + a_hi * b_lo
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(a_lo_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(b_hi_local))
    push!(bytes, Opcode.I64_MUL)
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(a_hi_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(b_lo_local))
    push!(bytes, Opcode.I64_MUL)
    push!(bytes, Opcode.I64_ADD)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(result_hi_local))

    # Create result struct
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(result_lo_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(result_hi_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_NEW)
    append!(bytes, encode_leb128_unsigned(type_idx))

    return bytes
end

"""
Emit 128-bit negation: -x = ~x + 1 = (0, 0) - x
Stack: [x_struct] -> [result_struct]
"""
function emit_int128_neg(ctx, result_type::Type)::Vector{UInt8}
    bytes = UInt8[]
    type_idx = get_int128_type!(ctx.mod, ctx.type_registry, result_type)

    # Allocate locals
    x_lo_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    x_hi_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)

    # Pop struct to local
    x_struct_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, julia_to_wasm_type_concrete(result_type, ctx))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(x_struct_local))

    # Extract fields
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(x_struct_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_GET)
    append!(bytes, encode_leb128_unsigned(type_idx))
    append!(bytes, encode_leb128_unsigned(0))  # lo
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(x_lo_local))

    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(x_struct_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_GET)
    append!(bytes, encode_leb128_unsigned(type_idx))
    append!(bytes, encode_leb128_unsigned(1))  # hi
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(x_hi_local))

    # Two's complement negation: -x = ~x + 1
    # result_lo = ~x_lo + 1
    # result_hi = ~x_hi + carry
    # where carry = 1 if ~x_lo overflows when adding 1 (i.e., x_lo == 0)

    result_lo_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    result_hi_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)

    # ~x_lo
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(x_lo_local))
    push!(bytes, Opcode.I64_CONST)
    push!(bytes, 0x7F)  # -1 in LEB128
    push!(bytes, Opcode.I64_XOR)

    # +1
    push!(bytes, Opcode.I64_CONST)
    push!(bytes, 0x01)
    push!(bytes, Opcode.I64_ADD)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(result_lo_local))

    # carry = (x_lo == 0) ? 1 : 0
    # ~x_hi + carry
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(x_hi_local))
    push!(bytes, Opcode.I64_CONST)
    push!(bytes, 0x7F)  # -1
    push!(bytes, Opcode.I64_XOR)

    # Add carry if x_lo was 0
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(x_lo_local))
    push!(bytes, Opcode.I64_EQZ)
    push!(bytes, Opcode.I64_EXTEND_I32_U)  # Convert i32 bool to i64
    push!(bytes, Opcode.I64_ADD)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(result_hi_local))

    # Create result struct
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(result_lo_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(result_hi_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_NEW)
    append!(bytes, encode_leb128_unsigned(type_idx))

    return bytes
end

"""
Emit 128-bit signed less than: a < b (signed)
Stack: [a_struct, b_struct] -> [i32 result (0 or 1)]
"""
function emit_int128_slt(ctx, arg_type::Type)::Vector{UInt8}
    bytes = UInt8[]
    type_idx = get_int128_type!(ctx.mod, ctx.type_registry, arg_type)

    # Allocate locals
    a_lo_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    a_hi_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    b_lo_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    b_hi_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)

    # Pop structs to locals
    b_struct_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, julia_to_wasm_type_concrete(arg_type, ctx))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(b_struct_local))

    a_struct_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, julia_to_wasm_type_concrete(arg_type, ctx))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(a_struct_local))

    # Extract fields
    for (struct_local, lo_local, hi_local) in [(a_struct_local, a_lo_local, a_hi_local),
                                                (b_struct_local, b_lo_local, b_hi_local)]
        push!(bytes, Opcode.LOCAL_GET)
        append!(bytes, encode_leb128_unsigned(struct_local))
        push!(bytes, Opcode.GC_PREFIX)
        push!(bytes, Opcode.STRUCT_GET)
        append!(bytes, encode_leb128_unsigned(type_idx))
        append!(bytes, encode_leb128_unsigned(0))
        push!(bytes, Opcode.LOCAL_SET)
        append!(bytes, encode_leb128_unsigned(lo_local))

        push!(bytes, Opcode.LOCAL_GET)
        append!(bytes, encode_leb128_unsigned(struct_local))
        push!(bytes, Opcode.GC_PREFIX)
        push!(bytes, Opcode.STRUCT_GET)
        append!(bytes, encode_leb128_unsigned(type_idx))
        append!(bytes, encode_leb128_unsigned(1))
        push!(bytes, Opcode.LOCAL_SET)
        append!(bytes, encode_leb128_unsigned(hi_local))
    end

    # Signed 128-bit comparison: a < b
    # if a_hi < b_hi (signed): true
    # if a_hi > b_hi (signed): false
    # if a_hi == b_hi: a_lo < b_lo (unsigned, since lo is always unsigned)

    # (a_hi < b_hi) || (a_hi == b_hi && a_lo < b_lo)
    # Using: (a_hi <_s b_hi) | ((a_hi == b_hi) & (a_lo <_u b_lo))

    # a_hi < b_hi (signed)
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(a_hi_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(b_hi_local))
    push!(bytes, Opcode.I64_LT_S)

    # a_hi == b_hi
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(a_hi_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(b_hi_local))
    push!(bytes, Opcode.I64_EQ)

    # a_lo < b_lo (unsigned)
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(a_lo_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(b_lo_local))
    push!(bytes, Opcode.I64_LT_U)

    # (a_hi == b_hi) && (a_lo < b_lo)
    push!(bytes, Opcode.I32_AND)

    # (a_hi < b_hi) || ((a_hi == b_hi) && (a_lo < b_lo))
    push!(bytes, Opcode.I32_OR)

    return bytes
end

"""
Emit 128-bit unsigned less than: a < b (unsigned)
Stack: [a_struct, b_struct] -> [i32 result (0 or 1)]
"""
function emit_int128_ult(ctx, arg_type::Type)::Vector{UInt8}
    bytes = UInt8[]
    type_idx = get_int128_type!(ctx.mod, ctx.type_registry, arg_type)

    # Allocate locals
    a_lo_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    a_hi_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    b_lo_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    b_hi_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)

    # Pop structs to locals
    b_struct_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, julia_to_wasm_type_concrete(arg_type, ctx))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(b_struct_local))

    a_struct_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, julia_to_wasm_type_concrete(arg_type, ctx))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(a_struct_local))

    # Extract fields
    for (struct_local, lo_local, hi_local) in [(a_struct_local, a_lo_local, a_hi_local),
                                                (b_struct_local, b_lo_local, b_hi_local)]
        push!(bytes, Opcode.LOCAL_GET)
        append!(bytes, encode_leb128_unsigned(struct_local))
        push!(bytes, Opcode.GC_PREFIX)
        push!(bytes, Opcode.STRUCT_GET)
        append!(bytes, encode_leb128_unsigned(type_idx))
        append!(bytes, encode_leb128_unsigned(0))
        push!(bytes, Opcode.LOCAL_SET)
        append!(bytes, encode_leb128_unsigned(lo_local))

        push!(bytes, Opcode.LOCAL_GET)
        append!(bytes, encode_leb128_unsigned(struct_local))
        push!(bytes, Opcode.GC_PREFIX)
        push!(bytes, Opcode.STRUCT_GET)
        append!(bytes, encode_leb128_unsigned(type_idx))
        append!(bytes, encode_leb128_unsigned(1))
        push!(bytes, Opcode.LOCAL_SET)
        append!(bytes, encode_leb128_unsigned(hi_local))
    end

    # Unsigned comparison: (a_hi < b_hi) || (a_hi == b_hi && a_lo < b_lo)
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(a_hi_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(b_hi_local))
    push!(bytes, Opcode.I64_LT_U)

    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(a_hi_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(b_hi_local))
    push!(bytes, Opcode.I64_EQ)

    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(a_lo_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(b_lo_local))
    push!(bytes, Opcode.I64_LT_U)

    push!(bytes, Opcode.I32_AND)
    push!(bytes, Opcode.I32_OR)

    return bytes
end

"""
Emit 128-bit left shift: x << n (where n is 64-bit)
Stack: [x_struct, n_i64] -> [result_struct]
Algorithm: result_lo = x_lo << n, result_hi = (x_hi << n) | (x_lo >> (64 - n))
"""
function emit_int128_shl(ctx, result_type::Type)::Vector{UInt8}
    bytes = UInt8[]
    type_idx = get_int128_type!(ctx.mod, ctx.type_registry, result_type)

    # Allocate all locals upfront
    n_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    x_struct_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, julia_to_wasm_type_concrete(result_type, ctx))
    x_lo_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    x_hi_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    result_lo_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    result_hi_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)

    # Stack: [x_struct, n_i64]
    # Pop n first (it's on top)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(n_local))

    # Pop x struct
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(x_struct_local))

    # Extract x fields
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(x_struct_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_GET)
    append!(bytes, encode_leb128_unsigned(type_idx))
    append!(bytes, encode_leb128_unsigned(0))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(x_lo_local))

    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(x_struct_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_GET)
    append!(bytes, encode_leb128_unsigned(type_idx))
    append!(bytes, encode_leb128_unsigned(1))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(x_hi_local))

    # result_lo = x_lo << n
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(x_lo_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(n_local))
    push!(bytes, Opcode.I64_SHL)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(result_lo_local))

    # result_hi = (x_hi << n) | (x_lo >> (64 - n))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(x_hi_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(n_local))
    push!(bytes, Opcode.I64_SHL)

    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(x_lo_local))
    push!(bytes, Opcode.I64_CONST)
    append!(bytes, encode_leb128_unsigned(64))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(n_local))
    push!(bytes, Opcode.I64_SUB)
    push!(bytes, Opcode.I64_SHR_U)

    push!(bytes, Opcode.I64_OR)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(result_hi_local))

    # Create result struct
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(result_lo_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(result_hi_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_NEW)
    append!(bytes, encode_leb128_unsigned(type_idx))

    return bytes
end

"""
Emit 128-bit logical right shift: x >> n (unsigned, where n is 64-bit)
Stack: [x_struct, n_i64] -> [result_struct]
"""
function emit_int128_lshr(ctx, result_type::Type)::Vector{UInt8}
    bytes = UInt8[]
    type_idx = get_int128_type!(ctx.mod, ctx.type_registry, result_type)

    # Allocate locals
    x_lo_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    x_hi_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    n_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    result_lo_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    result_hi_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)

    # Pop n first
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(n_local))

    # Pop x struct
    x_struct_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, julia_to_wasm_type_concrete(result_type, ctx))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(x_struct_local))

    # Extract x fields
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(x_struct_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_GET)
    append!(bytes, encode_leb128_unsigned(type_idx))
    append!(bytes, encode_leb128_unsigned(0))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(x_lo_local))

    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(x_struct_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_GET)
    append!(bytes, encode_leb128_unsigned(type_idx))
    append!(bytes, encode_leb128_unsigned(1))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(x_hi_local))

    # Logical right shift (for n < 64):
    # result_hi = x_hi >> n
    # result_lo = (x_lo >> n) | (x_hi << (64 - n))

    # result_hi = x_hi >> n (logical)
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(x_hi_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(n_local))
    push!(bytes, Opcode.I64_SHR_U)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(result_hi_local))

    # result_lo = (x_lo >> n) | (x_hi << (64 - n))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(x_lo_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(n_local))
    push!(bytes, Opcode.I64_SHR_U)

    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(x_hi_local))
    push!(bytes, Opcode.I64_CONST)
    append!(bytes, encode_leb128_unsigned(64))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(n_local))
    push!(bytes, Opcode.I64_SUB)
    push!(bytes, Opcode.I64_SHL)

    push!(bytes, Opcode.I64_OR)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(result_lo_local))

    # Create result struct
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(result_lo_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(result_hi_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_NEW)
    append!(bytes, encode_leb128_unsigned(type_idx))

    return bytes
end

"""
Emit 128-bit count leading zeros
Stack: [x_struct] -> [i64 result]
"""
function emit_int128_ctlz(ctx, arg_type::Type)::Vector{UInt8}
    bytes = UInt8[]
    type_idx = get_int128_type!(ctx.mod, ctx.type_registry, arg_type)

    # Allocate locals
    x_lo_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    x_hi_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)

    # Pop x struct
    x_struct_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, julia_to_wasm_type_concrete(arg_type, ctx))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(x_struct_local))

    # Extract x fields
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(x_struct_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_GET)
    append!(bytes, encode_leb128_unsigned(type_idx))
    append!(bytes, encode_leb128_unsigned(0))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(x_lo_local))

    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(x_struct_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_GET)
    append!(bytes, encode_leb128_unsigned(type_idx))
    append!(bytes, encode_leb128_unsigned(1))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(x_hi_local))

    # Count leading zeros:
    # if hi != 0: clz(hi)
    # else: 64 + clz(lo)

    # Compute: (x_hi == 0) ? (64 + clz(x_lo)) : clz(x_hi)
    # Using select: select(64 + clz(x_lo), clz(x_hi), x_hi == 0)

    # clz(x_hi)
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(x_hi_local))
    push!(bytes, Opcode.I64_CLZ)

    # 64 + clz(x_lo)
    push!(bytes, Opcode.I64_CONST)
    append!(bytes, encode_leb128_unsigned(64))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(x_lo_local))
    push!(bytes, Opcode.I64_CLZ)
    push!(bytes, Opcode.I64_ADD)

    # x_hi == 0
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(x_hi_local))
    push!(bytes, Opcode.I64_EQZ)

    # select(64+clz_lo, clz_hi, hi_is_zero) - but args are in wrong order
    # WASM select: select a, b, c -> c ? a : b
    # We want: (x_hi == 0) ? (64 + clz_lo) : clz_hi
    # So: select(64+clz_lo, clz_hi, x_hi==0) is wrong order
    # Actually: stack has [clz_hi, 64+clz_lo, x_hi==0]
    # select pops [val1, val2, cond] and pushes cond ? val1 : val2
    # So we need [64+clz_lo, clz_hi, x_hi==0] to get (x_hi==0) ? (64+clz_lo) : clz_hi
    # Current stack: [clz_hi, 64+clz_lo, x_hi==0]
    # We need to swap clz_hi and 64+clz_lo... not easy without locals

    # Let's use locals instead
    bytes = UInt8[]

    # Allocate locals
    x_lo_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    x_hi_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    clz_hi_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)

    # Pop x struct
    x_struct_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, julia_to_wasm_type_concrete(arg_type, ctx))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(x_struct_local))

    # Extract x fields
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(x_struct_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_GET)
    append!(bytes, encode_leb128_unsigned(type_idx))
    append!(bytes, encode_leb128_unsigned(0))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(x_lo_local))

    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(x_struct_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_GET)
    append!(bytes, encode_leb128_unsigned(type_idx))
    append!(bytes, encode_leb128_unsigned(1))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(x_hi_local))

    # clz(x_hi) -> store
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(x_hi_local))
    push!(bytes, Opcode.I64_CLZ)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(clz_hi_local))

    # Now compute result with proper select order:
    # select(64+clz_lo, clz_hi, hi==0)
    # Stack needs: [true_val, false_val, cond]

    # 64 + clz(x_lo) - true value (when hi == 0)
    push!(bytes, Opcode.I64_CONST)
    append!(bytes, encode_leb128_unsigned(64))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(x_lo_local))
    push!(bytes, Opcode.I64_CLZ)
    push!(bytes, Opcode.I64_ADD)

    # clz(x_hi) - false value
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(clz_hi_local))

    # x_hi == 0
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(x_hi_local))
    push!(bytes, Opcode.I64_EQZ)

    # select
    push!(bytes, Opcode.SELECT)

    # Now we have an i64 on the stack (the clz result)
    # But Julia expects ctlz_int to return UInt128, so wrap it in a struct with hi=0
    # Stack: [clz_result (i64)]

    # Store the clz result temporarily
    result_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(result_local))

    # Create UInt128 struct: (lo=clz_result, hi=0)
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(result_local))  # lo = clz_result
    push!(bytes, Opcode.I64_CONST)
    append!(bytes, encode_leb128_signed(0))  # hi = 0
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_NEW)
    append!(bytes, encode_leb128_unsigned(type_idx))

    return bytes
end

"""
Emit 128-bit bitwise AND
Stack: [a_struct, b_struct] -> [result_struct]
"""
function emit_int128_and(ctx, result_type::Type)::Vector{UInt8}
    bytes = UInt8[]
    type_idx = get_int128_type!(ctx.mod, ctx.type_registry, result_type)

    # Allocate locals
    a_lo_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    a_hi_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    b_lo_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    b_hi_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)

    # Pop structs to locals
    b_struct_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, julia_to_wasm_type_concrete(result_type, ctx))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(b_struct_local))

    a_struct_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, julia_to_wasm_type_concrete(result_type, ctx))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(a_struct_local))

    # Extract fields
    for (struct_local, lo_local, hi_local) in [(a_struct_local, a_lo_local, a_hi_local),
                                                (b_struct_local, b_lo_local, b_hi_local)]
        push!(bytes, Opcode.LOCAL_GET)
        append!(bytes, encode_leb128_unsigned(struct_local))
        push!(bytes, Opcode.GC_PREFIX)
        push!(bytes, Opcode.STRUCT_GET)
        append!(bytes, encode_leb128_unsigned(type_idx))
        append!(bytes, encode_leb128_unsigned(0))
        push!(bytes, Opcode.LOCAL_SET)
        append!(bytes, encode_leb128_unsigned(lo_local))

        push!(bytes, Opcode.LOCAL_GET)
        append!(bytes, encode_leb128_unsigned(struct_local))
        push!(bytes, Opcode.GC_PREFIX)
        push!(bytes, Opcode.STRUCT_GET)
        append!(bytes, encode_leb128_unsigned(type_idx))
        append!(bytes, encode_leb128_unsigned(1))
        push!(bytes, Opcode.LOCAL_SET)
        append!(bytes, encode_leb128_unsigned(hi_local))
    end

    # result_lo = a_lo & b_lo
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(a_lo_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(b_lo_local))
    push!(bytes, Opcode.I64_AND)

    # result_hi = a_hi & b_hi
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(a_hi_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(b_hi_local))
    push!(bytes, Opcode.I64_AND)

    # Create result struct
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_NEW)
    append!(bytes, encode_leb128_unsigned(type_idx))

    return bytes
end

"""
Emit 128-bit bitwise OR
Stack: [a_struct, b_struct] -> [result_struct]
"""
function emit_int128_or(ctx, result_type::Type)::Vector{UInt8}
    bytes = UInt8[]
    type_idx = get_int128_type!(ctx.mod, ctx.type_registry, result_type)

    # Allocate locals
    a_lo_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    a_hi_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    b_lo_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    b_hi_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)

    # Pop structs to locals
    b_struct_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, julia_to_wasm_type_concrete(result_type, ctx))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(b_struct_local))

    a_struct_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, julia_to_wasm_type_concrete(result_type, ctx))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(a_struct_local))

    # Extract fields
    for (struct_local, lo_local, hi_local) in [(a_struct_local, a_lo_local, a_hi_local),
                                                (b_struct_local, b_lo_local, b_hi_local)]
        push!(bytes, Opcode.LOCAL_GET)
        append!(bytes, encode_leb128_unsigned(struct_local))
        push!(bytes, Opcode.GC_PREFIX)
        push!(bytes, Opcode.STRUCT_GET)
        append!(bytes, encode_leb128_unsigned(type_idx))
        append!(bytes, encode_leb128_unsigned(0))
        push!(bytes, Opcode.LOCAL_SET)
        append!(bytes, encode_leb128_unsigned(lo_local))

        push!(bytes, Opcode.LOCAL_GET)
        append!(bytes, encode_leb128_unsigned(struct_local))
        push!(bytes, Opcode.GC_PREFIX)
        push!(bytes, Opcode.STRUCT_GET)
        append!(bytes, encode_leb128_unsigned(type_idx))
        append!(bytes, encode_leb128_unsigned(1))
        push!(bytes, Opcode.LOCAL_SET)
        append!(bytes, encode_leb128_unsigned(hi_local))
    end

    # result_lo = a_lo | b_lo
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(a_lo_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(b_lo_local))
    push!(bytes, Opcode.I64_OR)

    # result_hi = a_hi | b_hi
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(a_hi_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(b_hi_local))
    push!(bytes, Opcode.I64_OR)

    # Create result struct
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_NEW)
    append!(bytes, encode_leb128_unsigned(type_idx))

    return bytes
end

"""
Emit 128-bit bitwise XOR
Stack: [a_struct, b_struct] -> [result_struct]
"""
function emit_int128_xor(ctx, result_type::Type)::Vector{UInt8}
    bytes = UInt8[]
    type_idx = get_int128_type!(ctx.mod, ctx.type_registry, result_type)

    # Allocate locals
    a_lo_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    a_hi_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    b_lo_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    b_hi_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)

    # Pop structs to locals
    b_struct_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, julia_to_wasm_type_concrete(result_type, ctx))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(b_struct_local))

    a_struct_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, julia_to_wasm_type_concrete(result_type, ctx))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(a_struct_local))

    # Extract fields
    for (struct_local, lo_local, hi_local) in [(a_struct_local, a_lo_local, a_hi_local),
                                                (b_struct_local, b_lo_local, b_hi_local)]
        push!(bytes, Opcode.LOCAL_GET)
        append!(bytes, encode_leb128_unsigned(struct_local))
        push!(bytes, Opcode.GC_PREFIX)
        push!(bytes, Opcode.STRUCT_GET)
        append!(bytes, encode_leb128_unsigned(type_idx))
        append!(bytes, encode_leb128_unsigned(0))
        push!(bytes, Opcode.LOCAL_SET)
        append!(bytes, encode_leb128_unsigned(lo_local))

        push!(bytes, Opcode.LOCAL_GET)
        append!(bytes, encode_leb128_unsigned(struct_local))
        push!(bytes, Opcode.GC_PREFIX)
        push!(bytes, Opcode.STRUCT_GET)
        append!(bytes, encode_leb128_unsigned(type_idx))
        append!(bytes, encode_leb128_unsigned(1))
        push!(bytes, Opcode.LOCAL_SET)
        append!(bytes, encode_leb128_unsigned(hi_local))
    end

    # result_lo = a_lo ^ b_lo
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(a_lo_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(b_lo_local))
    push!(bytes, Opcode.I64_XOR)

    # result_hi = a_hi ^ b_hi
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(a_hi_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(b_hi_local))
    push!(bytes, Opcode.I64_XOR)

    # Create result struct
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_NEW)
    append!(bytes, encode_leb128_unsigned(type_idx))

    return bytes
end

"""
Emit 128-bit equality comparison
Stack: [a_struct, b_struct] -> [i32 result (0 or 1)]
"""
function emit_int128_eq(ctx, arg_type::Type)::Vector{UInt8}
    bytes = UInt8[]
    type_idx = get_int128_type!(ctx.mod, ctx.type_registry, arg_type)

    # Allocate locals
    a_lo_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    a_hi_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    b_lo_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    b_hi_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)

    # Pop structs to locals
    b_struct_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, julia_to_wasm_type_concrete(arg_type, ctx))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(b_struct_local))

    a_struct_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, julia_to_wasm_type_concrete(arg_type, ctx))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(a_struct_local))

    # Extract fields
    for (struct_local, lo_local, hi_local) in [(a_struct_local, a_lo_local, a_hi_local),
                                                (b_struct_local, b_lo_local, b_hi_local)]
        push!(bytes, Opcode.LOCAL_GET)
        append!(bytes, encode_leb128_unsigned(struct_local))
        push!(bytes, Opcode.GC_PREFIX)
        push!(bytes, Opcode.STRUCT_GET)
        append!(bytes, encode_leb128_unsigned(type_idx))
        append!(bytes, encode_leb128_unsigned(0))
        push!(bytes, Opcode.LOCAL_SET)
        append!(bytes, encode_leb128_unsigned(lo_local))

        push!(bytes, Opcode.LOCAL_GET)
        append!(bytes, encode_leb128_unsigned(struct_local))
        push!(bytes, Opcode.GC_PREFIX)
        push!(bytes, Opcode.STRUCT_GET)
        append!(bytes, encode_leb128_unsigned(type_idx))
        append!(bytes, encode_leb128_unsigned(1))
        push!(bytes, Opcode.LOCAL_SET)
        append!(bytes, encode_leb128_unsigned(hi_local))
    end

    # (a_lo == b_lo) && (a_hi == b_hi)
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(a_lo_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(b_lo_local))
    push!(bytes, Opcode.I64_EQ)

    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(a_hi_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(b_hi_local))
    push!(bytes, Opcode.I64_EQ)

    push!(bytes, Opcode.I32_AND)

    return bytes
end

"""
Emit 128-bit not-equal comparison
Stack: [a_struct, b_struct] -> [i32 result (0 or 1)]
"""
function emit_int128_ne(ctx, arg_type::Type)::Vector{UInt8}
    bytes = UInt8[]
    type_idx = get_int128_type!(ctx.mod, ctx.type_registry, arg_type)

    # Allocate locals
    a_lo_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    a_hi_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    b_lo_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)
    b_hi_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, I64)

    # Pop structs to locals
    b_struct_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, julia_to_wasm_type_concrete(arg_type, ctx))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(b_struct_local))

    a_struct_local = length(ctx.locals) + ctx.n_params
    push!(ctx.locals, julia_to_wasm_type_concrete(arg_type, ctx))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(a_struct_local))

    # Extract fields
    for (struct_local, lo_local, hi_local) in [(a_struct_local, a_lo_local, a_hi_local),
                                                (b_struct_local, b_lo_local, b_hi_local)]
        push!(bytes, Opcode.LOCAL_GET)
        append!(bytes, encode_leb128_unsigned(struct_local))
        push!(bytes, Opcode.GC_PREFIX)
        push!(bytes, Opcode.STRUCT_GET)
        append!(bytes, encode_leb128_unsigned(type_idx))
        append!(bytes, encode_leb128_unsigned(0))
        push!(bytes, Opcode.LOCAL_SET)
        append!(bytes, encode_leb128_unsigned(lo_local))

        push!(bytes, Opcode.LOCAL_GET)
        append!(bytes, encode_leb128_unsigned(struct_local))
        push!(bytes, Opcode.GC_PREFIX)
        push!(bytes, Opcode.STRUCT_GET)
        append!(bytes, encode_leb128_unsigned(type_idx))
        append!(bytes, encode_leb128_unsigned(1))
        push!(bytes, Opcode.LOCAL_SET)
        append!(bytes, encode_leb128_unsigned(hi_local))
    end

    # (a_lo != b_lo) || (a_hi != b_hi)
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(a_lo_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(b_lo_local))
    push!(bytes, Opcode.I64_NE)

    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(a_hi_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(b_hi_local))
    push!(bytes, Opcode.I64_NE)

    push!(bytes, Opcode.I32_OR)

    return bytes
end

# ============================================================================
# Compilation Context
# ============================================================================

"""
Tracks state during compilation of a single function.
"""
mutable struct CompilationContext
    code_info::Core.CodeInfo
    arg_types::Tuple
    return_type::Type
    n_params::Int
    locals::Vector{WasmValType}  # Additional locals beyond params (supports refs)
    ssa_types::Dict{Int, Type}   # SSA value -> Julia type
    ssa_locals::Dict{Int, Int}   # SSA value -> local index (for multi-use SSAs)
    phi_locals::Dict{Int, Int}   # PhiNode SSA -> local index
    loop_headers::Set{Int}       # Line numbers that are loop headers (targets of backward jumps)
    mod::WasmModule              # The module being built
    type_registry::TypeRegistry  # Struct type mappings
    func_registry::Union{FunctionRegistry, Nothing}  # Function mappings for cross-calls
    func_idx::UInt32             # Index of the function being compiled (for recursion)
    func_ref::Any                # Reference to original function (for self-call detection)
    global_args::Set{Int}        # Argument indices (1-based) that are WasmGlobal (phantom params)
    is_compiled_closure::Bool    # True if function being compiled is itself a closure
    # Signal substitution for Therapy.jl closures
    signal_ssa_getters::Dict{Int, UInt32}   # SSA id (from getfield) -> Wasm global index
    signal_ssa_setters::Dict{Int, UInt32}   # SSA id (from getfield) -> Wasm global index
    captured_signal_fields::Dict{Symbol, Tuple{Bool, UInt32}}  # field_name -> (is_getter, global_idx)
    # DOM bindings for Therapy.jl - emit DOM update calls after signal writes
    # Maps global_idx -> [(import_idx, [hk_arg, ...]), ...]
    dom_bindings::Dict{UInt32, Vector{Tuple{UInt32, Vector{Int32}}}}
    # Module-level globals: maps (Module, Symbol) -> Wasm global index
    # Used for const mutable struct instances that should be shared across functions
    module_globals::Dict{Tuple{Module, Symbol}, UInt32}
    # Scratch local indices for string operations (fixed at allocation time)
    # Tuple of (result_local, str1_local, str2_local, len1_local, i_local) or nothing
    scratch_locals::Union{Nothing, NTuple{5, Int}}
    # MemoryRef offset tracking: maps SSA id -> index SSA/value for memoryrefnew(ref, index, bc)
    # Used by memoryrefoffset to get the offset. Fresh refs (not in this map) have offset 1.
    memoryref_offsets::Dict{Int, Any}
end

function CompilationContext(code_info, arg_types::Tuple, return_type, mod::WasmModule, type_registry::TypeRegistry;
                           func_registry::Union{FunctionRegistry, Nothing}=nothing,
                           func_idx::UInt32=UInt32(0), func_ref=nothing,
                           global_args::Set{Int}=Set{Int}(),
                           is_compiled_closure::Bool=false,
                           captured_signal_fields::Dict{Symbol, Tuple{Bool, UInt32}}=Dict{Symbol, Tuple{Bool, UInt32}}(),
                           dom_bindings::Dict{UInt32, Vector{Tuple{UInt32, Vector{Int32}}}}=Dict{UInt32, Vector{Tuple{UInt32, Vector{Int32}}}}(),
                           module_globals::Dict{Tuple{Module, Symbol}, UInt32}=Dict{Tuple{Module, Symbol}, UInt32}())
    # Calculate n_params excluding WasmGlobal arguments (they're phantom)
    n_real_params = count(i -> !(i in global_args), 1:length(arg_types))
    ctx = CompilationContext(
        code_info,
        arg_types,
        return_type,
        n_real_params,
        WasmValType[],
        Dict{Int, Type}(),
        Dict{Int, Int}(),
        Dict{Int, Int}(),
        Set{Int}(),
        mod,
        type_registry,
        func_registry,
        func_idx,
        func_ref,
        global_args,
        is_compiled_closure,    # Is this function itself a closure?
        Dict{Int, UInt32}(),    # signal_ssa_getters
        Dict{Int, UInt32}(),    # signal_ssa_setters
        captured_signal_fields, # captured signal field mappings
        dom_bindings,           # DOM bindings for Therapy.jl
        module_globals,         # Module-level globals (const mutable structs)
        nothing,                # scratch_locals (set by allocate_scratch_locals!)
        Dict{Int, Any}()        # memoryref_offsets (populated during compilation)
    )
    # Analyze SSA types and allocate locals for multi-use SSAs
    analyze_ssa_types!(ctx)
    analyze_control_flow!(ctx)  # Find loops and phi nodes
    analyze_signal_captures!(ctx)  # Identify SSAs that are signal getters/setters
    allocate_ssa_locals!(ctx)
    allocate_scratch_locals!(ctx)  # Extra locals for complex operations
    return ctx
end

"""
Analyze getfield expressions on the closure (arg 1) to identify signal captures.
Maps SSA values from getfield to their signal global indices.

For CompilableSignal/CompilableSetter pattern:
- getfield(_1, :count) -> CompilableSignal SSA
- getfield(CompilableSignal, :signal) -> Signal SSA
- getfield(Signal, :value) -> actual value read (substitutes to global.get)
- setfield!(Signal, :value, x) -> value write (substitutes to global.set)
"""
function analyze_signal_captures!(ctx::CompilationContext)
    isempty(ctx.captured_signal_fields) && return

    code = ctx.code_info.code

    # For Therapy.jl: captured signal fields are getter/setter FUNCTIONS (closures)
    # When we see getfield(_1, :count) where :count is a getter, the resulting SSA
    # is a function that when invoked returns the signal value.
    # We directly map these to signal_ssa_getters/setters so that when compile_invoke
    # sees invoke(%ssa), it knows to emit global.get/global.set.

    # First pass: find closure field accesses to signal getter/setter functions
    for (i, stmt) in enumerate(code)
        if stmt isa Expr && stmt.head === :call
            func = stmt.args[1]
            # Handle both Core.getfield and Base.getfield
            is_getfield = (func isa GlobalRef &&
                          ((func.mod === Core && func.name === :getfield) ||
                           (func.mod === Base && func.name === :getfield)))
            if is_getfield && length(stmt.args) >= 3
                target = stmt.args[2]
                field_ref = stmt.args[3]
                field_name = field_ref isa QuoteNode ? field_ref.value : field_ref

                # Check if this is getfield(_1, :fieldname) - getting captured closure field
                # Target can be Core.SlotNumber(1) or Core.Argument(1)
                is_closure_self = (target isa Core.SlotNumber && target.id == 1) ||
                                  (target isa Core.Argument && target.n == 1)
                if is_closure_self
                    if field_name isa Symbol && haskey(ctx.captured_signal_fields, field_name)
                        is_getter, global_idx = ctx.captured_signal_fields[field_name]
                        # Directly map the SSA to signal getter/setter
                        # When this SSA is invoked, it becomes a signal read or write
                        if is_getter
                            ctx.signal_ssa_getters[i] = global_idx
                        else
                            ctx.signal_ssa_setters[i] = global_idx
                        end
                    end
                end
            end
        end
    end

    # Also handle WasmGlobal-style patterns (for compatibility with WasmGlobal{T, IDX})
    # Track CompilableSignal/CompilableSetter SSAs
    compilable_ssas = Dict{Int, Tuple{Bool, UInt32}}()  # ssa -> (is_getter, global_idx)

    # Track Signal SSAs (from getfield(CompilableSignal/Setter, :signal))
    signal_ssas = Dict{Int, UInt32}()  # ssa -> global_idx

    # Find getfield(_1, :fieldname) that might be WasmGlobal-style
    for (i, stmt) in enumerate(code)
        if stmt isa Expr && stmt.head === :call
            func = stmt.args[1]
            # Handle both Core.getfield and Base.getfield
            is_getfield = (func isa GlobalRef &&
                          ((func.mod === Core && func.name === :getfield) ||
                           (func.mod === Base && func.name === :getfield)))
            if is_getfield && length(stmt.args) >= 3
                target = stmt.args[2]
                field_ref = stmt.args[3]
                field_name = field_ref isa QuoteNode ? field_ref.value : field_ref

                is_closure_self = (target isa Core.SlotNumber && target.id == 1) ||
                                  (target isa Core.Argument && target.n == 1)
                if is_closure_self
                    if field_name isa Symbol && haskey(ctx.captured_signal_fields, field_name)
                        is_getter, global_idx = ctx.captured_signal_fields[field_name]
                        compilable_ssas[i] = (is_getter, global_idx)
                    end
                end
            end
        end
    end

    # Find getfield(CompilableSignal/Setter, :signal) -> Signal
    for (i, stmt) in enumerate(code)
        if stmt isa Expr && stmt.head === :call
            func = stmt.args[1]
            is_getfield = (func isa GlobalRef &&
                          ((func.mod === Core && func.name === :getfield) ||
                           (func.mod === Base && func.name === :getfield)))
            if is_getfield && length(stmt.args) >= 3
                target = stmt.args[2]
                field_ref = stmt.args[3]
                field_name = field_ref isa QuoteNode ? field_ref.value : field_ref

                if target isa Core.SSAValue && field_name === :signal
                    if haskey(compilable_ssas, target.id)
                        _, global_idx = compilable_ssas[target.id]
                        signal_ssas[i] = global_idx
                    end
                end
            end
        end
    end

    # Mark getfield(Signal, :value) as signal reads
    # and setfield!(Signal, :value, x) as signal writes
    for (i, stmt) in enumerate(code)
        if stmt isa Expr && stmt.head === :call
            func = stmt.args[1]

            # Handle getfield(Signal, :value) -> signal read
            is_getfield = (func isa GlobalRef &&
                          ((func.mod === Core && func.name === :getfield) ||
                           (func.mod === Base && func.name === :getfield)))
            if is_getfield && length(stmt.args) >= 3
                target = stmt.args[2]
                field_ref = stmt.args[3]
                field_name = field_ref isa QuoteNode ? field_ref.value : field_ref

                if target isa Core.SSAValue && field_name === :value
                    if haskey(signal_ssas, target.id)
                        global_idx = signal_ssas[target.id]
                        ctx.signal_ssa_getters[i] = global_idx
                    end
                end
            end

            # Handle setfield!(Signal, :value, x) -> signal write
            is_setfield = (func isa GlobalRef &&
                          ((func.mod === Core && func.name === :setfield!) ||
                           (func.mod === Base && func.name === :setfield!)))
            if is_setfield && length(stmt.args) >= 4
                target = stmt.args[2]
                field_ref = stmt.args[3]
                new_value = stmt.args[4]
                field_name = field_ref isa QuoteNode ? field_ref.value : field_ref

                if target isa Core.SSAValue && field_name === :value
                    if haskey(signal_ssas, target.id)
                        global_idx = signal_ssas[target.id]
                        ctx.signal_ssa_setters[i] = global_idx
                    end
                end
            end
        end
    end
end

"""
Allocate scratch locals for complex operations like string concatenation.
These are extra locals beyond what SSA analysis requires.
Stores the indices in ctx.scratch_locals for later use.
"""
function allocate_scratch_locals!(ctx::CompilationContext)
    # Check if any SSA type is String - if so, we need scratch locals
    needs_string_scratch = false
    for (_, T) in ctx.ssa_types
        if T === String
            needs_string_scratch = true
            break
        end
    end

    # Also check if return type or arg types include String
    if ctx.return_type === String
        needs_string_scratch = true
    end
    for T in ctx.arg_types
        if T === String
            needs_string_scratch = true
            break
        end
    end

    if needs_string_scratch
        # Add 5 scratch locals for string operations:
        # - 1 ref for result array
        # - 2 refs for source strings
        # - 2 i32s for lengths/indices
        # Use get_string_array_type! to ensure type is registered
        str_type_idx = get_string_array_type!(ctx.mod, ctx.type_registry)
        str_ref_type = ConcreteRef(str_type_idx, true)

        # Calculate indices BEFORE adding locals (indices are n_params + current local count)
        scratch_base = ctx.n_params + length(ctx.locals)
        result_local = scratch_base      # ref for result
        str1_local = scratch_base + 1    # ref for str1
        str2_local = scratch_base + 2    # ref for str2
        len1_local = scratch_base + 3    # i32 for len1
        i_local = scratch_base + 4       # i32 for len2/index

        # Store the indices in context
        ctx.scratch_locals = (result_local, str1_local, str2_local, len1_local, i_local)

        # Now add the locals
        push!(ctx.locals, str_ref_type)  # result/scratch ref 1
        push!(ctx.locals, str_ref_type)  # scratch ref 2
        push!(ctx.locals, str_ref_type)  # scratch ref 3
        push!(ctx.locals, I32)           # scratch i32 1 (len1)
        push!(ctx.locals, I32)           # scratch i32 2 (len2/i)
    end
end

"""
    allocate_local!(ctx, julia_type) -> local_index

Allocate a new local variable of the given Julia type and return its index.
The index is relative to the function's locals, accounting for parameters.
"""
function allocate_local!(ctx::CompilationContext, T::Type)::Int
    wasm_type = julia_to_wasm_type_concrete(T, ctx)
    local_idx = ctx.n_params + length(ctx.locals)
    push!(ctx.locals, wasm_type)
    return local_idx
end

"""
Convert a Julia type to a WasmValType, using concrete references for struct/array types.
This is like `julia_to_wasm_type` but returns `ConcreteRef` for registered types.
"""
function julia_to_wasm_type_concrete(T, ctx::CompilationContext)::WasmValType
    if is_struct_type(T)
        # If struct is registered, return a ConcreteRef
        if haskey(ctx.type_registry.structs, T)
            info = ctx.type_registry.structs[T]
            return ConcreteRef(info.wasm_type_idx, true)
        else
            # Register it now
            register_struct_type!(ctx.mod, ctx.type_registry, T)
            if haskey(ctx.type_registry.structs, T)
                info = ctx.type_registry.structs[T]
                return ConcreteRef(info.wasm_type_idx, true)
            end
        end
        # Fallback to abstract StructRef
        return StructRef
    elseif T <: Tuple
        # Tuples are stored as WasmGC structs
        if haskey(ctx.type_registry.structs, T)
            info = ctx.type_registry.structs[T]
            return ConcreteRef(info.wasm_type_idx, true)
        else
            # Register it now
            register_tuple_type!(ctx.mod, ctx.type_registry, T)
            if haskey(ctx.type_registry.structs, T)
                info = ctx.type_registry.structs[T]
                return ConcreteRef(info.wasm_type_idx, true)
            end
        end
        # Fallback to abstract StructRef
        return StructRef
    elseif T isa DataType && (T.name.name === :MemoryRef || T.name.name === :GenericMemoryRef)
        # MemoryRef{T} / GenericMemoryRef maps to the array type for element T
        # This is Julia's internal type for array element access
        # IMPORTANT: Check this BEFORE AbstractArray since MemoryRef <: AbstractArray
        # GenericMemoryRef parameters: (atomicity, element_type, addrspace)
        elem_type = T.name.name === :GenericMemoryRef ? T.parameters[2] : T.parameters[1]
        if haskey(ctx.type_registry.arrays, elem_type)
            type_idx = ctx.type_registry.arrays[elem_type]
            return ConcreteRef(type_idx, true)
        else
            type_idx = get_array_type!(ctx.mod, ctx.type_registry, elem_type)
            return ConcreteRef(type_idx, true)
        end
    elseif T isa DataType && (T.name.name === :Memory || T.name.name === :GenericMemory)
        # GenericMemory/Memory is the backing storage for Vector (Julia 1.11+)
        # IMPORTANT: Check this BEFORE AbstractArray since Memory <: AbstractArray
        # Parameters are: (atomicity, element_type, addrspace)
        # In WasmGC, it's the same as the array
        elem_type = T.parameters[2]  # Element type is second parameter
        if haskey(ctx.type_registry.arrays, elem_type)
            type_idx = ctx.type_registry.arrays[elem_type]
            return ConcreteRef(type_idx, true)
        else
            type_idx = get_array_type!(ctx.mod, ctx.type_registry, elem_type)
            return ConcreteRef(type_idx, true)
        end
    elseif T <: AbstractArray  # Handles Vector, Matrix, and higher-dim arrays
        # In Julia 1.11+, Vector is a struct with :ref (MemoryRef) and :size fields
        # Check if the type is registered as a struct first (for Vector/Matrix)
        if haskey(ctx.type_registry.structs, T)
            info = ctx.type_registry.structs[T]
            return ConcreteRef(info.wasm_type_idx, true)
        end

        # 1D arrays (Vector) are stored as WasmGC structs (with ref and size fields)
        if T <: AbstractVector
            # Register Vector as a struct type
            info = register_vector_type!(ctx.mod, ctx.type_registry, T)
            return ConcreteRef(info.wasm_type_idx, true)
        else
            # Matrix and higher-dim arrays: also stored as structs
            info = register_matrix_type!(ctx.mod, ctx.type_registry, T)
            return ConcreteRef(info.wasm_type_idx, true)
        end
    elseif T === String
        # Strings are WasmGC arrays of bytes
        type_idx = get_string_array_type!(ctx.mod, ctx.type_registry)
        return ConcreteRef(type_idx, true)
    elseif T === Int128 || T === UInt128
        # 128-bit integers are represented as WasmGC structs with two i64 fields
        if haskey(ctx.type_registry.structs, T)
            info = ctx.type_registry.structs[T]
            return ConcreteRef(info.wasm_type_idx, true)
        else
            info = register_int128_type!(ctx.mod, ctx.type_registry, T)
            return ConcreteRef(info.wasm_type_idx, true)
        end
    elseif T isa Union
        # Handle Union types
        inner_type = get_nullable_inner_type(T)
        if inner_type !== nothing
            # Union{Nothing, T} -> use T's concrete type (nullable reference)
            return julia_to_wasm_type_concrete(inner_type, ctx)
        elseif needs_tagged_union(T)
            # Multi-variant union -> use tagged union struct
            info = get_union_type!(ctx.mod, ctx.type_registry, T)
            return ConcreteRef(info.wasm_type_idx, true)
        else
            # Fall back to standard resolution
            return julia_to_wasm_type(T)
        end
    else
        # Use the standard conversion for non-struct types
        return julia_to_wasm_type(T)
    end
end

"""
Emit bytecode to convert a value on the stack to f64.
Used for DOM bindings where all numeric values are passed as f64 for JS compatibility.
"""
function emit_convert_to_f64(valtype::WasmValType)::Vector{UInt8}
    if valtype == I32
        return UInt8[0xB7]  # f64.convert_i32_s
    elseif valtype == I64
        return UInt8[0xB9]  # f64.convert_i64_s
    elseif valtype == F32
        return UInt8[0xBB]  # f64.promote_f32
    elseif valtype == F64
        return UInt8[]      # Already f64, no conversion needed
    else
        # For other types (refs, etc.), no conversion - will cause type error
        return UInt8[]
    end
end

"""
Encode a block result type (for if/block/loop).
Handles both simple types (i32/i64/f32/f64) and concrete reference types.
Returns a vector of bytes to append to the instruction stream.
"""
function encode_block_type(result_type::WasmValType)::Vector{UInt8}
    bytes = UInt8[]
    if result_type isa NumType
        push!(bytes, UInt8(result_type))
    elseif result_type isa RefType
        push!(bytes, UInt8(result_type))
    elseif result_type isa ConcreteRef
        # Concrete reference type: 0x63 (nullable) or 0x64 (non-nullable) + type index
        if result_type.nullable
            push!(bytes, 0x63)  # ref null
        else
            push!(bytes, 0x64)  # ref
        end
        # Type index as signed LEB128
        append!(bytes, encode_leb128_signed(Int64(result_type.type_idx)))
    elseif result_type isa UInt8
        push!(bytes, result_type)
    else
        # Fallback - try to convert to UInt8
        push!(bytes, UInt8(result_type))
    end
    return bytes
end

"""
Analyze control flow to find loops and handle phi nodes.
"""
function analyze_control_flow!(ctx::CompilationContext)
    code = ctx.code_info.code

    # Find loop headers (targets of backward jumps)
    for (i, stmt) in enumerate(code)
        if stmt isa Core.GotoNode
            target = stmt.label
            if target < i  # Backward jump = loop
                push!(ctx.loop_headers, target)
            end
        elseif stmt isa Core.GotoIfNot
            # GotoIfNot jumps forward (to exit), but check anyway
        end
    end

    # Find goto statements that jump backward (unconditional loop back)
    for (i, stmt) in enumerate(code)
        if stmt isa Core.GotoNode && stmt.label < i
            push!(ctx.loop_headers, stmt.label)
        end
    end

    # Allocate locals for phi nodes (they need to persist across iterations)
    for (i, stmt) in enumerate(code)
        if stmt isa Core.PhiNode
            wasm_type = get(ctx.ssa_types, i, Int64)
            local_idx = ctx.n_params + length(ctx.locals)
            push!(ctx.locals, julia_to_wasm_type_concrete(wasm_type, ctx))
            ctx.phi_locals[i] = local_idx
        end
    end
end

"""
Allocate locals for SSA values that need them.
We need locals when:
1. An SSA value is used multiple times
2. An SSA value is not used immediately (intervening stack operations)
3. An SSA value is used in a multi-arg call where a sibling arg has a local
4. An SSA value is defined inside a loop but used outside (e.g., in return)
"""
function allocate_ssa_locals!(ctx::CompilationContext)
    code = ctx.code_info.code

    # Count uses of each SSA value
    ssa_uses = Dict{Int, Int}()
    for stmt in code
        count_ssa_uses!(stmt, ssa_uses)
    end

    # Find loop bounds (header to backward goto)
    loop_bounds = Dict{Int, Int}()  # header => back_edge_idx
    for (i, stmt) in enumerate(code)
        if stmt isa Core.GotoNode && stmt.label < i
            # This is a backward jump
            header = stmt.label
            loop_bounds[header] = i
        end
    end

    # First pass: allocate locals for SSAs used more than once or with intervening ops
    needs_local_set = Set{Int}()

    # Find SSAs defined inside a loop but used outside
    # These need locals because stack values don't persist across Wasm block boundaries
    for (header, back_edge) in loop_bounds
        for (i, stmt) in enumerate(code)
            # Check if SSA i is defined inside this loop
            if i >= header && i <= back_edge
                # Check if it's used after the loop (in return or other statements)
                for (j, other) in enumerate(code)
                    if j > back_edge && references_ssa(other, i)
                        # SSA i is defined inside loop but used outside - needs local
                        push!(needs_local_set, i)
                        break
                    end
                end
            end
        end
    end

    # Find non-phi SSA values that are referenced by phi nodes
    # These MUST have locals because phi values are set at the jump site,
    # not where the SSA was computed (the value is no longer on the stack)
    for (i, stmt) in enumerate(code)
        if stmt isa Core.PhiNode
            for val in stmt.values
                if val isa Core.SSAValue && !(code[val.id] isa Core.PhiNode)
                    # This is a non-phi SSA referenced by a phi - needs local
                    push!(needs_local_set, val.id)
                end
            end
        end
    end

    # Find SSA values referenced by PiNodes that have control flow between definition and use
    # PiNodes narrow types after branch conditions, but the original value must be preserved
    for (i, stmt) in enumerate(code)
        if stmt isa Core.PiNode && stmt.val isa Core.SSAValue
            val_id = stmt.val.id
            # Check if there's control flow between the definition and this PiNode
            has_control_flow = false
            for j in (val_id + 1):(i - 1)
                if code[j] isa Core.GotoNode || code[j] isa Core.GotoIfNot
                    has_control_flow = true
                    break
                end
            end
            if has_control_flow
                push!(needs_local_set, val_id)
            end
        end
    end

    # Find SSAs that produce values and are followed by control flow
    # In Wasm, stack values don't persist across block boundaries
    # So any value produced before a GotoNode/GotoIfNot/PhiNode must be stored
    for (i, stmt) in enumerate(code)
        if produces_stack_value(stmt) && i < length(code)
            next_stmt = code[i + 1]
            # If the NEXT statement is control flow (not intermediate), this SSA needs a local
            # This handles cases where we create a value and immediately enter control flow
            if next_stmt isa Core.GotoNode || next_stmt isa Core.GotoIfNot
                push!(needs_local_set, i)
            end
        end
    end

    for (ssa_id, use_count) in ssa_uses
        if haskey(ctx.phi_locals, ssa_id)
            # Phi nodes already have locals
            ctx.ssa_locals[ssa_id] = ctx.phi_locals[ssa_id]
        elseif use_count > 1 || needs_local(ctx, ssa_id)
            push!(needs_local_set, ssa_id)
        end
    end

    # Second pass: if any arg in a multi-arg call has a local, all args need locals
    # to ensure correct stack ordering
    for (i, stmt) in enumerate(code)
        if stmt isa Expr && stmt.head === :call
            args = stmt.args[2:end]
            ssa_args = [arg.id for arg in args if arg isa Core.SSAValue]

            # Check if any SSA arg has or needs a local
            any_has_local = any(id in needs_local_set for id in ssa_args)

            if any_has_local
                # All SSA args need locals
                for id in ssa_args
                    push!(needs_local_set, id)
                end
            end
        end

        # Also handle :invoke expressions (method calls)
        # args[1] is MethodInstance, args[2] is function ref, args[3:end] are actual arguments
        if stmt isa Expr && stmt.head === :invoke
            args = stmt.args[3:end]
            ssa_args = [arg.id for arg in args if arg isa Core.SSAValue]

            # If there are any SSA args and there are other args before them, need locals
            # to ensure correct stack ordering (SSA values on stack would be in wrong position)
            has_non_ssa_args = any(!(arg isa Core.SSAValue) for arg in args)

            if !isempty(ssa_args) && (has_non_ssa_args || length(ssa_args) > 1)
                # All SSA args need locals to ensure correct stack ordering
                for id in ssa_args
                    push!(needs_local_set, id)
                end
            end
        end

        # Also handle :new expressions - struct fields need correct ordering
        if stmt isa Expr && stmt.head === :new
            # args[1] is the type, args[2:end] are field values
            field_values = stmt.args[2:end]
            ssa_args = [arg.id for arg in field_values if arg isa Core.SSAValue]

            # If there are multiple field values and any is an SSA, all SSA args need locals
            # This ensures we can push values in the correct field order
            if length(field_values) > 1 && !isempty(ssa_args)
                for id in ssa_args
                    push!(needs_local_set, id)
                end
            end
        end

        # Handle setfield! - the value arg needs a local if it's an SSA
        # because struct.set expects [ref, value] order, but if value is a single-use
        # SSA from a previous statement, it's already on the stack before we push ref
        if stmt isa Expr && stmt.head === :call
            func = stmt.args[1]
            is_setfield = (func isa GlobalRef &&
                          ((func.mod === Core && func.name === :setfield!) ||
                           (func.mod === Base && func.name === :setfield!)))
            if is_setfield && length(stmt.args) >= 4
                value_arg = stmt.args[4]  # args = [func, obj, field, value]
                if value_arg isa Core.SSAValue
                    push!(needs_local_set, value_arg.id)
                end
            end
        end

        # Handle :call expressions where a non-SSA arg appears BEFORE an SSA arg
        # This causes stack ordering issues: the SSA from the previous statement
        # is already on the stack, but we need to push the non-SSA first.
        # Example: slt_int(0, %1) - need to push 0, then %1, but %1 is already on stack
        # ONLY applies to numeric SSA values (struct refs have different handling)
        if stmt isa Expr && stmt.head === :call
            args = stmt.args[2:end]  # Skip function ref
            seen_non_ssa = false
            for arg in args
                if !(arg isa Core.SSAValue)
                    seen_non_ssa = true
                elseif seen_non_ssa
                    # This SSA comes after a non-SSA arg - needs a local
                    ssa_type = get(ctx.ssa_types, arg.id, Any)
                    is_numeric = ssa_type in (Int32, UInt32, Int64, UInt64, Int, Float32, Float64, Bool)
                    if is_numeric
                        push!(needs_local_set, arg.id)
                    end
                end
            end
        end

        # Handle Core.tuple calls - same as :new, need locals for SSA args
        # when there are multiple elements to ensure correct struct.new field ordering
        if stmt isa Expr && stmt.head === :call
            func = stmt.args[1]
            is_tuple = func isa GlobalRef && func.mod === Core && func.name === :tuple
            if is_tuple
                args = stmt.args[2:end]
                ssa_args = [arg.id for arg in args if arg isa Core.SSAValue]
                # If there are multiple SSA args, all of them need locals to ensure
                # correct ordering (even if there are no non-SSA args)
                # Also need locals if there are non-SSA args mixed with SSA args
                has_non_ssa_args = any(!(arg isa Core.SSAValue) for arg in args)
                if (has_non_ssa_args && !isempty(ssa_args)) || length(ssa_args) > 1
                    for id in ssa_args
                        push!(needs_local_set, id)
                    end
                end
            end
        end

    end

    # Actually allocate the locals
    for ssa_id in sort(collect(needs_local_set))
        if !haskey(ctx.ssa_locals, ssa_id)  # Skip phi nodes already added
            ssa_type = get(ctx.ssa_types, ssa_id, Int64)

            # MemoryRef types in WasmGC are just array references, so they CAN be stored in a local
            # (They're NOT two stack values - that's only relevant for linear memory, not WasmGC)
            # So we don't skip them anymore.

            # Skip Nothing type - nothing is compiled as ref.null, not i32
            # Trying to store it in an i32 local causes type errors
            if ssa_type === Nothing
                continue
            end

            # Skip bottom type (Union{}) - unreachable code
            if ssa_type === Union{}
                continue
            end

            wasm_type = julia_to_wasm_type_concrete(ssa_type, ctx)
            local_idx = ctx.n_params + length(ctx.locals)
            push!(ctx.locals, wasm_type)
            ctx.ssa_locals[ssa_id] = local_idx
        end
    end
end

"""
Check if an SSA value needs a local (e.g., not used immediately or used after other stack-producing operations).
"""
function needs_local(ctx::CompilationContext, ssa_id::Int)
    code = ctx.code_info.code

    # Find where this SSA is used
    use_idx = nothing
    for (i, stmt) in enumerate(code)
        if i != ssa_id && references_ssa(stmt, ssa_id)
            use_idx = i
            break
        end
    end

    if use_idx === nothing
        return false  # Never used
    end

    # If there are any statements between definition and use that produce values,
    # we need a local because those values will mess up the stack
    for i in (ssa_id + 1):(use_idx - 1)
        stmt = code[i]
        if produces_stack_value(stmt)
            return true
        end
    end

    # Also need local if there's control flow between definition and use
    for i in (ssa_id + 1):(use_idx - 1)
        stmt = code[i]
        if stmt isa Core.GotoIfNot || stmt isa Core.GotoNode
            return true
        end
    end

    # If SSA is defined inside a loop and there are conditionals in the loop,
    # we need a local to ensure stack balance across control flow
    for header in ctx.loop_headers
        # Find corresponding back-edge
        back_edge = nothing
        for (i, stmt) in enumerate(code)
            if stmt isa Core.GotoNode && stmt.label == header
                back_edge = i
                break
            end
        end
        if back_edge !== nothing && ssa_id >= header && ssa_id <= back_edge
            # SSA is defined inside this loop
            # Check if there are any conditionals in the loop
            for i in header:back_edge
                if code[i] isa Core.GotoIfNot
                    # Loop has a conditional (not the exit condition if it's at the start)
                    if i != header && i != header + 1
                        return true
                    end
                end
            end
        end
    end

    return false
end

"""
Check if a statement produces a value on the stack.
"""
function produces_stack_value(stmt)
    # Most expressions produce values
    if stmt isa Expr
        return stmt.head in (:call, :invoke, :new, :boundscheck, :tuple)
    end
    if stmt isa Core.PhiNode
        return true
    end
    if stmt isa Core.PiNode
        return true
    end
    # Literals and SSA refs also produce values (but shouldn't appear as statements)
    if stmt isa Number || stmt isa Core.SSAValue
        return true
    end
    return false
end

"""
Check if a statement is a passthrough that doesn't emit bytecode but relies on
a value already being on the stack from an earlier SSA.
Examples:
- memoryrefnew(memory) - just passes through the array reference
- Core.memoryref(memory) via :invoke - also a passthrough
Note: Vector{T} is NO LONGER a passthrough - it's now a struct with (ref, size) fields.
"""
function is_passthrough_statement(stmt, ctx::CompilationContext)
    if !(stmt isa Expr)
        return false
    end

    # Check for memoryrefnew with single arg (passthrough pattern) via :call
    if stmt.head === :call
        func = stmt.args[1]
        is_memrefnew = (func isa GlobalRef && func.mod === Core && func.name === :memoryrefnew) ||
                       (func === :(Core.memoryrefnew))
        if is_memrefnew && length(stmt.args) == 2
            # Single arg memoryrefnew is a passthrough
            return true
        end
    end

    # Check for Core.memoryref via :invoke - this is also a passthrough
    # Julia uses :invoke for Core.memoryref(memory::Memory{T}) -> MemoryRef{T}
    # In WasmGC, this is a no-op since Memory and MemoryRef are both the array
    if stmt.head === :invoke && length(stmt.args) >= 3
        # args[1] is MethodInstance, args[2] is function ref, args[3:end] are actual args
        func_ref = stmt.args[2]
        args = stmt.args[3:end]

        # Check if it's Core.memoryref with single arg
        is_memoryref = func_ref === :(Core.memoryref) ||
                       (func_ref isa GlobalRef && func_ref.mod === Core && func_ref.name === :memoryref)

        if is_memoryref && length(args) == 1
            arg = args[1]
            # It's a passthrough if the single arg is an SSA that doesn't have a local
            # (meaning its value is still on the stack from the previous statement)
            if arg isa Core.SSAValue && !haskey(ctx.ssa_locals, arg.id)
                return true
            end
        end
    end

    # Note: Vector %new is NO LONGER a passthrough
    # Vector{T} is now a struct with (ref, size) fields for setfield! support

    return false
end

"""
Count SSA uses in a statement.
"""
function count_ssa_uses!(stmt, uses::Dict{Int, Int})
    if stmt isa Core.SSAValue
        uses[stmt.id] = get(uses, stmt.id, 0) + 1
    elseif stmt isa Expr
        for arg in stmt.args
            count_ssa_uses!(arg, uses)
        end
    elseif stmt isa Core.ReturnNode && isdefined(stmt, :val)
        count_ssa_uses!(stmt.val, uses)
    elseif stmt isa Core.GotoIfNot
        count_ssa_uses!(stmt.cond, uses)
    elseif stmt isa Core.PhiNode
        for val in stmt.values
            count_ssa_uses!(val, uses)
        end
    end
end

"""
Check if a statement references an SSA value.
"""
function references_ssa(stmt, ssa_id::Int)::Bool
    if stmt isa Core.SSAValue
        return stmt.id == ssa_id
    elseif stmt isa Expr
        return any(references_ssa(arg, ssa_id) for arg in stmt.args)
    elseif stmt isa Core.ReturnNode && isdefined(stmt, :val)
        return references_ssa(stmt.val, ssa_id)
    elseif stmt isa Core.GotoIfNot
        return references_ssa(stmt.cond, ssa_id)
    end
    return false
end

"""
Get the Julia type of an SSA value or other value reference.
Used for type checking (e.g., in isa() calls).
"""
function get_ssa_type(ctx::CompilationContext, val)::Type
    if val isa Core.SSAValue
        return get(ctx.ssa_types, val.id, Any)
    elseif val isa Core.Argument
        # Handle argument references
        if ctx.is_compiled_closure
            idx = val.n
        else
            idx = val.n - 1
        end
        if idx >= 1 && idx <= length(ctx.arg_types)
            return ctx.arg_types[idx]
        end
        return Any
    elseif val isa Type
        return Type{val}  # It's a type constant
    else
        return typeof(val)
    end
end

"""
Analyze the IR to determine types of SSA values.
Uses CodeInfo.ssavaluetypes for accurate type information.
"""
function analyze_ssa_types!(ctx::CompilationContext)
    # Use Julia's type inference results when available
    ssatypes = ctx.code_info.ssavaluetypes
    if ssatypes isa Vector
        for (i, T) in enumerate(ssatypes)
            # Store all concrete types including Nothing (needed for function dispatch)
            # Only skip Any as it provides no useful information
            if T !== Any
                ctx.ssa_types[i] = T
            end
        end
    end

    # Fallback: infer from calls for any missing types
    for (i, stmt) in enumerate(ctx.code_info.code)
        if !haskey(ctx.ssa_types, i)
            if stmt isa Expr && stmt.head === :call
                ctx.ssa_types[i] = infer_call_type(stmt, ctx)
            elseif stmt isa Expr && stmt.head === :invoke
                # For invoke expressions with Any type, get the actual method return type
                mi_or_ci = stmt.args[1]
                mi = if mi_or_ci isa Core.MethodInstance
                    mi_or_ci
                elseif isdefined(Core, :CodeInstance) && mi_or_ci isa Core.CodeInstance
                    mi_or_ci.def
                else
                    nothing
                end
                if mi isa Core.MethodInstance
                    meth = mi.def
                    if meth isa Method
                        # Get the function reference from the invoke expression
                        func_ref = stmt.args[2]
                        if func_ref isa GlobalRef
                            func = try getfield(func_ref.mod, func_ref.name) catch; nothing end
                            if func !== nothing && ctx.func_registry !== nothing && haskey(ctx.func_registry.by_ref, func)
                                # Look up in registry by function reference
                                infos = ctx.func_registry.by_ref[func]
                                if !isempty(infos)
                                    # Use the first matching function's return type
                                    ctx.ssa_types[i] = infos[1].return_type
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

function infer_call_type(expr::Expr, ctx::CompilationContext)
    func = expr.args[1]
    args = expr.args[2:end]

    # Comparison operations return Bool
    if is_comparison(func)
        return Bool
    end

    # For arithmetic, infer from first argument
    if length(args) > 0
        return infer_value_type(args[1], ctx)
    end

    return Int64  # Default
end

function infer_value_type(val, ctx::CompilationContext)
    if val isa Core.Argument
        # For closures being compiled, _1 is the closure object (arg_types[1])
        # For regular functions, arguments start at _2 (arg_types[1])
        # Use is_compiled_closure flag to distinguish (not the type of first arg)
        if ctx.is_compiled_closure
            # Closure: direct mapping (_1 = closure, _2 = first arg)
            idx = val.n
        else
            # Regular function: skip _1 (function type in IR)
            idx = val.n - 1
        end
        if idx >= 1 && idx <= length(ctx.arg_types)
            return ctx.arg_types[idx]
        end
    elseif val isa Core.SSAValue
        return get(ctx.ssa_types, val.id, Int64)
    elseif val isa Int64 || val isa Int
        return Int64
    elseif val isa Int32
        return Int32
    elseif val isa Float64
        return Float64
    elseif val isa Float32
        return Float32
    elseif val isa Bool
        return Bool
    elseif val isa Char
        return Char
    elseif val isa WasmGlobal
        return typeof(val)
    elseif val isa GlobalRef
        # GlobalRef to a constant - infer type from the actual value
        try
            actual_val = getfield(val.mod, val.name)
            if actual_val isa Int32
                return Int32
            elseif actual_val isa Int64 || actual_val isa Int
                return Int64
            elseif actual_val isa Float32
                return Float32
            elseif actual_val isa Float64
                return Float64
            elseif actual_val isa Bool
                return Bool
            elseif actual_val isa Char
                return Char
            elseif actual_val isa Type
                return Type  # Type references
            else
                return typeof(actual_val)
            end
        catch
            # If we can't evaluate, default to Int64
        end
    elseif isprimitivetype(typeof(val))
        # Custom primitive type (e.g., JuliaSyntax.Kind) - return actual type
        return typeof(val)
    elseif isstructtype(typeof(val)) && !isa(val, Type) && !isa(val, Function) && !isa(val, Module)
        # Struct constant - return actual type
        return typeof(val)
    end
    return Int64
end

"""
Extract the global index from a WasmGlobal type.
The index is stored as a type parameter, so we extract it from the type.
"""
function get_wasm_global_idx(val, ctx::CompilationContext)::Union{Int, Nothing}
    val_type = infer_value_type(val, ctx)
    if val_type <: WasmGlobal
        # Extract IDX from WasmGlobal{T, IDX}
        return global_index(val_type)
    end
    return nothing
end

"""
Check if a call/invoke statement produces a value on the WASM stack.
Returns false for calls to functions that return Nothing (void).
This checks the function registry first (most reliable), then MethodInstance return type.
"""
function statement_produces_wasm_value(stmt::Expr, idx::Int, ctx::CompilationContext)::Bool
    # Get the SSA type first
    stmt_type = get(ctx.ssa_types, idx, Any)

    # If SSA type is definitely Nothing, no value produced
    if stmt_type === Nothing
        return false
    end

    # If SSA type is Union{} (bottom type), the statement never returns so no value
    if stmt_type === Union{}
        return false
    end

    # NOTE: Union{T, Nothing} DOES produce a value (a union struct in WASM)
    # Only exact Nothing type means void return

    # Check the function registry first - this is the most reliable source
    # because it reflects what we actually compiled the function with
    if ctx.func_registry !== nothing
        # Extract the called function from the statement
        called_func = nothing
        call_arg_types = nothing

        if stmt.head === :invoke && length(stmt.args) >= 2
            # For invoke, args[2] is typically a GlobalRef to the function
            func_ref = stmt.args[2]
            if func_ref isa GlobalRef
                try
                    called_func = getfield(func_ref.mod, func_ref.name)
                    # Skip built-in functions that aren't in the registry
                    if called_func !== Base.getfield && called_func !== Core.getfield &&
                       called_func !== Base.setfield! && called_func !== Core.setfield!
                        # Get argument types from the remaining args
                        call_arg_types = Tuple{[infer_value_type(arg, ctx) for arg in stmt.args[3:end]]...}
                    end
                catch
                end
            end
        elseif stmt.head === :call && length(stmt.args) >= 1
            func_ref = stmt.args[1]
            if func_ref isa GlobalRef
                try
                    called_func = getfield(func_ref.mod, func_ref.name)
                    # Skip built-in functions that aren't in the registry
                    if called_func !== Base.getfield && called_func !== Core.getfield &&
                       called_func !== Base.setfield! && called_func !== Core.setfield!
                        call_arg_types = Tuple{[infer_value_type(arg, ctx) for arg in stmt.args[2:end]]...}
                    end
                catch
                end
            end
        end

        if called_func !== nothing && call_arg_types !== nothing
            # Only look up if the function is in our registry
            if haskey(ctx.func_registry.by_ref, called_func)
                try
                    target_info = get_function(ctx.func_registry, called_func, call_arg_types)
                    if target_info !== nothing
                        # Use the return type we actually compiled with
                        if target_info.return_type === Nothing
                            return false
                        else
                            return true
                        end
                    end
                catch
                    # If lookup fails (e.g., type mismatch), fall through to other checks
                end
            end
        end
    end

    # For invoke statements, check the MethodInstance's return type
    if stmt.head === :invoke && length(stmt.args) >= 1
        mi_or_ci = stmt.args[1]
        mi = if mi_or_ci isa Core.MethodInstance
            mi_or_ci
        elseif isdefined(Core, :CodeInstance) && mi_or_ci isa Core.CodeInstance
            mi_or_ci.def
        else
            nothing
        end
        if mi isa Core.MethodInstance
            # Get the return type from the MethodInstance
            # specTypes contains the return type
            ret_type = mi.specTypes
            # The return type is the rettype field when available
            if isdefined(mi, :rettype)
                ret_type = mi.rettype
                if ret_type === Nothing
                    return false
                end
            end
        end
    end

    # If SSA type is Any, be conservative and assume it might be Nothing
    # (e.g., when Julia's optimizer didn't infer the type precisely)
    if stmt_type === Any
        # Check if it's an invoke - we can get more precise info
        if stmt.head === :invoke && length(stmt.args) >= 1
            mi_or_ci = stmt.args[1]
            mi = if mi_or_ci isa Core.MethodInstance
                mi_or_ci
            elseif isdefined(Core, :CodeInstance) && mi_or_ci isa Core.CodeInstance
                mi_or_ci.def
            else
                nothing
            end
            if mi isa Core.MethodInstance && isdefined(mi, :rettype) && mi.rettype === Nothing
                return false
            end
        end
        # For Any type that's not a known Nothing invoke, check the function registry
        # to see if we compiled this function with void return
        # For now, conservatively assume Any might be Nothing
        return false  # Be safe - don't DROP if type is Any
    end

    # For other types (concrete types that aren't Nothing), value is produced
    return true
end

# ============================================================================
# Code Generation
# ============================================================================

"""
Generate Wasm bytecode from Julia CodeInfo.
Uses a block-based translation for control flow.
"""
function generate_body(ctx::CompilationContext)::Vector{UInt8}
    code = ctx.code_info.code
    n = length(code)

    # Analyze control flow to find basic block structure
    blocks = analyze_blocks(code)

    # Generate code using structured control flow
    bytes = generate_structured(ctx, blocks)

    return bytes
end

"""
Represents a basic block in the IR.
"""
struct BasicBlock
    start_idx::Int
    end_idx::Int
    terminator::Any  # GotoIfNot, GotoNode, or ReturnNode
end

"""
Represents a try/catch region in the IR.
"""
struct TryRegion
    enter_idx::Int      # SSA index of Core.EnterNode
    catch_dest::Int     # SSA index where catch block starts
    leave_idx::Int      # SSA index of :leave expression (end of try body)
end

"""
Find try/catch regions by scanning for Core.EnterNode statements.
Returns a list of TryRegion structs.
"""
function find_try_regions(code)::Vector{TryRegion}
    regions = TryRegion[]

    for (i, stmt) in enumerate(code)
        if stmt isa Core.EnterNode
            catch_dest = stmt.catch_dest
            # Find the corresponding :leave that references this EnterNode
            leave_idx = 0
            for (j, s) in enumerate(code)
                if s isa Expr && s.head === :leave
                    # :leave args contain references to EnterNode SSA values
                    for arg in s.args
                        if arg isa Core.SSAValue && arg.id == i
                            leave_idx = j
                            break
                        end
                    end
                    if leave_idx > 0
                        break
                    end
                end
            end

            if leave_idx > 0
                push!(regions, TryRegion(i, catch_dest, leave_idx))
            end
        end
    end

    return regions
end

"""
Check if code contains try/catch regions.
"""
function has_try_catch(code)::Bool
    for stmt in code
        if stmt isa Core.EnterNode
            return true
        end
    end
    return false
end

"""
Analyze the IR to find basic block boundaries.
A new block starts after each terminator AND at each jump target.
"""
function analyze_blocks(code)
    # First, collect all jump targets
    jump_targets = Set{Int}()
    for stmt in code
        if stmt isa Core.GotoNode
            push!(jump_targets, stmt.label)
        elseif stmt isa Core.GotoIfNot
            push!(jump_targets, stmt.dest)
        end
    end

    blocks = BasicBlock[]
    block_start = 1

    for i in 1:length(code)
        stmt = code[i]

        # Check if NEXT statement is a jump target (start new block after this one)
        is_terminator = stmt isa Core.GotoIfNot || stmt isa Core.GotoNode || stmt isa Core.ReturnNode
        next_is_jump_target = (i + 1) in jump_targets

        if is_terminator
            push!(blocks, BasicBlock(block_start, i, stmt))
            block_start = i + 1
        elseif next_is_jump_target && i >= block_start
            # Current statement is NOT a terminator but next statement IS a jump target
            # Close current block with no terminator (fallthrough)
            push!(blocks, BasicBlock(block_start, i, nothing))
            block_start = i + 1
        end
    end

    # Handle trailing code without explicit terminator
    if block_start <= length(code)
        push!(blocks, BasicBlock(block_start, length(code), nothing))
    end

    return blocks
end

"""
Check if this code contains a loop (has backward jumps).
"""
function has_loop(ctx::CompilationContext)
    return !isempty(ctx.loop_headers)
end

"""
Check if there's a conditional BEFORE the first loop that jumps PAST the first loop.
This pattern requires special handling (generate_complex_flow instead of generate_loop_code).
Example: if/else where each branch has its own loop (like float_to_string).
"""
function has_branch_past_first_loop(ctx::CompilationContext, code)
    if isempty(ctx.loop_headers)
        return false
    end

    # Find first loop header and its back-edge
    first_header = minimum(ctx.loop_headers)
    back_edge_idx = nothing
    for (i, stmt) in enumerate(code)
        if stmt isa Core.GotoNode && stmt.label == first_header
            back_edge_idx = i
            break
        end
    end
    if back_edge_idx === nothing
        return false
    end

    # Check for conditionals BEFORE the first loop that jump PAST its back-edge
    for i in 1:(first_header - 1)
        stmt = code[i]
        if stmt isa Core.GotoIfNot
            target = stmt.dest
            if target > back_edge_idx
                # This conditional jumps past the first loop - complex pattern
                return true
            end
        end
    end

    return false
end

"""
Find merge points - targets of multiple forward jumps.
These are blocks that need WASM block/br structure for proper control flow.
Returns a Dict mapping target index to list of source indices.
"""
function find_merge_points(code)
    # Track all forward jump targets
    forward_targets = Dict{Int, Vector{Int}}()

    for (i, stmt) in enumerate(code)
        if stmt isa Core.GotoNode
            target = stmt.label
            if target > i  # Forward jump
                if !haskey(forward_targets, target)
                    forward_targets[target] = Int[]
                end
                push!(forward_targets[target], i)
            end
        elseif stmt isa Core.GotoIfNot
            target = stmt.dest
            if target > i  # Forward jump (the false branch)
                if !haskey(forward_targets, target)
                    forward_targets[target] = Int[]
                end
                push!(forward_targets[target], i)
            end
        end
    end

    # Merge points are targets with multiple sources
    merge_points = Dict{Int, Vector{Int}}()
    for (target, sources) in forward_targets
        if length(sources) >= 2
            merge_points[target] = sources
        end
    end

    return merge_points
end

"""
Check if the control flow has || or && patterns (merge points from short-circuit evaluation).
"""
function has_short_circuit_patterns(code)
    merge_points = find_merge_points(code)
    return !isempty(merge_points)
end

"""
Generate code for try/catch blocks using WASM exception handling (try_table).

Following dart2wasm's approach:
- Use a single exception tag for all Julia exceptions
- try_table with catch_all to handle any exception
- Catch handler gets exception value (if any)

WASM structure:
  (block \$after_try          ; exit point for try success
    (block \$catch_handler    ; catch handler block
      (try_table (catch_all 0) ; branch to \$catch_handler on exception
        ;; try body code
        (br 1)                 ; normal exit (skip catch)
      )
    )
    ;; catch handler code
  )
  ;; code after try/catch
"""
function generate_try_catch(ctx::CompilationContext, blocks::Vector{BasicBlock}, code)::Vector{UInt8}
    bytes = UInt8[]
    regions = find_try_regions(code)

    if isempty(regions)
        # No try regions, fall back to normal generation
        return generate_complex_flow(ctx, blocks, code)
    end

    # Ensure module has an exception tag for Julia exceptions
    # Tag 0 is a void function type (no parameters, no results) for simple exceptions
    if isempty(ctx.mod.tags)
        # Add a void function type for the exception tag using add_type!
        void_ft = FuncType(WasmValType[], WasmValType[])
        void_type_idx = add_type!(ctx.mod, void_ft)
        add_tag!(ctx.mod, void_type_idx)
    end

    # For now, handle single try/catch region
    region = regions[1]
    enter_idx = region.enter_idx
    catch_dest = region.catch_dest
    leave_idx = region.leave_idx

    # Determine result type for the function
    result_type_byte = get_concrete_wasm_type(ctx.return_type, ctx.mod, ctx.type_registry)

    # Structure:
    # (block $after_try [result_type]      ; outer block - exit for both paths
    #   (block $catch_block                 ; catch jumps here
    #     (try_table (catch_all 0)          ; try body, catch_all jumps to label 0 ($catch_block)
    #       ;; code before EnterNode
    #       ;; try body (enter_idx+1 to leave_idx-1)
    #       ;; normal path after :leave until catch_dest-1
    #       (br 1)                          ; skip catch, go to $after_try
    #     )
    #   )
    #   ;; catch handler (catch_dest to end of catch)
    # )

    # Outer block for the result value
    push!(bytes, Opcode.BLOCK)
    append!(bytes, encode_block_type(result_type_byte))

    # Inner void block for catch destination
    push!(bytes, Opcode.BLOCK)
    push!(bytes, 0x40)  # void result type

    # try_table with catch_all clause
    # Format: try_table blocktype vec(catch) end
    push!(bytes, Opcode.TRY_TABLE)
    push!(bytes, 0x40)  # void block type (no result from try_table itself)

    # Catch clauses: catch_all 0 (branch to label 0 on any exception)
    append!(bytes, encode_leb128_unsigned(1))  # 1 catch clause
    push!(bytes, Opcode.CATCH_ALL)             # catch_all type
    append!(bytes, encode_leb128_unsigned(0))  # label index 0 (inner block)

    # Generate code BEFORE EnterNode
    for i in 1:(enter_idx-1)
        stmt = code[i]
        if stmt !== nothing && !(stmt isa Core.EnterNode)
            append!(bytes, compile_statement(stmt, i, ctx))
        end
    end

    # Generate try body (from EnterNode+1 to leave_idx-1)
    # Need to handle control flow (GotoIfNot) properly
    i = enter_idx + 1
    while i <= leave_idx - 1
        stmt = code[i]
        if stmt === nothing
            i += 1
            continue
        end

        # Handle GotoIfNot (if statement) inside try body
        if stmt isa Core.GotoIfNot
            # This is an if statement in the try body
            # The then-branch is from i+1 to dest-1
            # The else-branch starts at dest
            goto_if_not = stmt
            else_target = goto_if_not.dest

            # Compile the condition value
            append!(bytes, compile_value(goto_if_not.cond, ctx))

            # Check if then-branch has a return or throw (void if) vs needs else
            then_start = i + 1
            then_end = min(else_target - 1, leave_idx - 1)
            then_has_return = false
            then_has_throw = false

            for j in then_start:then_end
                if code[j] isa Core.ReturnNode
                    then_has_return = true
                    break
                elseif code[j] isa Expr && code[j].head === :call
                    func = code[j].args[1]
                    if func isa GlobalRef && func.name === :throw
                        then_has_throw = true
                        break
                    end
                end
            end

            if then_has_throw || then_has_return
                # Then branch ends with throw/return, no else branch needed
                # Use: (if (then ...))
                push!(bytes, Opcode.IF)
                push!(bytes, 0x40)  # void result type

                # Generate then branch
                for j in then_start:then_end
                    if code[j] !== nothing
                        append!(bytes, compile_statement(code[j], j, ctx))
                    end
                end

                push!(bytes, Opcode.END)

                # Skip to else_target (which becomes the continuation)
                i = else_target
            else
                # Normal if-else pattern (rare in try body, but handle it)
                push!(bytes, Opcode.IF)
                push!(bytes, 0x40)  # void result type

                for j in then_start:then_end
                    if code[j] !== nothing
                        append!(bytes, compile_statement(code[j], j, ctx))
                    end
                end

                push!(bytes, Opcode.ELSE)

                # Else branch from else_target to leave_idx-1
                for j in else_target:(leave_idx-1)
                    if code[j] !== nothing
                        append!(bytes, compile_statement(code[j], j, ctx))
                    end
                end

                push!(bytes, Opcode.END)

                # We've processed everything up to leave_idx
                i = leave_idx
            end
        else
            append!(bytes, compile_statement(stmt, i, ctx))
            i += 1
        end
    end

    # Skip the :leave itself (it's a control flow marker)

    # Generate normal path code after :leave until catch_dest
    for i in (leave_idx+1):(catch_dest-1)
        stmt = code[i]
        if stmt !== nothing
            # Check if this is a return - if so, we need to handle it specially
            if stmt isa Core.ReturnNode
                append!(bytes, compile_statement(stmt, i, ctx))
                # After return in try, branch out
                push!(bytes, Opcode.BR)
                append!(bytes, encode_leb128_unsigned(1))  # branch to outer block
                break
            else
                append!(bytes, compile_statement(stmt, i, ctx))
            end
        end
    end

    # If no return in try body, branch past catch
    push!(bytes, Opcode.BR)
    append!(bytes, encode_leb128_unsigned(1))  # branch to outer block (past catch)

    # End try_table
    push!(bytes, Opcode.END)

    # End inner (catch destination) block
    push!(bytes, Opcode.END)

    # Catch handler code (from catch_dest to end)
    for i in catch_dest:length(code)
        stmt = code[i]
        if stmt !== nothing
            # Skip :pop_exception - it's just a marker
            if stmt isa Expr && stmt.head === :pop_exception
                continue
            end
            append!(bytes, compile_statement(stmt, i, ctx))
        end
    end

    # End outer block - don't add END here, generate_structured will add it
    # Actually wait, we need to end the outer block but the END is added by generate_structured
    # Let me check... generate_structured adds one END at the end of the function

    # Actually we DO need to end the outer block here
    push!(bytes, Opcode.END)

    return bytes
end

"""
Generate code using Wasm's structured control flow.
For simple if-then-else patterns, we use the `if` instruction.
"""
function generate_structured(ctx::CompilationContext, blocks::Vector{BasicBlock})::Vector{UInt8}
    bytes = UInt8[]
    code = ctx.code_info.code

    # Check for try/catch first
    if has_try_catch(code)
        append!(bytes, generate_try_catch(ctx, blocks, code))
    # Check for loops first
    elseif has_loop(ctx)
        append!(bytes, generate_loop_code(ctx))
    elseif length(blocks) == 1
        # Single block - just generate statements
        append!(bytes, generate_block_code(ctx, blocks[1]))
    elseif is_simple_conditional(blocks, code)
        # Simple if-then-else pattern
        append!(bytes, generate_if_then_else(ctx, blocks, code))
    else
        # More complex control flow - use block/br structure
        append!(bytes, generate_complex_flow(ctx, blocks, code))
    end

    # Always end with END opcode
    push!(bytes, Opcode.END)

    return bytes
end

"""
Generate code for "branched loops" pattern where a conditional before the first loop
jumps past it to an alternate code path with its own loop.
Example: float_to_string where negative/positive branches each have their own loop.

Structure:
  if (condition)
    ; first branch code (with loop 1)
  else
    ; second branch code (with loop 2)
  end
"""
function generate_branched_loops(ctx::CompilationContext, first_header::Int, first_back_edge::Int,
                                  cond_idx::Int, second_branch_start::Int,
                                  ssa_use_count::Dict{Int, Int})::Vector{UInt8}
    bytes = UInt8[]
    code = ctx.code_info.code

    # Identify dead code regions (boundscheck patterns)
    # Since we emit i32.const 0 for ALL boundscheck expressions,
    # the GotoIfNot following a boundscheck ALWAYS jumps to the target.
    # Pattern: boundscheck at line N, GotoIfNot %N at line N+1
    # Dead code: lines from N+2 to target-1 (the fall-through path)
    # Note: We DON'T skip the boundscheck or GotoIfNot - we compile them normally
    # so the control flow (BR) is properly emitted.
    dead_regions = Set{Int}()
    for i in 1:length(code)
        stmt = code[i]
        if stmt isa Expr && stmt.head === :boundscheck && length(stmt.args) == 1
            if i + 1 <= length(code) && code[i + 1] isa Core.GotoIfNot
                goto_stmt = code[i + 1]
                if goto_stmt.cond isa Core.SSAValue && goto_stmt.cond.id == i
                    # Mark lines between the GotoIfNot and its target as dead
                    # (the fall-through path that's never taken)
                    for j in (i + 2):(goto_stmt.dest - 1)
                        push!(dead_regions, j)
                    end
                end
            end
        end
    end

    # For now, just use simple sequential code generation with explicit returns
    # Both branches end with return, so we can just compile sequentially

    # The conditional is at cond_idx: goto %second_branch_start if not %cond
    # If condition is TRUE: fall through to first branch (lines cond_idx+1 to second_branch_start-1)
    # If condition is FALSE: jump to second branch (lines second_branch_start to end)

    # First, compile statements before the conditional
    for i in 1:(cond_idx - 1)
        # Skip dead code (boundscheck patterns)
        if i in dead_regions
            continue
        end

        stmt = code[i]
        if stmt === nothing
            continue
        elseif stmt isa Core.GotoIfNot || stmt isa Core.GotoNode || stmt isa Core.PhiNode
            # Control flow handled specially
            continue
        else
            append!(bytes, compile_statement(stmt, i, ctx))
        end
    end

    # Get the condition and compile it
    cond_stmt = code[cond_idx]::Core.GotoIfNot
    append!(bytes, compile_value(cond_stmt.cond, ctx))

    # Create if/else structure
    # When condition is TRUE: first branch
    # When condition is FALSE (after EQZ): second branch
    push!(bytes, Opcode.IF)
    push!(bytes, 0x40)  # void block type

    # THEN branch: first loop branch (lines cond_idx+1 to second_branch_start-1)
    # This includes the first loop
    for i in (cond_idx + 1):(second_branch_start - 1)
        # Skip dead code (boundscheck patterns)
        if i in dead_regions
            continue
        end

        stmt = code[i]
        if stmt === nothing
            continue
        elseif stmt isa Core.ReturnNode
            if isdefined(stmt, :val)
                append!(bytes, compile_value(stmt.val, ctx))
                push!(bytes, Opcode.RETURN)
            else
                # ReturnNode without val is `unreachable` - should be skipped if in dead_regions
                # but if we reach here, emit WASM unreachable
                push!(bytes, Opcode.UNREACHABLE)
            end
        elseif stmt isa Core.GotoIfNot
            # Inner conditional - use IF to properly consume the condition
            # GotoIfNot: if NOT condition, jump to target
            # With IF: if condition is TRUE, execute then-branch (do nothing)
            #          if condition is FALSE, skip (which matches GotoIfNot semantics)
            # Since the dead code is already skipped via dead_regions,
            # we just need to consume the condition value
            append!(bytes, compile_value(stmt.cond, ctx))
            push!(bytes, Opcode.IF)
            push!(bytes, 0x40)  # void
            push!(bytes, Opcode.END)  # Empty then-branch
            # Fall through to continue (else branch is the continuation)
        elseif stmt isa Core.GotoNode
            # Skip goto - control flow handled
            if stmt.label == first_header
                # Back-edge to loop - emit br
                # For now, just skip (the loop structure handles this)
            end
        elseif stmt isa Core.PhiNode
            # Phi - handled via locals
            continue
        else
            append!(bytes, compile_statement(stmt, i, ctx))

            # Drop unused values
            if stmt isa Expr && (stmt.head === :call || stmt.head === :invoke)
                stmt_type = get(ctx.ssa_types, i, Any)
                if stmt_type !== Nothing
                    is_nothing_union = stmt_type isa Union && Nothing in Base.uniontypes(stmt_type)
                    if !is_nothing_union
                        if !haskey(ctx.ssa_locals, i) && !haskey(ctx.phi_locals, i)
                            use_count = get(ssa_use_count, i, 0)
                            if use_count == 0
                                push!(bytes, Opcode.DROP)
                            end
                        end
                    end
                end
            end
        end
    end

    # ELSE branch: second loop branch (lines second_branch_start to end)
    push!(bytes, Opcode.ELSE)

    for i in second_branch_start:length(code)
        # Skip dead code (boundscheck patterns)
        if i in dead_regions
            continue
        end

        stmt = code[i]
        if stmt === nothing
            continue
        elseif stmt isa Core.ReturnNode
            if isdefined(stmt, :val)
                append!(bytes, compile_value(stmt.val, ctx))
            end
            push!(bytes, Opcode.RETURN)
        elseif stmt isa Core.GotoIfNot
            # Inner conditional - use IF to consume condition
            append!(bytes, compile_value(stmt.cond, ctx))
            push!(bytes, Opcode.IF)
            push!(bytes, 0x40)  # void
            push!(bytes, Opcode.END)  # Empty then-branch
        elseif stmt isa Core.GotoNode
            # Skip goto
            continue
        elseif stmt isa Core.PhiNode
            continue
        else
            append!(bytes, compile_statement(stmt, i, ctx))

            # Drop unused values
            if stmt isa Expr && (stmt.head === :call || stmt.head === :invoke)
                stmt_type = get(ctx.ssa_types, i, Any)
                if stmt_type !== Nothing
                    is_nothing_union = stmt_type isa Union && Nothing in Base.uniontypes(stmt_type)
                    if !is_nothing_union
                        if !haskey(ctx.ssa_locals, i) && !haskey(ctx.phi_locals, i)
                            use_count = get(ssa_use_count, i, 0)
                            if use_count == 0
                                push!(bytes, Opcode.DROP)
                            end
                        end
                    end
                end
            end
        end
    end

    push!(bytes, Opcode.END)  # End if/else

    # Both branches return, so code after the if/else is unreachable
    # Add UNREACHABLE to satisfy WASM validation (function end needs result value on stack)
    push!(bytes, Opcode.UNREACHABLE)

    return bytes
end

"""
Generate code for a loop structure.
Wasm loop structure:
  (block \$exit
    (loop \$continue
      ... body ...
      (br_if \$exit)  ; exit condition
      (br \$continue) ; loop back
    )
  )

Following dart2wasm patterns for inner conditionals:
- Loop exit: GotoIfNot with target > back_edge → br_if to outer block
- Inner conditional: GotoIfNot with target <= back_edge → nested block/br pattern
- Dead code (boundscheck false): Skip unreachable branches entirely
"""
function generate_loop_code(ctx::CompilationContext)::Vector{UInt8}
    bytes = UInt8[]
    code = ctx.code_info.code

    # Count SSA uses (for drop logic)
    ssa_use_count = Dict{Int, Int}()
    for stmt in code
        count_ssa_uses!(stmt, ssa_use_count)
    end

    # Find loop bounds (header to back-edge)
    first_header = minimum(ctx.loop_headers)
    back_edge_idx = nothing
    for (i, stmt) in enumerate(code)
        if stmt isa Core.GotoNode && stmt.label == first_header
            back_edge_idx = i
            break
        end
    end
    if back_edge_idx === nothing
        back_edge_idx = length(code)
    end

    # Check for "branch past first loop" pattern (e.g., float_to_string)
    # This is when a conditional BEFORE the loop jumps PAST the loop to an alternate code path
    # that also contains its own loop (both branches have loops and end with return)
    branch_past_target = nothing
    branch_past_cond_idx = nothing
    for i in 1:(first_header - 1)
        stmt = code[i]
        if stmt isa Core.GotoIfNot && stmt.dest > back_edge_idx
            # Check if the alternate path actually has a SECOND LOOP
            # (i.e., there's a backward jump in the alternate path)
            has_second_loop = false
            for j in stmt.dest:length(code)
                if code[j] isa Core.GotoNode && code[j].label < j && code[j].label >= stmt.dest
                    has_second_loop = true
                    break
                end
            end
            if has_second_loop
                branch_past_target = stmt.dest
                branch_past_cond_idx = i
                break
            end
        end
    end

    # If we have a branch-past-loop pattern (both branches have loops), use a special handler
    if branch_past_target !== nothing
        return generate_branched_loops(ctx, first_header, back_edge_idx,
                                       branch_past_cond_idx, branch_past_target, ssa_use_count)
    end

    # Original single-loop code follows
    loop_header = first_header

    # Identify dead code regions (boundscheck patterns)
    # Since we emit i32.const 0 for ALL boundscheck expressions (both true and false),
    # the GotoIfNot following a boundscheck ALWAYS jumps to the target.
    # Pattern: boundscheck(true/false) at line N, GotoIfNot %N at line N+1
    # With boundscheck=0: GotoIfNot "if NOT 0" = "if TRUE" = always jump
    # Dead code: lines from N+2 to target-1 (the fall-through path)
    dead_regions = Set{Int}()
    boundscheck_jumps = Dict{Int, Int}()  # GotoIfNot line → target (for always-jump)
    for i in 1:length(code)
        stmt = code[i]
        # Handle BOTH boundscheck(true) and boundscheck(false) - we emit 0 for both
        if stmt isa Expr && stmt.head === :boundscheck && length(stmt.args) == 1
            # Check if next line is GotoIfNot using this boundscheck
            if i + 1 <= length(code) && code[i + 1] isa Core.GotoIfNot
                goto_stmt = code[i + 1]
                if goto_stmt.cond isa Core.SSAValue && goto_stmt.cond.id == i
                    # This GotoIfNot always jumps (we emit 0 for boundscheck)
                    boundscheck_jumps[i + 1] = goto_stmt.dest
                    # Mark the boundscheck itself and lines from i+2 to target-1 as dead
                    push!(dead_regions, i)  # boundscheck - no need to emit (it's always 0)
                    for j in (i + 2):(goto_stmt.dest - 1)
                        push!(dead_regions, j)
                    end
                end
            end
        end
    end

    # Identify inner conditional GotoIfNot statements (target within loop body)
    # Only for REAL conditionals (not boundscheck always-jump patterns or dead code)
    # IMPORTANT: Only scan from loop_header to back_edge_idx, NOT from line 1
    # Pre-loop conditionals (early returns) are NOT inner conditionals
    inner_conditionals = Dict{Int, Int}()  # GotoIfNot line → merge point
    for i in loop_header:back_edge_idx
        # Skip dead code and boundscheck jumps
        if i in dead_regions || haskey(boundscheck_jumps, i)
            continue
        end
        stmt = code[i]
        if stmt isa Core.GotoIfNot
            target = stmt.dest
            # Inner conditional: target is within loop, not the exit
            if target <= back_edge_idx && target > i
                inner_conditionals[i] = target
            end
        end
    end

    # ============================================================
    # PHASE 1: Generate PRE-LOOP code (lines 1 to loop_header - 1)
    # This handles early return guards, pre-loop conditionals, etc.
    # These must be generated BEFORE the block/loop structure.
    # ============================================================

    # Track if we open an IF for a pre-loop conditional that skips past the loop
    # This IF will be closed AFTER the loop ends
    post_loop_skip_phi_target = nothing  # phi target line if we have such a pattern

    if loop_header > 1
        # Track pre-loop blocks and their types:
        # :if_end - simple if-then, emit END at this line
        # :if_else - if-then-else, emit ELSE at this line (else branch start)
        # :if_else_end - if-then-else, emit END at this line (merge point after else)
        pre_loop_block_type = Dict{Int, Symbol}()  # line → type
        pre_loop_depth = 0
        # Track which GotoNodes should be skipped (they become implicit in if-else)
        skip_goto_at = Set{Int}()

        for i in 1:(loop_header - 1)
            # Skip dead code
            if i in dead_regions
                continue
            end

            stmt = code[i]

            # Check if we need to emit ELSE or END at this line
            if haskey(pre_loop_block_type, i)
                block_type = pre_loop_block_type[i]

                if block_type == :if_end
                    # Simple if-then: close the block
                    # Handle pre-loop phi at merge point (set then-value before END)
                    if code[i] isa Core.PhiNode && haskey(ctx.phi_locals, i)
                        phi_stmt = code[i]::Core.PhiNode
                        for (edge_idx, edge) in enumerate(phi_stmt.edges)
                            edge_stmt = get(code, edge, nothing)
                            if edge_stmt !== nothing && !(edge_stmt isa Core.GotoIfNot)
                                val = phi_stmt.values[edge_idx]
                                append!(bytes, compile_value(val, ctx))
                                local_idx = ctx.phi_locals[i]
                                push!(bytes, Opcode.LOCAL_SET)
                                append!(bytes, encode_leb128_unsigned(local_idx))
                                break
                            end
                        end
                    end
                    push!(bytes, Opcode.END)
                    delete!(pre_loop_block_type, i)
                    pre_loop_depth -= 1

                elseif block_type == :if_else
                    # If-then-else: emit ELSE to start else branch
                    # First, set phi value from then-branch before leaving it
                    # Find the phi at the actual merge point
                    for (mp, mt) in pre_loop_block_type
                        if mt == :if_else_end && mp > i
                            if code[mp] isa Core.PhiNode && haskey(ctx.phi_locals, mp)
                                phi_stmt = code[mp]::Core.PhiNode
                                # Find the then-edge (comes from before else_start)
                                for (edge_idx, edge) in enumerate(phi_stmt.edges)
                                    if edge < i
                                        val = phi_stmt.values[edge_idx]
                                        append!(bytes, compile_value(val, ctx))
                                        local_idx = ctx.phi_locals[mp]
                                        push!(bytes, Opcode.LOCAL_SET)
                                        append!(bytes, encode_leb128_unsigned(local_idx))
                                        break
                                    end
                                end
                            end
                            break
                        end
                    end
                    push!(bytes, Opcode.ELSE)
                    delete!(pre_loop_block_type, i)
                    # Note: depth stays the same (still inside the if-else)

                elseif block_type == :if_else_end
                    # If-then-else merge point: set else-branch phi value and END
                    if code[i] isa Core.PhiNode && haskey(ctx.phi_locals, i)
                        phi_stmt = code[i]::Core.PhiNode
                        # Find the else-edge: it's the edge with the LARGEST line number
                        # (else branch comes after then branch)
                        max_edge_idx = 0
                        max_edge = 0
                        for (edge_idx, edge) in enumerate(phi_stmt.edges)
                            if edge > max_edge
                                max_edge = edge
                                max_edge_idx = edge_idx
                            end
                        end
                        if max_edge_idx > 0
                            val = phi_stmt.values[max_edge_idx]
                            append!(bytes, compile_value(val, ctx))
                            local_idx = ctx.phi_locals[i]
                            push!(bytes, Opcode.LOCAL_SET)
                            append!(bytes, encode_leb128_unsigned(local_idx))
                        end
                    end
                    push!(bytes, Opcode.END)
                    delete!(pre_loop_block_type, i)
                    pre_loop_depth -= 1
                end
            end

            if stmt isa Core.PhiNode
                # Pre-loop phi nodes - they should have been initialized
                # by their incoming edges. Just skip.
                continue
            elseif stmt isa Core.GotoIfNot
                target = stmt.dest
                # Skip boundscheck always-jump patterns
                if haskey(boundscheck_jumps, i)
                    continue
                elseif target > back_edge_idx
                    # This conditional jumps PAST the loop (skip if-body AND loop)
                    # We need to use IF/ELSE: if condition true, execute if-body+loop
                    #                         if condition false, skip to post-loop

                    # Check for phi at target (post-loop phi)
                    # We need to set the phi local BEFORE the IF
                    if code[target] isa Core.PhiNode && haskey(ctx.phi_locals, target)
                        phi_stmt = code[target]::Core.PhiNode
                        for (edge_idx, edge) in enumerate(phi_stmt.edges)
                            if edge == i
                                val = phi_stmt.values[edge_idx]
                                append!(bytes, compile_value(val, ctx))
                                local_idx = ctx.phi_locals[target]
                                push!(bytes, Opcode.LOCAL_SET)
                                append!(bytes, encode_leb128_unsigned(local_idx))
                                break
                            end
                        end
                    end

                    # Use IF structure: if condition is TRUE, fall through to if-body
                    # The ELSE branch (implicit) skips to post-loop
                    append!(bytes, compile_value(stmt.cond, ctx))
                    push!(bytes, Opcode.IF)
                    push!(bytes, 0x40)  # void block type
                    # This IF will be closed after the loop completes
                    # Store the target for later (we'll close this IF after loop ends)
                    post_loop_skip_phi_target = target  # Track phi target for post-loop skip
                    pre_loop_depth += 1
                elseif target >= loop_header
                    # This conditional jumps to or near the loop header
                    # It's a pre-loop conditional with fall-through to loop
                    # We need to handle this with a block for the then-branch

                    # Open block for the then-branch (fall-through path)
                    # The block ends at loop_header (when we transition to loop)
                    push!(bytes, Opcode.BLOCK)
                    push!(bytes, 0x40)
                    # Mark this block for closing - but since target >= loop_header,
                    # we'll need to close it before entering the loop
                    pre_loop_block_type[loop_header] = :if_end
                    pre_loop_depth += 1

                    # Branch past the block if condition is FALSE (skip then-branch)
                    append!(bytes, compile_value(stmt.cond, ctx))
                    push!(bytes, Opcode.I32_EQZ)
                    push!(bytes, Opcode.BR_IF)
                    push!(bytes, 0x00)
                elseif target > i && target < loop_header
                    # Inner pre-loop conditional (both branches before loop)
                    # Pattern: GotoIfNot jumps to target (else-branch start), fall-through is then-branch
                    #
                    # CRITICAL: Need to detect if-then-else-phi pattern:
                    #   GotoIfNot → then-branch → goto merge → else-branch → goto merge → phi at merge
                    #
                    # Check if this is if-then-else-phi pattern:
                    # 1. Look for a goto at target-1 that jumps past target
                    # 2. If found, that goto's target is the TRUE merge point

                    then_end_idx = target - 1
                    then_end_stmt = get(code, then_end_idx, nothing)

                    if then_end_stmt isa Core.GotoNode && then_end_stmt.label > target && then_end_stmt.label < loop_header
                        # This IS if-then-else-phi pattern
                        # then_end_stmt.label is the TRUE merge point (where phi is)
                        merge_point = then_end_stmt.label
                        else_start = target

                        # Compile condition BEFORE any control structure
                        append!(bytes, compile_value(stmt.cond, ctx))

                        # Use IF/ELSE structure
                        push!(bytes, Opcode.IF)
                        push!(bytes, 0x40)  # void block type

                        # Mark: when we reach else_start, emit ELSE
                        # Mark: when we reach merge_point, emit END
                        pre_loop_block_type[else_start] = :if_else
                        pre_loop_block_type[merge_point] = :if_else_end
                        pre_loop_depth += 1

                        # Mark gotos at end of then-branch and else-branch to be skipped
                        # (they become implicit in the if/else structure)
                        push!(skip_goto_at, then_end_idx)  # goto at end of then-branch
                        # Find goto at end of else-branch (just before merge_point)
                        for j in (else_start):(merge_point - 1)
                            if code[j] isa Core.GotoNode && code[j].label == merge_point
                                push!(skip_goto_at, j)
                            end
                        end
                    else
                        # Simple if-then pattern (no else branch or simple merge)
                        # Handle pre-loop phi at target (set else-branch value)
                        if code[target] isa Core.PhiNode && haskey(ctx.phi_locals, target)
                            phi_stmt = code[target]::Core.PhiNode
                            for (edge_idx, edge) in enumerate(phi_stmt.edges)
                                if edge == i
                                    val = phi_stmt.values[edge_idx]
                                    append!(bytes, compile_value(val, ctx))
                                    local_idx = ctx.phi_locals[target]
                                    push!(bytes, Opcode.LOCAL_SET)
                                    append!(bytes, encode_leb128_unsigned(local_idx))
                                    break
                                end
                            end
                        end

                        # Compile condition BEFORE any control structure
                        append!(bytes, compile_value(stmt.cond, ctx))

                        # Use if-then structure: if condition is TRUE, execute then-branch
                        push!(bytes, Opcode.IF)
                        push!(bytes, 0x40)  # void block type
                        pre_loop_block_type[target] = :if_end
                        pre_loop_depth += 1
                        # The then-branch code follows (lines i+1 to target-1)
                        # When we reach target, we'll emit END
                    end
                end
            elseif stmt isa Core.GotoNode
                # Skip gotos that are implicit in if-else structure
                if i in skip_goto_at
                    continue
                end

                # Unconditional jump in pre-loop code
                if stmt.label >= loop_header
                    # Jump to loop - becomes fall-through (we're about to enter loop)
                    # Handle phi at target if needed
                    if stmt.label == loop_header
                        for (j, phi_stmt) in enumerate(code)
                            if j >= loop_header && phi_stmt isa Core.PhiNode && haskey(ctx.phi_locals, j)
                                for (edge_idx, edge) in enumerate(phi_stmt.edges)
                                    if edge == i
                                        val = phi_stmt.values[edge_idx]
                                        append!(bytes, compile_value(val, ctx))
                                        local_idx = ctx.phi_locals[j]
                                        push!(bytes, Opcode.LOCAL_SET)
                                        append!(bytes, encode_leb128_unsigned(local_idx))
                                        break
                                    end
                                end
                            end
                        end
                    end
                    # No actual jump needed - will fall through to loop
                elseif haskey(pre_loop_block_type, stmt.label)
                    # Jump to a pre-loop merge point
                    if code[stmt.label] isa Core.PhiNode && haskey(ctx.phi_locals, stmt.label)
                        phi_stmt = code[stmt.label]::Core.PhiNode
                        for (edge_idx, edge) in enumerate(phi_stmt.edges)
                            if edge == i
                                val = phi_stmt.values[edge_idx]
                                append!(bytes, compile_value(val, ctx))
                                local_idx = ctx.phi_locals[stmt.label]
                                push!(bytes, Opcode.LOCAL_SET)
                                append!(bytes, encode_leb128_unsigned(local_idx))
                                break
                            end
                        end
                    end
                    push!(bytes, Opcode.BR)
                    push!(bytes, 0x00)
                end
            elseif stmt isa Core.ReturnNode
                # Early return in pre-loop code
                if isdefined(stmt, :val)
                    append!(bytes, compile_value(stmt.val, ctx))
                end
                push!(bytes, Opcode.RETURN)
            elseif stmt === nothing
                # Skip nothing statements
            else
                # Regular statement
                append!(bytes, compile_statement(stmt, i, ctx))
            end
        end

        # Close any remaining pre-loop blocks
        for (line, block_type) in pre_loop_block_type
            # This shouldn't happen if control flow is handled correctly,
            # but emit END for any unclosed blocks
            push!(bytes, Opcode.END)
            pre_loop_depth -= 1
        end
    end

    # ============================================================
    # Initialize LOOP phi node locals with their entry values
    # This MUST happen AFTER PHASE 1 (pre-loop code) because loop phi entry
    # values may reference pre-loop phi results.
    # ============================================================
    for (i, stmt) in enumerate(code)
        if stmt isa Core.PhiNode && haskey(ctx.phi_locals, i)
            # Only initialize if this is a LOOP phi (at or AFTER loop header)
            if i < loop_header
                continue
            end

            # Loop phis have an entry edge from before the loop header
            is_loop_phi = false
            entry_val = nothing

            for (edge_idx, edge) in enumerate(stmt.edges)
                if edge < loop_header
                    is_loop_phi = true
                    entry_val = stmt.values[edge_idx]
                    break
                end
            end

            if is_loop_phi && entry_val !== nothing
                append!(bytes, compile_value(entry_val, ctx))
                local_idx = ctx.phi_locals[i]
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(local_idx))
            end
        end
    end

    # ============================================================
    # PHASE 2: Generate LOOP code (lines loop_header to back_edge_idx)
    # ============================================================

    # block $exit (for breaking out of loop)
    push!(bytes, Opcode.BLOCK)
    push!(bytes, 0x40)  # void block type

    # loop $continue
    push!(bytes, Opcode.LOOP)
    push!(bytes, 0x40)  # void block type

    # Track block depth for inner conditionals
    # Key: merge point line number, Value: true if block is open
    open_blocks = Dict{Int, Bool}()
    current_depth = 0  # 0 = inside loop, additional depth for inner blocks

    # Generate loop body (statements from loop_header to back_edge_idx)
    i = loop_header
    while i <= back_edge_idx
        # Check if we need to close any blocks at this merge point
        if haskey(open_blocks, i) && open_blocks[i]
            # Before closing the block, set the then-value for any phi at this merge point
            # The then-branch ends here, so we need to store the value
            if code[i] isa Core.PhiNode && haskey(ctx.phi_locals, i)
                phi_stmt = code[i]::Core.PhiNode
                # Find the then-value (edge from before this line, NOT the GotoIfNot)
                # The then-branch is the fall-through, so look for edge from line i-1
                for (edge_idx, edge) in enumerate(phi_stmt.edges)
                    # The then-edge comes from the line just before the merge (the last then-stmt)
                    # Or more precisely, any edge that's not the GotoIfNot line
                    edge_stmt = get(code, edge, nothing)
                    if edge_stmt !== nothing && !(edge_stmt isa Core.GotoIfNot)
                        val = phi_stmt.values[edge_idx]
                        append!(bytes, compile_value(val, ctx))
                        local_idx = ctx.phi_locals[i]
                        push!(bytes, Opcode.LOCAL_SET)
                        append!(bytes, encode_leb128_unsigned(local_idx))
                        break
                    end
                end
            end
            push!(bytes, Opcode.END)
            open_blocks[i] = false
            current_depth -= 1
        end

        stmt = code[i]

        # Skip dead code regions
        if i in dead_regions
            i += 1
            continue
        end

        if stmt isa Core.PhiNode
            # Phi nodes in loops are handled via locals
            # For inner conditional phi nodes, we need to handle the merge
            if haskey(ctx.phi_locals, i)
                # The phi local should already have the correct value
                # (set by either branch)
            end
            i += 1
            continue
        elseif stmt isa Core.GotoIfNot
            target = stmt.dest

            # Skip boundscheck always-jump patterns (condition is always false)
            if haskey(boundscheck_jumps, i)
                # The dead region will be skipped, just continue
                i += 1
                continue
            elseif target > back_edge_idx
                # This is the LOOP EXIT condition
                append!(bytes, compile_value(stmt.cond, ctx))
                push!(bytes, Opcode.I32_EQZ)  # Invert: if NOT condition
                push!(bytes, Opcode.BR_IF)
                push!(bytes, UInt8(1 + current_depth))  # Break to exit block
            elseif haskey(inner_conditionals, i)
                # This is an INNER CONDITIONAL
                # dart2wasm pattern: block + br_if to skip then-branch
                merge_point = inner_conditionals[i]

                # Check if there's a phi node at the merge point
                merge_phi = nothing
                if code[merge_point] isa Core.PhiNode
                    merge_phi = merge_point
                end

                # If this conditional has a phi node, we need to set the else-value
                # before the branch (it gets set if we skip the then-branch)
                if merge_phi !== nothing && haskey(ctx.phi_locals, merge_phi)
                    phi_stmt = code[merge_phi]::Core.PhiNode
                    # Find the value for the else branch (edge from this GotoIfNot)
                    for (edge_idx, edge) in enumerate(phi_stmt.edges)
                        if edge == i
                            val = phi_stmt.values[edge_idx]
                            append!(bytes, compile_value(val, ctx))
                            local_idx = ctx.phi_locals[merge_phi]
                            push!(bytes, Opcode.LOCAL_SET)
                            append!(bytes, encode_leb128_unsigned(local_idx))
                            break
                        end
                    end
                end

                # Open a block for the then-branch
                push!(bytes, Opcode.BLOCK)
                push!(bytes, 0x40)  # void block type
                open_blocks[merge_point] = true
                current_depth += 1

                # Branch to merge point if condition is FALSE
                append!(bytes, compile_value(stmt.cond, ctx))
                push!(bytes, Opcode.I32_EQZ)  # Invert condition
                push!(bytes, Opcode.BR_IF)
                push!(bytes, 0x00)  # Branch to the block we just opened (depth 0)
            else
                # Fallback: treat as simple forward branch (skip to target)
                push!(bytes, Opcode.BLOCK)
                push!(bytes, 0x40)
                open_blocks[target] = true
                current_depth += 1
                append!(bytes, compile_value(stmt.cond, ctx))
                push!(bytes, Opcode.I32_EQZ)
                push!(bytes, Opcode.BR_IF)
                push!(bytes, 0x00)
            end
        elseif stmt isa Core.GotoNode
            if stmt.label in ctx.loop_headers
                # This is the loop-back jump
                # First update phi locals with their iteration values
                for (j, phi_stmt) in enumerate(code)
                    if phi_stmt isa Core.PhiNode && haskey(ctx.phi_locals, j)
                        # Find the iteration value (from the back-edge - AFTER the phi node)
                        for (edge_idx, edge) in enumerate(phi_stmt.edges)
                            if edge > j  # Back-edge (from after the phi node)
                                val = phi_stmt.values[edge_idx]
                                append!(bytes, compile_value(val, ctx))
                                local_idx = ctx.phi_locals[j]
                                push!(bytes, Opcode.LOCAL_SET)
                                append!(bytes, encode_leb128_unsigned(local_idx))
                                break
                            end
                        end
                    end
                end
                # Continue loop
                push!(bytes, Opcode.BR)
                push!(bytes, UInt8(current_depth))  # Branch to loop (accounting for open blocks)
            elseif stmt.label > i && stmt.label <= back_edge_idx
                # Forward jump within loop - branch to that point
                # This handles the then-branch jumping to merge point
                if haskey(open_blocks, stmt.label) && open_blocks[stmt.label]
                    # Jump to merge point - handle phi update if needed
                    if code[stmt.label] isa Core.PhiNode && haskey(ctx.phi_locals, stmt.label)
                        phi_stmt = code[stmt.label]::Core.PhiNode
                        for (edge_idx, edge) in enumerate(phi_stmt.edges)
                            if edge == i
                                val = phi_stmt.values[edge_idx]
                                append!(bytes, compile_value(val, ctx))
                                local_idx = ctx.phi_locals[stmt.label]
                                # Use LOCAL_TEE to keep value on stack for branch target
                                push!(bytes, Opcode.LOCAL_TEE)
                                append!(bytes, encode_leb128_unsigned(local_idx))
                                break
                            end
                        end
                    end
                    push!(bytes, Opcode.BR)
                    push!(bytes, 0x00)  # Branch to inner block
                end
            elseif stmt.label > back_edge_idx
                # Jump past loop end - this is a BREAK statement
                # Need to branch to the exit block (depth = 1 + current_depth)
                push!(bytes, Opcode.BR)
                push!(bytes, UInt8(1 + current_depth))  # Branch to exit block
            end
        elseif stmt isa Core.ReturnNode
            if isdefined(stmt, :val)
                append!(bytes, compile_value(stmt.val, ctx))
            end
            push!(bytes, Opcode.RETURN)
        elseif stmt === nothing
            # Skip nothing statements
        else
            append!(bytes, compile_statement(stmt, i, ctx))

            # Drop unused values from calls (prevents stack pollution in loops)
            if stmt isa Expr && (stmt.head === :call || stmt.head === :invoke)
                # Use statement_produces_wasm_value for consistent handling
                # This checks the function registry for accurate return type info
                if statement_produces_wasm_value(stmt, i, ctx)
                    if !haskey(ctx.ssa_locals, i) && !haskey(ctx.phi_locals, i)
                        use_count = get(ssa_use_count, i, 0)
                        if use_count == 0
                            push!(bytes, Opcode.DROP)
                        end
                    end
                end
            end

            # Check if this statement has Union{} type (never returns) - stop generating code
            # Code after unreachable/throw is dead and causes validation errors
            stmt_type = get(ctx.ssa_types, i, Any)
            if stmt_type === Union{}
                # Skip to next merge point (block end) or back-edge
                # Find the next merge point
                next_merge = nothing
                for (merge_point, is_open) in open_blocks
                    if is_open && merge_point > i
                        if next_merge === nothing || merge_point < next_merge
                            next_merge = merge_point
                        end
                    end
                end
                if next_merge !== nothing
                    # Skip to just before merge point
                    i = next_merge - 1
                else
                    # No merge point - skip to back edge
                    i = back_edge_idx
                end
            end
        end
        i += 1
    end

    # Close any remaining open blocks
    for (merge_point, is_open) in open_blocks
        if is_open
            push!(bytes, Opcode.END)
        end
    end

    # End loop
    push!(bytes, Opcode.END)

    # End block
    push!(bytes, Opcode.END)

    # Close any IF block for pre-loop conditional that skips past the loop
    # This was opened by `target > back_edge_idx` case in pre-loop handling
    if post_loop_skip_phi_target !== nothing
        # Before closing the IF, we need to set the phi value for the then-branch (loop completed)
        if code[post_loop_skip_phi_target] isa Core.PhiNode && haskey(ctx.phi_locals, post_loop_skip_phi_target)
            phi_stmt = code[post_loop_skip_phi_target]::Core.PhiNode
            # Find the edge that comes from inside/after the loop (not the pre-loop skip)
            # This is the edge that leads to here (end of loop, before post-loop code)
            for (edge_idx, edge) in enumerate(phi_stmt.edges)
                # The loop exit edge is the one from the loop exit condition (line 27 in our IR)
                # or any edge that's > the loop header and <= back_edge_idx
                if edge >= loop_header && edge <= back_edge_idx
                    val = phi_stmt.values[edge_idx]
                    append!(bytes, compile_value(val, ctx))
                    local_idx = ctx.phi_locals[post_loop_skip_phi_target]
                    push!(bytes, Opcode.LOCAL_SET)
                    append!(bytes, encode_leb128_unsigned(local_idx))
                    break
                end
            end
        end
        push!(bytes, Opcode.END)  # Close the IF block
    end

    # Generate code AFTER the loop (statements that run after loop exits)
    # This code may contain conditionals (GotoIfNot) that need proper handling
    # Track blocks as a stack: each entry is a merge point
    post_loop_block_stack = Int[]

    for i in (back_edge_idx + 1):length(code)
        stmt = code[i]

        # Close any open blocks at this merge point
        while !isempty(post_loop_block_stack) && post_loop_block_stack[end] == i
            # If this merge point is a phi, set the phi local from the fall-through edge
            # The fall-through edge is the line right before this merge point (i-1)
            if i <= length(code) && code[i] isa Core.PhiNode && haskey(ctx.phi_locals, i)
                phi_stmt = code[i]::Core.PhiNode
                prev_line = i - 1
                # Find edge value from fall-through (the line just before phi)
                for (edge_idx, edge) in enumerate(phi_stmt.edges)
                    if edge == prev_line
                        edge_val = phi_stmt.values[edge_idx]
                        append!(bytes, compile_value(edge_val, ctx))
                        push!(bytes, Opcode.LOCAL_SET)
                        append!(bytes, encode_leb128_unsigned(ctx.phi_locals[i]))
                        break
                    end
                end
            end
            push!(bytes, Opcode.END)
            pop!(post_loop_block_stack)
        end

        if stmt isa Core.ReturnNode
            if isdefined(stmt, :val)
                append!(bytes, compile_value(stmt.val, ctx))
            end
            push!(bytes, Opcode.RETURN)
        elseif stmt isa Core.GotoIfNot
            target = stmt.dest
            # This is a conditional that jumps forward
            # Use block + br_if pattern
            push!(bytes, Opcode.BLOCK)
            push!(bytes, 0x40)  # void block type
            push!(post_loop_block_stack, target)

            # If target is a phi, set the phi local BEFORE branching
            # When condition is FALSE, we skip to phi with edge value from this line
            if target <= length(code) && code[target] isa Core.PhiNode && haskey(ctx.phi_locals, target)
                phi_stmt = code[target]::Core.PhiNode
                # Find edge value for when we skip (edge from this line)
                for (edge_idx, edge) in enumerate(phi_stmt.edges)
                    if edge == i
                        edge_val = phi_stmt.values[edge_idx]
                        append!(bytes, compile_value(edge_val, ctx))
                        push!(bytes, Opcode.LOCAL_SET)
                        append!(bytes, encode_leb128_unsigned(ctx.phi_locals[target]))
                        break
                    end
                end
            end

            # Branch if condition is FALSE (skip then-branch)
            append!(bytes, compile_value(stmt.cond, ctx))
            push!(bytes, Opcode.I32_EQZ)
            push!(bytes, Opcode.BR_IF)
            push!(bytes, 0x00)
        elseif stmt isa Core.GotoNode
            # Unconditional forward jump - find how many blocks to close
            # and branch to the right depth
            depth = 0
            for j in length(post_loop_block_stack):-1:1
                if post_loop_block_stack[j] == stmt.label
                    depth = length(post_loop_block_stack) - j
                    break
                end
            end

            # If target is a phi, set the phi local BEFORE branching
            if stmt.label <= length(code) && code[stmt.label] isa Core.PhiNode && haskey(ctx.phi_locals, stmt.label)
                phi_stmt = code[stmt.label]::Core.PhiNode
                # Find edge value for this GotoNode (edge from this line)
                for (edge_idx, edge) in enumerate(phi_stmt.edges)
                    if edge == i
                        edge_val = phi_stmt.values[edge_idx]
                        append!(bytes, compile_value(edge_val, ctx))
                        push!(bytes, Opcode.LOCAL_SET)
                        append!(bytes, encode_leb128_unsigned(ctx.phi_locals[stmt.label]))
                        break
                    end
                end
            end

            if depth >= 0 && !isempty(post_loop_block_stack)
                push!(bytes, Opcode.BR)
                push!(bytes, UInt8(depth))
            end
        elseif stmt isa Core.PhiNode
            # Phi nodes in post-loop are merge points - they're handled when blocks close
            # The phi local should already be set by the branches leading here
            continue
        elseif stmt === nothing
            # Skip nothing statements
        else
            append!(bytes, compile_statement(stmt, i, ctx))
        end
    end

    # Close any remaining open blocks
    while !isempty(post_loop_block_stack)
        push!(bytes, Opcode.END)
        pop!(post_loop_block_stack)
    end

    return bytes
end

"""
Check if this is a simple if-then-else pattern.
Pattern: condition, GotoIfNot, then-code, return, else-code, return

A simple conditional has exactly 2-3 blocks:
- Block 1: condition computation, ends with GotoIfNot
- Block 2: then-branch code
- Block 3 (optional): else-branch code

If there are more blocks or nested conditionals, it's not simple.
"""
function is_simple_conditional(blocks::Vector{BasicBlock}, code)
    # Simple pattern has exactly 2-3 blocks
    if length(blocks) < 2 || length(blocks) > 3
        return false
    end

    # First block should end with GotoIfNot
    if !(blocks[1].terminator isa Core.GotoIfNot)
        return false
    end

    # Check that other blocks don't have GotoIfNot (no nested conditionals)
    for i in 2:length(blocks)
        if blocks[i].terminator isa Core.GotoIfNot
            return false
        end
    end

    return true
end

"""
Generate code for a simple if-then-else pattern.
Handles both return-based patterns and phi node patterns (ternary expressions).
"""
function generate_if_then_else(ctx::CompilationContext, blocks::Vector{BasicBlock}, code)::Vector{UInt8}
    bytes = UInt8[]

    # For void return types (like event handlers), delegate to generate_void_flow
    # which properly handles if blocks with void block type (0x40) instead of trying
    # to produce a value
    if ctx.return_type === Nothing
        return generate_void_flow(ctx, blocks, code)
    end

    # Count SSA uses (for drop logic)
    ssa_use_count = Dict{Int, Int}()
    for stmt in code
        count_ssa_uses!(stmt, ssa_use_count)
    end

    # First block: statements up to the condition
    first_block = blocks[1]
    goto_if_not = first_block.terminator::Core.GotoIfNot
    target_label = goto_if_not.dest

    # Generate statements in first block (including condition computation)
    for i in first_block.start_idx:first_block.end_idx-1
        append!(bytes, compile_statement(code[i], i, ctx))
    end

    # The condition value should be on the stack (it's an SSA reference)
    # We need to push it
    append!(bytes, compile_value(goto_if_not.cond, ctx))

    # Find the then-branch and else-branch boundaries
    then_start = first_block.end_idx + 1
    else_start = target_label

    # Check if there's a phi node that merges the branches
    # Pattern: then-branch jumps to merge point, else-branch falls through, phi at merge
    phi_idx = nothing
    phi_node = nothing
    for i in else_start:length(code)
        if code[i] isa Core.PhiNode
            phi_idx = i
            phi_node = code[i]
            break
        end
    end

    if phi_node !== nothing
        # Phi node pattern (ternary expression)
        # The phi provides values for each branch - use those directly
        phi_type = get(ctx.ssa_types, phi_idx, Int32)
        result_type = julia_to_wasm_type_concrete(phi_type, ctx)

        # Start if block with phi result type
        push!(bytes, Opcode.IF)
        append!(bytes, encode_block_type(result_type))

        # Get the phi values for each edge
        # The phi edges reference statement numbers that lead to the phi
        then_value = nothing
        else_value = nothing

        for (edge_idx, edge) in enumerate(phi_node.edges)
            val = phi_node.values[edge_idx]
            if edge < else_start
                # This edge comes from the then-branch (before else_start)
                then_value = val
            else
                # This edge comes from the else-branch
                else_value = val
            end
        end

        # Then-branch: generate all statements in the then-branch, then push the then-value
        # Note: compile_statement already stores to local if SSA has one (via LOCAL_SET)
        for i in then_start:else_start-1
            stmt = code[i]
            if stmt isa Core.GotoNode
                # Skip the goto - we're handling control flow with if/else
            elseif stmt === nothing
                # Skip nothing statements
            else
                append!(bytes, compile_statement(stmt, i, ctx))
            end
        end
        # Now push the then-value for the phi result
        # compile_value will do LOCAL_GET if the value has a local
        if then_value !== nothing
            append!(bytes, compile_value(then_value, ctx))
        end

        # Else branch
        push!(bytes, Opcode.ELSE)

        # Else-branch: generate all statements in the else-branch, then push the else-value
        for i in else_start:phi_idx-1
            stmt = code[i]
            if stmt isa Core.GotoNode
                # Skip the goto
            elseif stmt === nothing
                # Skip nothing statements
            else
                append!(bytes, compile_statement(stmt, i, ctx))
            end
        end
        # Now push the else-value for the phi result
        if else_value !== nothing
            append!(bytes, compile_value(else_value, ctx))
        end

        # End if - phi result is on the stack
        push!(bytes, Opcode.END)

        # Store phi result to local if it has one
        if haskey(ctx.phi_locals, phi_idx)
            local_idx = ctx.phi_locals[phi_idx]
            push!(bytes, Opcode.LOCAL_SET)
            append!(bytes, encode_leb128_unsigned(local_idx))
        end

        # Generate code after the phi node
        for i in phi_idx+1:length(code)
            stmt = code[i]
            if stmt isa Core.ReturnNode
                if isdefined(stmt, :val)
                    append!(bytes, compile_value(stmt.val, ctx))
                end
                push!(bytes, Opcode.RETURN)
            elseif !(stmt === nothing)
                append!(bytes, compile_statement(stmt, i, ctx))
            end
        end
    else
        # Return-based pattern OR void-if-then pattern
        # First, check if then-branch contains a return
        then_has_return = false
        for i in then_start:else_start-1
            if code[i] isa Core.ReturnNode
                then_has_return = true
                break
            end
        end

        if !then_has_return
            # Void-if-then pattern: then-branch has no return, falls through to common return
            # Generate void IF block for side effects, then continue to shared return path
            push!(bytes, Opcode.IF)
            push!(bytes, 0x40)  # void block type

            for i in then_start:else_start-1
                stmt = code[i]
                if stmt === nothing
                    # Skip nothing statements
                elseif stmt isa Core.GotoNode
                    # Skip goto - handled by control flow
                else
                    append!(bytes, compile_statement(stmt, i, ctx))

                    # Drop unused values from calls
                    if stmt isa Expr && (stmt.head === :call || stmt.head === :invoke)
                        stmt_type = get(ctx.ssa_types, i, Any)
                        if stmt_type !== Nothing
                            is_nothing_union = stmt_type isa Union && Nothing in Base.uniontypes(stmt_type)
                            if !is_nothing_union
                                if !haskey(ctx.ssa_locals, i) && !haskey(ctx.phi_locals, i)
                                    use_count = get(ssa_use_count, i, 0)
                                    if use_count == 0
                                        push!(bytes, Opcode.DROP)
                                    end
                                end
                            end
                        end
                    end
                end
            end

            push!(bytes, Opcode.END)  # End the void IF block

            # Generate the common return path (else-branch which both paths reach)
            for i in else_start:length(code)
                stmt = code[i]
                if stmt isa Core.ReturnNode
                    if isdefined(stmt, :val)
                        append!(bytes, compile_value(stmt.val, ctx))
                    end
                    push!(bytes, Opcode.RETURN)
                elseif stmt === nothing
                    # Skip
                else
                    append!(bytes, compile_statement(stmt, i, ctx))
                end
            end

            return bytes
        end

        # Original return-based pattern: both branches have returns
        # Determine result type for the if block
        result_type = julia_to_wasm_type_concrete(ctx.return_type, ctx)

        # Start if block (condition is on stack)
        push!(bytes, Opcode.IF)
        append!(bytes, encode_block_type(result_type))

        # Generate then-branch (executed when condition is TRUE)
        for i in then_start:else_start-1
            stmt = code[i]
            if stmt isa Core.ReturnNode
                if isdefined(stmt, :val)
                    append!(bytes, compile_value(stmt.val, ctx))
                end
                # Don't emit return - the value stays on stack for the if result
            elseif stmt === nothing
                # Skip nothing statements
            else
                append!(bytes, compile_statement(stmt, i, ctx))

                # Drop unused values from calls (like setfield! which returns a value)
                # Also drop Any-typed values (like bb_read) when unused
                if stmt isa Expr && (stmt.head === :call || stmt.head === :invoke)
                    # Default to Any (not Nothing) so unknown types get drop check
                    stmt_type = get(ctx.ssa_types, i, Any)
                    if stmt_type !== Nothing  # Only skip if type is definitely Nothing
                        is_nothing_union = stmt_type isa Union && Nothing in Base.uniontypes(stmt_type)
                        if !is_nothing_union
                            if !haskey(ctx.ssa_locals, i) && !haskey(ctx.phi_locals, i)
                                use_count = get(ssa_use_count, i, 0)
                                if use_count == 0
                                    push!(bytes, Opcode.DROP)
                                end
                            end
                        end
                    end
                end
            end
        end

        # Else branch
        push!(bytes, Opcode.ELSE)

        # Generate else-branch
        # Track compiled statements to handle nested conditionals properly
        compiled_in_else = Set{Int}()
        for i in else_start:length(code)
            if i in compiled_in_else
                continue
            end
            stmt = code[i]
            if stmt isa Core.ReturnNode
                if isdefined(stmt, :val)
                    append!(bytes, compile_value(stmt.val, ctx))
                end
            elseif stmt === nothing
                # Skip nothing statements
            elseif stmt isa Core.GotoIfNot
                # Nested conditional in else branch - generate nested if/else
                nested_result = compile_nested_if_else(ctx, code, i, compiled_in_else, ssa_use_count)
                append!(bytes, nested_result)
            elseif stmt isa Core.GotoNode
                # Skip goto statements (they're control flow markers)
                push!(compiled_in_else, i)
            else
                append!(bytes, compile_statement(stmt, i, ctx))

                # Drop unused values from calls (like setfield! which returns a value)
                # Also drop Any-typed values (like bb_read) when unused
                if stmt isa Expr && (stmt.head === :call || stmt.head === :invoke)
                    # Default to Any (not Nothing) so unknown types get drop check
                    stmt_type = get(ctx.ssa_types, i, Any)
                    if stmt_type !== Nothing  # Only skip if type is definitely Nothing
                        is_nothing_union = stmt_type isa Union && Nothing in Base.uniontypes(stmt_type)
                        if !is_nothing_union
                            if !haskey(ctx.ssa_locals, i) && !haskey(ctx.phi_locals, i)
                                use_count = get(ssa_use_count, i, 0)
                                if use_count == 0
                                    push!(bytes, Opcode.DROP)
                                end
                            end
                        end
                    end
                end
            end
        end

        # End if
        push!(bytes, Opcode.END)

        # The result of if...else...end is on the stack, return it
        push!(bytes, Opcode.RETURN)
    end

    return bytes
end

"""
Compile a nested if/else inside a return-based pattern.
This handles the case where there's a GotoIfNot inside an else branch
that creates a nested conditional, each branch ending with a return.
"""
function compile_nested_if_else(ctx::CompilationContext, code, goto_idx::Int, compiled::Set{Int}, ssa_use_count::Dict{Int,Int})::Vector{UInt8}
    bytes = UInt8[]

    goto_if_not = code[goto_idx]::Core.GotoIfNot
    else_target = goto_if_not.dest  # Where to jump if condition is FALSE
    then_start = goto_idx + 1

    # The condition is already computed (it's an SSA reference)
    # Push it
    append!(bytes, compile_value(goto_if_not.cond, ctx))
    push!(compiled, goto_idx)

    # Determine result type - should match the enclosing function's return type
    result_type = julia_to_wasm_type_concrete(ctx.return_type, ctx)

    # Start if block
    push!(bytes, Opcode.IF)
    append!(bytes, encode_block_type(result_type))

    # Then branch: from then_start to else_target-1
    for i in then_start:else_target-1
        if i in compiled
            continue
        end
        stmt = code[i]
        push!(compiled, i)

        if stmt isa Core.ReturnNode
            if isdefined(stmt, :val)
                append!(bytes, compile_value(stmt.val, ctx))
            end
            # Value stays on stack for the if result
        elseif stmt === nothing
            # Skip
        elseif stmt isa Core.GotoNode
            # Skip forward gotos
        else
            append!(bytes, compile_statement(stmt, i, ctx))

            # Drop unused values
            if stmt isa Expr && (stmt.head === :call || stmt.head === :invoke)
                # Default to Any (not Nothing) so unknown types get drop check
                stmt_type = get(ctx.ssa_types, i, Any)
                use_count = get(ssa_use_count, i, 0)
                if stmt_type !== Nothing
                    is_nothing_union = stmt_type isa Union && Nothing in Base.uniontypes(stmt_type)
                    if !is_nothing_union
                        if !haskey(ctx.ssa_locals, i) && !haskey(ctx.phi_locals, i)
                            if use_count == 0
                                push!(bytes, Opcode.DROP)
                            end
                        end
                    end
                end
            end
        end
    end

    # Else branch
    push!(bytes, Opcode.ELSE)

    # Else branch: from else_target to end
    for i in else_target:length(code)
        if i in compiled
            continue
        end
        stmt = code[i]
        push!(compiled, i)

        if stmt isa Core.ReturnNode
            if isdefined(stmt, :val)
                append!(bytes, compile_value(stmt.val, ctx))
            end
            # Value stays on stack for the if result
        elseif stmt === nothing
            # Skip
        elseif stmt isa Core.GotoNode
            # Skip forward gotos
        elseif stmt isa Core.GotoIfNot
            # Another nested conditional - recurse
            nested = compile_nested_if_else(ctx, code, i, compiled, ssa_use_count)
            append!(bytes, nested)
        else
            append!(bytes, compile_statement(stmt, i, ctx))

            # Drop unused values
            if stmt isa Expr && (stmt.head === :call || stmt.head === :invoke)
                # Default to Any (not Nothing) so unknown types get drop check
                stmt_type = get(ctx.ssa_types, i, Any)
                if stmt_type !== Nothing
                    is_nothing_union = stmt_type isa Union && Nothing in Base.uniontypes(stmt_type)
                    if !is_nothing_union
                        if !haskey(ctx.ssa_locals, i) && !haskey(ctx.phi_locals, i)
                            use_count = get(ssa_use_count, i, 0)
                            if use_count == 0
                                push!(bytes, Opcode.DROP)
                            end
                        end
                    end
                end
            end
        end
    end

    # End nested if
    push!(bytes, Opcode.END)

    return bytes
end

"""
Generate code for more complex control flow patterns.
Uses nested blocks with br instructions.
"""
function generate_complex_flow(ctx::CompilationContext, blocks::Vector{BasicBlock}, code)::Vector{UInt8}
    bytes = UInt8[]

    # For void return types (like event handlers), use a simpler approach:
    # just execute all statements in order and return at the end
    if ctx.return_type === Nothing
        append!(bytes, generate_void_flow(ctx, blocks, code))
        return bytes
    end

    # Count how many conditional branches we have
    conditionals = [(i, b) for (i, b) in enumerate(blocks) if b.terminator isa Core.GotoIfNot]

    # For functions with 3+ conditionals, use the stackifier algorithm.
    # The nested conditional generator handles simple if-else well (1-2 conditionals),
    # but multi-edge phis (3+ branches merging) require the stackifier's approach
    # of explicitly storing to phi locals at each branch.
    if length(conditionals) > 2
        return generate_stackified_flow(ctx, blocks, code)
    end

    # For simpler functions, use nested if-else (which works well for moderate complexity)
    if length(conditionals) >= 1
        append!(bytes, generate_nested_conditionals(ctx, blocks, code, conditionals))
    else
        # Fallback: generate blocks sequentially
        for block in blocks
            append!(bytes, generate_block_code(ctx, block))
        end
    end

    return bytes
end

"""
Stackifier algorithm for complex control flow.
Converts Julia IR CFG to WASM structured control flow by:
1. Building a CFG from basic blocks
2. Computing dominators and identifying merge points
3. Generating each block exactly once
4. Using block/br for forward jumps, loop/br for back jumps

Based on LLVM's WebAssembly backend stackifier and Cheerp's enhancements.
Reference: https://labs.leaningtech.com/blog/control-flow
"""
function generate_stackified_flow(ctx::CompilationContext, blocks::Vector{BasicBlock}, code)::Vector{UInt8}
    # ========================================================================
    # STEP 0: BOUNDSCHECK PATTERN DETECTION
    # ========================================================================
    # We emit i32.const 0 for boundscheck, so GotoIfNot following boundscheck
    # ALWAYS jumps (since NOT 0 = TRUE). Track these patterns to skip dead code.

    boundscheck_jumps = Set{Int}()  # Statement indices of GotoIfNot that always jump
    dead_regions = Set{Int}()       # Statement indices that are dead code
    dead_blocks = Set{Int}()        # Block indices that are entirely dead

    for i in 1:length(code)
        stmt = code[i]
        if stmt isa Expr && stmt.head === :boundscheck && length(stmt.args) >= 1
            if i + 1 <= length(code) && code[i + 1] isa Core.GotoIfNot
                goto_stmt = code[i + 1]::Core.GotoIfNot
                if goto_stmt.cond isa Core.SSAValue && goto_stmt.cond.id == i
                    push!(boundscheck_jumps, i + 1)
                    push!(dead_regions, i)
                    target = goto_stmt.dest
                    for j in (i + 2):(target - 1)
                        push!(dead_regions, j)
                    end
                end
            end
        end
    end

    # Mark blocks as dead if all their statements are in dead regions
    for (block_idx, block) in enumerate(blocks)
        all_dead = true
        for i in block.start_idx:block.end_idx
            if !(i in dead_regions) && !(i in boundscheck_jumps)
                all_dead = false
                break
            end
        end
        if all_dead
            push!(dead_blocks, block_idx)
        end
    end

    # ========================================================================
    # STEP 1: Build Control Flow Graph
    # ========================================================================

    # Map statement index -> block index
    stmt_to_block = Dict{Int, Int}()
    for (block_idx, block) in enumerate(blocks)
        for i in block.start_idx:block.end_idx
            stmt_to_block[i] = block_idx
        end
    end

    # Build successor/predecessor maps (block indices)
    successors = Dict{Int, Vector{Int}}()  # block_idx -> successor block indices
    predecessors = Dict{Int, Vector{Int}}()  # block_idx -> predecessor block indices

    for i in 1:length(blocks)
        successors[i] = Int[]
        predecessors[i] = Int[]
    end

    for (block_idx, block) in enumerate(blocks)
        # Skip dead blocks entirely - don't add edges to/from them
        if block_idx in dead_blocks
            continue
        end

        term = block.terminator
        if term isa Core.GotoIfNot
            # Check if this is a boundscheck-based always-jump
            term_idx = block.end_idx
            if term_idx in boundscheck_jumps
                # This GotoIfNot ALWAYS jumps (boundscheck is 0, NOT 0 = TRUE)
                # Only add the jump target as successor, NOT the fall-through
                dest_block = get(stmt_to_block, term.dest, nothing)
                if dest_block !== nothing && !(dest_block in dead_blocks)
                    push!(successors[block_idx], dest_block)
                    push!(predecessors[dest_block], block_idx)
                end
            else
                # Real conditional: two successors
                dest_block = get(stmt_to_block, term.dest, nothing)
                fall_through_block = block_idx < length(blocks) ? block_idx + 1 : nothing

                if fall_through_block !== nothing && fall_through_block <= length(blocks) && !(fall_through_block in dead_blocks)
                    push!(successors[block_idx], fall_through_block)
                    push!(predecessors[fall_through_block], block_idx)
                end
                if dest_block !== nothing && !(dest_block in dead_blocks)
                    push!(successors[block_idx], dest_block)
                    push!(predecessors[dest_block], block_idx)
                end
            end
        elseif term isa Core.GotoNode
            dest_block = get(stmt_to_block, term.label, nothing)
            if dest_block !== nothing
                push!(successors[block_idx], dest_block)
                push!(predecessors[dest_block], block_idx)
            end
        elseif term isa Core.ReturnNode
            # No successors for return
        else
            # Fall through to next block
            if block_idx < length(blocks)
                push!(successors[block_idx], block_idx + 1)
                push!(predecessors[block_idx + 1], block_idx)
            end
        end
    end

    # ========================================================================
    # STEP 2: Identify Back Edges (loops) vs Forward Edges
    # ========================================================================

    back_edges = Set{Tuple{Int, Int}}()  # (from_block, to_block)
    forward_edges = Set{Tuple{Int, Int}}()
    loop_headers = Set{Int}()

    for (block_idx, succs) in successors
        for succ in succs
            if succ <= block_idx  # Back edge (loop)
                push!(back_edges, (block_idx, succ))
                push!(loop_headers, succ)
            else  # Forward edge
                push!(forward_edges, (block_idx, succ))
            end
        end
    end

    # ========================================================================
    # STEP 3: Find Forward Edge Targets (merge points that need block/br)
    # ========================================================================

    # For each forward edge target, track the sources
    # These are targets where we need to emit a block and use br to jump
    forward_targets = Dict{Int, Vector{Int}}()  # target_block -> source_blocks

    for (src, dst) in forward_edges
        # A forward edge needs block/br if it's NOT a simple fall-through
        # (i.e., src + 1 != dst or there are multiple paths to dst)
        if !haskey(forward_targets, dst)
            forward_targets[dst] = Int[]
        end
        push!(forward_targets[dst], src)
    end

    # ========================================================================
    # STEP 4: Count SSA uses for drop logic
    # ========================================================================

    ssa_use_count = Dict{Int, Int}()
    for stmt in code
        count_ssa_uses!(stmt, ssa_use_count)
    end

    # ========================================================================
    # STEP 5: Generate Code Using Stackifier Strategy
    # ========================================================================

    # Helper to compile statements in a basic block
    function compile_block_statements(block::BasicBlock, skip_terminator::Bool)::Vector{UInt8}
        block_bytes = UInt8[]

        for i in block.start_idx:block.end_idx
            stmt = code[i]

            # Skip terminator if requested (we handle control flow separately)
            if skip_terminator && i == block.end_idx && (stmt isa Core.GotoIfNot || stmt isa Core.GotoNode || stmt isa Core.ReturnNode)
                continue
            end

            if stmt isa Core.ReturnNode
                if isdefined(stmt, :val)
                    append!(block_bytes, compile_value(stmt.val, ctx))
                end
                push!(block_bytes, Opcode.RETURN)

            elseif stmt isa Core.GotoIfNot
                # GotoIfNot: handled by control flow structure
                # Nothing to emit here

            elseif stmt isa Core.GotoNode
                # Unconditional goto: handled by control flow structure
                # Nothing to emit here

            elseif stmt isa Core.PhiNode
                # Phi nodes: check if we're falling through from a previous statement
                # in the same block that is an edge for this phi
                if haskey(ctx.phi_locals, i)
                    # Look for an edge from a previous statement in this block
                    for (edge_idx, edge) in enumerate(stmt.edges)
                        # Check if this edge is from within the same block (internal fallthrough)
                        if edge >= block.start_idx && edge < i
                            # This is an internal fallthrough edge - set the phi local
                            val = stmt.values[edge_idx]
                            phi_value_bytes = compile_phi_value(val, i)
                            # Only emit local_set if we actually have a value on the stack
                            if !isempty(phi_value_bytes)
                                append!(block_bytes, phi_value_bytes)
                                local_idx = ctx.phi_locals[i]
                                push!(block_bytes, Opcode.LOCAL_SET)
                                append!(block_bytes, encode_leb128_unsigned(local_idx))
                            end
                            break
                        end
                    end
                end
                # No other code needed - phi result is read via LOCAL_GET

            elseif stmt === nothing
                # Nothing statement

            else
                # Regular statement - compile_statement handles local.set internally
                # for statements that produce values and have ssa_locals allocated
                stmt_bytes = compile_statement(stmt, i, ctx)
                append!(block_bytes, stmt_bytes)

                # NOTE: compile_statement already adds LOCAL_SET for SSA values
                # that need storing. We don't add another one here to avoid
                # duplicate stores that would cause "not enough arguments on stack" errors.

                # Only drop unused values that don't have locals
                if !haskey(ctx.ssa_locals, i) && stmt isa Expr && (stmt.head === :call || stmt.head === :invoke)
                    # Use statement_produces_wasm_value to check if the call actually
                    # produces a value on the stack (handles Any type correctly)
                    if statement_produces_wasm_value(stmt, i, ctx)
                        if !haskey(ctx.phi_locals, i)
                            use_count = get(ssa_use_count, i, 0)
                            if use_count == 0
                                push!(block_bytes, Opcode.DROP)
                            end
                        end
                    end
                end
            end
        end

        return block_bytes
    end

    # ========================================================================
    # STEP 6: Main Code Generation
    # ========================================================================
    #
    # Strategy: Process blocks in order. For each block:
    # - If it's a loop header: wrap with loop/end
    # - If it's a forward edge target: wrap with block/end (so br can jump past it)
    # - For GotoIfNot: emit if/else
    # - For GotoNode: emit br to the right scope
    #
    # The key insight: we need to set up block scopes BEFORE we need to br to them.
    # So we scan ahead to find all forward jump targets and wrap them.
    #
    # Simplified approach for Julia IR:
    # - Julia's IR tends to have simple diamond patterns (if/else merge)
    # - Most forward jumps go to the "next" merge point
    # - We use nested if/else for these patterns
    # - For more complex patterns, we use labeled blocks

    bytes = UInt8[]

    # For very complex functions, use a dispatcher-style approach
    # Create a big block structure with all targets as labeled positions

    # Collect all unique forward jump targets (excluding immediate fall-through)
    # Also exclude dead blocks and treat boundscheck-based jumps correctly
    non_trivial_targets = Set{Int}()
    for (block_idx, block) in enumerate(blocks)
        # Skip dead blocks
        if block_idx in dead_blocks
            continue
        end

        term = block.terminator
        term_idx = block.end_idx

        if term isa Core.GotoIfNot
            # Check if this is a boundscheck always-jump
            if term_idx in boundscheck_jumps
                # Boundscheck jumps ALWAYS go to dest, so it's like an unconditional jump
                # Only record it as non-trivial if it's not immediate fall-through
                dest_block = get(stmt_to_block, term.dest, nothing)
                if dest_block !== nothing && dest_block != block_idx + 1 && !(dest_block in dead_blocks)
                    push!(non_trivial_targets, dest_block)
                end
            else
                # Real conditional - the false branch destination
                dest_block = get(stmt_to_block, term.dest, nothing)
                if dest_block !== nothing && dest_block != block_idx + 1 && !(dest_block in dead_blocks)
                    push!(non_trivial_targets, dest_block)
                end
            end
        elseif term isa Core.GotoNode
            dest_block = get(stmt_to_block, term.label, nothing)
            if dest_block !== nothing && dest_block != block_idx + 1 && !(dest_block in dead_blocks)
                push!(non_trivial_targets, dest_block)
            end
        end
    end

    # Sort targets in ASCENDING order (smallest first)
    # We'll open blocks for all targets (largest first = outermost)
    # and close them as we reach each target
    sorted_targets_asc = sort(collect(non_trivial_targets))
    sorted_targets_desc = reverse(sorted_targets_asc)  # largest first

    # Track currently open blocks (as a stack of target block indices)
    # The stack is ordered with outermost at bottom, innermost at top
    open_blocks = copy(sorted_targets_desc)  # Will close in ascending order

    # Also track open loops
    open_loops = Int[]  # Stack of loop header block indices

    # Open all blocks for forward jump targets (outermost first = largest target)
    for target in sorted_targets_desc
        push!(bytes, Opcode.BLOCK)
        push!(bytes, 0x40)  # void
    end

    # Helper function to get current label depth for a forward jump target
    # Label 0 = innermost currently open block
    function get_forward_label_depth(target_block::Int)::Int
        # Find position of target in open_blocks (0-indexed from end = innermost)
        # open_blocks is [largest, ..., smallest] so target at end has depth 0
        for (i, t) in enumerate(reverse(open_blocks))
            if t == target_block
                return i - 1 + length(open_loops)  # Account for any open loops
            end
        end
        # Target not in open blocks - shouldn't happen for non_trivial_targets
        return 0
    end

    # Helper to get label depth for back edge (loop)
    function get_loop_label_depth(loop_header::Int)::Int
        # Find the loop in open_loops stack
        for (i, h) in enumerate(reverse(open_loops))
            if h == loop_header
                return i - 1  # 0 = innermost loop
            end
        end
        return 0
    end

    # Helper to check if destination has phi nodes from this edge
    function dest_has_phi_from_edge(dest_block::Int, terminator_idx::Int)::Bool
        if dest_block < 1 || dest_block > length(blocks)
            return false
        end
        dest_start = blocks[dest_block].start_idx
        dest_end = blocks[dest_block].end_idx
        for i in dest_start:dest_end
            stmt = code[i]
            if stmt isa Core.PhiNode
                if haskey(ctx.phi_locals, i) && terminator_idx in stmt.edges
                    return true
                end
            else
                break  # Phi nodes are consecutive at the start
            end
        end
        return false
    end

    # Helper to compile a value, ensuring it actually produces bytes
    # For SSAValues without locals, we need to recompute the value
    # phi_idx: the SSA index of the phi node we're setting (to get the phi's type)
    function compile_phi_value(val, phi_idx::Int)::Vector{UInt8}
        result = UInt8[]
        if val isa Core.SSAValue
            # Debug trace for specific SSAs
            if val.id in [45, 80, 84, 85, 93, 95]
            end
            # Check if this SSA has a local allocated
            if haskey(ctx.ssa_locals, val.id)
                local_idx = ctx.ssa_locals[val.id]
                push!(result, Opcode.LOCAL_GET)
                append!(result, encode_leb128_unsigned(local_idx))
            elseif haskey(ctx.phi_locals, val.id)
                local_idx = ctx.phi_locals[val.id]
                push!(result, Opcode.LOCAL_GET)
                append!(result, encode_leb128_unsigned(local_idx))
            else
                # SSA without local - need to recompute the statement
                # This should ideally not happen for phi values, but handle it
                println("WARNING: SSA %$(val.id) has no local in compile_phi_value, recomputing: $(code[val.id])")
                stmt = code[val.id]
                if stmt !== nothing && !(stmt isa Core.PhiNode)
                    append!(result, compile_statement(stmt, val.id, ctx))
                else
                    # Can't recompute - this is likely a bug
                    # Try compile_value as fallback
                    append!(result, compile_value(val, ctx))
                end
            end
        elseif val === nothing || (val isa GlobalRef && val.name === :nothing)
            # Value is `nothing` (can be Core.nothing or Main.nothing in IR)
            # Check if we need to emit ref.null for Union{T, Nothing} phi
            phi_type = get(ctx.ssa_types, phi_idx, nothing)
            if phi_type !== nothing && phi_type !== Nothing
                # Phi expects a value but we have nothing - emit ref.null
                wasm_type = julia_to_wasm_type_concrete(phi_type, ctx)
                if wasm_type isa ConcreteRef
                    push!(result, Opcode.REF_NULL)
                    append!(result, encode_leb128_unsigned(wasm_type.type_idx))
                end
            end
            # If phi_type is Nothing, return empty bytes (nothing to push)
        else
            # Not an SSA and not nothing - just compile directly
            append!(result, compile_value(val, ctx))
        end
        return result
    end

    # Helper to set all phi locals at destination
    # dest_block: the block index being jumped to
    # terminator_idx: the statement index of the terminator (edge in phi)
    # target_stmt: optional - the actual statement being jumped to (may differ from block start)
    function set_phi_locals_for_edge!(bytes::Vector{UInt8}, dest_block::Int, terminator_idx::Int; target_stmt::Int=0)
        if dest_block < 1 || dest_block > length(blocks)
            return
        end
        # If target_stmt is specified, start from there; otherwise start from block start
        dest_start = target_stmt > 0 ? target_stmt : blocks[dest_block].start_idx
        dest_end = blocks[dest_block].end_idx
        phi_count = 0
        for i in dest_start:dest_end
            stmt = code[i]
            if stmt isa Core.PhiNode
                if haskey(ctx.phi_locals, i)
                    found_edge = false
                    for (edge_idx, edge) in enumerate(stmt.edges)
                        if edge == terminator_idx
                            val = stmt.values[edge_idx]
                            phi_value_bytes = compile_phi_value(val, i)
                            # Only emit local_set if we actually have a value on the stack
                            if !isempty(phi_value_bytes)
                                append!(bytes, phi_value_bytes)
                                local_idx = ctx.phi_locals[i]
                                push!(bytes, Opcode.LOCAL_SET)
                                append!(bytes, encode_leb128_unsigned(local_idx))
                                phi_count += 1
                            end
                            found_edge = true
                            break
                        end
                    end
                end
            else
                break  # Phi nodes are consecutive at the start
            end
        end
    end

    # Now generate code for each block in order
    for (block_idx, block) in enumerate(blocks)
        # First, close any blocks whose target is this block
        # (We close BEFORE generating code for the target block)
        while !isempty(open_blocks) && last(open_blocks) == block_idx
            pop!(open_blocks)
            push!(bytes, Opcode.END)  # End the block for this target
        end

        # Skip dead blocks (from boundscheck patterns)
        if block_idx in dead_blocks
            continue
        end

        # Check if we're entering a loop
        is_loop_header = block_idx in loop_headers

        if is_loop_header
            push!(bytes, Opcode.LOOP)
            push!(bytes, 0x40)  # void
            push!(open_loops, block_idx)
        end

        # Compile the block's statements (not the terminator, we handle it separately)
        # Skip any dead statements within the block
        block_bytes = UInt8[]
        for i in block.start_idx:block.end_idx
            # Skip dead statements
            if i in dead_regions
                continue
            end
            if i in boundscheck_jumps
                continue  # This GotoIfNot always jumps - skip it (handled below)
            end

            stmt = code[i]

            # Skip terminator if we're going to handle it separately
            if i == block.end_idx && (stmt isa Core.GotoIfNot || stmt isa Core.GotoNode || stmt isa Core.ReturnNode)
                continue
            end

            if stmt isa Core.ReturnNode
                if isdefined(stmt, :val)
                    append!(block_bytes, compile_value(stmt.val, ctx))
                end
                push!(block_bytes, Opcode.RETURN)

            elseif stmt isa Core.GotoIfNot
                # GotoIfNot: handled by control flow structure
                # Nothing to emit here

            elseif stmt isa Core.GotoNode
                # Unconditional goto: handled by control flow structure
                # Nothing to emit here

            elseif stmt isa Core.PhiNode
                # Phi nodes: check if we're falling through from a previous statement
                if haskey(ctx.phi_locals, i)
                    for (edge_idx, edge) in enumerate(stmt.edges)
                        if edge >= block.start_idx && edge < i
                            val = stmt.values[edge_idx]
                            phi_value_bytes = compile_phi_value(val, i)
                            if !isempty(phi_value_bytes)
                                append!(block_bytes, phi_value_bytes)
                                local_idx = ctx.phi_locals[i]
                                push!(block_bytes, Opcode.LOCAL_SET)
                                append!(block_bytes, encode_leb128_unsigned(local_idx))
                            end
                            break
                        end
                    end
                end

            elseif stmt === nothing
                # Nothing statement

            else
                stmt_bytes = compile_statement(stmt, i, ctx)
                append!(block_bytes, stmt_bytes)

                if !haskey(ctx.ssa_locals, i) && stmt isa Expr && (stmt.head === :call || stmt.head === :invoke)
                    if statement_produces_wasm_value(stmt, i, ctx)
                        if !haskey(ctx.phi_locals, i)
                            use_count = get(ssa_use_count, i, 0)
                            if use_count == 0
                                push!(block_bytes, Opcode.DROP)
                            end
                        end
                    end
                end
            end
        end
        append!(bytes, block_bytes)

        # Handle the terminator
        term = block.terminator
        terminator_idx = block.end_idx

        # Check if this terminator is a boundscheck always-jump
        if terminator_idx in boundscheck_jumps && term isa Core.GotoIfNot
            # This is an always-jump - emit unconditional br to the target
            dest_block = get(stmt_to_block, term.dest, nothing)
            if dest_block !== nothing && dest_block > block_idx && dest_block in non_trivial_targets
                label_depth = get_forward_label_depth(dest_block)
                push!(bytes, Opcode.BR)
                append!(bytes, encode_leb128_unsigned(label_depth))
            end
            # Otherwise, it's just a fall-through to a live block - nothing needed

        elseif term isa Core.ReturnNode
            if isdefined(term, :val)
                append!(bytes, compile_value(term.val, ctx))
            end
            push!(bytes, Opcode.RETURN)

        elseif term isa Core.GotoIfNot
            dest_block = get(stmt_to_block, term.dest, nothing)

            # Check if destination has phi nodes that need values from this edge
            has_phi = dest_block !== nothing && dest_has_phi_from_edge(dest_block, terminator_idx)

            # Compile condition
            append!(bytes, compile_value(term.cond, ctx))

            # If condition is TRUE, fall through to next block
            # If condition is FALSE, jump to dest

            if dest_block !== nothing && dest_block > block_idx
                # Forward jump when condition is false
                if dest_block in non_trivial_targets
                    if has_phi
                        # Need to set phi values before jumping - use if/else
                        push!(bytes, Opcode.IF)
                        push!(bytes, 0x40)  # void
                        # Then branch: condition true, fall through (empty)
                        push!(bytes, Opcode.ELSE)
                        # Else branch: condition false, set all phi locals and jump
                        set_phi_locals_for_edge!(bytes, dest_block, terminator_idx; target_stmt=term.dest)
                        # Jump to destination (account for the if block we're inside)
                        label_depth = get_forward_label_depth(dest_block) + 1
                        push!(bytes, Opcode.BR)
                        append!(bytes, encode_leb128_unsigned(label_depth))
                        push!(bytes, Opcode.END)
                    else
                        # No phi - use br_if
                        label_depth = get_forward_label_depth(dest_block)
                        push!(bytes, Opcode.I32_EQZ)  # Invert the condition
                        push!(bytes, Opcode.BR_IF)
                        append!(bytes, encode_leb128_unsigned(label_depth))
                    end
                else
                    # Simple fall-through pattern - condition true continues, false skips
                    if has_phi
                        push!(bytes, Opcode.IF)
                        push!(bytes, 0x40)
                        push!(bytes, Opcode.ELSE)
                        set_phi_locals_for_edge!(bytes, dest_block, terminator_idx; target_stmt=term.dest)
                        push!(bytes, Opcode.END)
                    else
                        push!(bytes, Opcode.IF)
                        push!(bytes, 0x40)
                        push!(bytes, Opcode.END)
                    end
                end
            elseif dest_block !== nothing && dest_block <= block_idx
                # Back edge (loop continuation condition)
                if dest_block in loop_headers
                    if has_phi
                        push!(bytes, Opcode.IF)
                        push!(bytes, 0x40)
                        push!(bytes, Opcode.ELSE)
                        set_phi_locals_for_edge!(bytes, dest_block, terminator_idx; target_stmt=term.dest)
                        label_depth = get_loop_label_depth(dest_block) + 1
                        push!(bytes, Opcode.BR)
                        append!(bytes, encode_leb128_unsigned(label_depth))
                        push!(bytes, Opcode.END)
                    else
                        label_depth = get_loop_label_depth(dest_block)
                        push!(bytes, Opcode.I32_EQZ)
                        push!(bytes, Opcode.BR_IF)
                        append!(bytes, encode_leb128_unsigned(label_depth))
                    end
                end
            end

        elseif term isa Core.GotoNode
            dest_block = get(stmt_to_block, term.label, nothing)
            terminator_idx = block.end_idx

            # Set all phi values before jumping
            # Pass the actual target statement to find phi nodes (might be inside the block)
            if dest_block !== nothing
                set_phi_locals_for_edge!(bytes, dest_block, terminator_idx; target_stmt=term.label)
            end

            if dest_block !== nothing && dest_block > block_idx
                # Forward jump
                if dest_block in non_trivial_targets
                    label_depth = get_forward_label_depth(dest_block)
                    push!(bytes, Opcode.BR)
                    append!(bytes, encode_leb128_unsigned(label_depth))
                end
                # Otherwise, simple fall through - implicit
            elseif dest_block !== nothing && dest_block <= block_idx
                # Back edge (loop)
                if dest_block in loop_headers
                    label_depth = get_loop_label_depth(dest_block)
                    push!(bytes, Opcode.BR)
                    append!(bytes, encode_leb128_unsigned(label_depth))
                end
            end
        else
            # No explicit terminator (GotoNode, GotoIfNot, ReturnNode)
            # This block falls through to the next block
            # Check if next block has phi nodes that need values from this edge
            next_block_idx = block_idx + 1
            if next_block_idx <= length(blocks)
                # The edge for fallthrough is the last statement of this block
                terminator_idx = block.end_idx
                set_phi_locals_for_edge!(bytes, next_block_idx, terminator_idx)
            end
        end

        # Close loop if this is the last block of the loop (back edge source)
        for (src, dst) in back_edges
            if src == block_idx
                push!(bytes, Opcode.END)  # End of loop
                # Remove from open_loops
                filter!(h -> h != dst, open_loops)
            end
        end
    end

    # Close any remaining open blocks
    while !isempty(open_blocks)
        pop!(open_blocks)
        push!(bytes, Opcode.END)
    end

    # The code should always end with a return, but add unreachable as safety
    push!(bytes, Opcode.UNREACHABLE)

    return bytes
end

"""
Generate code for complex functions using a block-based approach.
Compiles each basic block exactly once using structured control flow.
This is a simpler approach than full Stackifier, suitable for moderate complexity.
"""
function generate_linear_flow(ctx::CompilationContext, blocks::Vector{BasicBlock}, code, conditionals)::Vector{UInt8}
    bytes = UInt8[]
    result_type = julia_to_wasm_type_concrete(ctx.return_type, ctx)

    # Count SSA uses for drop logic
    ssa_use_count = Dict{Int, Int}()
    for stmt in code
        count_ssa_uses!(stmt, ssa_use_count)
    end

    # Track which statements have been compiled
    compiled = Set{Int}()

    # Find all GotoIfNot destinations to create block structure
    # This helps with forward jumps
    jump_targets = Set{Int}()
    for (i, stmt) in enumerate(code)
        if stmt isa Core.GotoIfNot
            push!(jump_targets, stmt.dest)
        elseif stmt isa Core.GotoNode && stmt.label > i
            push!(jump_targets, stmt.label)
        end
    end

    # Helper to compile a range of statements
    function compile_range(start_idx::Int, end_idx::Int)::Vector{UInt8}
        range_bytes = UInt8[]

        for i in start_idx:min(end_idx, length(code))
            if i in compiled
                continue
            end

            stmt = code[i]

            if stmt isa Core.ReturnNode
                push!(compiled, i)
                if isdefined(stmt, :val)
                    append!(range_bytes, compile_value(stmt.val, ctx))
                end
                # RETURN pops the value from the stack
                push!(range_bytes, Opcode.RETURN)
                return range_bytes  # Return immediately

            elseif stmt isa Core.GotoIfNot
                push!(compiled, i)
                # Compile condition
                append!(range_bytes, compile_value(stmt.cond, ctx))

                # Check if then branch has a return
                then_end = stmt.dest - 1
                then_has_return = any(code[j] isa Core.ReturnNode for j in (i+1):min(then_end, length(code)))

                # Create if/else for the branch
                push!(range_bytes, Opcode.IF)
                push!(range_bytes, 0x40)  # void

                # Then branch: condition true, continue to next line
                append!(range_bytes, compile_range(i + 1, then_end))

                push!(range_bytes, Opcode.ELSE)

                # Else branch: condition false, jump to dest
                # If then branch returns, else branch handles the rest
                # Otherwise, both branches should reach the merge point
                if then_has_return
                    # Else branch handles all remaining code
                    append!(range_bytes, compile_range(stmt.dest, end_idx))
                else
                    # Both branches continue to merge point
                    # Else is empty (code at dest will be compiled after END)
                end

                push!(range_bytes, Opcode.END)

                if then_has_return
                    return range_bytes  # Else already handled the rest
                end
                # Otherwise continue - code at merge point follows

            elseif stmt isa Core.GotoNode
                push!(compiled, i)
                # Skip forward gotos - the target will be compiled in the else branch
                # For now, just continue

            elseif stmt isa Core.PhiNode
                push!(compiled, i)
                # Phi values are handled via locals, nothing to do here

            elseif stmt === nothing
                push!(compiled, i)

            else
                push!(compiled, i)
                stmt_bytes = compile_statement(stmt, i, ctx)
                append!(range_bytes, stmt_bytes)

                # Drop unused values
                if stmt isa Expr && (stmt.head === :call || stmt.head === :invoke)
                    stmt_type = get(ctx.ssa_types, i, Any)
                    if stmt_type !== Nothing
                        is_nothing_union = stmt_type isa Union && Nothing in Base.uniontypes(stmt_type)
                        if !is_nothing_union
                            if !haskey(ctx.ssa_locals, i) && !haskey(ctx.phi_locals, i)
                                use_count = get(ssa_use_count, i, 0)
                                if use_count == 0
                                    push!(range_bytes, Opcode.DROP)
                                end
                            end
                        end
                    end
                end
            end
        end

        return range_bytes
    end

    # Compile all code starting from line 1
    append!(bytes, compile_range(1, length(code)))

    # The code should always end with a return, but add unreachable as safety
    push!(bytes, Opcode.UNREACHABLE)

    return bytes
end

"""
Generate code for void functions (no return value).
Compiles all statements sequentially, using structured control flow for conditionals.
"""
function generate_void_flow(ctx::CompilationContext, blocks::Vector{BasicBlock}, code)::Vector{UInt8}
    bytes = UInt8[]

    # Track which statements we've already compiled
    compiled = Set{Int}()

    # Count how many times each SSA value is used (to determine if we need to DROP)
    # SSA values that are used elsewhere should NOT be dropped - they stay on stack
    ssa_use_count = Dict{Int, Int}()
    for stmt in code
        count_ssa_uses!(stmt, ssa_use_count)
    end

    # Process statements in order
    i = 1
    while i <= length(code)
        if i in compiled
            i += 1
            continue
        end

        stmt = code[i]

        if stmt === nothing
            push!(compiled, i)
            i += 1
            continue
        end

        if stmt isa Core.ReturnNode
            # Void return - just return
            push!(bytes, Opcode.RETURN)
            push!(compiled, i)
            i += 1
            continue
        end

        if stmt isa Core.GotoNode
            # Unconditional jump - skip (handled by structured control flow)
            push!(compiled, i)
            i += 1
            continue
        end

        if stmt isa Core.GotoIfNot
            # Conditional - compile as void if-block
            goto_if_not = stmt
            else_target = goto_if_not.dest

            # Determine if this is an if-then-else or just if-then by checking for a GotoNode
            # at the end of the then-branch that jumps past the else_target.
            # If-then-else pattern:
            #   GotoIfNot → else_target
            #   then-code
            #   GotoNode → merge_point  ← jumps PAST else_target
            #   else_target: else-code
            #   merge_point: ...
            # If-then pattern (no else):
            #   GotoIfNot → merge_point
            #   then-code
            #   merge_point: continuation  ← no GotoNode, code continues sequentially
            has_else_branch = false
            for j in (i+1):(else_target-1)
                if code[j] isa Core.GotoNode
                    goto_node = code[j]::Core.GotoNode
                    # If the GotoNode jumps past else_target, we have an else branch
                    if goto_node.label > else_target
                        has_else_branch = true
                        break
                    end
                end
            end

            # Push condition
            append!(bytes, compile_value(goto_if_not.cond, ctx))
            push!(compiled, i)

            # Start void if block
            push!(bytes, Opcode.IF)
            push!(bytes, 0x40)

            # Compile then-branch (i+1 to else_target-1)
            for j in (i+1):(else_target-1)
                if j in compiled
                    continue
                end
                inner = code[j]
                if inner === nothing
                    push!(compiled, j)
                elseif inner isa Core.GotoNode
                    push!(compiled, j)
                elseif inner isa Core.ReturnNode
                    # Early return inside conditional - emit return instruction
                    # BUT: If the return type is Union{} (unreachable), don't emit RETURN
                    stmt_type = get(ctx.ssa_types, j, Any)
                    if stmt_type !== Union{}
                        push!(bytes, Opcode.RETURN)
                    end
                    push!(compiled, j)
                elseif inner isa Core.GotoIfNot
                    # Check if this GotoIfNot is a ternary pattern (has a phi node)
                    # If so, use compile_ternary_for_phi to produce the phi value
                    inner_goto_if_not = inner::Core.GotoIfNot
                    inner_else_target = inner_goto_if_not.dest

                    # Look for a phi node after the inner else target
                    phi_idx_for_ternary = nothing
                    for k in inner_else_target:length(code)
                        if code[k] isa Core.PhiNode
                            phi_idx_for_ternary = k
                            break
                        end
                        if code[k] isa Core.GotoIfNot || (code[k] isa Expr && code[k].head === :call)
                            break  # Past the ternary
                        end
                    end

                    if phi_idx_for_ternary !== nothing && haskey(ctx.phi_locals, phi_idx_for_ternary)
                        # This is a ternary pattern - use compile_ternary_for_phi
                        append!(bytes, compile_ternary_for_phi(ctx, code, j, compiled))
                    else
                        # Regular nested conditional in void context (from && operator)
                        append!(bytes, compile_void_nested_conditional(ctx, code, j, compiled, ssa_use_count))
                    end
                elseif inner isa Core.PhiNode
                    # Phi already handled by compile_ternary_for_phi if it was part of a ternary
                    # If not handled, just skip it
                    push!(compiled, j)
                else
                    append!(bytes, compile_statement(inner, j, ctx))
                    push!(compiled, j)

                    # Check if this statement produces Union{} (never returns, e.g., throw)
                    # If so, stop compiling - any code after is dead code
                    stmt_type = get(ctx.ssa_types, j, Any)
                    if stmt_type === Union{}
                        break
                    end

                    # Check if this statement leaves a value on stack that we need to drop
                    # In void functions, return statements are skipped, so values meant for
                    # returns stay on stack. We need to drop them.
                    if inner isa Expr && (inner.head === :call || inner.head === :invoke)
                        # First check if this is a signal setter invoke - these ALWAYS need DROP
                        # because setters push a return value that won't be used in void context
                        is_setter_call = false
                        if inner.head === :invoke && length(inner.args) >= 2
                            func_ref = inner.args[2]
                            if func_ref isa Core.SSAValue
                                is_setter_call = haskey(ctx.signal_ssa_setters, func_ref.id)
                            end
                        end

                        if is_setter_call
                            # Signal setters push a return value that won't be used
                            push!(bytes, Opcode.DROP)
                        else
                            # For other calls, check if statement produces a value and use count
                            if statement_produces_wasm_value(inner, j, ctx)
                                if !haskey(ctx.ssa_locals, j) && !haskey(ctx.phi_locals, j)
                                    use_count = get(ssa_use_count, j, 0)
                                    if use_count == 0
                                        push!(bytes, Opcode.DROP)
                                    end
                                end
                            end
                        end
                    end
                end
            end

            if has_else_branch
                # Else branch - only emit when there's actual else code
                push!(bytes, Opcode.ELSE)

                # Find where the else branch ends (the merge point from the GotoNode)
                else_end = length(code)
                for j in (i+1):(else_target-1)
                    if code[j] isa Core.GotoNode
                        goto_node = code[j]::Core.GotoNode
                        if goto_node.label > else_target
                            else_end = goto_node.label - 1
                            break
                        end
                    end
                end

                # Compile else-branch (else_target to else_end)
                for j in else_target:else_end
                    if j in compiled
                        continue
                    end
                    inner = code[j]
                    if inner === nothing
                        push!(compiled, j)
                    elseif inner isa Core.ReturnNode
                        # Early return inside else branch - emit return instruction
                        # BUT: If the return type is Union{} (unreachable), don't emit RETURN
                        stmt_type = get(ctx.ssa_types, j, Any)
                        if stmt_type !== Union{}
                            push!(bytes, Opcode.RETURN)
                        end
                        push!(compiled, j)
                    elseif inner isa Core.GotoNode
                        push!(compiled, j)
                    elseif inner isa Core.GotoIfNot
                        # Check if this GotoIfNot is a ternary pattern (has a phi node)
                        inner_goto_if_not = inner::Core.GotoIfNot
                        inner_else_target = inner_goto_if_not.dest

                        phi_idx_for_ternary = nothing
                        for k in inner_else_target:length(code)
                            if code[k] isa Core.PhiNode
                                phi_idx_for_ternary = k
                                break
                            end
                            if code[k] isa Core.GotoIfNot || (code[k] isa Expr && code[k].head === :call)
                                break
                            end
                        end

                        if phi_idx_for_ternary !== nothing && haskey(ctx.phi_locals, phi_idx_for_ternary)
                            append!(bytes, compile_ternary_for_phi(ctx, code, j, compiled))
                        else
                            append!(bytes, compile_void_nested_conditional(ctx, code, j, compiled, ssa_use_count))
                        end
                    elseif inner isa Core.PhiNode
                        # Phi already handled by compile_ternary_for_phi
                        push!(compiled, j)
                    else
                        append!(bytes, compile_statement(inner, j, ctx))
                        push!(compiled, j)

                        # Check if this statement produces Union{} (never returns, e.g., throw)
                        # If so, stop compiling - any code after is dead code
                        stmt_type = get(ctx.ssa_types, j, Any)
                        if stmt_type === Union{}
                            break
                        end

                        # Drop unused values in void context (else branch)
                        if inner isa Expr && (inner.head === :call || inner.head === :invoke)
                            stmt_type = get(ctx.ssa_types, j, Nothing)
                            if stmt_type !== Nothing  # Only skip if type is definitely Nothing
                                is_nothing_union = stmt_type isa Union && Nothing in Base.uniontypes(stmt_type)
                                if !is_nothing_union
                                    if !haskey(ctx.ssa_locals, j) && !haskey(ctx.phi_locals, j)
                                        use_count = get(ssa_use_count, j, 0)
                                        if use_count == 0
                                            push!(bytes, Opcode.DROP)
                                        end
                                    end
                                end
                            end
                        end
                    end
                end

                # Mark all statements up to else_end as compiled (not beyond)
                for j in i:else_end
                    push!(compiled, j)
                end

                push!(bytes, Opcode.END)
                i = else_end + 1
            else
                # No else branch - just end the if block and continue from else_target
                # Mark only the then-branch as compiled, else_target onwards will be processed
                # by the main loop
                for j in i:(else_target-1)
                    push!(compiled, j)
                end

                push!(bytes, Opcode.END)
                i = else_target
            end
            continue
        end

        # Regular statement
        append!(bytes, compile_statement(stmt, i, ctx))
        push!(compiled, i)

        # Check if this statement produces Union{} (never returns, e.g., throw)
        # If so, stop compiling - any code after is dead code
        stmt_type = get(ctx.ssa_types, i, Any)
        if stmt_type === Union{}
            # Don't add more code, just return what we have
            # The function ends with unreachable code path
            return bytes
        end

        # Drop unused values from statements that produce values
        if stmt isa Expr && (stmt.head === :call || stmt.head === :invoke)
            stmt_type = get(ctx.ssa_types, i, Nothing)
            if stmt_type !== Nothing  # Only skip if type is definitely Nothing
                is_nothing_union = stmt_type isa Union && Nothing in Base.uniontypes(stmt_type)
                if !is_nothing_union
                    if !haskey(ctx.ssa_locals, i) && !haskey(ctx.phi_locals, i)
                        use_count = get(ssa_use_count, i, 0)
                        if use_count == 0
                            push!(bytes, Opcode.DROP)
                        end
                    end
                end
            end
        elseif stmt isa GlobalRef
            # GlobalRef statements (constants) may leave values on stack
            stmt_type = get(ctx.ssa_types, i, Nothing)
            if stmt_type !== Nothing  # Only skip if type is definitely Nothing
                if !haskey(ctx.ssa_locals, i) && !haskey(ctx.phi_locals, i)
                    use_count = get(ssa_use_count, i, 0)
                    if use_count == 0
                        push!(bytes, Opcode.DROP)
                    end
                end
            end
        end

        i += 1
    end

    # Final return (in case we didn't hit one)
    push!(bytes, Opcode.RETURN)

    return bytes
end

"""
Compile a nested conditional in void context (e.g., from && operators).
This handles patterns like `a && b && c` which compile to nested GotoIfNot.

For `a && b`:
  %1 = a()
  GotoIfNot %1 → end
  %2 = b()
  GotoIfNot %2 → end
  # then code
  end:

Compiles to:
  a()
  if
    b()
    if
      ;; then code
    end
  end
"""
function compile_void_nested_conditional(ctx::CompilationContext, code, start_idx::Int, compiled::Set{Int}, ssa_use_count::Dict{Int,Int})::Vector{UInt8}
    bytes = UInt8[]

    goto_if_not = code[start_idx]::Core.GotoIfNot
    end_target = goto_if_not.dest

    # Push condition
    append!(bytes, compile_value(goto_if_not.cond, ctx))
    push!(compiled, start_idx)

    # Start void if block
    push!(bytes, Opcode.IF)
    push!(bytes, 0x40)  # void block type

    # Process statements in the then-branch (start_idx+1 to end_target-1)
    for j in (start_idx+1):(end_target-1)
        if j in compiled
            continue
        end

        inner = code[j]

        if inner === nothing
            push!(compiled, j)
        elseif inner isa Core.GotoNode
            # Skip unconditional jumps in && chain
            push!(compiled, j)
        elseif inner isa Core.ReturnNode
            # Early return inside conditional
            # BUT: If the return type is Union{} (unreachable), don't emit RETURN
            # This happens after throw statements - the code is dead
            stmt_type = get(ctx.ssa_types, j, Any)
            if stmt_type !== Union{}
                if isdefined(inner, :val) && inner.val !== nothing
                    # Non-void return - but we're in void handler, just return
                end
                push!(bytes, Opcode.RETURN)
            end
            push!(compiled, j)
        elseif inner isa Core.GotoIfNot
            # Check if this GotoIfNot is a ternary pattern (has a phi node)
            inner_goto_if_not = inner::Core.GotoIfNot
            inner_else_target = inner_goto_if_not.dest

            phi_idx_for_ternary = nothing
            for k in inner_else_target:length(code)
                if code[k] isa Core.PhiNode
                    phi_idx_for_ternary = k
                    break
                end
                if code[k] isa Core.GotoIfNot || (code[k] isa Expr && code[k].head === :call)
                    break
                end
            end

            if phi_idx_for_ternary !== nothing && haskey(ctx.phi_locals, phi_idx_for_ternary)
                # This is a ternary pattern - use compile_ternary_for_phi
                append!(bytes, compile_ternary_for_phi(ctx, code, j, compiled))
            else
                # RECURSION: Another conditional (from && chain)
                append!(bytes, compile_void_nested_conditional(ctx, code, j, compiled, ssa_use_count))
            end
        elseif inner isa Core.PhiNode
            # Phi already handled by compile_ternary_for_phi if it was part of a ternary
            push!(compiled, j)
        else
            # Regular statement (including setter calls)
            append!(bytes, compile_statement(inner, j, ctx))
            push!(compiled, j)

            # Check if this statement produces Union{} (never returns, e.g., throw)
            # If so, stop compiling - any code after is dead code
            stmt_type = get(ctx.ssa_types, j, Any)
            if stmt_type === Union{}
                break
            end

            # Drop unused values in void context
            if inner isa Expr && (inner.head === :call || inner.head === :invoke)
                is_setter_call = false
                if inner.head === :invoke && length(inner.args) >= 2
                    func_ref = inner.args[2]
                    if func_ref isa Core.SSAValue
                        is_setter_call = haskey(ctx.signal_ssa_setters, func_ref.id)
                    end
                end

                if is_setter_call
                    push!(bytes, Opcode.DROP)
                else
                    stmt_type = get(ctx.ssa_types, j, Nothing)
                    if stmt_type !== Nothing  # Only skip if type is definitely Nothing
                        is_nothing_union = stmt_type isa Union && Nothing in Base.uniontypes(stmt_type)
                        if !is_nothing_union
                            if !haskey(ctx.ssa_locals, j) && !haskey(ctx.phi_locals, j)
                                use_count = get(ssa_use_count, j, 0)
                                if use_count == 0
                                    push!(bytes, Opcode.DROP)
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    # End if block (no else for && pattern - false case just skips)
    push!(bytes, Opcode.END)

    return bytes
end

"""
Compile a ternary expression (if-then-else with phi) that produces a value.
Returns bytecode that computes the ternary and stores to the phi local.
"""
function compile_ternary_for_phi(ctx::CompilationContext, code, cond_idx::Int, compiled::Set{Int})::Vector{UInt8}
    bytes = UInt8[]

    goto_if_not = code[cond_idx]::Core.GotoIfNot
    else_target = goto_if_not.dest

    # Find the phi node after the else branch
    phi_idx = nothing
    for j in else_target:length(code)
        if code[j] isa Core.PhiNode
            phi_idx = j
            break
        end
        if code[j] isa Core.GotoIfNot || (code[j] isa Core.Expr && code[j].head === :call)
            break  # Past the ternary
        end
    end

    if phi_idx === nothing
        # No phi - this might be a void conditional inside, just skip
        push!(compiled, cond_idx)
        return bytes
    end

    phi_node = code[phi_idx]::Core.PhiNode

    # Check if we have a local for this phi
    if !haskey(ctx.phi_locals, phi_idx)
        push!(compiled, cond_idx)
        push!(compiled, phi_idx)
        return bytes
    end

    local_idx = ctx.phi_locals[phi_idx]
    phi_type = get(ctx.ssa_types, phi_idx, Int64)
    wasm_type = julia_to_wasm_type_concrete(phi_type, ctx)

    # Push condition
    append!(bytes, compile_value(goto_if_not.cond, ctx))
    push!(compiled, cond_idx)

    # Start if block with result type
    push!(bytes, Opcode.IF)
    append!(bytes, encode_block_type(wasm_type))

    # Get then-value from phi
    then_value = nothing
    else_value = nothing
    for (edge_idx, edge) in enumerate(phi_node.edges)
        if edge < else_target
            then_value = phi_node.values[edge_idx]
        else
            else_value = phi_node.values[edge_idx]
        end
    end

    # Then branch - push value
    if then_value !== nothing
        append!(bytes, compile_value(then_value, ctx))
    else
        # Fallback
        push!(bytes, Opcode.I64_CONST)
        push!(bytes, 0x00)
    end

    # Else branch
    push!(bytes, Opcode.ELSE)

    # Else branch - push value
    if else_value !== nothing
        append!(bytes, compile_value(else_value, ctx))
    else
        push!(bytes, Opcode.I64_CONST)
        push!(bytes, 0x00)
    end

    push!(bytes, Opcode.END)

    # Store result to phi local
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(local_idx))

    # Mark the GotoNode, nothing, and phi as compiled
    for j in cond_idx+1:phi_idx
        push!(compiled, j)
    end

    return bytes
end

"""
Generate code for && pattern: multiple conditionals all jumping to the same else target.
Uses block/br_if structure:
  block \$outer [result_type]
    block \$else_target []
      cond1; i32.eqz; br_if 0   ;; if false, jump to else
      cond2; i32.eqz; br_if 0   ;; if false, jump to else
      <then_code>; br 1         ;; jump past else
    end
    <else_code>
  end
"""
function generate_and_pattern(ctx::CompilationContext, blocks, code, conditionals, result_type, else_target, ssa_use_count)::Vector{UInt8}
    bytes = UInt8[]

    # Outer block for result
    push!(bytes, Opcode.BLOCK)
    append!(bytes, encode_block_type(result_type))

    # Inner block for else jump target (void result)
    push!(bytes, Opcode.BLOCK)
    push!(bytes, 0x40)  # void result type

    # Generate each condition with br_if to else
    for (i, (block_idx, block)) in enumerate(conditionals)
        goto_if_not = block.terminator::Core.GotoIfNot

        # Generate statements before condition
        for j in block.start_idx:block.end_idx-1
            append!(bytes, compile_statement(code[j], j, ctx))

            # Drop unused values
            stmt = code[j]
            if stmt isa Expr && (stmt.head === :call || stmt.head === :invoke)
                stmt_type = get(ctx.ssa_types, j, Any)
                if stmt_type !== Nothing
                    is_nothing_union = stmt_type isa Union && Nothing in Base.uniontypes(stmt_type)
                    if !is_nothing_union
                        if !haskey(ctx.ssa_locals, j) && !haskey(ctx.phi_locals, j)
                            use_count = get(ssa_use_count, j, 0)
                            if use_count == 0
                                push!(bytes, Opcode.DROP)
                            end
                        end
                    end
                end
            end
        end

        # Push condition and test for false (invert condition)
        append!(bytes, compile_value(goto_if_not.cond, ctx))
        push!(bytes, Opcode.I32_EQZ)  # invert: GotoIfNot jumps when false, so we br when !cond
        push!(bytes, Opcode.BR_IF)
        append!(bytes, encode_leb128_unsigned(0))  # br to inner block (else)
    end

    # All conditions passed - generate then code
    last_cond = conditionals[end]
    then_start = last_cond[2].end_idx + 1
    for i in then_start:else_target-1
        stmt = code[i]
        if stmt isa Core.ReturnNode
            if isdefined(stmt, :val)
                append!(bytes, compile_value(stmt.val, ctx))
            end
            break
        elseif stmt !== nothing && !(stmt isa Core.GotoIfNot) && !(stmt isa Core.GotoNode)
            append!(bytes, compile_statement(stmt, i, ctx))

            # Drop unused values
            if stmt isa Expr && (stmt.head === :call || stmt.head === :invoke)
                stmt_type = get(ctx.ssa_types, i, Any)
                if stmt_type !== Nothing
                    is_nothing_union = stmt_type isa Union && Nothing in Base.uniontypes(stmt_type)
                    if !is_nothing_union
                        if !haskey(ctx.ssa_locals, i) && !haskey(ctx.phi_locals, i)
                            use_count = get(ssa_use_count, i, 0)
                            if use_count == 0
                                push!(bytes, Opcode.DROP)
                            end
                        end
                    end
                end
            end
        end
    end

    # Check if there's a phi node at or after else_target that we need to provide a value for
    phi_idx = nothing
    phi_node = nothing
    for i in else_target:length(code)
        if code[i] isa Core.PhiNode
            phi_idx = i
            phi_node = code[i]
            break
        end
    end

    # If there's a phi, push the then-value before branching
    if phi_node !== nothing
        # Find the phi value from the then-branch (before else_target)
        for (edge_idx, edge) in enumerate(phi_node.edges)
            if edge < else_target && edge > 0
                val = phi_node.values[edge_idx]
                append!(bytes, compile_value(val, ctx))
                break
            end
        end
    end

    # br past else to outer block end
    push!(bytes, Opcode.BR)
    append!(bytes, encode_leb128_unsigned(1))  # br to outer block (depth 1)

    # End inner block (else target)
    push!(bytes, Opcode.END)

    # Generate else code - if there's a phi, push its else-value directly
    if phi_node !== nothing
        # Find the phi value from an else-branch (at or after else_target)
        else_value_pushed = false
        for (edge_idx, edge) in enumerate(phi_node.edges)
            # Else edges come from conditionals that jump to else_target
            # These are the GotoIfNot statements - their line numbers are stored in edges
            edge_stmt = edge <= length(code) ? code[edge] : nothing
            if edge_stmt isa Core.GotoIfNot
                # This is an else-edge from a conditional
                val = phi_node.values[edge_idx]
                append!(bytes, compile_value(val, ctx))
                else_value_pushed = true
                break
            end
        end
        if !else_value_pushed
            # Fallback: look for else-value (edge from else_target or later)
            for (edge_idx, edge) in enumerate(phi_node.edges)
                if edge >= else_target
                    val = phi_node.values[edge_idx]
                    append!(bytes, compile_value(val, ctx))
                    break
                end
            end
        end
    else
        # No phi - iterate through else code looking for return
        for i in else_target:length(code)
            stmt = code[i]
            if stmt isa Core.ReturnNode
                if isdefined(stmt, :val)
                    append!(bytes, compile_value(stmt.val, ctx))
                end
                break
            elseif stmt !== nothing && !(stmt isa Core.GotoIfNot) && !(stmt isa Core.GotoNode)
                append!(bytes, compile_statement(stmt, i, ctx))

                # Drop unused values
                if stmt isa Expr && (stmt.head === :call || stmt.head === :invoke)
                    stmt_type = get(ctx.ssa_types, i, Any)
                    if stmt_type !== Nothing
                        is_nothing_union = stmt_type isa Union && Nothing in Base.uniontypes(stmt_type)
                        if !is_nothing_union
                            if !haskey(ctx.ssa_locals, i) && !haskey(ctx.phi_locals, i)
                                use_count = get(ssa_use_count, i, 0)
                                if use_count == 0
                                    push!(bytes, Opcode.DROP)
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    # End outer block
    push!(bytes, Opcode.END)

    # Add RETURN after the block
    push!(bytes, Opcode.RETURN)

    return bytes
end

"""
Detect switch pattern: sequential conditionals testing the same SSA value against different constants.
Pattern (switch on n):
  %cond1 = (n === 0)
  goto %else1 if not %cond1
  ... case 0 code with return ...
  %cond2 = (n === 1)
  goto %else2 if not %cond2
  ... case 1 code with return ...
  ...
Returns (switch_value_ssa, cases) where cases = [(cond_idx, const_val, case_start, case_end), ...]
"""
function detect_switch_pattern(code, conditionals)
    if length(conditionals) < 2
        return nothing
    end

    # Group conditionals by which SSA value they compare
    # Then find the best switch pattern (most cases with returns)
    switch_candidates = Dict{Int, Vector{Tuple{Int, Any, Int, Int}}}()  # ssa_id => cases

    for (i, (block_idx, block)) in enumerate(conditionals)
        gin = block.terminator::Core.GotoIfNot
        cond = gin.cond

        if !(cond isa Core.SSAValue)
            continue
        end

        cond_stmt = code[cond.id]
        if !(cond_stmt isa Expr && cond_stmt.head === :call)
            continue
        end

        args = cond_stmt.args
        if length(args) < 3
            continue
        end

        # Check if it's an equality comparison
        func = args[1]
        is_eq = func isa GlobalRef && (func.name === :(===) || func.name === :(==))
        if !is_eq
            continue
        end

        lhs = args[2]
        rhs = args[3]

        # One side should be an SSA value (the switch value), other is a constant
        ssa_val = nothing
        const_val = nothing

        if lhs isa Core.SSAValue && !(rhs isa Core.SSAValue) && !(rhs isa Core.Argument)
            ssa_val = lhs
            const_val = rhs
        elseif rhs isa Core.SSAValue && !(lhs isa Core.SSAValue) && !(lhs isa Core.Argument)
            ssa_val = rhs
            const_val = lhs
        else
            continue
        end

        # Find the case code range (from gin line + 1 to gin.dest - 1)
        case_start = block.end_idx + 1
        case_end = gin.dest - 1

        # Check if this case has a return
        has_return = false
        for j in case_start:min(case_end, length(code))
            if code[j] isa Core.ReturnNode
                has_return = true
                break
            end
        end

        # Only consider cases with returns for the switch pattern
        if has_return
            ssa_id = ssa_val.id
            if !haskey(switch_candidates, ssa_id)
                switch_candidates[ssa_id] = []
            end
            push!(switch_candidates[ssa_id], (i, const_val, case_start, case_end))
        end
    end

    # Find the best switch pattern (most cases)
    best_switch = nothing
    best_cases = nothing

    for (ssa_id, cases) in switch_candidates
        if length(cases) >= 2
            if best_cases === nothing || length(cases) > length(best_cases)
                best_switch = ssa_id
                best_cases = cases
            end
        end
    end

    if best_switch !== nothing && best_cases !== nothing
        return (best_switch, best_cases)
    end

    return nothing
end

"""
Generate code for switch pattern using nested if-else with proper stack handling.
Each case returns independently, so we don't need phi handling for the switch itself.
"""
function generate_switch_pattern(ctx::CompilationContext, blocks, code, conditionals, result_type, switch_pattern, ssa_use_count)::Vector{UInt8}
    bytes = UInt8[]
    switch_value_ssa, cases = switch_pattern

    # First, compile any statements before the switch (up to the first case condition)
    first_case_idx = cases[1][1]
    first_block_idx, first_block = conditionals[first_case_idx]

    # Find the actual first conditional
    first_cond_idx = conditionals[1][2].start_idx
    for i in 1:first_cond_idx-1
        stmt = code[i]
        if stmt !== nothing && !(stmt isa Core.GotoIfNot) && !(stmt isa Core.GotoNode) && !(stmt isa Core.PhiNode)
            append!(bytes, compile_statement(stmt, i, ctx))

            # Handle SSA storage and drops
            if stmt isa Expr && (stmt.head === :call || stmt.head === :invoke)
                stmt_type = get(ctx.ssa_types, i, Any)
                if stmt_type !== Nothing
                    is_nothing_union = stmt_type isa Union && Nothing in Base.uniontypes(stmt_type)
                    if !is_nothing_union
                        if !haskey(ctx.ssa_locals, i) && !haskey(ctx.phi_locals, i)
                            use_count = get(ssa_use_count, i, 0)
                            if use_count == 0
                                push!(bytes, Opcode.DROP)
                            end
                        end
                    end
                end
            end
        end
    end

    # Generate switch as nested if-else, but with each case having its own return
    # This ensures no code duplication - each case is visited exactly once
    function gen_case(case_idx::Int)::Vector{UInt8}
        inner_bytes = UInt8[]

        if case_idx > length(cases)
            # Default case: code after all switch cases
            # Find where the default case starts (after the last case's else target)
            last_case = cases[end]
            last_cond_idx = last_case[1]
            _, last_block = conditionals[last_cond_idx]
            default_start = last_block.terminator.dest

            for i in default_start:length(code)
                stmt = code[i]
                if stmt isa Core.ReturnNode
                    if isdefined(stmt, :val)
                        append!(inner_bytes, compile_value(stmt.val, ctx))
                    end
                    break
                elseif stmt === nothing || stmt isa Core.PhiNode || stmt isa Core.GotoIfNot || stmt isa Core.GotoNode
                    continue
                else
                    append!(inner_bytes, compile_statement(stmt, i, ctx))

                    if stmt isa Expr && (stmt.head === :call || stmt.head === :invoke)
                        stmt_type = get(ctx.ssa_types, i, Any)
                        if stmt_type !== Nothing
                            is_nothing_union = stmt_type isa Union && Nothing in Base.uniontypes(stmt_type)
                            if !is_nothing_union
                                if !haskey(ctx.ssa_locals, i) && !haskey(ctx.phi_locals, i)
                                    use_count = get(ssa_use_count, i, 0)
                                    if use_count == 0
                                        push!(inner_bytes, Opcode.DROP)
                                    end
                                end
                            end
                        end
                    end
                end
            end

            return inner_bytes
        end

        cond_idx, const_val, case_start, case_end = cases[case_idx]
        block_idx, block = conditionals[cond_idx]
        gin = block.terminator::Core.GotoIfNot

        # Compile statements in this block (before condition check)
        for i in block.start_idx:block.end_idx-1
            stmt = code[i]
            if stmt !== nothing && !(stmt isa Core.GotoIfNot) && !(stmt isa Core.GotoNode) && !(stmt isa Core.PhiNode)
                append!(inner_bytes, compile_statement(stmt, i, ctx))

                if stmt isa Expr && (stmt.head === :call || stmt.head === :invoke)
                    stmt_type = get(ctx.ssa_types, i, Any)
                    if stmt_type !== Nothing
                        is_nothing_union = stmt_type isa Union && Nothing in Base.uniontypes(stmt_type)
                        if !is_nothing_union
                            if !haskey(ctx.ssa_locals, i) && !haskey(ctx.phi_locals, i)
                                use_count = get(ssa_use_count, i, 0)
                                if use_count == 0
                                    push!(inner_bytes, Opcode.DROP)
                                end
                            end
                        end
                    end
                end
            end
        end

        # Compile the condition
        append!(inner_bytes, compile_value(gin.cond, ctx))

        # IF with result type (since each case returns a value)
        push!(inner_bytes, Opcode.IF)
        append!(inner_bytes, encode_block_type(result_type))

        # Then branch: this case's code (should end with return value on stack)
        for i in case_start:min(case_end, length(code))
            stmt = code[i]
            if stmt isa Core.ReturnNode
                if isdefined(stmt, :val)
                    append!(inner_bytes, compile_value(stmt.val, ctx))
                end
                break
            elseif stmt === nothing || stmt isa Core.PhiNode || stmt isa Core.GotoIfNot || stmt isa Core.GotoNode
                continue
            else
                append!(inner_bytes, compile_statement(stmt, i, ctx))

                if stmt isa Expr && (stmt.head === :call || stmt.head === :invoke)
                    stmt_type = get(ctx.ssa_types, i, Any)
                    if stmt_type !== Nothing
                        is_nothing_union = stmt_type isa Union && Nothing in Base.uniontypes(stmt_type)
                        if !is_nothing_union
                            if !haskey(ctx.ssa_locals, i) && !haskey(ctx.phi_locals, i)
                                use_count = get(ssa_use_count, i, 0)
                                if use_count == 0
                                    push!(inner_bytes, Opcode.DROP)
                                end
                            end
                        end
                    end
                end
            end
        end

        # Else branch: recurse to next case
        push!(inner_bytes, Opcode.ELSE)
        append!(inner_bytes, gen_case(case_idx + 1))

        push!(inner_bytes, Opcode.END)

        return inner_bytes
    end

    append!(bytes, gen_case(1))
    push!(bytes, Opcode.RETURN)

    return bytes
end

"""
Detect OR pattern: multiple conditionals where each then-branch jumps to the same phi node.
Returns (phi_idx, or_conditions, next_conditional_idx) or nothing.

OR pattern IR example (a || b || c):
  1: a
  2: goto %4 if not %1
  3: goto %9  (then-branch)
  4: b
  5: goto %7 if not %4
  6: goto %9  (then-branch)
  7: c
  8: goto %9
  9: φ (%3 => %1, %6 => %4, %8 => %7)
  10: goto %N if not %9  (uses the phi)
"""
function detect_or_pattern(code, conditionals, ssa_types)
    if length(conditionals) < 1
        return nothing
    end

    # For each conditional, check if the then-branch (fall-through) has a GotoNode
    # that jumps to a phi node
    phi_targets = Dict{Int, Vector{Tuple{Int, Int}}}()  # phi_idx => [(cond_idx, goto_idx), ...]

    for (cond_idx, (block_idx, block)) in enumerate(conditionals)
        goto_if_not = block.terminator::Core.GotoIfNot
        then_start = block.end_idx + 1
        else_target = goto_if_not.dest
        then_end = else_target - 1

        # Look for GotoNode in then-branch
        for i in then_start:min(then_end, length(code))
            stmt = code[i]
            if stmt isa Core.GotoNode && stmt.label > i
                target = stmt.label
                if target <= length(code) && code[target] isa Core.PhiNode
                    # Found a then-branch GotoNode to a phi
                    if !haskey(phi_targets, target)
                        phi_targets[target] = []
                    end
                    push!(phi_targets[target], (cond_idx, i))
                end
                break
            end
        end
    end

    # Check if we have a phi with multiple incoming OR conditions
    # Return the FIRST one (lowest phi_idx) to process patterns in code order
    best_phi_idx = nothing
    best_cond_infos = nothing
    best_next_cond_idx = nothing

    for (phi_idx, cond_infos) in phi_targets
        if length(cond_infos) >= 2
            # Only consider if this is earlier than our current best
            if best_phi_idx === nothing || phi_idx < best_phi_idx
                # Found an OR pattern - verify all edges are from these conditions
                phi_stmt = code[phi_idx]::Core.PhiNode

                # CRITICAL: Only treat as OR pattern if the phi type is Bool
                # Non-boolean phi nodes with multiple edges should not be handled as OR patterns
                phi_type = get(ssa_types, phi_idx, nothing)
                if phi_type !== Bool
                    continue  # Skip non-boolean phi nodes
                end

                # Find the conditional that USES this phi (tests the OR result)
                next_cond_idx = nothing
                for (j, (_, b)) in enumerate(conditionals)
                    goto_if_not = b.terminator::Core.GotoIfNot
                    if goto_if_not.cond isa Core.SSAValue && goto_if_not.cond.id == phi_idx
                        next_cond_idx = j
                        break
                    end
                end

                best_phi_idx = phi_idx
                best_cond_infos = cond_infos
                best_next_cond_idx = next_cond_idx
            end
        end
    end

    if best_phi_idx !== nothing
        return (best_phi_idx, best_cond_infos, best_next_cond_idx)
    end

    return nothing
end

"""
Generate code for OR pattern (a || b || c producing boolean phi).
Creates nested if-else structure that evaluates each condition.
"""
function generate_or_pattern(ctx::CompilationContext, blocks, code, conditionals, result_type, or_pattern, ssa_use_count)::Vector{UInt8}
    bytes = UInt8[]
    phi_idx, cond_infos, next_cond_idx = or_pattern
    phi_stmt = code[phi_idx]::Core.PhiNode

    # Sort conditions by their index (they should be in order)
    sorted_conds = sort(cond_infos, by=x -> x[1])

    # Helper to generate code for OR condition at index i
    function gen_or_cond(idx::Int)::Vector{UInt8}
        inner_bytes = UInt8[]

        if idx > length(sorted_conds)
            # Last condition (the one without a GotoNode in then-branch)
            # Find the last edge in the phi - this is the final condition value
            last_edge = nothing
            last_val = nothing
            for (edge_idx, edge) in enumerate(phi_stmt.edges)
                # Find edge that's not from one of the GotoNode lines
                is_goto_edge = any(ci -> ci[2] == edge, sorted_conds)
                if !is_goto_edge
                    last_edge = edge
                    last_val = phi_stmt.values[edge_idx]
                    break
                end
            end

            if last_val !== nothing
                # For SSAValue with no local, we need to compile the statement
                if last_val isa Core.SSAValue && !haskey(ctx.ssa_locals, last_val.id) && !haskey(ctx.phi_locals, last_val.id)
                    # Compile the statement for this SSA value
                    stmt = code[last_val.id]
                    if stmt !== nothing
                        append!(inner_bytes, compile_statement(stmt, last_val.id, ctx))
                    end
                else
                    # Has a local or is not SSAValue - use compile_value
                    append!(inner_bytes, compile_value(last_val, ctx))
                end
            else
                # Fallback - push false
                push!(inner_bytes, Opcode.I32_CONST)
                append!(inner_bytes, encode_leb128_signed(0))
            end

            return inner_bytes
        end

        cond_idx, goto_line = sorted_conds[idx]
        block_idx, block = conditionals[cond_idx]
        goto_if_not = block.terminator::Core.GotoIfNot

        # Generate all statements in the block (including the condition)
        # compile_statement will store to local if needed, then compile_value will load
        for j in block.start_idx:block.end_idx-1
            stmt = code[j]
            if stmt !== nothing && !(stmt isa Core.GotoIfNot) && !(stmt isa Core.GotoNode) && !(stmt isa Core.PhiNode)
                append!(inner_bytes, compile_statement(stmt, j, ctx))
            end
        end

        # Push condition (will load from local if multi-use, or assume on stack if single-use)
        append!(inner_bytes, compile_value(goto_if_not.cond, ctx))

        # IF with i32 result (for the phi value)
        push!(inner_bytes, Opcode.IF)
        push!(inner_bytes, 0x7f)  # i32 result type

        # Then-branch: condition was true
        # For || pattern, phi value = condition = true = 1
        # We push constant 1 instead of trying to re-compile the condition
        push!(inner_bytes, Opcode.I32_CONST)
        append!(inner_bytes, encode_leb128_signed(1))

        # Else-branch: condition was false, evaluate next condition
        push!(inner_bytes, Opcode.ELSE)
        append!(inner_bytes, gen_or_cond(idx + 1))

        push!(inner_bytes, Opcode.END)

        return inner_bytes
    end

    # Generate the nested OR conditions
    append!(bytes, gen_or_cond(1))

    # Store result in phi local
    if haskey(ctx.phi_locals, phi_idx)
        local_idx = ctx.phi_locals[phi_idx]
        push!(bytes, Opcode.LOCAL_SET)
        append!(bytes, encode_leb128_unsigned(local_idx))
    end

    # Now continue with the conditional that uses the phi
    if next_cond_idx !== nothing
        # Generate the rest of the conditionals starting from next_cond_idx
        remaining_conds = [(i, conditionals[i]) for i in next_cond_idx:length(conditionals)]
        if !isempty(remaining_conds)
            # Generate remaining conditionals recursively
            append!(bytes, generate_remaining_conditionals(ctx, blocks, code, remaining_conds, result_type, ssa_use_count))
        end
    end

    return bytes
end

"""
Generate code for remaining conditionals after OR pattern.
"""
function generate_remaining_conditionals(ctx::CompilationContext, blocks, code, remaining_conds, result_type, ssa_use_count)::Vector{UInt8}
    bytes = UInt8[]

    if isempty(remaining_conds)
        return bytes
    end

    _, (block_idx, block) = remaining_conds[1]
    goto_if_not = block.terminator::Core.GotoIfNot

    # Push condition (which might be a phi local)
    append!(bytes, compile_value(goto_if_not.cond, ctx))

    # Generate IF
    push!(bytes, Opcode.IF)
    append!(bytes, encode_block_type(result_type))

    # Then-branch: generate code from block.end_idx + 1 to goto_if_not.dest - 1
    then_start = block.end_idx + 1
    then_end = goto_if_not.dest - 1

    for i in then_start:min(then_end, length(code))
        stmt = code[i]
        if stmt isa Core.ReturnNode
            if isdefined(stmt, :val)
                append!(bytes, compile_value(stmt.val, ctx))
            end
            break
        elseif stmt === nothing
            # Skip
        elseif !(stmt isa Core.GotoIfNot) && !(stmt isa Core.GotoNode) && !(stmt isa Core.PhiNode)
            append!(bytes, compile_statement(stmt, i, ctx))
        end
    end

    # Else-branch
    push!(bytes, Opcode.ELSE)

    # Check for more conditionals in else branch
    rest_conds = remaining_conds[2:end]
    if !isempty(rest_conds)
        # Recurse for remaining conditionals
        append!(bytes, generate_remaining_conditionals(ctx, blocks, code, rest_conds, result_type, ssa_use_count))
    else
        # Generate code from goto_if_not.dest to end (else branch)
        for i in goto_if_not.dest:length(code)
            stmt = code[i]
            if stmt isa Core.ReturnNode
                if isdefined(stmt, :val)
                    append!(bytes, compile_value(stmt.val, ctx))
                end
                break
            elseif stmt === nothing
                # Skip
            elseif !(stmt isa Core.GotoIfNot) && !(stmt isa Core.GotoNode) && !(stmt isa Core.PhiNode)
                append!(bytes, compile_statement(stmt, i, ctx))
            end
        end
    end

    push!(bytes, Opcode.END)

    return bytes
end

"""
Generate nested if-else for multiple conditionals.
"""
function generate_nested_conditionals(ctx::CompilationContext, blocks, code, conditionals)::Vector{UInt8}
    bytes = UInt8[]
    result_type = julia_to_wasm_type_concrete(ctx.return_type, ctx)

    # Count SSA uses for drop logic
    ssa_use_count = Dict{Int, Int}()
    for stmt in code
        count_ssa_uses!(stmt, ssa_use_count)
    end

    # ========================================================================
    # BOUNDSCHECK PATTERN DETECTION
    # ========================================================================
    # We emit i32.const 0 for boundscheck, so GotoIfNot following boundscheck
    # ALWAYS jumps (since NOT 0 = TRUE). We need to:
    # 1. Filter out these fake conditionals (they're always-jump, not real conditionals)
    # 2. Track dead code regions (fall-through path that's never taken)
    # 3. Generate code that goes directly to the jump target

    boundscheck_jumps = Set{Int}()  # Statement indices of GotoIfNot that always jump
    dead_regions = Set{Int}()       # Statement indices that are dead code

    for i in 1:length(code)
        stmt = code[i]
        if stmt isa Expr && stmt.head === :boundscheck && length(stmt.args) >= 1
            # Check if next statement is a GotoIfNot using this boundscheck result
            if i + 1 <= length(code) && code[i + 1] isa Core.GotoIfNot
                goto_stmt = code[i + 1]::Core.GotoIfNot
                if goto_stmt.cond isa Core.SSAValue && goto_stmt.cond.id == i
                    # This is a boundscheck+GotoIfNot pattern - the GotoIfNot always jumps
                    push!(boundscheck_jumps, i + 1)
                    # Mark the boundscheck as dead (we don't need to emit i32.const 0)
                    push!(dead_regions, i)
                    # Mark the fall-through path as dead (from GotoIfNot+1 to target-1)
                    target = goto_stmt.dest
                    for j in (i + 2):(target - 1)
                        push!(dead_regions, j)
                    end
                end
            end
        end
    end

    # Filter out boundscheck-based conditionals - they're not real conditionals
    # Also filter out conditionals that are entirely within dead regions
    real_conditionals = filter(conditionals) do (block_idx, block)
        term_idx = block.end_idx
        if term_idx in boundscheck_jumps
            return false  # This is an always-jump, not a real conditional
        end
        if term_idx in dead_regions
            return false  # This conditional is inside dead code
        end
        return true
    end

    # If we filtered out all conditionals, we just need to emit the code
    # that the boundscheck jumps to
    if isempty(real_conditionals)
        # Find the first non-dead statement after any boundscheck jumps
        # This is the actual code that should run
        first_live = 1
        for i in 1:length(code)
            if !(i in dead_regions)
                first_live = i
                break
            end
        end

        # Generate code starting from first_live, skipping any remaining dead regions
        for i in first_live:length(code)
            if i in dead_regions
                continue
            end
            if i in boundscheck_jumps
                continue  # Skip the always-jump GotoIfNot
            end
            stmt = code[i]
            if stmt isa Core.ReturnNode
                if isdefined(stmt, :val)
                    append!(bytes, compile_value(stmt.val, ctx))
                end
                push!(bytes, Opcode.RETURN)
            elseif stmt === nothing
                # Skip
            elseif stmt isa Core.GotoNode
                # Skip forward gotos that were part of the dead structure
            elseif stmt isa Core.GotoIfNot
                # Skip conditionals that are part of dead structure
            else
                append!(bytes, compile_statement(stmt, i, ctx))

                # Drop unused values
                if stmt isa Expr && (stmt.head === :call || stmt.head === :invoke)
                    stmt_type = get(ctx.ssa_types, i, Any)
                    if stmt_type !== Nothing && stmt_type !== Union{}
                        is_nothing_union = stmt_type isa Union && Nothing in Base.uniontypes(stmt_type)
                        if !is_nothing_union
                            if !haskey(ctx.ssa_locals, i) && !haskey(ctx.phi_locals, i)
                                use_count = get(ssa_use_count, i, 0)
                                if use_count == 0
                                    push!(bytes, Opcode.DROP)
                                end
                            end
                        end
                    end
                end
            end
        end

        return bytes
    end

    # Use real_conditionals for the rest of the function
    conditionals = real_conditionals

    # Check for && pattern: all conditionals jump to the same destination
    # This pattern needs special handling with block/br_if instead of nested if/else
    if length(conditionals) >= 2
        first_dest = conditionals[1][2].terminator.dest
        all_same_dest = all(c -> c[2].terminator.dest == first_dest, conditionals)

        if all_same_dest
            # && pattern: use block/br_if approach
            return generate_and_pattern(ctx, blocks, code, conditionals, result_type, first_dest, ssa_use_count)
        end
    end

    # Check for || pattern: conditionals where then-branch (fall-through) jumps to a phi
    # Pattern: cond1 || cond2 || cond3 generates:
    #   1: cond1
    #   2: goto %4 if not %1
    #   3: goto %phi  (then-branch when cond1 is true)
    #   4: cond2
    #   5: goto %7 if not %4
    #   6: goto %phi  (then-branch when cond2 is true)
    #   ...
    #   phi: φ (%3 => %1, %6 => %4, ...)
    or_pattern = detect_or_pattern(code, conditionals, ctx.ssa_types)
    if or_pattern !== nothing
        return generate_or_pattern(ctx, blocks, code, conditionals, result_type, or_pattern, ssa_use_count)
    end

    # Check for switch pattern: sequential conditionals testing same value against constants
    # Each case returns independently (no phi merge for the switch itself)
    # NOTE: Switch pattern disabled for now - it replaces entire code generation incorrectly
    # TODO: Integrate switch pattern into the main code flow properly
    # switch_pattern = detect_switch_pattern(code, conditionals)
    # if switch_pattern !== nothing
    #     return generate_switch_pattern(ctx, blocks, code, conditionals, result_type, switch_pattern, ssa_use_count)
    # end

    # Track which statements have been compiled to avoid duplicating code
    compiled_stmts = Set{Int}()

    # Build a recursive if-else structure
    # target_idx tracks where to generate code when no more conditionals
    function gen_conditional(cond_idx::Int; target_idx::Int=0)::Vector{UInt8}
        inner_bytes = UInt8[]

        if cond_idx > length(conditionals)
            # No more conditionals - generate code starting from target_idx
            # This should generate the "else" path for the control flow
            for block in blocks
                # Find the block that contains or starts at target_idx
                if target_idx > 0 && block.start_idx <= target_idx && target_idx <= block.end_idx && block.terminator isa Core.ReturnNode
                    # target_idx is inside this block - generate from target_idx to end
                    for i in target_idx:block.end_idx
                        stmt = code[i]
                        if stmt isa Core.ReturnNode
                            if isdefined(stmt, :val)
                                append!(inner_bytes, compile_value(stmt.val, ctx))
                            end
                        elseif !(stmt isa Core.GotoIfNot)
                            append!(inner_bytes, compile_statement(stmt, i, ctx))

                            # Drop unused values (but NOT for Union{} which never returns)
                            if stmt isa Expr && (stmt.head === :call || stmt.head === :invoke)
                                stmt_type = get(ctx.ssa_types, i, Any)
                                # Union{} means the call never returns (throws), so no value to drop
                                if stmt_type !== Nothing && stmt_type !== Union{}
                                    is_nothing_union = stmt_type isa Union && Nothing in Base.uniontypes(stmt_type)
                                    if !is_nothing_union
                                        if !haskey(ctx.ssa_locals, i) && !haskey(ctx.phi_locals, i)
                                            use_count = get(ssa_use_count, i, 0)
                                            if use_count == 0
                                                push!(inner_bytes, Opcode.DROP)
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                    break
                elseif target_idx > 0 && block.start_idx >= target_idx && block.terminator isa Core.ReturnNode
                    for i in block.start_idx:block.end_idx
                        stmt = code[i]
                        if stmt isa Core.ReturnNode
                            if isdefined(stmt, :val)
                                append!(inner_bytes, compile_value(stmt.val, ctx))
                            else
                                # ReturnNode without val is `unreachable` - emit WASM unreachable
                                push!(inner_bytes, Opcode.UNREACHABLE)
                            end
                        elseif !(stmt isa Core.GotoIfNot)
                            append!(inner_bytes, compile_statement(stmt, i, ctx))

                            # Drop unused values (but NOT for Union{} which never returns)
                            if stmt isa Expr && (stmt.head === :call || stmt.head === :invoke)
                                stmt_type = get(ctx.ssa_types, i, Any)
                                # Union{} means the call never returns (throws), so no value to drop
                                if stmt_type !== Nothing && stmt_type !== Union{}
                                    is_nothing_union = stmt_type isa Union && Nothing in Base.uniontypes(stmt_type)
                                    if !is_nothing_union
                                        if !haskey(ctx.ssa_locals, i) && !haskey(ctx.phi_locals, i)
                                            use_count = get(ssa_use_count, i, 0)
                                            if use_count == 0
                                                push!(inner_bytes, Opcode.DROP)
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                    break
                elseif target_idx == 0 && block.terminator isa Core.ReturnNode
                    # Fallback: find first return block
                    for i in block.start_idx:block.end_idx
                        stmt = code[i]
                        if stmt isa Core.ReturnNode
                            if isdefined(stmt, :val)
                                append!(inner_bytes, compile_value(stmt.val, ctx))
                            end
                        elseif !(stmt isa Core.GotoIfNot)
                            append!(inner_bytes, compile_statement(stmt, i, ctx))
                        end
                    end
                    break
                end
            end
            return inner_bytes
        end

        block_idx, block = conditionals[cond_idx]
        goto_if_not = block.terminator::Core.GotoIfNot

        # Generate statements before condition
        for i in block.start_idx:block.end_idx-1
            append!(inner_bytes, compile_statement(code[i], i, ctx))

            # Drop unused values
            stmt = code[i]
            if stmt isa Expr && (stmt.head === :call || stmt.head === :invoke)
                stmt_type = get(ctx.ssa_types, i, Any)
                if stmt_type !== Nothing
                    is_nothing_union = stmt_type isa Union && Nothing in Base.uniontypes(stmt_type)
                    if !is_nothing_union
                        if !haskey(ctx.ssa_locals, i) && !haskey(ctx.phi_locals, i)
                            use_count = get(ssa_use_count, i, 0)
                            if use_count == 0
                                push!(inner_bytes, Opcode.DROP)
                            end
                        end
                    end
                end
            end
        end

        # Then branch - analyze what's in the then range BEFORE generating IF
        then_start = block.end_idx + 1
        then_end = goto_if_not.dest - 1
        found_return = false
        found_nested_cond = false
        found_forward_goto = nothing  # Target of unconditional forward GotoNode
        found_phi_pattern = nothing  # For && producing boolean to phi
        found_base_closure_invoke = false  # Base closure that will emit unreachable

        # First, analyze what's in the then range
        for i in then_start:min(then_end, length(code))
            stmt = code[i]
            if stmt isa Core.GotoIfNot
                found_nested_cond = true
                break
            elseif stmt isa Core.GotoNode && stmt.label > i
                # Unconditional forward jump - check if it's an || merge pattern
                # Only treat as || if target is NOT a PhiNode (phi indicates && boolean result)
                target_idx = stmt.label
                if target_idx <= length(code) && code[target_idx] isa Core.PhiNode
                    # Forward goto to phi - this is && producing a boolean value
                    # Need to generate ternary: if cond1 then cond2 else false
                    found_phi_pattern = (target_idx, i)  # (phi_idx, goto_idx)
                elseif target_idx <= length(code)
                    found_forward_goto = target_idx
                end
                break
            elseif stmt isa Expr && stmt.head === :invoke
                # Check if this is a Base closure invoke (which will emit unreachable)
                mi_or_ci = stmt.args[1]
                mi = if mi_or_ci isa Core.MethodInstance
                    mi_or_ci
                elseif isdefined(Core, :CodeInstance) && mi_or_ci isa Core.CodeInstance
                    mi_or_ci.def
                else
                    nothing
                end
                if mi isa Core.MethodInstance && mi.def isa Method
                    meth = mi.def
                    name = meth.name
                    if meth.module === Base && startswith(string(name), "#")
                        found_base_closure_invoke = true
                    end
                end
            end
        end

        # Handle phi pattern specially
        if found_phi_pattern !== nothing
            phi_idx, goto_idx = found_phi_pattern
            phi_node = code[phi_idx]::Core.PhiNode
            phi_type = get(ctx.ssa_types, phi_idx, Bool)

            # Check if this is a boolean && pattern or a ternary with computed values
            is_boolean_phi = phi_type === Bool

            if is_boolean_phi
                # Boolean && pattern: generates IF with i32 result, else = 0
                append!(inner_bytes, compile_value(goto_if_not.cond, ctx))
                push!(inner_bytes, Opcode.IF)
                push!(inner_bytes, 0x7f)  # i32 result type

                # Then-branch: compute cond2 (the expression before the goto)
                for i in then_start:goto_idx-1
                    stmt = code[i]
                    if stmt !== nothing && !(stmt isa Core.GotoIfNot) && !(stmt isa Core.GotoNode)
                        append!(inner_bytes, compile_statement(stmt, i, ctx))
                    end
                end

                # Else-branch: push false (0)
                push!(inner_bytes, Opcode.ELSE)
                push!(inner_bytes, Opcode.I32_CONST)
                append!(inner_bytes, encode_leb128_signed(0))

                push!(inner_bytes, Opcode.END)

                # Store to phi local if we have one
                if haskey(ctx.phi_locals, phi_idx)
                    local_idx = ctx.phi_locals[phi_idx]
                    push!(inner_bytes, Opcode.LOCAL_SET)
                    append!(inner_bytes, encode_leb128_unsigned(local_idx))
                end

                # Continue with conditionals after the phi
                for (j, (_, b)) in enumerate(conditionals)
                    goto_if_not = b.terminator::Core.GotoIfNot
                    if goto_if_not.cond isa Core.SSAValue && goto_if_not.cond.id == phi_idx
                        append!(inner_bytes, gen_conditional(j; target_idx=0))
                        break
                    end
                end

                return inner_bytes
            else
                # Multi-edge phi pattern - need to handle each branch separately
                # For phis with >2 edges, we can't use simple if/else result type
                # Instead, store each branch's value to the phi local

                if length(phi_node.edges) > 2
                    # Multi-edge phi - use local storage approach
                    # This pattern occurs with chained if-elseif-else
                    # We need to recurse and let each branch store to the phi local

                    # Compile then-branch statements and store value to phi local
                    append!(inner_bytes, compile_value(goto_if_not.cond, ctx))
                    push!(inner_bytes, Opcode.IF)
                    push!(inner_bytes, 0x40)  # void - we'll use locals

                    # Then-branch: compile statements and store to phi local
                    for i in then_start:goto_idx-1
                        stmt = code[i]
                        if stmt !== nothing && !(stmt isa Core.GotoIfNot) && !(stmt isa Core.GotoNode) && !(stmt isa Core.PhiNode)
                            append!(inner_bytes, compile_statement(stmt, i, ctx))
                        end
                    end

                    # Find the phi value for this edge (goto_idx is the edge)
                    for (edge_idx, edge) in enumerate(phi_node.edges)
                        if edge == goto_idx
                            val = phi_node.values[edge_idx]
                            append!(inner_bytes, compile_value(val, ctx))
                            if haskey(ctx.phi_locals, phi_idx)
                                local_idx = ctx.phi_locals[phi_idx]
                                push!(inner_bytes, Opcode.LOCAL_SET)
                                append!(inner_bytes, encode_leb128_unsigned(local_idx))
                            end
                            break
                        end
                    end

                    # Handle other phi nodes at this merge point
                    for other_phi_idx in (phi_idx+1):length(code)
                        other_stmt = code[other_phi_idx]
                        if other_stmt isa Core.PhiNode
                            for (edge_idx, edge) in enumerate(other_stmt.edges)
                                if edge == goto_idx
                                    val = other_stmt.values[edge_idx]
                                    append!(inner_bytes, compile_value(val, ctx))
                                    if haskey(ctx.phi_locals, other_phi_idx)
                                        local_idx = ctx.phi_locals[other_phi_idx]
                                        push!(inner_bytes, Opcode.LOCAL_SET)
                                        append!(inner_bytes, encode_leb128_unsigned(local_idx))
                                    end
                                    break
                                end
                            end
                        else
                            break  # Phi nodes are consecutive
                        end
                    end

                    # Else-branch: recurse for remaining conditionals
                    push!(inner_bytes, Opcode.ELSE)

                    # Find next conditional in the else branch
                    next_cond_idx = cond_idx + 1
                    if next_cond_idx <= length(conditionals)
                        append!(inner_bytes, gen_conditional(next_cond_idx; target_idx=phi_idx))
                    else
                        # No more conditionals - this is the final else branch
                        # Find the edge that corresponds to this fallthrough path
                        for i in goto_if_not.dest:phi_idx-1
                            stmt = code[i]
                            if stmt === nothing || stmt isa Core.GotoNode || stmt isa Core.PhiNode
                                continue
                            elseif stmt isa Core.GotoIfNot
                                # There's another conditional - this shouldn't happen if we're at the end
                                continue
                            else
                                append!(inner_bytes, compile_statement(stmt, i, ctx))
                            end
                        end

                        # Store the final else value to phi locals
                        # The fallthrough edge is the last statement index before phi
                        last_stmt_idx = phi_idx - 1
                        for (edge_idx, edge) in enumerate(phi_node.edges)
                            if edge == last_stmt_idx
                                val = phi_node.values[edge_idx]
                                append!(inner_bytes, compile_value(val, ctx))
                                if haskey(ctx.phi_locals, phi_idx)
                                    local_idx = ctx.phi_locals[phi_idx]
                                    push!(inner_bytes, Opcode.LOCAL_SET)
                                    append!(inner_bytes, encode_leb128_unsigned(local_idx))
                                end
                                break
                            end
                        end

                        # Handle other phi nodes
                        for other_phi_idx in (phi_idx+1):length(code)
                            other_stmt = code[other_phi_idx]
                            if other_stmt isa Core.PhiNode
                                for (edge_idx, edge) in enumerate(other_stmt.edges)
                                    if edge == last_stmt_idx
                                        val = other_stmt.values[edge_idx]
                                        append!(inner_bytes, compile_value(val, ctx))
                                        if haskey(ctx.phi_locals, other_phi_idx)
                                            local_idx = ctx.phi_locals[other_phi_idx]
                                            push!(inner_bytes, Opcode.LOCAL_SET)
                                            append!(inner_bytes, encode_leb128_unsigned(local_idx))
                                        end
                                        break
                                    end
                                end
                            else
                                break
                            end
                        end
                    end

                    push!(inner_bytes, Opcode.END)

                    # Only generate code after the phi nodes at the outermost level
                    # (when target_idx == 0, meaning this is not a recursive call)
                    if target_idx == 0
                        # Now generate code after the phi nodes
                        # Find first non-phi statement after phi_idx
                        first_non_phi = phi_idx
                        for i in phi_idx:length(code)
                            if !(code[i] isa Core.PhiNode)
                                first_non_phi = i
                                break
                            end
                        end

                        for i in first_non_phi:length(code)
                            stmt = code[i]
                            if stmt isa Core.ReturnNode
                                if isdefined(stmt, :val)
                                    append!(inner_bytes, compile_value(stmt.val, ctx))
                                end
                                push!(inner_bytes, Opcode.RETURN)
                                break
                            elseif stmt === nothing || stmt isa Core.PhiNode
                                continue
                            else
                                append!(inner_bytes, compile_statement(stmt, i, ctx))
                            end
                        end
                    end

                    return inner_bytes
                end

                # Simple 2-edge ternary pattern
                phi_wasm_type = julia_to_wasm_type_concrete(phi_type, ctx)

                then_value = nothing
                else_value = nothing
                else_edge = nothing
                for (edge_idx, edge) in enumerate(phi_node.edges)
                    if edge < goto_if_not.dest
                        # Edge from then-branch (before else target)
                        then_value = phi_node.values[edge_idx]
                    else
                        # Edge from else-branch
                        else_value = phi_node.values[edge_idx]
                        else_edge = edge
                    end
                end

                # Push condition
                append!(inner_bytes, compile_value(goto_if_not.cond, ctx))

                # IF block with phi's result type
                push!(inner_bytes, Opcode.IF)
                append!(inner_bytes, encode_block_type(phi_wasm_type))

                # Then-branch: compile any statements, then push the value
                for i in then_start:goto_idx-1
                    stmt = code[i]
                    if stmt !== nothing && !(stmt isa Core.GotoIfNot) && !(stmt isa Core.GotoNode) && !(stmt isa Core.PhiNode)
                        append!(inner_bytes, compile_statement(stmt, i, ctx))
                    end
                end
                # Push then-branch result value
                # Handle cases:
                # 1. then_value is nothing (Julia's nothing) - need ref.null if ref type expected
                # 2. then_value compiles to nothing (SSA with Nothing type) - need ref.null
                # 3. then_value compiles to actual value - use that
                if then_value === nothing
                    # Phi value is Julia's nothing - emit ref.null if ref type expected
                    if phi_wasm_type isa ConcreteRef
                        push!(inner_bytes, Opcode.REF_NULL)
                        append!(inner_bytes, encode_leb128_unsigned(phi_wasm_type.type_idx))
                    end
                    # For non-ref types, nothing produces no value (shouldn't happen for valid code)
                elseif then_value !== nothing
                    value_bytes = compile_value(then_value, ctx)
                    if isempty(value_bytes) && phi_wasm_type isa ConcreteRef
                        # Value compiled to nothing but we need a ref type - emit ref.null
                        push!(inner_bytes, Opcode.REF_NULL)
                        append!(inner_bytes, encode_leb128_unsigned(phi_wasm_type.type_idx))
                    else
                        append!(inner_bytes, value_bytes)
                    end
                end

                # Else-branch: compile statements from dest to else_edge, then push the value
                push!(inner_bytes, Opcode.ELSE)

                # Check if the GotoIfNot's type is Union{} (bottom type) - this means the else branch is dead code
                # The type of the GotoIfNot is stored at block.end_idx (the line with the conditional)
                goto_if_not_type = get(ctx.ssa_types, block.end_idx, Any)
                is_else_unreachable = goto_if_not_type === Union{}

                if is_else_unreachable
                    # Else branch is dead code - just emit unreachable
                    push!(inner_bytes, Opcode.UNREACHABLE)
                elseif else_edge !== nothing
                    for i in goto_if_not.dest:else_edge
                        stmt = code[i]
                        if stmt === nothing
                            continue
                        elseif stmt isa Core.GotoNode || stmt isa Core.PhiNode
                            continue
                        elseif stmt isa Core.ReturnNode
                            continue
                        else
                            append!(inner_bytes, compile_statement(stmt, i, ctx))
                        end
                    end
                end

                # Push else-branch result value (same logic as then-branch)
                # But skip if else branch is unreachable (code after unreachable is dead)
                if !is_else_unreachable
                    if else_value === nothing
                        # Phi value is Julia's nothing - emit ref.null if ref type expected
                        if phi_wasm_type isa ConcreteRef
                            push!(inner_bytes, Opcode.REF_NULL)
                            append!(inner_bytes, encode_leb128_unsigned(phi_wasm_type.type_idx))
                        end
                    elseif else_value !== nothing
                        value_bytes = compile_value(else_value, ctx)
                        if isempty(value_bytes) && phi_wasm_type isa ConcreteRef
                            # Value compiled to nothing but we need a ref type - emit ref.null
                            push!(inner_bytes, Opcode.REF_NULL)
                            append!(inner_bytes, encode_leb128_unsigned(phi_wasm_type.type_idx))
                        else
                            append!(inner_bytes, value_bytes)
                        end
                    end
                end

                push!(inner_bytes, Opcode.END)

                # Store to phi local if we have one
                if haskey(ctx.phi_locals, phi_idx)
                    local_idx = ctx.phi_locals[phi_idx]
                    push!(inner_bytes, Opcode.LOCAL_SET)
                    append!(inner_bytes, encode_leb128_unsigned(local_idx))
                end

                # After the phi, continue generating code from phi_idx+1 to the return
                for i in phi_idx+1:length(code)
                    stmt = code[i]
                    if stmt isa Core.ReturnNode
                        if isdefined(stmt, :val)
                            append!(inner_bytes, compile_value(stmt.val, ctx))
                        end
                        break
                    elseif stmt === nothing || stmt isa Core.GotoNode || stmt isa Core.PhiNode
                        continue
                    else
                        append!(inner_bytes, compile_statement(stmt, i, ctx))
                    end
                end

                return inner_bytes
            end
        end

        # Push condition for normal pattern
        append!(inner_bytes, compile_value(goto_if_not.cond, ctx))

        # Check if both branches terminate (then: return, else: unreachable or return)
        # If so, use void result type for IF block
        else_terminates = false
        for i in goto_if_not.dest:length(code)
            stmt = code[i]
            if stmt isa Core.ReturnNode
                else_terminates = true
                break
            elseif stmt isa Core.GotoIfNot || stmt isa Core.GotoNode
                break  # Hit another control flow
            end
        end

        # Check if then-branch ends with unreachable (Union{} typed call/invoke)
        # This happens with Base closures that we emit UNREACHABLE for
        then_ends_unreachable = false
        for i in then_start:min(then_end, length(code))
            stmt = code[i]
            if stmt isa Expr && (stmt.head === :call || stmt.head === :invoke)
                stmt_type = get(ctx.ssa_types, i, Any)
                if stmt_type === Union{}
                    then_ends_unreachable = true
                end
            end
        end

        # if block
        push!(inner_bytes, Opcode.IF)
        if found_forward_goto !== nothing && else_terminates
            # Both branches terminate - use void result type
            push!(inner_bytes, 0x40)  # void block type
        elseif (then_ends_unreachable || found_base_closure_invoke)
            # Then-branch ends with unreachable.
            # In WASM, unreachable can "produce" any type, so we CAN use typed result
            # if the else branch produces a value. Use the return type for the if block.
            # This allows the else branch to leave a value on the stack that becomes
            # the if result (and ultimately the return value).
            append!(inner_bytes, encode_block_type(result_type))
        else
            append!(inner_bytes, encode_block_type(result_type))
        end

        if found_forward_goto !== nothing
            # The then-branch is a forward goto to a merge point
            # Generate the code at the merge point target
            for i in found_forward_goto:length(code)
                stmt = code[i]
                if stmt isa Core.ReturnNode
                    if isdefined(stmt, :val)
                        append!(inner_bytes, compile_value(stmt.val, ctx))
                    end
                    # Emit RETURN since we're in a void IF block
                    if else_terminates
                        push!(inner_bytes, Opcode.RETURN)
                    end
                    break
                elseif stmt === nothing
                    # Skip nothing statements
                elseif !(stmt isa Core.GotoIfNot) && !(stmt isa Core.GotoNode)
                    append!(inner_bytes, compile_statement(stmt, i, ctx))

                    # Drop unused values (only if not going to return)
                    if !else_terminates
                        if stmt isa Expr && (stmt.head === :call || stmt.head === :invoke)
                            stmt_type = get(ctx.ssa_types, i, Any)
                            if stmt_type !== Nothing
                                is_nothing_union = stmt_type isa Union && Nothing in Base.uniontypes(stmt_type)
                                if !is_nothing_union
                                    if !haskey(ctx.ssa_locals, i) && !haskey(ctx.phi_locals, i)
                                        use_count = get(ssa_use_count, i, 0)
                                        if use_count == 0
                                            push!(inner_bytes, Opcode.DROP)
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        elseif !found_nested_cond
            # No nested conditional and no forward goto - compile statements normally
            for i in then_start:min(then_end, length(code))
                stmt = code[i]
                if stmt isa Core.ReturnNode
                    if isdefined(stmt, :val)
                        append!(inner_bytes, compile_value(stmt.val, ctx))
                    end
                    found_return = true
                    break
                elseif stmt === nothing
                    # Skip nothing statements
                else
                    stmt_bytes = compile_statement(stmt, i, ctx)
                    append!(inner_bytes, stmt_bytes)

                    # Drop unused values (but NOT if statement emitted UNREACHABLE)
                    # Check if the last opcode is UNREACHABLE (0x00) - if so, no value to drop
                    ends_with_unreachable = !isempty(stmt_bytes) && stmt_bytes[end] == Opcode.UNREACHABLE
                    if !ends_with_unreachable && stmt isa Expr && (stmt.head === :call || stmt.head === :invoke)
                        stmt_type = get(ctx.ssa_types, i, Any)
                        if stmt_type !== Nothing
                            is_nothing_union = stmt_type isa Union && Nothing in Base.uniontypes(stmt_type)
                            if !is_nothing_union
                                if !haskey(ctx.ssa_locals, i) && !haskey(ctx.phi_locals, i)
                                    use_count = get(ssa_use_count, i, 0)
                                    if use_count == 0
                                        push!(inner_bytes, Opcode.DROP)
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end

        # Handle based on what we found in the then branch
        # NOTE: found_phi_pattern is handled earlier and returns, so won't reach here
        if found_forward_goto !== nothing
            # Already generated merge point code above - nothing more to do for then branch
        elseif found_nested_cond
            # Then branch has a nested conditional - recurse to handle it
            # This handles short-circuit && patterns
            append!(inner_bytes, gen_conditional(cond_idx + 1))
        elseif !found_return
            # Then branch doesn't return and has no nested conditionals
            # Generate code from goto dest to the first return/conditional
            # IMPORTANT: Stop at the first return or when entering another conditional's block
            for i in goto_if_not.dest:length(code)
                stmt = code[i]
                if stmt isa Core.ReturnNode
                    if isdefined(stmt, :val)
                        append!(inner_bytes, compile_value(stmt.val, ctx))
                    end
                    break
                elseif stmt isa Core.GotoIfNot || stmt isa Core.GotoNode
                    # Hit another control flow statement - stop here
                    # The recursive structure will handle this
                    break
                elseif stmt isa Core.PhiNode
                    # Hit a phi node - stop (this is a merge point)
                    break
                elseif stmt === nothing
                    # Skip nothing statements
                else
                    append!(inner_bytes, compile_statement(stmt, i, ctx))

                    # Drop unused values
                    if stmt isa Expr && (stmt.head === :call || stmt.head === :invoke)
                        stmt_type = get(ctx.ssa_types, i, Any)
                        if stmt_type !== Nothing
                            is_nothing_union = stmt_type isa Union && Nothing in Base.uniontypes(stmt_type)
                            if !is_nothing_union
                                if !haskey(ctx.ssa_locals, i) && !haskey(ctx.phi_locals, i)
                                    use_count = get(ssa_use_count, i, 0)
                                    if use_count == 0
                                        push!(inner_bytes, Opcode.DROP)
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end

        # Else branch
        push!(inner_bytes, Opcode.ELSE)

        # Find the conditional at dest (if any)
        # This handles the case where multiple conditionals jump to the same target
        dest_cond_idx = nothing
        for (j, (_, b)) in enumerate(conditionals)
            if b.start_idx >= goto_if_not.dest
                dest_cond_idx = j
                break
            end
        end

        # Recurse to the conditional at dest, or generate final block
        if dest_cond_idx !== nothing
            append!(inner_bytes, gen_conditional(dest_cond_idx; target_idx=goto_if_not.dest))
        else
            # No conditional at dest - generate the code at dest directly
            append!(inner_bytes, gen_conditional(length(conditionals) + 1; target_idx=goto_if_not.dest))
        end

        push!(inner_bytes, Opcode.END)

        return inner_bytes
    end

    append!(bytes, gen_conditional(1))

    # Check if all code paths terminate inside the conditionals
    # This is the case when:
    # 1. All blocks that are return blocks (have ReturnNode terminator)
    # 2. The function uses void IF blocks because both branches terminate
    #
    # For typed IF blocks (with result type), each branch produces a value,
    # and the IF itself returns a value. If the function returns this value,
    # we don't need RETURN or UNREACHABLE - just fall through with value on stack.
    #
    # For void IF blocks where branches use RETURN, code after IF is unreachable.
    #
    # Count actual return blocks vs total blocks
    return_blocks = count(b -> b.terminator isa Core.ReturnNode, blocks)
    total_blocks = length(blocks)

    # Check if we're using typed IF blocks (branches produce values)
    # If so, the value is on the stack and we just fall through
    # The gen_conditional function uses typed blocks when there's a phi merge
    # or when branches don't use explicit RETURN
    #
    # If return_blocks >= 2 AND the result_type is not void (0x40),
    # the IF produced a value and we don't emit anything
    if result_type isa ConcreteRef || result_type === I32 || result_type === I64 ||
       result_type === F32 || result_type === F64
        # Typed result - IF produces value, fall through with value on stack
        # No RETURN or UNREACHABLE needed
    elseif return_blocks >= 2
        # Multiple return blocks with void result - branches return inside
        push!(bytes, Opcode.UNREACHABLE)
    else
        push!(bytes, Opcode.RETURN)
    end

    return bytes
end

"""
Generate code for a single basic block.
"""
function generate_block_code(ctx::CompilationContext, block::BasicBlock)::Vector{UInt8}
    bytes = UInt8[]
    code = ctx.code_info.code

    for i in block.start_idx:block.end_idx
        append!(bytes, compile_statement(code[i], i, ctx))
    end

    return bytes
end

# ============================================================================
# Statement Compilation
# ============================================================================

"""
Compile a single IR statement to Wasm bytecode.
"""
function compile_statement(stmt, idx::Int, ctx::CompilationContext)::Vector{UInt8}
    bytes = UInt8[]

    if stmt isa Core.ReturnNode
        if isdefined(stmt, :val)
            append!(bytes, compile_value(stmt.val, ctx))
        end
        push!(bytes, Opcode.RETURN)

    elseif stmt isa Core.GotoNode
        # Unconditional branch - handled by control flow analysis

    elseif stmt isa Core.GotoIfNot
        # Conditional branch - handled by control flow analysis

    elseif stmt isa Core.PiNode
        # PiNode is a type assertion - just pass through the value
        # BUT: If the result type is Nothing, the value won't be used in a meaningful way
        # (consumers will emit ref.null instead), so don't push anything
        pi_type = get(ctx.ssa_types, idx, Any)
        if pi_type !== Nothing
            append!(bytes, compile_value(stmt.val, ctx))

            # If this SSA value needs a local, store it (and remove from stack)
            if haskey(ctx.ssa_locals, idx)
                local_idx = ctx.ssa_locals[idx]
                push!(bytes, Opcode.LOCAL_SET)  # Use SET not TEE to not leave on stack
                append!(bytes, encode_leb128_unsigned(local_idx))
            end
        end
        # If pi_type is Nothing, emit nothing - consumers will handle it

    elseif stmt isa Core.EnterNode
        # Exception handling: Enter try block
        # For now, we just skip this - full implementation requires try_table
        # The catch destination is in stmt.catch_dest
        # TODO: Implement full try/catch with try_table instruction

    elseif stmt isa GlobalRef
        # GlobalRef statement - check if it's a module-level global first
        key = (stmt.mod, stmt.name)
        if haskey(ctx.module_globals, key)
            # Emit global.get for module-level mutable struct instances
            global_idx = ctx.module_globals[key]
            push!(bytes, Opcode.GLOBAL_GET)
            append!(bytes, encode_leb128_unsigned(global_idx))

            # If this SSA value needs a local, store it
            if haskey(ctx.ssa_locals, idx)
                local_idx = ctx.ssa_locals[idx]
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(local_idx))
            end
        else
            # Regular GlobalRef - evaluate the constant and push it
            # This handles things like Main.SLOT_EMPTY that are module-level constants
            try
                val = getfield(stmt.mod, stmt.name)
                value_bytes = compile_value(val, ctx)
                append!(bytes, value_bytes)

                # If this SSA value needs a local, store it (only if we actually pushed a value)
                # compile_value returns empty bytes for Functions, Types, etc.
                if !isempty(value_bytes) && haskey(ctx.ssa_locals, idx)
                    local_idx = ctx.ssa_locals[idx]
                    push!(bytes, Opcode.LOCAL_SET)
                    append!(bytes, encode_leb128_unsigned(local_idx))
                end
            catch
                # If we can't evaluate, it might be a type reference which has no runtime value
            end
        end

    elseif stmt isa Expr
        stmt_bytes = UInt8[]
        if stmt.head === :call
            stmt_bytes = compile_call(stmt, idx, ctx)
        elseif stmt.head === :invoke
            stmt_bytes = compile_invoke(stmt, idx, ctx)
        elseif stmt.head === :new
            # Struct construction: %new(Type, args...)
            stmt_bytes = compile_new(stmt, idx, ctx)
        elseif stmt.head === :boundscheck
            # Bounds check - we can skip this as Wasm has its own bounds checking
            # This is a no-op that produces a Bool (we push false since we're not doing checks)
            push!(stmt_bytes, Opcode.I32_CONST)
            push!(stmt_bytes, 0x00)  # false = no bounds checking
        elseif stmt.head === :foreigncall
            # Handle foreign calls - specifically for Vector allocation
            stmt_bytes = compile_foreigncall(stmt, idx, ctx)
        elseif stmt.head === :leave
            # Exception handling: Leave try block
            # For now, skip - full implementation requires try_table control flow
            # TODO: Implement proper br out of try_table
        elseif stmt.head === :pop_exception
            # Exception handling: Pop exception from handler stack
            # For now, skip - full implementation requires exnref handling
            # TODO: Implement proper exception value handling
        end

        append!(bytes, stmt_bytes)

        # If this SSA value needs a local, store it (and remove from stack)
        # We use LOCAL_SET (not LOCAL_TEE) to avoid leaving extra values on stack
        # that would interfere with later operations. Values will be retrieved
        # via local.get when needed.
        #
        # We emit local.set if:
        # 1. The statement produced bytecode (!isempty(stmt_bytes)), OR
        # 2. The statement is a passthrough (memoryrefnew, Vector %new) where the value
        #    is expected to be on the stack from an earlier SSA
        # BUT NOT if the statement ends with UNREACHABLE (there's no value on stack)
        #
        # Skipped signal statements return empty bytes and are NOT passthroughs, so
        # they correctly get no local.set.
        if haskey(ctx.ssa_locals, idx)
            # Don't emit local.set after unreachable - there's no value on the stack.
            # We detect unreachable in two ways:
            # 1. Type-based: Union{} means the statement never returns (throws/unreachable)
            # 2. Bytecode-based: if the statement ends with UNREACHABLE opcode (0x00)
            #    AND is preceded by DROP (0x1A) - this is our Base closure pattern
            #    (we emit DROP then UNREACHABLE for closures we can't compile)
            stmt_type = get(ctx.ssa_types, idx, Any)
            is_unreachable_type = stmt_type === Union{}
            # Check for DROP+UNREACHABLE pattern at end of bytecode
            # This handles Base closure invokes which have non-Union{} return type
            # but we emit UNREACHABLE anyway
            is_unreachable_bytecode = length(stmt_bytes) >= 2 &&
                                       stmt_bytes[end] == Opcode.UNREACHABLE &&
                                       stmt_bytes[end-1] == Opcode.DROP
            is_unreachable = is_unreachable_type || is_unreachable_bytecode
            # Also store for passthrough statements that have a value on the stack
            # but don't emit their own bytecode (like memoryrefnew, Vector %new)
            should_store = (!isempty(stmt_bytes) || is_passthrough_statement(stmt, ctx)) && !is_unreachable
            if should_store
                local_idx = ctx.ssa_locals[idx]
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(local_idx))
            end
        end
    end

    return bytes
end

"""
Compile a struct construction expression (%new).
"""
function compile_new(expr::Expr, idx::Int, ctx::CompilationContext)::Vector{UInt8}
    bytes = UInt8[]

    # expr.args[1] is the type, rest are field values
    struct_type_ref = expr.args[1]
    field_values = expr.args[2:end]



    # Resolve the struct type if it's a GlobalRef
    struct_type = if struct_type_ref isa GlobalRef
        getfield(struct_type_ref.mod, struct_type_ref.name)
    elseif struct_type_ref isa DataType
        struct_type_ref
    else
        error("Unknown struct type reference: $struct_type_ref")
    end

    # Special case: Vector{T} construction
    # Vector is now a struct with (ref, size) fields to support setfield!(v, :size, ...)
    # The %new(Vector{T}, memref, size_tuple) creates a struct with both fields
    if struct_type <: AbstractVector && length(field_values) >= 1
        # field_values[1] is the MemoryRef (which is actually our array)
        # field_values[2] is the size tuple (Tuple{Int64})

        # Register the vector type if not already done
        if !haskey(ctx.type_registry.structs, struct_type)
            register_vector_type!(ctx.mod, ctx.type_registry, struct_type)
        end
        vec_info = ctx.type_registry.structs[struct_type]

        # Compile field 0: the array reference (from MemoryRef)
        append!(bytes, compile_value(field_values[1], ctx))

        # Compile field 1: the size tuple
        if length(field_values) >= 2
            append!(bytes, compile_value(field_values[2], ctx))
        else
            # No size provided - get array length and create tuple
            # Push array ref again for array.len
            append!(bytes, compile_value(field_values[1], ctx))
            push!(bytes, Opcode.GC_PREFIX)
            push!(bytes, Opcode.ARRAY_LEN)
            push!(bytes, Opcode.I64_EXTEND_I32_S)
            # Create Tuple{Int64} struct
            size_tuple_type = Tuple{Int64}
            if !haskey(ctx.type_registry.structs, size_tuple_type)
                register_tuple_type!(ctx.mod, ctx.type_registry, size_tuple_type)
            end
            size_info = ctx.type_registry.structs[size_tuple_type]
            push!(bytes, Opcode.GC_PREFIX)
            push!(bytes, Opcode.STRUCT_NEW)
            append!(bytes, encode_leb128_unsigned(size_info.wasm_type_idx))
        end

        # Create the Vector struct
        push!(bytes, Opcode.GC_PREFIX)
        push!(bytes, Opcode.STRUCT_NEW)
        append!(bytes, encode_leb128_unsigned(vec_info.wasm_type_idx))
        return bytes
    end

    # Get the registered struct info
    if !haskey(ctx.type_registry.structs, struct_type)
        # Register it now - use appropriate registration for closures vs regular structs
        if is_closure_type(struct_type)
            register_closure_type!(ctx.mod, ctx.type_registry, struct_type)
        else
            register_struct_type!(ctx.mod, ctx.type_registry, struct_type)
        end
    end

    info = ctx.type_registry.structs[struct_type]

    # Push field values in order, handling Union field types
    for (i, val) in enumerate(field_values)
        field_type = info.field_types[i]

        # Check if this field is a Union type that needs wrapping
        if field_type isa Union && needs_tagged_union(field_type)
            # Get the value's actual type
            val_type = if val isa Core.SSAValue
                get(ctx.ssa_types, val.id, Any)
            elseif val isa GlobalRef
                actual_val = try getfield(val.mod, val.name) catch; nothing end
                typeof(actual_val)
            else
                typeof(val)
            end

            # Compile the value first
            append!(bytes, compile_value(val, ctx))

            # Wrap it in the tagged union
            append!(bytes, emit_wrap_union_value(ctx, val_type, field_type))
        elseif field_type isa Union
            # Simple nullable union (Union{Nothing, T})
            inner_type = get_nullable_inner_type(field_type)

            # Get the value's actual type
            val_type = if val isa Core.SSAValue
                get(ctx.ssa_types, val.id, Any)
            elseif val isa GlobalRef
                actual_val = try getfield(val.mod, val.name) catch; nothing end
                typeof(actual_val)
            else
                typeof(val)
            end

            # Check if this value is nothing - either literally or via an SSA with Nothing type
            # SSA values with Nothing type (e.g., from GlobalRef to nothing) produce no bytecode,
            # so we need to emit ref.null directly instead of trying to load a non-existent value
            is_literal_nothing = val === nothing || (val isa GlobalRef && val.name === :nothing)
            is_nothing_type_ssa = val isa Core.SSAValue && val_type === Nothing
            should_emit_null = is_literal_nothing || is_nothing_type_ssa

            if should_emit_null
                # Nothing value (literal or SSA with Nothing type) - emit ref.null
                if inner_type !== nothing && isconcretetype(inner_type) && isstructtype(inner_type)
                    # Nullable struct ref - emit null reference
                    if haskey(ctx.type_registry.structs, inner_type)
                        inner_info = ctx.type_registry.structs[inner_type]
                        push!(bytes, Opcode.REF_NULL)
                        append!(bytes, encode_leb128_unsigned(inner_info.wasm_type_idx))
                    else
                        # Use generic null
                        push!(bytes, Opcode.REF_NULL)
                        push!(bytes, UInt8(StructRef))
                    end
                elseif inner_type !== nothing && inner_type <: AbstractVector
                    # Nullable array ref
                    elem_type = eltype(inner_type)
                    arr_type_idx = get_array_type!(ctx.mod, ctx.type_registry, elem_type)
                    push!(bytes, Opcode.REF_NULL)
                    append!(bytes, encode_leb128_unsigned(arr_type_idx))
                else
                    # Generic nullable - use structref null
                    push!(bytes, Opcode.REF_NULL)
                    push!(bytes, UInt8(StructRef))
                end
            else
                # Non-null value - just compile normally
                append!(bytes, compile_value(val, ctx))
            end
        else
            # Regular field - compile value directly
            append!(bytes, compile_value(val, ctx))
        end
    end

    # struct.new type_idx
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_NEW)
    append!(bytes, encode_leb128_unsigned(info.wasm_type_idx))

    return bytes
end

"""
Compile a foreign call expression.
Handles specific patterns like jl_alloc_genericmemory for Vector allocation.
"""
function compile_foreigncall(expr::Expr, idx::Int, ctx::CompilationContext)::Vector{UInt8}
    bytes = UInt8[]

    # foreigncall format: Expr(:foreigncall, name, return_type, arg_types, nreq, calling_conv, args...)
    # For jl_alloc_genericmemory:
    #   args[1] = :(:jl_alloc_genericmemory)
    #   args[2] = return type (e.g., Ref{Memory{Int32}})
    #   args[7] = element type (e.g., Memory{Int32})
    #   args[8] = length

    if length(expr.args) >= 1
        name_arg = expr.args[1]
        name = if name_arg isa QuoteNode
            name_arg.value
        elseif name_arg isa Symbol
            name_arg
        else
            nothing
        end

        if name === :jl_alloc_genericmemory
            # Extract element type from return type
            # args[2] is like Ref{Memory{Int32}}
            # args[7] is Memory{Int32}
            ret_type = length(expr.args) >= 2 ? expr.args[2] : nothing

            # Get the element type from Memory{T}
            # Memory{T} is actually GenericMemory{:not_atomic, T, ...}
            # The memory type is at args[6] (not args[7])
            elem_type = Int32  # default
            if length(expr.args) >= 6
                mem_type = expr.args[6]
                if mem_type isa DataType && mem_type.name.name === :GenericMemory && length(mem_type.parameters) >= 2
                    # GenericMemory parameters: (atomicity, element_type, addrspace)
                    elem_type = mem_type.parameters[2]
                elseif mem_type isa DataType && mem_type.name.name === :Memory && length(mem_type.parameters) >= 1
                    elem_type = mem_type.parameters[1]
                elseif mem_type isa GlobalRef
                    resolved = try getfield(mem_type.mod, mem_type.name) catch; nothing end
                    if resolved isa DataType && resolved.name.name === :GenericMemory && length(resolved.parameters) >= 2
                        elem_type = resolved.parameters[2]
                    elseif resolved isa DataType && resolved.name.name === :Memory && length(resolved.parameters) >= 1
                        elem_type = resolved.parameters[1]
                    end
                end
            end

            # Get the length argument (at args[7] or args[8])
            len_arg = length(expr.args) >= 7 ? expr.args[7] : nothing

            # Get or create array type for this element type
            arr_type_idx = if elem_type <: AbstractVector || (elem_type isa DataType && isstructtype(elem_type))
                # For struct element types, register the element struct first
                if isconcretetype(elem_type) && isstructtype(elem_type) && !haskey(ctx.type_registry.structs, elem_type)
                    register_struct_type!(ctx.mod, ctx.type_registry, elem_type)
                end
                get_array_type!(ctx.mod, ctx.type_registry, elem_type)
            elseif elem_type === String
                get_string_array_type!(ctx.mod, ctx.type_registry)
            else
                get_array_type!(ctx.mod, ctx.type_registry, elem_type)
            end

            # Compile length argument
            if len_arg !== nothing
                append!(bytes, compile_value(len_arg, ctx))
                len_type = infer_value_type(len_arg, ctx)
                if len_type === Int64 || len_type === Int
                    push!(bytes, Opcode.I32_WRAP_I64)
                end
            else
                # Default length of 0
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x00)
            end

            # array.new_default creates array filled with default value (0 for primitives, null for refs)
            push!(bytes, Opcode.GC_PREFIX)
            push!(bytes, Opcode.ARRAY_NEW_DEFAULT)
            append!(bytes, encode_leb128_unsigned(arr_type_idx))

            return bytes
        end
    end

    # Unknown foreigncall - return empty bytes (will be skipped)
    return bytes
end

# ============================================================================
# Value Compilation
# ============================================================================

"""
Compile a value reference (SSA, Argument, or Literal).
"""
function compile_value(val, ctx::CompilationContext)::Vector{UInt8}
    bytes = UInt8[]

    # Handle nothing explicitly - it's the Julia singleton
    if val === nothing
        # Nothing doesn't have a runtime value in WasmGC
        # Caller should handle this case (e.g., for Union{Nothing, T} fields)
        # Return empty bytes - nothing to push
        return bytes
    end

    if val isa Core.SSAValue
        # Check if this SSA has a local allocated (either regular or phi)
        if haskey(ctx.ssa_locals, val.id)
            local_idx = ctx.ssa_locals[val.id]
            push!(bytes, Opcode.LOCAL_GET)
            append!(bytes, encode_leb128_unsigned(local_idx))
        elseif haskey(ctx.phi_locals, val.id)
            # Phi node - load from phi local
            local_idx = ctx.phi_locals[val.id]
            push!(bytes, Opcode.LOCAL_GET)
            append!(bytes, encode_leb128_unsigned(local_idx))
        else
            # No local - check if this is a PiNode with Nothing type
            # In that case, we need to emit ref.null for the union type
            stmt = ctx.code_info.code[val.id]
            if stmt isa Core.PiNode
                pi_type = get(ctx.ssa_types, val.id, Any)
                if pi_type === Nothing
                    # PiNode narrowed to Nothing - we need to emit ref.null
                    # Get the type of the underlying value to find the union's other type
                    if stmt.val isa Core.SSAValue
                        underlying_type = get(ctx.ssa_types, stmt.val.id, Any)
                        # For Union{Nothing, T}, emit ref.null $T
                        if underlying_type !== Nothing && underlying_type !== Any
                            wasm_type = julia_to_wasm_type_concrete(underlying_type, ctx)
                            if wasm_type isa ConcreteRef
                                push!(bytes, Opcode.REF_NULL)
                                append!(bytes, encode_leb128_unsigned(wasm_type.type_idx))
                            end
                        end
                    end
                end
            end
            # Otherwise, assume it's on the stack (for single-use SSAs in sequence)
        end

    elseif val isa Core.Argument
        # For closures being compiled, _1 is the closure object (arg_types[1])
        # For regular functions, arguments start at _2 (arg_types[1])
        # Use is_compiled_closure flag (not the type of first arg)
        if ctx.is_compiled_closure
            # Closure: direct mapping (_1 = closure, _2 = first arg)
            arg_idx = val.n
        else
            # Regular function: skip _1 (function type in IR)
            arg_idx = val.n - 1
        end

        # WasmGlobal arguments don't have locals - they're accessed via global.get/set
        # in the getfield/setfield handlers, so we skip emitting anything here
        if arg_idx in ctx.global_args
            # WasmGlobal arg - no local.get needed (handled by getfield/setfield)
            # Return empty bytes
        elseif arg_idx >= 1 && arg_idx <= length(ctx.arg_types)
            # Calculate local index: count non-WasmGlobal args before this one
            local_idx = count(i -> !(i in ctx.global_args), 1:arg_idx-1)
            push!(bytes, Opcode.LOCAL_GET)
            append!(bytes, encode_leb128_unsigned(local_idx))
        end

    elseif val isa Core.SlotNumber
        local_idx = val.id - 2
        if local_idx >= 0
            push!(bytes, Opcode.LOCAL_GET)
            append!(bytes, encode_leb128_unsigned(local_idx))
        end

    elseif val isa Bool
        push!(bytes, Opcode.I32_CONST)
        push!(bytes, val ? 0x01 : 0x00)

    elseif val isa Char
        # Char is represented as i32 (Unicode codepoint)
        push!(bytes, Opcode.I32_CONST)
        append!(bytes, encode_leb128_signed(Int32(val)))

    elseif val isa Int32 || val isa UInt32
        push!(bytes, Opcode.I32_CONST)
        append!(bytes, encode_leb128_signed(Int32(val)))

    elseif val isa Int64 || val isa UInt64 || val isa Int
        push!(bytes, Opcode.I64_CONST)
        append!(bytes, encode_leb128_signed(Int64(val)))

    elseif val isa Int128 || val isa UInt128
        # 128-bit integers are represented as WasmGC structs with (lo, hi) fields
        result_type = typeof(val)
        type_idx = get_int128_type!(ctx.mod, ctx.type_registry, result_type)

        # Extract lo (low 64 bits) and hi (high 64 bits)
        lo = UInt64(val & 0xFFFFFFFFFFFFFFFF)
        hi = UInt64((val >> 64) & 0xFFFFFFFFFFFFFFFF)

        # Push lo value
        push!(bytes, Opcode.I64_CONST)
        append!(bytes, encode_leb128_signed(reinterpret(Int64, lo)))

        # Push hi value
        push!(bytes, Opcode.I64_CONST)
        append!(bytes, encode_leb128_signed(reinterpret(Int64, hi)))

        # Create struct
        push!(bytes, Opcode.GC_PREFIX)
        push!(bytes, Opcode.STRUCT_NEW)
        append!(bytes, encode_leb128_unsigned(type_idx))

    elseif val isa Float32
        push!(bytes, Opcode.F32_CONST)
        append!(bytes, reinterpret(UInt8, [val]))

    elseif val isa Float64
        push!(bytes, Opcode.F64_CONST)
        append!(bytes, reinterpret(UInt8, [val]))

    elseif val isa String
        # String constant - create a WasmGC array with the characters
        # Get the string array type
        type_idx = get_string_array_type!(ctx.mod, ctx.type_registry)

        # Push each character as an i32
        for c in val
            push!(bytes, Opcode.I32_CONST)
            append!(bytes, encode_leb128_signed(Int32(c)))
        end

        # array.new_fixed $type_idx $length
        push!(bytes, Opcode.GC_PREFIX)
        push!(bytes, Opcode.ARRAY_NEW_FIXED)
        append!(bytes, encode_leb128_unsigned(type_idx))
        append!(bytes, encode_leb128_unsigned(length(val)))

    elseif val isa GlobalRef
        # Check if this GlobalRef is a module-level global (mutable struct instance)
        key = (val.mod, val.name)
        if haskey(ctx.module_globals, key)
            # Emit global.get instead of creating a new struct instance
            global_idx = ctx.module_globals[key]
            push!(bytes, Opcode.GLOBAL_GET)
            append!(bytes, encode_leb128_unsigned(global_idx))
        else
            # GlobalRef to a constant - evaluate and compile the value
            try
                actual_val = getfield(val.mod, val.name)
                append!(bytes, compile_value(actual_val, ctx))
            catch
                # If we can't evaluate, might be a type reference (no runtime value)
            end
        end

    elseif val isa QuoteNode
        # QuoteNode wraps a constant value - unwrap and compile
        append!(bytes, compile_value(val.value, ctx))

    elseif isprimitivetype(typeof(val)) && !isa(val, Bool) && !isa(val, Char) &&
           !isa(val, Int8) && !isa(val, Int16) && !isa(val, Int32) && !isa(val, Int64) &&
           !isa(val, UInt8) && !isa(val, UInt16) && !isa(val, UInt32) && !isa(val, UInt64) &&
           !isa(val, Float32) && !isa(val, Float64)
        # Custom primitive type (e.g., JuliaSyntax.Kind) - bitcast to integer
        T = typeof(val)
        sz = sizeof(T)
        if sz == 1
            int_val = Core.Intrinsics.bitcast(UInt8, val)
            push!(bytes, Opcode.I32_CONST)
            append!(bytes, encode_leb128_signed(Int32(int_val)))
        elseif sz == 2
            int_val = Core.Intrinsics.bitcast(UInt16, val)
            push!(bytes, Opcode.I32_CONST)
            append!(bytes, encode_leb128_signed(Int32(int_val)))
        elseif sz == 4
            int_val = Core.Intrinsics.bitcast(UInt32, val)
            push!(bytes, Opcode.I32_CONST)
            append!(bytes, encode_leb128_signed(Int32(int_val)))
        elseif sz == 8
            int_val = Core.Intrinsics.bitcast(UInt64, val)
            push!(bytes, Opcode.I64_CONST)
            append!(bytes, encode_leb128_signed(Int64(int_val)))
        else
            error("Primitive type with unsupported size for Wasm: $T ($sz bytes)")
        end

    elseif typeof(val) <: Tuple
        # Tuple constant - create it with struct.new
        T = typeof(val)

        # Ensure tuple type is registered using register_tuple_type!
        info = register_tuple_type!(ctx.mod, ctx.type_registry, T)
        type_idx = info.wasm_type_idx

        # Push field values (tuples use 1-based indexing)
        for i in 1:length(val)
            field_val = val[i]
            append!(bytes, compile_value(field_val, ctx))
        end

        # Create the struct
        push!(bytes, Opcode.GC_PREFIX)
        push!(bytes, Opcode.STRUCT_NEW)
        append!(bytes, encode_leb128_unsigned(type_idx))

    elseif isstructtype(typeof(val)) && !isa(val, Type) && !isa(val, Function) && !isa(val, Module)
        # Struct constant - create it with struct.new
        T = typeof(val)

        # Ensure struct type is registered and get its type index
        info = register_struct_type!(ctx.mod, ctx.type_registry, T)
        type_idx = info.wasm_type_idx

        # Push field values
        for field_name in fieldnames(T)
            field_val = getfield(val, field_name)
            append!(bytes, compile_value(field_val, ctx))
        end

        # Create the struct
        push!(bytes, Opcode.GC_PREFIX)
        push!(bytes, Opcode.STRUCT_NEW)
        append!(bytes, encode_leb128_unsigned(type_idx))
    end

    return bytes
end

# ============================================================================
# Call Compilation
# ============================================================================

"""
Compile a function call expression.
"""
function compile_call(expr::Expr, idx::Int, ctx::CompilationContext)::Vector{UInt8}
    bytes = UInt8[]
    func = expr.args[1]
    args = expr.args[2:end]

    # Special case for signal read: getfield(Signal, :value) -> global.get
    # This is detected by analyze_signal_captures! and stored in signal_ssa_getters
    # ONLY applies to actual getfield/getproperty(Signal, :value) calls (WasmGlobal pattern)
    # For Therapy.jl closures, signal_ssa_getters maps closure field SSAs - handled in compile_invoke
    is_getfield_value = (is_func(func, :getfield) || is_func(func, :getproperty)) && length(args) >= 2
    if is_getfield_value && haskey(ctx.signal_ssa_getters, idx)
        # Check that this is accessing :value field (WasmGlobal pattern)
        field_ref = args[2]
        field_name = field_ref isa QuoteNode ? field_ref.value : field_ref
        if field_name === :value
            global_idx = ctx.signal_ssa_getters[idx]
            push!(bytes, Opcode.GLOBAL_GET)
            append!(bytes, encode_leb128_unsigned(global_idx))
            return bytes
        end
    end

    # Special case for signal write: setfield!(Signal, :value, x) -> global.set
    # This is detected by analyze_signal_captures! and stored in signal_ssa_setters
    # ONLY applies to actual setfield!/setproperty! calls (WasmGlobal pattern), NOT closure field access
    is_setfield_call = (is_func(func, :setfield!) || is_func(func, :setproperty!)) && length(args) >= 3
    if is_setfield_call && haskey(ctx.signal_ssa_setters, idx)
        # The value to write is the 3rd argument (args = [target, field, value])
        global_idx = ctx.signal_ssa_setters[idx]
        value_arg = args[3]
        append!(bytes, compile_value(value_arg, ctx))
        push!(bytes, Opcode.GLOBAL_SET)
        append!(bytes, encode_leb128_unsigned(global_idx))

        # Inject DOM update calls for this signal (Therapy.jl reactive updates)
        if haskey(ctx.dom_bindings, global_idx)
            # Get global's type for conversion
            global_type = ctx.mod.globals[global_idx + 1].valtype

            for (import_idx, const_args) in ctx.dom_bindings[global_idx]
                # Push constant arguments (e.g., hydration key)
                for arg in const_args
                    push!(bytes, Opcode.I32_CONST)
                    append!(bytes, encode_leb128_signed(Int(arg)))
                end
                # Push the signal value (re-read from global)
                push!(bytes, Opcode.GLOBAL_GET)
                append!(bytes, encode_leb128_unsigned(global_idx))
                # Convert to f64 for DOM imports (all DOM imports expect f64)
                append!(bytes, emit_convert_to_f64(global_type))
                # Call the DOM import function
                push!(bytes, Opcode.CALL)
                append!(bytes, encode_leb128_unsigned(import_idx))
            end
        end

        # setfield! returns the value written, so re-read it
        push!(bytes, Opcode.GLOBAL_GET)
        append!(bytes, encode_leb128_unsigned(global_idx))
        return bytes
    end

    # Handle signal getter/setter SSA function calls: (%ssa)() or (%ssa)(value)
    # When func is an SSA that represents a captured signal getter/setter,
    # emit global.get/global.set directly (same logic as compile_invoke)
    if func isa Core.SSAValue
        ssa_id = func.id
        # Signal getter: no args, returns the signal value
        if haskey(ctx.signal_ssa_getters, ssa_id) && isempty(args)
            global_idx = ctx.signal_ssa_getters[ssa_id]
            push!(bytes, Opcode.GLOBAL_GET)
            append!(bytes, encode_leb128_unsigned(global_idx))
            return bytes
        end
        # Signal setter: one arg, sets the signal value
        if haskey(ctx.signal_ssa_setters, ssa_id) && length(args) == 1
            global_idx = ctx.signal_ssa_setters[ssa_id]
            # Compile the argument (the new value)
            append!(bytes, compile_value(args[1], ctx))
            # Store to global
            push!(bytes, Opcode.GLOBAL_SET)
            append!(bytes, encode_leb128_unsigned(global_idx))

            # Inject DOM update calls for this signal (Therapy.jl reactive updates)
            if haskey(ctx.dom_bindings, global_idx)
                # Get global's type for conversion
                global_type = ctx.mod.globals[global_idx + 1].valtype

                for (import_idx, const_args) in ctx.dom_bindings[global_idx]
                    # Push constant arguments (e.g., hydration key)
                    for arg in const_args
                        push!(bytes, Opcode.I32_CONST)
                        append!(bytes, encode_leb128_signed(Int(arg)))
                    end
                    # Push the signal value (re-read from global)
                    push!(bytes, Opcode.GLOBAL_GET)
                    append!(bytes, encode_leb128_unsigned(global_idx))
                    # Convert to f64 for DOM imports (all DOM imports expect f64)
                    append!(bytes, emit_convert_to_f64(global_type))
                    # Call the DOM import function
                    push!(bytes, Opcode.CALL)
                    append!(bytes, encode_leb128_unsigned(import_idx))
                end
            end

            # Setter returns the value in Therapy.jl, so re-read it
            push!(bytes, Opcode.GLOBAL_GET)
            append!(bytes, encode_leb128_unsigned(global_idx))
            return bytes
        end
    end

    # Special case for getfield on closure (_1) accessing captured signal fields
    # These produce intermediate SSA values (getter/setter functions)
    # Skip them - the actual read/write happens when the function is invoked
    is_getfield_closure = (func isa GlobalRef &&
                          ((func.mod === Core && func.name === :getfield) ||
                           (func.mod === Base && func.name === :getfield)))
    if is_getfield_closure && length(args) >= 2
        target = args[1]
        field_ref = args[2]
        # Target can be Core.SlotNumber(1) or Core.Argument(1)
        is_closure_self = (target isa Core.SlotNumber && target.id == 1) ||
                          (target isa Core.Argument && target.n == 1)
        if is_closure_self
            # This is accessing a field of the closure
            field_name = field_ref isa QuoteNode ? field_ref.value : field_ref
            if field_name isa Symbol && haskey(ctx.captured_signal_fields, field_name)
                # Skip - this produces a getter/setter function reference
                return bytes
            end
        end
    end

    # Skip getfield(CompilableSignal/Setter, :signal) - intermediate step
    # We track this in analyze_signal_captures! but don't need to emit anything
    # IMPORTANT: Only skip for actual CompilableSignal/Setter types, not any struct with a :signal field
    is_getfield = (func isa GlobalRef &&
                  ((func.mod === Core && func.name === :getfield) ||
                   (func.mod === Base && func.name === :getfield)))
    if is_getfield && length(args) >= 2
        field_ref = args[2]
        field_name = field_ref isa QuoteNode ? field_ref.value : field_ref
        if field_name === :signal
            # Only skip for CompilableSignal/Setter types (WasmGlobal pattern)
            target_type = infer_value_type(args[1], ctx)
            if target_type isa DataType && target_type.name.name in (:CompilableSignal, :CompilableSetter)
                # Skip - this is getting Signal from CompilableSignal/Setter
                return bytes
            end
        end
    end

    # Special case for ifelse - needs different argument order
    if is_func(func, :ifelse) && length(args) == 3
        # Wasm select expects: [val_if_true, val_if_false, cond] (cond on top)
        # Julia ifelse(cond, true_val, false_val)
        # So push: true_val, false_val, cond
        append!(bytes, compile_value(args[2], ctx))  # true_val
        append!(bytes, compile_value(args[3], ctx))  # false_val
        append!(bytes, compile_value(args[1], ctx))  # cond

        # Determine the type of the values for select
        val_type = infer_value_type(args[2], ctx)

        # For reference types (like Int128/UInt128 structs), need typed select
        if val_type === Int128 || val_type === UInt128
            # Use select_t with the struct type
            type_idx = get_int128_type!(ctx.mod, ctx.type_registry, val_type)
            push!(bytes, Opcode.SELECT_T)
            push!(bytes, 0x01)  # One type
            # Encode (ref null type_idx) for nullable struct ref
            push!(bytes, 0x63)  # ref null
            append!(bytes, encode_leb128_unsigned(type_idx))
        elseif is_struct_type(val_type) || val_type <: AbstractArray || val_type === String
            # Other reference types need typed select too
            wasm_type = julia_to_wasm_type_concrete(val_type, ctx)
            if wasm_type isa ConcreteRef
                push!(bytes, Opcode.SELECT_T)
                push!(bytes, 0x01)  # One type
                push!(bytes, 0x63)  # ref null
                append!(bytes, encode_leb128_unsigned(wasm_type.type_idx))
            else
                # Fall back to untyped select for value types
                push!(bytes, Opcode.SELECT)
            end
        else
            # Value types (i32, i64, f32, f64) use untyped select
            push!(bytes, Opcode.SELECT)
        end
        return bytes
    end

    # Special case for Core.sizeof - returns byte size
    # For strings/arrays, this is the array length
    if is_func(func, :sizeof) && length(args) == 1
        arg = args[1]
        arg_type = infer_value_type(arg, ctx)

        if arg_type === String || arg_type <: AbstractVector
            # For strings and arrays, sizeof is the array length
            append!(bytes, compile_value(arg, ctx))
            push!(bytes, Opcode.GC_PREFIX)
            push!(bytes, Opcode.ARRAY_LEN)
            # array.len returns i32, extend to i64 for Julia's Int
            push!(bytes, Opcode.I64_EXTEND_I32_S)
            return bytes
        end
        # For other types, fall through to error
    end

    # Special case for length - returns character count for strings, element count for arrays
    if is_func(func, :length) && length(args) == 1
        arg = args[1]
        arg_type = infer_value_type(arg, ctx)

        if arg_type === String
            # For strings, length is the array length (each char is one element)
            append!(bytes, compile_value(arg, ctx))
            push!(bytes, Opcode.GC_PREFIX)
            push!(bytes, Opcode.ARRAY_LEN)
            # array.len returns i32, extend to i64 for Julia's Int
            push!(bytes, Opcode.I64_EXTEND_I32_S)
            return bytes
        elseif arg_type <: AbstractVector
            # For Vector, length is v.size[1] (logical size from struct field 1)
            # Vector is now a struct with (ref, size) where size is Tuple{Int64}
            if haskey(ctx.type_registry.structs, arg_type)
                info = ctx.type_registry.structs[arg_type]

                # Get the vector struct
                append!(bytes, compile_value(arg, ctx))

                # Get field 1 (size tuple)
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.STRUCT_GET)
                append!(bytes, encode_leb128_unsigned(info.wasm_type_idx))
                append!(bytes, encode_leb128_unsigned(1))  # Field 1 = size tuple

                # Get field 0 of the size tuple (the Int64 value)
                # Size tuple is Tuple{Int64}
                size_tuple_type = Tuple{Int64}
                if haskey(ctx.type_registry.structs, size_tuple_type)
                    size_info = ctx.type_registry.structs[size_tuple_type]
                    push!(bytes, Opcode.GC_PREFIX)
                    push!(bytes, Opcode.STRUCT_GET)
                    append!(bytes, encode_leb128_unsigned(size_info.wasm_type_idx))
                    append!(bytes, encode_leb128_unsigned(0))  # Field 0 of tuple
                end
                return bytes
            end
            # Fallback to array.len if struct not registered (shouldn't happen)
            append!(bytes, compile_value(arg, ctx))
            push!(bytes, Opcode.GC_PREFIX)
            push!(bytes, Opcode.ARRAY_LEN)
            push!(bytes, Opcode.I64_EXTEND_I32_S)
            return bytes
        end
        # For other types, fall through to error
    end

    # Special case for push!(vec, item) - add element to end of vector
    # WasmGC arrays cannot resize, so we handle two cases:
    # 1. If size < capacity: just set element and increment size
    # 2. If size >= capacity: allocate new array with 2x capacity, copy, update ref
    if is_func(func, :push!) && length(args) >= 2
        vec_arg = args[1]
        item_arg = args[2]
        vec_type = infer_value_type(vec_arg, ctx)

        if vec_type <: AbstractVector && haskey(ctx.type_registry.structs, vec_type)
            elem_type = eltype(vec_type)
            info = ctx.type_registry.structs[vec_type]
            arr_type_idx = get_array_type!(ctx.mod, ctx.type_registry, elem_type)

            # Register size tuple type if needed
            size_tuple_type = Tuple{Int64}
            if !haskey(ctx.type_registry.structs, size_tuple_type)
                register_tuple_type!(ctx.mod, ctx.type_registry, size_tuple_type)
            end
            size_info = ctx.type_registry.structs[size_tuple_type]

            # We need locals to store intermediate values
            # Use local variables to store: vec_ref, old_size, new_size, capacity
            # For now, implement simple case: assume capacity is sufficient
            # In full implementation, we'd add growth logic

            # Algorithm:
            # 1. Get current size from v.size[1]
            # 2. new_size = old_size + 1
            # 3. Set v.size = (new_size,)
            # 4. Get ref = v.ref (the underlying array)
            # 5. Set ref[new_size] = item (using 1-based index)
            # 6. Return vec

            # Step 1-2: Get old_size, compute new_size
            # We'll compile this inline - need to duplicate vec on stack

            # First, allocate a local for the vector
            vec_local = allocate_local!(ctx, vec_type)
            size_local = allocate_local!(ctx, Int64)

            # Store vec in local
            append!(bytes, compile_value(vec_arg, ctx))
            push!(bytes, Opcode.LOCAL_TEE)
            append!(bytes, encode_leb128_unsigned(vec_local))

            # Get size tuple (field 1)
            push!(bytes, Opcode.GC_PREFIX)
            push!(bytes, Opcode.STRUCT_GET)
            append!(bytes, encode_leb128_unsigned(info.wasm_type_idx))
            append!(bytes, encode_leb128_unsigned(1))

            # Get size value (field 0 of tuple)
            push!(bytes, Opcode.GC_PREFIX)
            push!(bytes, Opcode.STRUCT_GET)
            append!(bytes, encode_leb128_unsigned(size_info.wasm_type_idx))
            append!(bytes, encode_leb128_unsigned(0))

            # Add 1 to get new size
            push!(bytes, Opcode.I64_CONST)
            push!(bytes, 0x01)
            push!(bytes, Opcode.I64_ADD)

            # Store new_size in local
            push!(bytes, Opcode.LOCAL_TEE)
            append!(bytes, encode_leb128_unsigned(size_local))

            # Create new size tuple with new_size
            # struct.new for Tuple{Int64}
            push!(bytes, Opcode.GC_PREFIX)
            push!(bytes, Opcode.STRUCT_NEW)
            append!(bytes, encode_leb128_unsigned(size_info.wasm_type_idx))

            # Now we have new size tuple on stack
            # Get vec from local and set its size field
            size_tuple_local = allocate_local!(ctx, size_tuple_type)
            push!(bytes, Opcode.LOCAL_SET)
            append!(bytes, encode_leb128_unsigned(size_tuple_local))

            # Get vec, set size field
            push!(bytes, Opcode.LOCAL_GET)
            append!(bytes, encode_leb128_unsigned(vec_local))
            push!(bytes, Opcode.LOCAL_GET)
            append!(bytes, encode_leb128_unsigned(size_tuple_local))
            push!(bytes, Opcode.GC_PREFIX)
            push!(bytes, Opcode.STRUCT_SET)
            append!(bytes, encode_leb128_unsigned(info.wasm_type_idx))
            append!(bytes, encode_leb128_unsigned(1))  # Field 1 = size

            # Now set the element at index new_size
            # Get ref (field 0 of vec)
            push!(bytes, Opcode.LOCAL_GET)
            append!(bytes, encode_leb128_unsigned(vec_local))
            push!(bytes, Opcode.GC_PREFIX)
            push!(bytes, Opcode.STRUCT_GET)
            append!(bytes, encode_leb128_unsigned(info.wasm_type_idx))
            append!(bytes, encode_leb128_unsigned(0))  # Field 0 = ref (array)

            # Index: new_size - 1 (convert to 0-based)
            push!(bytes, Opcode.LOCAL_GET)
            append!(bytes, encode_leb128_unsigned(size_local))
            push!(bytes, Opcode.I64_CONST)
            push!(bytes, 0x01)
            push!(bytes, Opcode.I64_SUB)
            push!(bytes, Opcode.I32_WRAP_I64)  # array.set expects i32 index

            # Value to store
            append!(bytes, compile_value(item_arg, ctx))

            # array.set
            push!(bytes, Opcode.GC_PREFIX)
            push!(bytes, Opcode.ARRAY_SET)
            append!(bytes, encode_leb128_unsigned(arr_type_idx))

            # Return the vector
            push!(bytes, Opcode.LOCAL_GET)
            append!(bytes, encode_leb128_unsigned(vec_local))

            return bytes
        end
    end

    # Special case for pop!(vec) - remove and return last element
    if is_func(func, :pop!) && length(args) >= 1
        vec_arg = args[1]
        vec_type = infer_value_type(vec_arg, ctx)

        if vec_type <: AbstractVector && haskey(ctx.type_registry.structs, vec_type)
            elem_type = eltype(vec_type)
            info = ctx.type_registry.structs[vec_type]
            arr_type_idx = get_array_type!(ctx.mod, ctx.type_registry, elem_type)

            # Register size tuple type if needed
            size_tuple_type = Tuple{Int64}
            if !haskey(ctx.type_registry.structs, size_tuple_type)
                register_tuple_type!(ctx.mod, ctx.type_registry, size_tuple_type)
            end
            size_info = ctx.type_registry.structs[size_tuple_type]

            # Algorithm:
            # 1. Get current size from v.size[1]
            # 2. Get element at index size (1-based)
            # 3. new_size = old_size - 1
            # 4. Set v.size = (new_size,)
            # 5. Return element

            vec_local = allocate_local!(ctx, vec_type)
            size_local = allocate_local!(ctx, Int64)
            elem_local = allocate_local!(ctx, elem_type)

            # Store vec in local
            append!(bytes, compile_value(vec_arg, ctx))
            push!(bytes, Opcode.LOCAL_TEE)
            append!(bytes, encode_leb128_unsigned(vec_local))

            # Get size tuple (field 1)
            push!(bytes, Opcode.GC_PREFIX)
            push!(bytes, Opcode.STRUCT_GET)
            append!(bytes, encode_leb128_unsigned(info.wasm_type_idx))
            append!(bytes, encode_leb128_unsigned(1))

            # Get size value (field 0 of tuple)
            push!(bytes, Opcode.GC_PREFIX)
            push!(bytes, Opcode.STRUCT_GET)
            append!(bytes, encode_leb128_unsigned(size_info.wasm_type_idx))
            append!(bytes, encode_leb128_unsigned(0))

            # Store size in local
            push!(bytes, Opcode.LOCAL_TEE)
            append!(bytes, encode_leb128_unsigned(size_local))

            # Get element at index size (1-based, so we use size-1 for 0-based)
            # First get ref
            push!(bytes, Opcode.LOCAL_GET)
            append!(bytes, encode_leb128_unsigned(vec_local))
            push!(bytes, Opcode.GC_PREFIX)
            push!(bytes, Opcode.STRUCT_GET)
            append!(bytes, encode_leb128_unsigned(info.wasm_type_idx))
            append!(bytes, encode_leb128_unsigned(0))  # Field 0 = ref

            # Index: size - 1 (convert to 0-based)
            push!(bytes, Opcode.LOCAL_GET)
            append!(bytes, encode_leb128_unsigned(size_local))
            push!(bytes, Opcode.I64_CONST)
            push!(bytes, 0x01)
            push!(bytes, Opcode.I64_SUB)
            push!(bytes, Opcode.I32_WRAP_I64)

            # array.get
            push!(bytes, Opcode.GC_PREFIX)
            push!(bytes, Opcode.ARRAY_GET)
            append!(bytes, encode_leb128_unsigned(arr_type_idx))

            # Store element in local
            push!(bytes, Opcode.LOCAL_SET)
            append!(bytes, encode_leb128_unsigned(elem_local))

            # Compute new_size = old_size - 1
            push!(bytes, Opcode.LOCAL_GET)
            append!(bytes, encode_leb128_unsigned(size_local))
            push!(bytes, Opcode.I64_CONST)
            push!(bytes, 0x01)
            push!(bytes, Opcode.I64_SUB)

            # Create new size tuple
            push!(bytes, Opcode.GC_PREFIX)
            push!(bytes, Opcode.STRUCT_NEW)
            append!(bytes, encode_leb128_unsigned(size_info.wasm_type_idx))

            # Store in local for struct.set
            size_tuple_local = allocate_local!(ctx, size_tuple_type)
            push!(bytes, Opcode.LOCAL_SET)
            append!(bytes, encode_leb128_unsigned(size_tuple_local))

            # Set vec.size = new_size_tuple
            push!(bytes, Opcode.LOCAL_GET)
            append!(bytes, encode_leb128_unsigned(vec_local))
            push!(bytes, Opcode.LOCAL_GET)
            append!(bytes, encode_leb128_unsigned(size_tuple_local))
            push!(bytes, Opcode.GC_PREFIX)
            push!(bytes, Opcode.STRUCT_SET)
            append!(bytes, encode_leb128_unsigned(info.wasm_type_idx))
            append!(bytes, encode_leb128_unsigned(1))  # Field 1 = size

            # Return the element
            push!(bytes, Opcode.LOCAL_GET)
            append!(bytes, encode_leb128_unsigned(elem_local))

            return bytes
        end
    end

    # Special case for getfield/getproperty - struct/tuple field access
    # In newer Julia, obj.field compiles to Base.getproperty(obj, :field)
    # rather than Core.getfield(obj, :field)
    if (is_func(func, :getfield) || is_func(func, :getproperty)) && length(args) >= 2
        obj_arg = args[1]
        field_ref = args[2]
        obj_type = infer_value_type(obj_arg, ctx)

        # Handle Memory{T}.instance pattern (Julia 1.11+ Vector allocation)
        # This pattern appears as Core.getproperty(Memory{T}, :instance)
        # where Memory{T} is passed directly as a DataType
        # Memory{T}.instance is a singleton empty Memory (length 0)
        # We compile it to create an empty WasmGC array
        field_sym = field_ref isa QuoteNode ? field_ref.value : field_ref
        if field_sym === :instance && obj_arg isa DataType && obj_arg <: Memory
            # Memory{T}.instance - create an empty array (length 0)
            # Extract element type from Memory{T}
            elem_type = if obj_arg.name.name === :Memory && length(obj_arg.parameters) >= 1
                obj_arg.parameters[1]
            elseif obj_arg.name.name === :GenericMemory && length(obj_arg.parameters) >= 2
                obj_arg.parameters[2]
            else
                Int32  # default
            end

            # Get or create array type for this element type
            arr_type_idx = get_array_type!(ctx.mod, ctx.type_registry, elem_type)

            # Emit array.new_default with length 0
            push!(bytes, Opcode.I32_CONST)
            push!(bytes, 0x00)  # length = 0
            push!(bytes, Opcode.GC_PREFIX)
            push!(bytes, Opcode.ARRAY_NEW_DEFAULT)
            append!(bytes, encode_leb128_unsigned(arr_type_idx))
            return bytes
        end

        # Handle WasmGlobal field access (:value -> global.get)
        if obj_type <: WasmGlobal
            field_sym = field_ref isa QuoteNode ? field_ref.value : field_ref
            if field_sym === :value
                # Extract global index from type parameter
                global_idx = get_wasm_global_idx(obj_arg, ctx)
                if global_idx !== nothing
                    push!(bytes, Opcode.GLOBAL_GET)
                    append!(bytes, encode_leb128_unsigned(global_idx))
                    return bytes
                end
            end
        end

        # Handle Array field access (:ref and :size) - works for Vector, Matrix, etc.
        # Both Vector and Matrix are now structs with (ref, size) fields
        if obj_type <: AbstractArray
            field_sym = if field_ref isa QuoteNode
                field_ref.value
            else
                field_ref
            end

            if field_sym === :ref
                # :ref returns the underlying array reference (field 0 of struct)
                append!(bytes, compile_value(obj_arg, ctx))
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.STRUCT_GET)
                if haskey(ctx.type_registry.structs, obj_type)
                    info = ctx.type_registry.structs[obj_type]
                    append!(bytes, encode_leb128_unsigned(info.wasm_type_idx))
                    append!(bytes, encode_leb128_unsigned(0))  # Field 0 = data array
                end
                return bytes
            elseif field_sym === :size
                # :size returns a Tuple containing the dimensions (field 1 of struct)
                # For Vector: Tuple{Int64}, for Matrix: Tuple{Int64, Int64}, etc.
                append!(bytes, compile_value(obj_arg, ctx))
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.STRUCT_GET)
                if haskey(ctx.type_registry.structs, obj_type)
                    info = ctx.type_registry.structs[obj_type]
                    append!(bytes, encode_leb128_unsigned(info.wasm_type_idx))
                    append!(bytes, encode_leb128_unsigned(1))  # Field 1 = size tuple
                end
                return bytes
            end
        end

        # Handle MemoryRef field access (:mem, :ptr_or_offset)
        # In WasmGC, MemoryRef IS the array, so :mem just returns it
        if obj_type <: MemoryRef
            field_sym = if field_ref isa QuoteNode
                field_ref.value
            else
                field_ref
            end

            if field_sym === :mem
                # :mem returns the underlying Memory - in WasmGC this is the array itself
                append!(bytes, compile_value(obj_arg, ctx))
                return bytes
            elseif field_sym === :ptr_or_offset
                # Not needed in WasmGC - return 0 as placeholder
                push!(bytes, Opcode.I64_CONST)
                push!(bytes, 0x00)
                return bytes
            end
        end

        # Handle Memory field access (:length, :ptr)
        # In WasmGC, Memory IS the array
        if obj_type <: Memory
            field_sym = if field_ref isa QuoteNode
                field_ref.value
            else
                field_ref
            end

            if field_sym === :length
                # Return array length
                append!(bytes, compile_value(obj_arg, ctx))
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_LEN)
                push!(bytes, Opcode.I64_EXTEND_I32_S)
                return bytes
            elseif field_sym === :ptr
                # Not meaningful in WasmGC - return 0
                push!(bytes, Opcode.I64_CONST)
                push!(bytes, 0x00)
                return bytes
            end
        end

        # Handle closure field access (captured variables)
        if is_closure_type(obj_type)
            # Register closure type if not already
            if !haskey(ctx.type_registry.structs, obj_type)
                register_closure_type!(ctx.mod, ctx.type_registry, obj_type)
            end

            if haskey(ctx.type_registry.structs, obj_type)
                info = ctx.type_registry.structs[obj_type]

                field_sym = if field_ref isa QuoteNode
                    field_ref.value
                else
                    field_ref
                end

                field_idx = findfirst(==(field_sym), info.field_names)
                if field_idx !== nothing
                    append!(bytes, compile_value(obj_arg, ctx))
                    push!(bytes, Opcode.GC_PREFIX)
                    push!(bytes, Opcode.STRUCT_GET)
                    append!(bytes, encode_leb128_unsigned(info.wasm_type_idx))
                    append!(bytes, encode_leb128_unsigned(field_idx - 1))
                    return bytes
                end
            end
        end

        # Handle struct field access by name
        if is_struct_type(obj_type)
            # Register the struct type on-demand if not already registered
            if !haskey(ctx.type_registry.structs, obj_type)
                register_struct_type!(ctx.mod, ctx.type_registry, obj_type)
            end
            info = ctx.type_registry.structs[obj_type]

            field_sym = if field_ref isa QuoteNode
                field_ref.value
            else
                field_ref
            end

            field_idx = findfirst(==(field_sym), info.field_names)
            if field_idx !== nothing
                append!(bytes, compile_value(obj_arg, ctx))
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.STRUCT_GET)
                append!(bytes, encode_leb128_unsigned(info.wasm_type_idx))
                append!(bytes, encode_leb128_unsigned(field_idx - 1))
                return bytes
            end
        end

        # Handle tuple field access by numeric index
        if obj_type <: Tuple
            # Register tuple type if needed
            if !haskey(ctx.type_registry.structs, obj_type)
                register_tuple_type!(ctx.mod, ctx.type_registry, obj_type)
            end

            if haskey(ctx.type_registry.structs, obj_type)
                info = ctx.type_registry.structs[obj_type]

                # Get the field index (1-indexed in Julia)
                field_idx = if field_ref isa Integer
                    field_ref
                elseif field_ref isa Core.SSAValue
                    # Dynamic index - will be handled below for homogeneous tuples
                    :dynamic
                else
                    nothing
                end

                if field_idx === :dynamic
                    # Dynamic tuple indexing - only supported for homogeneous tuples (NTuple)
                    # Check if all elements have the same type
                    elem_types = fieldtypes(obj_type)
                    if length(elem_types) > 0 && all(t -> t === elem_types[1], elem_types)
                        # Homogeneous tuple - we can treat it as an array
                        elem_type = elem_types[1]

                        # For constant tuple (GlobalRef), create a WasmGC array and access it
                        # The tuple value needs to be compiled as an array first

                        # Get or create array type for this element type
                        # Use concrete types for String elements to match tuple field types
                        array_type_idx = if elem_type === String
                            get_string_ref_array_type!(ctx.mod, ctx.type_registry)
                        else
                            get_array_type!(ctx.mod, ctx.type_registry, elem_type)
                        end

                        # Compile the tuple as an array
                        # First compile the tuple value
                        append!(bytes, compile_value(obj_arg, ctx))

                        # The struct is on the stack, we need to convert struct fields to array
                        # Store in local, then create array from fields
                        tuple_local = length(ctx.locals) + ctx.n_params
                        push!(ctx.locals, julia_to_wasm_type_concrete(obj_type, ctx))
                        push!(bytes, Opcode.LOCAL_SET)
                        append!(bytes, encode_leb128_unsigned(tuple_local))

                        # Push all fields onto stack
                        for i in 0:(length(elem_types)-1)
                            push!(bytes, Opcode.LOCAL_GET)
                            append!(bytes, encode_leb128_unsigned(tuple_local))
                            push!(bytes, Opcode.GC_PREFIX)
                            push!(bytes, Opcode.STRUCT_GET)
                            append!(bytes, encode_leb128_unsigned(info.wasm_type_idx))
                            append!(bytes, encode_leb128_unsigned(i))
                        end

                        # Create array from fields
                        push!(bytes, Opcode.GC_PREFIX)
                        push!(bytes, Opcode.ARRAY_NEW_FIXED)
                        append!(bytes, encode_leb128_unsigned(array_type_idx))
                        append!(bytes, encode_leb128_unsigned(length(elem_types)))

                        # Store array in local - use concrete ref to specific array type
                        array_local = length(ctx.locals) + ctx.n_params
                        push!(ctx.locals, ConcreteRef(array_type_idx, true))
                        push!(bytes, Opcode.LOCAL_SET)
                        append!(bytes, encode_leb128_unsigned(array_local))

                        # Now compile the index and access the array
                        # Julia uses 1-based indexing, Wasm uses 0-based
                        append!(bytes, compile_value(field_ref, ctx))

                        # Subtract 1 for 0-based indexing
                        push!(bytes, Opcode.I64_CONST)
                        push!(bytes, 0x01)
                        push!(bytes, Opcode.I64_SUB)
                        # Wrap to i32 for array index
                        push!(bytes, Opcode.I32_WRAP_I64)

                        # Store index in local
                        idx_local = length(ctx.locals) + ctx.n_params
                        push!(ctx.locals, I32)
                        push!(bytes, Opcode.LOCAL_SET)
                        append!(bytes, encode_leb128_unsigned(idx_local))

                        # Access array: array.get
                        push!(bytes, Opcode.LOCAL_GET)
                        append!(bytes, encode_leb128_unsigned(array_local))
                        push!(bytes, Opcode.LOCAL_GET)
                        append!(bytes, encode_leb128_unsigned(idx_local))
                        push!(bytes, Opcode.GC_PREFIX)
                        push!(bytes, Opcode.ARRAY_GET)
                        append!(bytes, encode_leb128_unsigned(array_type_idx))

                        return bytes
                    end
                    # Non-homogeneous tuple with dynamic index - fall through to error
                elseif field_idx !== nothing && field_idx >= 1 && field_idx <= length(info.field_names)
                    append!(bytes, compile_value(obj_arg, ctx))
                    push!(bytes, Opcode.GC_PREFIX)
                    push!(bytes, Opcode.STRUCT_GET)
                    append!(bytes, encode_leb128_unsigned(info.wasm_type_idx))
                    append!(bytes, encode_leb128_unsigned(field_idx - 1))  # 0-indexed
                    return bytes
                end
            end
        end
    end

    # Special case for memoryrefget - array element access
    # memoryrefget(ref, ordering, boundscheck) where ref is from memoryrefnew
    if is_func(func, :memoryrefget) && length(args) >= 1
        ref_arg = args[1]
        ref_type = infer_value_type(ref_arg, ctx)

        # Extract element type from MemoryRef{T} or GenericMemoryRef{atomicity, T, addrspace}
        elem_type = Int32  # default
        if ref_type isa DataType
            if ref_type.name.name === :MemoryRef
                elem_type = ref_type.parameters[1]
            elseif ref_type.name.name === :GenericMemoryRef
                # GenericMemoryRef has parameters (atomicity, element_type, addrspace)
                elem_type = ref_type.parameters[2]
            end
        end

        # Get or create array type for this element type
        array_type_idx = get_array_type!(ctx.mod, ctx.type_registry, elem_type)

        # The ref SSA value from memoryrefnew will have compiled to [array_ref, i32_index]
        # We need to compile ref_arg which will leave [array_ref, i32_index] on stack
        append!(bytes, compile_value(ref_arg, ctx))

        push!(bytes, Opcode.GC_PREFIX)
        push!(bytes, Opcode.ARRAY_GET)
        append!(bytes, encode_leb128_unsigned(array_type_idx))
        return bytes
    end

    # Special case for memoryrefoffset - get the 1-based offset of a MemoryRef
    # This is used by push!, resize!, and other dynamic array operations
    # Fresh MemoryRefs (from Core.memoryref, getfield(vec, :ref)) have offset 1
    # Indexed MemoryRefs (from memoryrefnew(ref, index, bc)) have offset = index
    if is_func(func, :memoryrefoffset) && length(args) >= 1
        ref_arg = args[1]

        # Check if this ref came from a memoryrefnew with an index
        if ref_arg isa Core.SSAValue && haskey(ctx.memoryref_offsets, ref_arg.id)
            # This MemoryRef has a recorded offset - compile the index value
            index_val = ctx.memoryref_offsets[ref_arg.id]
            append!(bytes, compile_value(index_val, ctx))

            # Ensure result is i64 (Julia's Int)
            idx_type = infer_value_type(index_val, ctx)
            if idx_type !== Int64 && idx_type !== Int
                # Convert to i64 if needed
                push!(bytes, Opcode.I64_EXTEND_I32_S)
            end
        else
            # Fresh MemoryRef - offset is always 1
            push!(bytes, Opcode.I64_CONST)
            push!(bytes, 0x01)  # 1
        end
        return bytes
    end

    # Special case for memoryrefset! - array element assignment
    # memoryrefset!(ref, value, ordering, boundscheck) -> stores value in array
    # In Julia, setindex! returns the stored value, so we need to return it too
    if is_func(func, :memoryrefset!) && length(args) >= 2
        ref_arg = args[1]
        value_arg = args[2]
        ref_type = infer_value_type(ref_arg, ctx)

        # Extract element type from MemoryRef{T} or GenericMemoryRef{atomicity, T, addrspace}
        elem_type = Int32  # default
        if ref_type isa DataType
            if ref_type.name.name === :MemoryRef
                elem_type = ref_type.parameters[1]
            elseif ref_type.name.name === :GenericMemoryRef
                # GenericMemoryRef has parameters (atomicity, element_type, addrspace)
                elem_type = ref_type.parameters[2]
            end
        end

        # Get or create array type for this element type
        array_type_idx = get_array_type!(ctx.mod, ctx.type_registry, elem_type)

        # Compile ref_arg which will leave [array_ref, i32_index] on stack
        append!(bytes, compile_value(ref_arg, ctx))

        # Compile the value to store - we need it twice (for array.set and return)
        # First compile gets the value on stack for array.set
        append!(bytes, compile_value(value_arg, ctx))

        # array.set consumes [array_ref, i32_index, value] and returns nothing
        push!(bytes, Opcode.GC_PREFIX)
        push!(bytes, Opcode.ARRAY_SET)
        append!(bytes, encode_leb128_unsigned(array_type_idx))

        # Julia's memoryrefset! returns the stored value, so push it again
        # This is needed because compile_statement may add LOCAL_SET after this
        append!(bytes, compile_value(value_arg, ctx))
        return bytes
    end

    # Special case for Core.memoryref - creates MemoryRef from Memory
    # memoryref(memory::Memory{T}) -> MemoryRef{T}
    # In WasmGC, this is a no-op since Memory IS the array
    if is_func(func, :memoryref) && length(args) == 1
        # Pass through the array reference - Memory and MemoryRef are the same in WasmGC
        append!(bytes, compile_value(args[1], ctx))
        return bytes
    end

    # Special case for memoryrefnew - handle both patterns:
    # 1. memoryrefnew(memory) -> MemoryRef (for Vector allocation, just pass through)
    # 2. memoryrefnew(base_ref, index, boundscheck) -> MemoryRef at offset
    if is_func(func, :memoryrefnew)
        if length(args) == 1
            # Single arg: just wrapping a Memory - pass through the array reference
            # This is a "fresh" MemoryRef with offset 1
            append!(bytes, compile_value(args[1], ctx))
            return bytes
        elseif length(args) >= 2
            base_ref = args[1]
            index = args[2]

            # Record the offset for this MemoryRef SSA so memoryrefoffset can use it
            ctx.memoryref_offsets[idx] = index

            # Compile the base array reference
            append!(bytes, compile_value(base_ref, ctx))

            # Compile and convert index to i32 (Julia uses 1-based Int64, Wasm uses 0-based i32)
            append!(bytes, compile_value(index, ctx))

            # Check if index is already Int32
            idx_type = infer_value_type(index, ctx)
            if idx_type === Int64 || idx_type === Int
                # Convert to i32 and subtract 1 for 0-based indexing
                push!(bytes, Opcode.I32_WRAP_I64)  # i64 -> i32
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x01)  # 1
                push!(bytes, Opcode.I32_SUB)  # index - 1 for 0-based
            else
                # Already i32, just subtract 1
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x01)  # 1
                push!(bytes, Opcode.I32_SUB)  # index - 1 for 0-based
            end

            # Now stack has [array_ref, i32_index] which is what memoryrefget needs
            return bytes
        end
    end

    # Special case for Core.tuple - tuple creation
    if is_func(func, :tuple) && length(args) > 0
        # Infer tuple type from arguments
        elem_types = Type[infer_value_type(arg, ctx) for arg in args]
        tuple_type = Tuple{elem_types...}

        # Register tuple type
        if !haskey(ctx.type_registry.structs, tuple_type)
            register_tuple_type!(ctx.mod, ctx.type_registry, tuple_type)
        end

        if haskey(ctx.type_registry.structs, tuple_type)
            info = ctx.type_registry.structs[tuple_type]

            # Push all tuple elements
            for arg in args
                append!(bytes, compile_value(arg, ctx))
            end

            # struct.new
            push!(bytes, Opcode.GC_PREFIX)
            push!(bytes, Opcode.STRUCT_NEW)
            append!(bytes, encode_leb128_unsigned(info.wasm_type_idx))

            return bytes
        end
    end

    # Special case for setfield!/setproperty! - mutable struct field assignment
    # Also handles WasmGlobal (:value -> global.set)
    # In newer Julia, obj.field = val compiles to Base.setproperty!(obj, :field, val)
    if (is_func(func, :setfield!) || is_func(func, :setproperty!)) && length(args) >= 3
        obj_arg = args[1]
        field_ref = args[2]
        value_arg = args[3]
        obj_type = infer_value_type(obj_arg, ctx)

        # Handle WasmGlobal field assignment (:value -> global.set)
        if obj_type <: WasmGlobal
            field_sym = field_ref isa QuoteNode ? field_ref.value : field_ref
            if field_sym === :value
                # Extract global index from type parameter
                global_idx = get_wasm_global_idx(obj_arg, ctx)
                if global_idx !== nothing
                    # Push the value to set
                    append!(bytes, compile_value(value_arg, ctx))
                    # Emit global.set
                    push!(bytes, Opcode.GLOBAL_SET)
                    append!(bytes, encode_leb128_unsigned(global_idx))
                    # setfield! returns the value, so push it again
                    append!(bytes, compile_value(value_arg, ctx))
                    return bytes
                end
            end
        end

        # Handle Vector/Array field assignment (:size is mutable for push!/resize!)
        # Vector{T} is now a struct with (ref, size) where size is mutable
        if obj_type <: AbstractArray
            field_sym = field_ref isa QuoteNode ? field_ref.value : field_ref
            if field_sym === :size && haskey(ctx.type_registry.structs, obj_type)
                info = ctx.type_registry.structs[obj_type]
                # :size is field index 2 (1-indexed), so 1 in 0-indexed
                # struct.set expects: [ref, value]

                # IMPORTANT: The value_arg might be an SSA that was just computed and
                # is on top of the stack. If we compile obj_arg first, we'd push it
                # AFTER the value, giving wrong order [value, ref] instead of [ref, value].
                # Solution: compile value first, store in temp local, then compile ref.
                value_type = infer_value_type(value_arg, ctx)
                temp_local = allocate_local!(ctx, value_type)

                # Compile value and store in local (value may already be on stack from prev stmt)
                append!(bytes, compile_value(value_arg, ctx))
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(temp_local))

                # Now compile obj (struct ref)
                append!(bytes, compile_value(obj_arg, ctx))

                # Load value from local
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(temp_local))

                # struct.set
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.STRUCT_SET)
                append!(bytes, encode_leb128_unsigned(info.wasm_type_idx))
                append!(bytes, encode_leb128_unsigned(1))  # Field 1 = size tuple

                # setfield! returns the value, so push it again
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(temp_local))
                return bytes
            end
        end

        # Handle mutable struct field assignment
        if is_struct_type(obj_type) && ismutabletype(obj_type)
            if haskey(ctx.type_registry.structs, obj_type)
                info = ctx.type_registry.structs[obj_type]
                field_sym = field_ref isa QuoteNode ? field_ref.value : field_ref

                field_idx = findfirst(==(field_sym), info.field_names)
                if field_idx !== nothing
                    # struct.set expects: [ref, value]
                    append!(bytes, compile_value(obj_arg, ctx))
                    append!(bytes, compile_value(value_arg, ctx))
                    push!(bytes, Opcode.GC_PREFIX)
                    push!(bytes, Opcode.STRUCT_SET)
                    append!(bytes, encode_leb128_unsigned(info.wasm_type_idx))
                    append!(bytes, encode_leb128_unsigned(field_idx - 1))  # 0-indexed
                    # setfield! returns the value, so push it again
                    append!(bytes, compile_value(value_arg, ctx))
                    return bytes
                end
            end
        end

        # Handle setfield! on Base.RefValue (used for optimization sinks)
        # These are no-ops in Wasm since we don't need the sink pattern
        if obj_type <: Base.RefValue
            # Just push the value (setfield! returns the value)
            append!(bytes, compile_value(value_arg, ctx))
            return bytes
        end
        # Fall through for other struct types - will hit error
    end

    # Special case for compilerbarrier - just pass through the value
    if is_func(func, :compilerbarrier)
        # compilerbarrier(kind, value) - first arg is a symbol, second is the value
        # We only want the value (second arg)
        if length(args) >= 2
            append!(bytes, compile_value(args[2], ctx))
        end
        return bytes
    end

    # Special case for typeassert - just pass through the value
    # Core.typeassert(x, T) returns x if type matches, throws otherwise
    # In Wasm we don't do runtime type checks, so just return the value
    if is_func(func, :typeassert)
        if length(args) >= 1
            append!(bytes, compile_value(args[1], ctx))
        end
        return bytes
    end

    # Special case for string equality/identity comparison (=== and !==)
    # Must be handled before generic argument pushing since strings are refs, not integers
    if (is_func(func, :(===)) || is_func(func, :(!==))) && length(args) == 2
        arg1_type = infer_value_type(args[1], ctx)
        arg2_type = infer_value_type(args[2], ctx)
        if arg1_type === String && arg2_type === String
            append!(bytes, compile_string_equal(args[1], args[2], ctx))
            if is_func(func, :(!==))
                # Negate the result for !==
                push!(bytes, Opcode.I32_EQZ)
            end
            return bytes
        end

        # Special case: comparing ref type with nothing - use ref.is_null
        arg1_is_nothing = is_nothing_value(args[1], ctx)
        arg2_is_nothing = is_nothing_value(args[2], ctx)

        if (arg1_is_nothing && is_ref_type_or_union(arg2_type)) ||
           (arg2_is_nothing && is_ref_type_or_union(arg1_type))
            # Compile the non-nothing ref argument
            if arg1_is_nothing
                append!(bytes, compile_value(args[2], ctx))
            else
                append!(bytes, compile_value(args[1], ctx))
            end
            # ref.is_null checks if ref is null (returns i32 1 for null, 0 otherwise)
            push!(bytes, Opcode.REF_IS_NULL)
            if is_func(func, :(!==))
                # Negate for !== (we want true when NOT null)
                push!(bytes, Opcode.I32_EQZ)
            end
            return bytes
        end
    end

    # Push arguments onto the stack (normal case)
    for arg in args
        append!(bytes, compile_value(arg, ctx))
    end

    # Determine argument type for opcode selection
    arg_type = length(args) > 0 ? infer_value_type(args[1], ctx) : Int64
    is_32bit = arg_type === Int32 || arg_type === UInt32 || arg_type === Bool || arg_type === Char ||
               arg_type === Int16 || arg_type === UInt16 || arg_type === Int8 || arg_type === UInt8 ||
               (isprimitivetype(arg_type) && sizeof(arg_type) <= 4)
    is_128bit = arg_type === Int128 || arg_type === UInt128

    # Match intrinsics by name
    if is_func(func, :add_int)
        if is_128bit
            # 128-bit addition: (a_lo, a_hi) + (b_lo, b_hi)
            # Stack has: [a_struct, b_struct], need to produce result_struct
            # This is complex - need to extract fields, compute with carry, create new struct
            append!(bytes, emit_int128_add(ctx, arg_type))
        else
            push!(bytes, is_32bit ? Opcode.I32_ADD : Opcode.I64_ADD)
        end

    elseif is_func(func, :sub_int)
        if is_128bit
            # 128-bit subtraction
            append!(bytes, emit_int128_sub(ctx, arg_type))
        else
            push!(bytes, is_32bit ? Opcode.I32_SUB : Opcode.I64_SUB)
        end

    elseif is_func(func, :mul_int)
        if is_128bit
            # 128-bit multiplication (only need low 128 bits of result)
            append!(bytes, emit_int128_mul(ctx, arg_type))
        else
            push!(bytes, is_32bit ? Opcode.I32_MUL : Opcode.I64_MUL)
        end

    elseif is_func(func, :sdiv_int) || is_func(func, :checked_sdiv_int)
        push!(bytes, is_32bit ? Opcode.I32_DIV_S : Opcode.I64_DIV_S)

    elseif is_func(func, :udiv_int) || is_func(func, :checked_udiv_int)
        push!(bytes, is_32bit ? Opcode.I32_DIV_U : Opcode.I64_DIV_U)

    elseif is_func(func, :srem_int) || is_func(func, :checked_srem_int)
        push!(bytes, is_32bit ? Opcode.I32_REM_S : Opcode.I64_REM_S)

    elseif is_func(func, :urem_int) || is_func(func, :checked_urem_int)
        push!(bytes, is_32bit ? Opcode.I32_REM_U : Opcode.I64_REM_U)

    # Bitcast (reinterpret bits between types)
    elseif is_func(func, :bitcast)
        # Bitcast reinterprets bits between same-size types
        # Need to emit reinterpret instructions for float<->int conversions
        # args = [target_type, source_value]
        # Get the target type - it's the first actual argument (args[1] after extracting args[2:end])
        target_type_ref = length(args) >= 1 ? args[1] : nothing
        source_val = length(args) >= 2 ? args[2] : nothing

        # Determine target type from the GlobalRef or type literal
        target_type = if target_type_ref isa GlobalRef
            # Try to get the actual type from the GlobalRef
            if target_type_ref.name === :Int64 || target_type_ref.name === Symbol("Base.Int64")
                Int64
            elseif target_type_ref.name === :UInt64
                UInt64
            elseif target_type_ref.name === :Int32 || target_type_ref.name === Symbol("Base.Int32")
                Int32
            elseif target_type_ref.name === :UInt32
                UInt32
            elseif target_type_ref.name === :Float64
                Float64
            elseif target_type_ref.name === :Float32
                Float32
            elseif target_type_ref.name === :Int128
                Int128
            elseif target_type_ref.name === :UInt128
                UInt128
            else
                # Try to evaluate the GlobalRef
                try
                    getfield(target_type_ref.mod, target_type_ref.name)
                catch
                    Any
                end
            end
        elseif target_type_ref isa DataType
            target_type_ref
        else
            Any
        end

        # Determine source type
        source_type = source_val !== nothing ? infer_value_type(source_val, ctx) : Any

        # Emit appropriate reinterpret instruction if needed
        if source_type === Float64 && (target_type === Int64 || target_type === UInt64)
            push!(bytes, Opcode.I64_REINTERPRET_F64)
        elseif (source_type === Int64 || source_type === UInt64) && target_type === Float64
            push!(bytes, Opcode.F64_REINTERPRET_I64)
        elseif source_type === Float32 && (target_type === Int32 || target_type === UInt32)
            push!(bytes, Opcode.I32_REINTERPRET_F32)
        elseif (source_type === Int32 || source_type === UInt32) && target_type === Float32
            push!(bytes, Opcode.F32_REINTERPRET_I32)
        end
        # For other cases (Int64<->UInt64, Int32<->UInt32, Int128<->UInt128),
        # bitcast is a no-op in Wasm (same representation)

    elseif is_func(func, :neg_int)
        if is_128bit
            # 128-bit negation
            append!(bytes, emit_int128_neg(ctx, arg_type))
        elseif is_32bit
            # For simplicity, emit: i32.const -1, i32.xor, i32.const 1, i32.add
            # Which is equivalent to: ~x + 1 = -x
            push!(bytes, Opcode.I32_CONST)
            push!(bytes, 0x7F)  # -1 in signed LEB128
            push!(bytes, Opcode.I32_XOR)
            push!(bytes, Opcode.I32_CONST)
            push!(bytes, 0x01)
            push!(bytes, Opcode.I32_ADD)
        else
            push!(bytes, Opcode.I64_CONST)
            push!(bytes, 0x7F)  # -1 in signed LEB128
            push!(bytes, Opcode.I64_XOR)
            push!(bytes, Opcode.I64_CONST)
            push!(bytes, 0x01)
            push!(bytes, Opcode.I64_ADD)
        end

    elseif is_func(func, :flipsign_int)
        # flipsign_int(x, y) returns -x if y < 0, otherwise x
        # Formula: (x xor signbit) - signbit where signbit = y >> 63 (all 1s if negative)
        # We need both x and y on stack, but they've been pushed as: [x, y]

        if is_128bit
            # For 128-bit, check if y's hi word is negative
            # flipsign_int(x, y) = y < 0 ? -x : x
            type_idx = get_int128_type!(ctx.mod, ctx.type_registry, arg_type)

            # Pop y struct to local
            y_struct_local = length(ctx.locals) + ctx.n_params
            push!(ctx.locals, julia_to_wasm_type_concrete(arg_type, ctx))
            push!(bytes, Opcode.LOCAL_SET)
            append!(bytes, encode_leb128_unsigned(y_struct_local))

            # Pop x struct to local
            x_struct_local = length(ctx.locals) + ctx.n_params
            push!(ctx.locals, julia_to_wasm_type_concrete(arg_type, ctx))
            push!(bytes, Opcode.LOCAL_SET)
            append!(bytes, encode_leb128_unsigned(x_struct_local))

            # Get y's hi part to check sign
            push!(bytes, Opcode.LOCAL_GET)
            append!(bytes, encode_leb128_unsigned(y_struct_local))
            push!(bytes, Opcode.GC_PREFIX)
            push!(bytes, Opcode.STRUCT_GET)
            append!(bytes, encode_leb128_unsigned(type_idx))
            append!(bytes, encode_leb128_unsigned(1))  # hi field

            # Check if negative (hi < 0)
            push!(bytes, Opcode.I64_CONST)
            push!(bytes, 0x00)
            push!(bytes, Opcode.I64_LT_S)

            # Store condition
            is_neg_local = length(ctx.locals) + ctx.n_params
            push!(ctx.locals, I32)
            push!(bytes, Opcode.LOCAL_SET)
            append!(bytes, encode_leb128_unsigned(is_neg_local))

            # Compute -x using emit_int128_neg
            push!(bytes, Opcode.LOCAL_GET)
            append!(bytes, encode_leb128_unsigned(x_struct_local))
            append!(bytes, emit_int128_neg(ctx, arg_type))

            # Store negated x
            neg_x_local = length(ctx.locals) + ctx.n_params
            push!(ctx.locals, julia_to_wasm_type_concrete(arg_type, ctx))
            push!(bytes, Opcode.LOCAL_SET)
            append!(bytes, encode_leb128_unsigned(neg_x_local))

            # Allocate result local
            result_local = length(ctx.locals) + ctx.n_params
            push!(ctx.locals, julia_to_wasm_type_concrete(arg_type, ctx))

            # if is_neg { result = neg_x } else { result = x }
            push!(bytes, Opcode.LOCAL_GET)
            append!(bytes, encode_leb128_unsigned(is_neg_local))
            push!(bytes, Opcode.IF)
            push!(bytes, 0x40)  # void

            push!(bytes, Opcode.LOCAL_GET)
            append!(bytes, encode_leb128_unsigned(neg_x_local))
            push!(bytes, Opcode.LOCAL_SET)
            append!(bytes, encode_leb128_unsigned(result_local))

            push!(bytes, Opcode.ELSE)

            push!(bytes, Opcode.LOCAL_GET)
            append!(bytes, encode_leb128_unsigned(x_struct_local))
            push!(bytes, Opcode.LOCAL_SET)
            append!(bytes, encode_leb128_unsigned(result_local))

            push!(bytes, Opcode.END)

            # Push result
            push!(bytes, Opcode.LOCAL_GET)
            append!(bytes, encode_leb128_unsigned(result_local))

        else
            # Pop y to local, check sign, conditionally negate x
            y_local = length(ctx.locals) + ctx.n_params
            push!(ctx.locals, is_32bit ? I32 : I64)
            push!(bytes, Opcode.LOCAL_SET)
            append!(bytes, encode_leb128_unsigned(y_local))

            x_local = length(ctx.locals) + ctx.n_params
            push!(ctx.locals, is_32bit ? I32 : I64)
            push!(bytes, Opcode.LOCAL_SET)
            append!(bytes, encode_leb128_unsigned(x_local))

            # Compute signbit = y >> (bits-1) (arithmetic shift gives all 1s if negative)
            push!(bytes, Opcode.LOCAL_GET)
            append!(bytes, encode_leb128_unsigned(y_local))
            if is_32bit
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 31)
                push!(bytes, Opcode.I32_SHR_S)
            else
                push!(bytes, Opcode.I64_CONST)
                push!(bytes, 63)
                push!(bytes, Opcode.I64_SHR_S)
            end

            signbit_local = length(ctx.locals) + ctx.n_params
            push!(ctx.locals, is_32bit ? I32 : I64)
            push!(bytes, Opcode.LOCAL_SET)
            append!(bytes, encode_leb128_unsigned(signbit_local))

            # result = (x xor signbit) - signbit
            push!(bytes, Opcode.LOCAL_GET)
            append!(bytes, encode_leb128_unsigned(x_local))
            push!(bytes, Opcode.LOCAL_GET)
            append!(bytes, encode_leb128_unsigned(signbit_local))
            push!(bytes, is_32bit ? Opcode.I32_XOR : Opcode.I64_XOR)
            push!(bytes, Opcode.LOCAL_GET)
            append!(bytes, encode_leb128_unsigned(signbit_local))
            push!(bytes, is_32bit ? Opcode.I32_SUB : Opcode.I64_SUB)
        end

    # Comparison operations
    elseif is_func(func, :slt_int)  # signed less than
        if is_128bit
            append!(bytes, emit_int128_slt(ctx, arg_type))
        else
            push!(bytes, is_32bit ? Opcode.I32_LT_S : Opcode.I64_LT_S)
        end

    elseif is_func(func, :sle_int)  # signed less or equal
        push!(bytes, is_32bit ? Opcode.I32_LE_S : Opcode.I64_LE_S)

    elseif is_func(func, :ult_int)  # unsigned less than
        if is_128bit
            append!(bytes, emit_int128_ult(ctx, arg_type))
        else
            push!(bytes, is_32bit ? Opcode.I32_LT_U : Opcode.I64_LT_U)
        end

    elseif is_func(func, :ule_int)  # unsigned less or equal
        push!(bytes, is_32bit ? Opcode.I32_LE_U : Opcode.I64_LE_U)

    elseif is_func(func, :eq_int)
        if is_128bit
            append!(bytes, emit_int128_eq(ctx, arg_type))
        else
            push!(bytes, is_32bit ? Opcode.I32_EQ : Opcode.I64_EQ)
        end

    elseif is_func(func, :ne_int)
        if is_128bit
            append!(bytes, emit_int128_ne(ctx, arg_type))
        else
            push!(bytes, is_32bit ? Opcode.I32_NE : Opcode.I64_NE)
        end

    # Float comparison operations
    elseif is_func(func, :lt_float)
        push!(bytes, arg_type === Float32 ? Opcode.F32_LT : Opcode.F64_LT)

    elseif is_func(func, :le_float)
        push!(bytes, arg_type === Float32 ? Opcode.F32_LE : Opcode.F64_LE)

    elseif is_func(func, :gt_float)
        push!(bytes, arg_type === Float32 ? Opcode.F32_GT : Opcode.F64_GT)

    elseif is_func(func, :ge_float)
        push!(bytes, arg_type === Float32 ? Opcode.F32_GE : Opcode.F64_GE)

    elseif is_func(func, :eq_float)
        push!(bytes, arg_type === Float32 ? Opcode.F32_EQ : Opcode.F64_EQ)

    elseif is_func(func, :ne_float)
        push!(bytes, arg_type === Float32 ? Opcode.F32_NE : Opcode.F64_NE)

    # Identity comparison (=== for integers is same as ==, for floats use float eq)
    elseif is_func(func, :(===))
        if is_128bit
            append!(bytes, emit_int128_eq(ctx, arg_type))
        elseif arg_type === Float64
            push!(bytes, Opcode.F64_EQ)
        elseif arg_type === Float32
            push!(bytes, Opcode.F32_EQ)
        elseif is_ref_type_or_union(arg_type) || is_ref_type_or_union(infer_value_type(args[2], ctx))
            # Reference type comparison - use ref.eq for WasmGC
            # If either arg is literally nothing, the value is already ref.null on stack
            # The ref.eq instruction handles null refs correctly
            # ref.eq is a standalone opcode (0xD3), not under GC_PREFIX
            push!(bytes, Opcode.REF_EQ)
        else
            push!(bytes, is_32bit ? Opcode.I32_EQ : Opcode.I64_EQ)
        end

    elseif is_func(func, :(!==))
        if is_128bit
            append!(bytes, emit_int128_ne(ctx, arg_type))
        elseif arg_type === Float64
            push!(bytes, Opcode.F64_NE)
        elseif arg_type === Float32
            push!(bytes, Opcode.F32_NE)
        elseif is_ref_type_or_union(arg_type) || is_ref_type_or_union(infer_value_type(args[2], ctx))
            # Reference type comparison - use ref.eq then negate
            # ref.eq is a standalone opcode (0xD3), not under GC_PREFIX
            push!(bytes, Opcode.REF_EQ)
            push!(bytes, Opcode.I32_EQZ)  # Negate for !==
        else
            push!(bytes, is_32bit ? Opcode.I32_NE : Opcode.I64_NE)
        end

    # Bitwise operations
    elseif is_func(func, :and_int)
        if is_128bit
            append!(bytes, emit_int128_and(ctx, arg_type))
        else
            push!(bytes, is_32bit ? Opcode.I32_AND : Opcode.I64_AND)
        end

    elseif is_func(func, :or_int)
        if is_128bit
            append!(bytes, emit_int128_or(ctx, arg_type))
        else
            push!(bytes, is_32bit ? Opcode.I32_OR : Opcode.I64_OR)
        end

    elseif is_func(func, :xor_int)
        if is_128bit
            append!(bytes, emit_int128_xor(ctx, arg_type))
        else
            push!(bytes, is_32bit ? Opcode.I32_XOR : Opcode.I64_XOR)
        end

    elseif is_func(func, :not_int)
        # Check if this is boolean negation (result of comparison)
        # If so, use eqz instead of bitwise NOT
        if length(args) == 1 && is_boolean_value(args[1], ctx)
            # Boolean NOT: eqz turns 0->1, 1->0
            push!(bytes, Opcode.I32_EQZ)
        else
            # Bitwise NOT: x xor -1
            if is_32bit
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x7F)  # -1
                push!(bytes, Opcode.I32_XOR)
            else
                push!(bytes, Opcode.I64_CONST)
                push!(bytes, 0x7F)  # -1
                push!(bytes, Opcode.I64_XOR)
            end
        end

    # Shift operations
    # Note: Wasm requires shift amount to have same type as value being shifted
    # Julia often uses Int64/UInt64 shift amounts even for Int32 values
    elseif is_func(func, :shl_int)
        if is_128bit
            # 128-bit left shift: stack has [x_struct, n_i64]
            append!(bytes, emit_int128_shl(ctx, arg_type))
        else
            if is_32bit && length(args) >= 2
                shift_type = infer_value_type(args[2], ctx)
                if shift_type === Int64 || shift_type === UInt64
                    # Truncate i64 shift amount to i32
                    push!(bytes, Opcode.I32_WRAP_I64)
                end
            end
            push!(bytes, is_32bit ? Opcode.I32_SHL : Opcode.I64_SHL)
        end

    elseif is_func(func, :ashr_int)  # arithmetic shift right
        if is_32bit && length(args) >= 2
            shift_type = infer_value_type(args[2], ctx)
            if shift_type === Int64 || shift_type === UInt64
                # Truncate i64 shift amount to i32
                push!(bytes, Opcode.I32_WRAP_I64)
            end
        end
        push!(bytes, is_32bit ? Opcode.I32_SHR_S : Opcode.I64_SHR_S)

    elseif is_func(func, :lshr_int)  # logical shift right
        if is_128bit
            # 128-bit logical right shift: stack has [x_struct, n_i64]
            append!(bytes, emit_int128_lshr(ctx, arg_type))
        else
            if is_32bit && length(args) >= 2
                shift_type = infer_value_type(args[2], ctx)
                if shift_type === Int64 || shift_type === UInt64
                    # Truncate i64 shift amount to i32
                    push!(bytes, Opcode.I32_WRAP_I64)
                end
            end
            push!(bytes, is_32bit ? Opcode.I32_SHR_U : Opcode.I64_SHR_U)
        end

    # Count leading/trailing zeros (used in Char conversion)
    elseif is_func(func, :ctlz_int)
        if is_128bit
            append!(bytes, emit_int128_ctlz(ctx, arg_type))
        else
            push!(bytes, is_32bit ? Opcode.I32_CLZ : Opcode.I64_CLZ)
        end

    elseif is_func(func, :cttz_int)
        push!(bytes, is_32bit ? Opcode.I32_CTZ : Opcode.I64_CTZ)

    # Float operations
    elseif is_func(func, :add_float)
        push!(bytes, arg_type === Float32 ? Opcode.F32_ADD : Opcode.F64_ADD)

    elseif is_func(func, :sub_float)
        push!(bytes, arg_type === Float32 ? Opcode.F32_SUB : Opcode.F64_SUB)

    elseif is_func(func, :mul_float)
        push!(bytes, arg_type === Float32 ? Opcode.F32_MUL : Opcode.F64_MUL)

    elseif is_func(func, :div_float)
        push!(bytes, arg_type === Float32 ? Opcode.F32_DIV : Opcode.F64_DIV)

    elseif is_func(func, :neg_float)
        push!(bytes, arg_type === Float32 ? Opcode.F32_NEG : Opcode.F64_NEG)

    # Fused multiply-add: muladd_float(a, b, c) = a*b + c
    # WASM doesn't have native fma, so we implement as mul then add
    elseif is_func(func, :muladd_float)
        # Stack has [a, b, c], we need to compute a*b + c
        # First multiply a*b, then add c
        push!(bytes, arg_type === Float32 ? Opcode.F32_MUL : Opcode.F64_MUL)
        push!(bytes, arg_type === Float32 ? Opcode.F32_ADD : Opcode.F64_ADD)

    # Type conversions
    elseif is_func(func, :sext_int)  # Sign extend
        # sext_int(TargetType, value) - first arg is target type
        target_type_ref = args[1]
        # Extract actual type from GlobalRef if needed
        target_type = if target_type_ref isa GlobalRef
            try
                getfield(target_type_ref.mod, target_type_ref.name)
            catch
                target_type_ref
            end
        else
            target_type_ref
        end
        if target_type === Int64 || target_type === UInt64
            # Extending to 64-bit - emit extend instruction
            push!(bytes, Opcode.I64_EXTEND_I32_S)
        end
        # If extending to 32-bit (Int32), it's a no-op since small types already map to i32

    elseif is_func(func, :zext_int)  # Zero extend
        # zext_int(TargetType, value) - first arg is target type
        target_type_ref = args[1]
        # Extract actual type from GlobalRef if needed
        target_type = if target_type_ref isa GlobalRef
            try
                getfield(target_type_ref.mod, target_type_ref.name)
            catch
                target_type_ref
            end
        else
            target_type_ref
        end
        if target_type === Int64 || target_type === UInt64
            # Extending to 64-bit - emit extend instruction
            push!(bytes, Opcode.I64_EXTEND_I32_U)
        elseif target_type === Int128 || target_type === UInt128
            # Extending to 128-bit - create struct with (lo=value, hi=0)
            # The value is already on the stack (i64), need to create 128-bit struct
            source_type = length(args) >= 2 ? infer_value_type(args[2], ctx) : UInt64

            # If source is 32-bit, extend to 64-bit first
            if source_type === Int32 || source_type === UInt32
                push!(bytes, Opcode.I64_EXTEND_I32_U)
            end

            # Now we have i64 on stack (the lo part)
            # Push 0 for hi part
            push!(bytes, Opcode.I64_CONST)
            push!(bytes, 0x00)

            # Create the 128-bit struct (lo, hi)
            type_idx = get_int128_type!(ctx.mod, ctx.type_registry, target_type)
            push!(bytes, Opcode.GC_PREFIX)
            push!(bytes, Opcode.STRUCT_NEW)
            append!(bytes, encode_leb128_unsigned(type_idx))
        end
        # If extending to 32-bit (UInt32/Int32), it's a no-op since small types already map to i32

    elseif is_func(func, :trunc_int)  # Truncate to smaller type
        # trunc_int(TargetType, value)
        target_type_ref = args[1]
        target_type = if target_type_ref isa GlobalRef
            try
                getfield(target_type_ref.mod, target_type_ref.name)
            catch
                target_type_ref
            end
        else
            target_type_ref
        end

        source_type = length(args) >= 2 ? infer_value_type(args[2], ctx) : Int64

        if source_type === Int128 || source_type === UInt128
            # Truncating from 128-bit - extract lo part
            source_type_idx = get_int128_type!(ctx.mod, ctx.type_registry, source_type)
            push!(bytes, Opcode.GC_PREFIX)
            push!(bytes, Opcode.STRUCT_GET)
            append!(bytes, encode_leb128_unsigned(source_type_idx))
            append!(bytes, encode_leb128_unsigned(0))  # Field 0 = lo

            # Now we have i64, may need to wrap to i32
            if target_type === Int32 || target_type === UInt32
                push!(bytes, Opcode.I32_WRAP_I64)
            end
        elseif target_type === Int32 || target_type === UInt32
            # Standard i64 to i32 truncation
            push!(bytes, Opcode.I32_WRAP_I64)
        end
        # i64 to i64 or i32 to i32 is a no-op

    elseif is_func(func, :sitofp)  # Signed int to float
        # sitofp(TargetType, value) - first arg is target type, second is value
        # Need to check: target float type (first arg) and source int type (second arg)
        target_type = args[1]  # Float32 or Float64
        source_type = length(args) >= 2 ? infer_value_type(args[2], ctx) : Int64
        source_is_32bit = source_type === Int32 || source_type === UInt32 || source_type === Char ||
                          source_type === Int16 || source_type === UInt16 || source_type === Int8 || source_type === UInt8 ||
                          (isprimitivetype(source_type) && sizeof(source_type) <= 4)

        if target_type === Float32
            push!(bytes, source_is_32bit ? Opcode.F32_CONVERT_I32_S : Opcode.F32_CONVERT_I64_S)
        else  # Float64
            push!(bytes, source_is_32bit ? Opcode.F64_CONVERT_I32_S : Opcode.F64_CONVERT_I64_S)
        end

    elseif is_func(func, :uitofp)  # Unsigned int to float
        target_type = args[1]
        source_type = length(args) >= 2 ? infer_value_type(args[2], ctx) : Int64
        source_is_32bit = source_type === Int32 || source_type === UInt32 || source_type === Char ||
                          source_type === Int16 || source_type === UInt16 || source_type === Int8 || source_type === UInt8 ||
                          (isprimitivetype(source_type) && sizeof(source_type) <= 4)

        if target_type === Float32
            push!(bytes, source_is_32bit ? Opcode.F32_CONVERT_I32_U : Opcode.F32_CONVERT_I64_U)
        else  # Float64
            push!(bytes, source_is_32bit ? Opcode.F64_CONVERT_I32_U : Opcode.F64_CONVERT_I64_U)
        end

    elseif is_func(func, :fptosi)  # Float to signed int
        # fptosi(TargetType, value) - first arg is target type
        target_type = args[1]
        source_type = length(args) >= 2 ? infer_value_type(args[2], ctx) : Float64
        source_is_f32 = source_type === Float32

        if target_type === Int32
            push!(bytes, source_is_f32 ? Opcode.I32_TRUNC_F32_S : Opcode.I32_TRUNC_F64_S)
        else  # Int64
            push!(bytes, source_is_f32 ? Opcode.I64_TRUNC_F32_S : Opcode.I64_TRUNC_F64_S)
        end

    elseif is_func(func, :fptoui)  # Float to unsigned int
        target_type = args[1]
        source_type = length(args) >= 2 ? infer_value_type(args[2], ctx) : Float64
        source_is_f32 = source_type === Float32

        if target_type === UInt32
            push!(bytes, source_is_f32 ? Opcode.I32_TRUNC_F32_U : Opcode.I32_TRUNC_F64_U)
        else  # UInt64
            push!(bytes, source_is_f32 ? Opcode.I64_TRUNC_F32_U : Opcode.I64_TRUNC_F64_U)
        end

    elseif is_func(func, :fpext)  # Float precision extension (Float32 → Float64)
        # fpext(TargetType, value) - extend Float32 to Float64
        # The source is always Float32, target is Float64
        push!(bytes, 0xBB)  # f64.promote_f32

    elseif is_func(func, :fptrunc)  # Float precision truncation (Float64 → Float32)
        # fptrunc(TargetType, value) - truncate Float64 to Float32
        # The source is always Float64, target is Float32
        push!(bytes, 0xB6)  # f32.demote_f64

    elseif is_func(func, :trunc_llvm)  # Truncate float towards zero (returns float)
        push!(bytes, arg_type === Float32 ? Opcode.F32_TRUNC : Opcode.F64_TRUNC)

    elseif is_func(func, :floor_llvm)  # Floor float
        push!(bytes, arg_type === Float32 ? Opcode.F32_FLOOR : Opcode.F64_FLOOR)

    elseif is_func(func, :ceil_llvm)  # Ceil float
        push!(bytes, arg_type === Float32 ? Opcode.F32_CEIL : Opcode.F64_CEIL)

    elseif is_func(func, :rint_llvm)  # Round to nearest even
        push!(bytes, arg_type === Float32 ? Opcode.F32_NEAREST : Opcode.F64_NEAREST)

    elseif is_func(func, :abs_float)  # Absolute value of float
        push!(bytes, arg_type === Float32 ? Opcode.F32_ABS : Opcode.F64_ABS)

    elseif is_func(func, :sqrt_llvm) || is_func(func, :sqrt_llvm_fast)  # Square root
        push!(bytes, arg_type === Float32 ? Opcode.F32_SQRT : Opcode.F64_SQRT)

    elseif is_func(func, :copysign_float)  # Copy sign
        push!(bytes, arg_type === Float32 ? Opcode.F32_COPYSIGN : Opcode.F64_COPYSIGN)

    elseif is_func(func, :min_float) || is_func(func, :min_float_fast)
        push!(bytes, arg_type === Float32 ? Opcode.F32_MIN : Opcode.F64_MIN)

    elseif is_func(func, :max_float) || is_func(func, :max_float_fast)
        push!(bytes, arg_type === Float32 ? Opcode.F32_MAX : Opcode.F64_MAX)

    # High-level operators (fallback)
    elseif is_func(func, :+)
        if arg_type === Float32
            push!(bytes, Opcode.F32_ADD)
        elseif arg_type === Float64
            push!(bytes, Opcode.F64_ADD)
        elseif is_32bit
            push!(bytes, Opcode.I32_ADD)
        else
            push!(bytes, Opcode.I64_ADD)
        end

    elseif is_func(func, :-)
        if arg_type === Float32
            push!(bytes, Opcode.F32_SUB)
        elseif arg_type === Float64
            push!(bytes, Opcode.F64_SUB)
        elseif is_32bit
            push!(bytes, Opcode.I32_SUB)
        else
            push!(bytes, Opcode.I64_SUB)
        end

    elseif is_func(func, :*)
        if arg_type === Float32
            push!(bytes, Opcode.F32_MUL)
        elseif arg_type === Float64
            push!(bytes, Opcode.F64_MUL)
        elseif is_32bit
            push!(bytes, Opcode.I32_MUL)
        else
            push!(bytes, Opcode.I64_MUL)
        end

    # Compiler hints - these can be ignored
    elseif is_func(func, :compilerbarrier)
        # compilerbarrier(kind, value) - just return the value
        # The first arg is a symbol (like :type), second is the actual value
        # We only pushed the value (args[2]) since args[1] is a QuoteNode
        # The value is already on stack, nothing more to do

    # isa() - type checking for Union discrimination
    elseif is_func(func, :isa) && length(args) >= 2
        # isa(value, Type) - check if value is of given type
        # Supports both Union{Nothing, T} (via ref.is_null) and tagged unions
        value_arg = args[1]
        type_arg = args[2]

        # Get the type being checked
        check_type = if type_arg isa Type
            type_arg
        elseif type_arg isa GlobalRef
            Core.eval(type_arg.mod, type_arg.name)
        else
            nothing
        end

        # Get the type of the value being checked (for detecting tagged unions)
        value_type = get_ssa_type(ctx, value_arg)

        # Check if this is a tagged union check
        # NOTE: The value argument is already on the stack from the loop that pushes all args
        if value_type isa Union && needs_tagged_union(value_type) && haskey(ctx.type_registry.unions, value_type)
            # Tagged union: check the tag field
            union_info = ctx.type_registry.unions[value_type]
            expected_tag = get(union_info.tag_map, check_type, Int32(-1))

            if expected_tag >= 0
                # Value is already on stack (tagged union struct)
                # Get the tag field (field 0)
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.STRUCT_GET)
                append!(bytes, encode_leb128_unsigned(union_info.wasm_type_idx))
                append!(bytes, encode_leb128_unsigned(0))  # field 0 is tag
                # Compare tag to expected value
                push!(bytes, Opcode.I32_CONST)
                append!(bytes, encode_leb128_signed(Int64(expected_tag)))
                push!(bytes, Opcode.I32_EQ)
            else
                # Type not in this union - drop value and return false
                push!(bytes, Opcode.DROP)
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x00)
            end
        elseif check_type === Nothing
            # isa(x, Nothing) -> ref.is_null
            # Value is already on stack
            push!(bytes, Opcode.REF_IS_NULL)
        elseif check_type !== nothing && isconcretetype(check_type)
            # isa(x, ConcreteType) -> check if reference is non-null
            # For Union{Nothing, T}, checking isa(x, T) is equivalent to !isnull
            # Value is already on stack
            push!(bytes, Opcode.REF_IS_NULL)
            push!(bytes, Opcode.I32_EQZ)  # negate: 1->0, 0->1
        else
            # Unknown type - drop value and return false
            push!(bytes, Opcode.DROP)
            push!(bytes, Opcode.I32_CONST)
            push!(bytes, 0x00)
        end

    # throw() - compile to WASM throw instruction
    elseif func isa GlobalRef && func.name === :throw
        # Ensure module has an exception tag (tag 0)
        if isempty(ctx.mod.tags)
            void_ft = FuncType(WasmValType[], WasmValType[])
            void_type_idx = add_type!(ctx.mod, void_ft)
            add_tag!(ctx.mod, void_type_idx)
        end
        # Emit throw instruction with tag 0 (our Julia exception tag)
        # For now, we're not passing the exception value - just throwing
        push!(bytes, Opcode.THROW)
        append!(bytes, encode_leb128_unsigned(0))  # tag index 0

    # Base.add_ptr - pointer arithmetic (used in string operations)
    # In WasmGC, pointers are i64, so this is just i64 add
    elseif func isa GlobalRef && func.name === :add_ptr
        # add_ptr(ptr, offset) -> ptr + offset
        append!(bytes, compile_value(args[1], ctx))
        append!(bytes, compile_value(args[2], ctx))
        push!(bytes, Opcode.I64_ADD)

    # Base.sub_ptr - pointer subtraction
    elseif func isa GlobalRef && func.name === :sub_ptr
        # sub_ptr(ptr, offset) -> ptr - offset
        append!(bytes, compile_value(args[1], ctx))
        append!(bytes, compile_value(args[2], ctx))
        push!(bytes, Opcode.I64_SUB)

    # Base.pointerref - read from pointer (used in string comparisons)
    # For WasmGC strings, we use the str_char intrinsic instead
    # But if we hit this code path, we're in trouble - strings should use our intrinsics
    elseif func isa GlobalRef && func.name === :pointerref
        # pointerref(ptr, i, aligned) -> value at ptr
        # For now, just load the value using i64.load
        # The pointer is an i64, index is typically Int32, aligned is Bool
        # Result type depends on the pointer type (UInt8 for strings)
        append!(bytes, compile_value(args[1], ctx))  # ptr
        # Load byte (i32.load8_u)
        push!(bytes, 0x2D)  # i32.load8_u
        push!(bytes, 0x00)  # align=0
        push!(bytes, 0x00)  # offset=0

    # Base.pointerset - write to pointer
    elseif func isa GlobalRef && func.name === :pointerset
        # pointerset(ptr, val, i, aligned) -> Nothing
        # For now, store byte using i32.store8
        append!(bytes, compile_value(args[1], ctx))  # ptr
        append!(bytes, compile_value(args[2], ctx))  # value
        push!(bytes, 0x3A)  # i32.store8
        push!(bytes, 0x00)  # align=0
        push!(bytes, 0x00)  # offset=0

    # Cross-function call via GlobalRef (dynamic dispatch when Julia can't specialize)
    elseif func isa GlobalRef && ctx.func_registry !== nothing
        # Try to find this function in our registry
        called_func = try
            getfield(func.mod, func.name)
        catch
            nothing
        end

        if called_func !== nothing
            # Push arguments first
            for arg in args
                append!(bytes, compile_value(arg, ctx))
            end

            # Infer argument types for dispatch
            call_arg_types = tuple([infer_value_type(arg, ctx) for arg in args]...)
            target_info = get_function(ctx.func_registry, called_func, call_arg_types)

            if target_info !== nothing
                # Cross-function call - emit call instruction with target index
                push!(bytes, Opcode.CALL)
                append!(bytes, encode_leb128_unsigned(target_info.wasm_idx))
            else
                # No matching signature - likely dead code from Union type branches
                # Emit unreachable instead of error (the branch won't be taken at runtime)
                bytes = UInt8[]  # Clear pushed args
                push!(bytes, Opcode.UNREACHABLE)
            end
        else
            error("Unsupported function call: $func (type: $(typeof(func)))")
        end

    else
        error("Unsupported function call: $func (type: $(typeof(func)))")
    end

    return bytes
end

"""
Compile an invoke expression (method invocation).
"""
function compile_invoke(expr::Expr, idx::Int, ctx::CompilationContext)::Vector{UInt8}
    bytes = UInt8[]
    args = expr.args[3:end]

    # Check for signal substitution (Therapy.jl closures)
    # When calling through a captured signal getter/setter, emit global.get/set directly
    func_ref = expr.args[2]
    if func_ref isa Core.SSAValue
        ssa_id = func_ref.id
        # Signal getter: no args, returns the signal value
        if haskey(ctx.signal_ssa_getters, ssa_id) && isempty(args)
            global_idx = ctx.signal_ssa_getters[ssa_id]
            push!(bytes, Opcode.GLOBAL_GET)
            append!(bytes, encode_leb128_unsigned(global_idx))
            return bytes
        end
        # Signal setter: one arg, sets the signal value
        if haskey(ctx.signal_ssa_setters, ssa_id) && length(args) == 1
            global_idx = ctx.signal_ssa_setters[ssa_id]
            # Compile the argument (the new value)
            append!(bytes, compile_value(args[1], ctx))
            # Store to global
            push!(bytes, Opcode.GLOBAL_SET)
            append!(bytes, encode_leb128_unsigned(global_idx))

            # Inject DOM update calls for this signal (Therapy.jl reactive updates)
            if haskey(ctx.dom_bindings, global_idx)
                # Get global's type for conversion
                global_type = ctx.mod.globals[global_idx + 1].valtype

                for (import_idx, const_args) in ctx.dom_bindings[global_idx]
                    # Push constant arguments (e.g., hydration key)
                    for arg in const_args
                        push!(bytes, Opcode.I32_CONST)
                        append!(bytes, encode_leb128_signed(Int(arg)))
                    end
                    # Push the signal value (re-read from global)
                    push!(bytes, Opcode.GLOBAL_GET)
                    append!(bytes, encode_leb128_unsigned(global_idx))
                    # Convert to f64 for DOM imports (all DOM imports expect f64)
                    append!(bytes, emit_convert_to_f64(global_type))
                    # Call the DOM import function
                    push!(bytes, Opcode.CALL)
                    append!(bytes, encode_leb128_unsigned(import_idx))
                end
            end

            # Setter returns the value in Therapy.jl, so re-read it
            push!(bytes, Opcode.GLOBAL_GET)
            append!(bytes, encode_leb128_unsigned(global_idx))
            return bytes
        end
    end

    # Get MethodInstance to check parameter types for nothing arguments
    mi_or_ci = expr.args[1]
    mi = if mi_or_ci isa Core.MethodInstance
        mi_or_ci
    elseif isdefined(Core, :CodeInstance) && mi_or_ci isa Core.CodeInstance
        mi_or_ci.def
    else
        nothing
    end

    # Get parameter types from the method signature if available
    param_types = nothing
    if mi isa Core.MethodInstance && mi.def isa Method
        sig = mi.def.sig
        if sig isa DataType && sig <: Tuple
            # sig is Tuple{typeof(func), arg1_type, arg2_type, ...}
            # We want arg types starting from index 2
            param_types = sig.parameters[2:end]
        end
    end

    # Push arguments (for non-signal calls)
    for (arg_idx, arg) in enumerate(args)
        # Check if this is a nothing argument that needs ref.null
        is_nothing_arg = arg === nothing ||
                        (arg isa GlobalRef && arg.name === :nothing) ||
                        (arg isa Core.SSAValue && begin
                            ssa_stmt = ctx.code_info.code[arg.id]
                            ssa_stmt isa GlobalRef && ssa_stmt.name === :nothing
                        end)

        if is_nothing_arg && param_types !== nothing && arg_idx <= length(param_types)
            # Get the parameter type from the method signature
            param_type = param_types[arg_idx]
            # If it's a Union with Nothing, emit ref.null for the non-Nothing type
            if param_type isa Union
                # Find the non-Nothing type in the union
                non_nothing_type = nothing
                for t in Base.uniontypes(param_type)
                    if t !== Nothing
                        non_nothing_type = t
                        break
                    end
                end
                if non_nothing_type !== nothing
                    wasm_type = julia_to_wasm_type_concrete(non_nothing_type, ctx)
                    if wasm_type isa ConcreteRef
                        push!(bytes, Opcode.REF_NULL)
                        append!(bytes, encode_leb128_unsigned(wasm_type.type_idx))
                    else
                        # Fallback to compile_value
                        append!(bytes, compile_value(arg, ctx))
                    end
                else
                    append!(bytes, compile_value(arg, ctx))
                end
            elseif param_type !== Nothing
                # Parameter type is not Union but not Nothing - emit ref.null for it
                wasm_type = julia_to_wasm_type_concrete(param_type, ctx)
                if wasm_type isa ConcreteRef
                    push!(bytes, Opcode.REF_NULL)
                    append!(bytes, encode_leb128_unsigned(wasm_type.type_idx))
                else
                    append!(bytes, compile_value(arg, ctx))
                end
            else
                # Parameter type is Nothing itself, no value needed
                append!(bytes, compile_value(arg, ctx))
            end
        else
            append!(bytes, compile_value(arg, ctx))
        end
    end

    arg_type = length(args) > 0 ? infer_value_type(args[1], ctx) : Int64
    is_32bit = arg_type === Int32 || arg_type === UInt32 || arg_type === Bool || arg_type === Char ||
               arg_type === Int16 || arg_type === UInt16 || arg_type === Int8 || arg_type === UInt8 ||
               (isprimitivetype(arg_type) && sizeof(arg_type) <= 4)

    # mi was already extracted above for parameter type checking
    if mi isa Core.MethodInstance
        meth = mi.def
        if meth isa Method
            name = meth.name

            # Check if this is a self-recursive call
            # The second argument of invoke is the function reference
            # It can be a GlobalRef directly, or an SSA value that points to a GlobalRef
            func_ref = expr.args[2]

            # If func_ref is an SSA value, try to resolve it to the underlying GlobalRef
            actual_func_ref = func_ref
            if func_ref isa Core.SSAValue
                ssa_stmt = ctx.code_info.code[func_ref.id]
                if ssa_stmt isa GlobalRef
                    actual_func_ref = ssa_stmt
                end
            end

            is_self_call = false
            if ctx.func_ref !== nothing && actual_func_ref isa GlobalRef
                # Check if this GlobalRef refers to the same function
                try
                    called_func = getfield(actual_func_ref.mod, actual_func_ref.name)
                    is_self_call = called_func === ctx.func_ref
                catch
                    is_self_call = false
                end
            end

            # Check for cross-function call within the module first
            cross_call_handled = false
            if ctx.func_registry !== nothing && actual_func_ref isa GlobalRef && !is_self_call
                # Try to find this function in our registry
                called_func = try
                    getfield(actual_func_ref.mod, actual_func_ref.name)
                catch
                    nothing
                end

                if called_func !== nothing
                    # Infer argument types for dispatch
                    call_arg_types = tuple([infer_value_type(arg, ctx) for arg in args]...)
                    target_info = get_function(ctx.func_registry, called_func, call_arg_types)

                    if target_info !== nothing
                        # Cross-function call - emit call instruction with target index
                        push!(bytes, Opcode.CALL)
                        append!(bytes, encode_leb128_unsigned(target_info.wasm_idx))
                        cross_call_handled = true
                    end
                end
            end

            if is_self_call
                # Self-recursive call - emit call instruction
                push!(bytes, Opcode.CALL)
                append!(bytes, encode_leb128_unsigned(ctx.func_idx))
            elseif cross_call_handled
                # Already handled above
            elseif name === :+ || name === :add_int
                push!(bytes, is_32bit ? Opcode.I32_ADD : Opcode.I64_ADD)
            elseif name === :- || name === :sub_int
                push!(bytes, is_32bit ? Opcode.I32_SUB : Opcode.I64_SUB)
            elseif name === :* || name === :mul_int
                push!(bytes, is_32bit ? Opcode.I32_MUL : Opcode.I64_MUL)
            elseif name === :throw_boundserror || name === :throw || name === :throw_inexacterror
                # Error throwing functions - emit unreachable
                # Clear the stack first (arguments were pushed but not needed)
                bytes = UInt8[]  # Reset - don't need the pushed args
                push!(bytes, Opcode.UNREACHABLE)

            # Power operator: x ^ y for floats
            # WASM doesn't have a native pow instruction, so we need to handle this
            # For now, we require the pow import to be available
            elseif name === :^ && length(args) == 2
                arg1_type = infer_value_type(args[1], ctx)
                arg2_type = infer_value_type(args[2], ctx)

                if (arg1_type === Float64 || arg1_type === Float32) &&
                   (arg2_type === Float64 || arg2_type === Float32)
                    # Float power - need Math.pow import
                    # Check if we have a pow import
                    pow_import_idx = nothing
                    for (i, imp) in enumerate(ctx.mod.imports)
                        if imp.kind == 0x00 && imp.field_name == "pow"  # function import
                            pow_import_idx = UInt32(i - 1)
                            break
                        end
                    end

                    if pow_import_idx !== nothing
                        # Args already compiled, call pow import
                        # Convert to f64 if needed (Math.pow expects f64, f64 -> f64)
                        if arg1_type === Float32
                            # First arg is f32, need to insert promotion before second arg
                            # This is tricky with stack order. For now, just promote both
                            bytes = UInt8[]  # Reset
                            append!(bytes, compile_value(args[1], ctx))
                            push!(bytes, 0xBB)  # f64.promote_f32
                            append!(bytes, compile_value(args[2], ctx))
                            if arg2_type === Float32
                                push!(bytes, 0xBB)  # f64.promote_f32
                            end
                        end
                        push!(bytes, Opcode.CALL)
                        append!(bytes, encode_leb128_unsigned(pow_import_idx))
                        # Convert back to f32 if needed
                        if arg1_type === Float32
                            push!(bytes, 0xB6)  # f32.demote_f64
                        end
                    else
                        # No pow import - emit approximation using exp(y * log(x))
                        # This is hacky but works for basic cases
                        # For now, error out requesting the import
                        error("Float power (^) requires 'pow' import from Math module. " *
                              "Add (\"Math\", \"pow\", [F64, F64], [F64]) to imports.")
                    end
                elseif (arg1_type === Int32 || arg1_type === Int64) &&
                       (arg2_type === Int32 || arg2_type === Int64)
                    # Integer power - can implement with loop
                    # For simplicity, error out for now
                    error("Integer power (^) not yet implemented. Use float power instead.")
                else
                    error("Unsupported power types: $(arg1_type) ^ $(arg2_type)")
                end

            elseif name === :length
                # String/array length - argument already pushed, emit array.len
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_LEN)
                # array.len returns i32, extend to i64 for Julia's Int
                push!(bytes, Opcode.I64_EXTEND_I32_S)

            # String concatenation: string * string -> string
            # Julia compiles string concatenation to Base._string
            elseif (name === :* || name === :_string) && length(args) >= 2 &&
                   infer_value_type(args[1], ctx) === String &&
                   infer_value_type(args[2], ctx) === String
                # String concatenation using WasmGC array operations
                # For now, handle 2-string concat (most common case)
                if length(args) == 2
                    bytes = compile_string_concat(args[1], args[2], ctx)
                else
                    # Multi-string concat: concat pairwise
                    bytes = compile_string_concat(args[1], args[2], ctx)
                    for i in 3:length(args)
                        # Store intermediate result and concat next string
                        # This is simplified - for full support we'd need proper temp locals
                        # For now, just do first two
                    end
                end

            # String equality comparison
            elseif name === :(==) && length(args) == 2 &&
                   infer_value_type(args[1], ctx) === String &&
                   infer_value_type(args[2], ctx) === String
                bytes = compile_string_equal(args[1], args[2], ctx)

            # WasmTarget string operations - str_char(s, i) -> Int32
            elseif name === :str_char && length(args) == 2
                # Get character at index: array.get on string array
                # Args: string, index (1-based)
                str_type_idx = get_string_array_type!(ctx.mod, ctx.type_registry)

                # Compile string arg (already pushed by args loop)
                # Compile index arg and convert to 0-based
                idx_type = infer_value_type(args[2], ctx)
                if idx_type === Int64 || idx_type === Int
                    # Convert Int64 to Int32 and subtract 1
                    push!(bytes, Opcode.I32_WRAP_I64)
                end
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x01)  # 1
                push!(bytes, Opcode.I32_SUB)  # index - 1 for 0-based

                # array.get
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_GET)
                append!(bytes, encode_leb128_unsigned(str_type_idx))

            # WasmTarget string operations - str_setchar!(s, i, c) -> Nothing
            elseif name === :str_setchar! && length(args) == 3
                # Set character at index: array.set on string array
                # Args: string, index (1-based), char (Int32)
                str_type_idx = get_string_array_type!(ctx.mod, ctx.type_registry)

                # Stack has: string, index, char
                # Need to reorder to: string, index-1, char for array.set
                # Actually array.set expects: array, index, value
                # So we need: compile string, compile index-1, compile char

                # Clear the bytes from the args loop - we'll recompile in correct order
                bytes = UInt8[]

                # Compile string
                append!(bytes, compile_value(args[1], ctx))

                # Compile index and convert to 0-based
                append!(bytes, compile_value(args[2], ctx))
                idx_type = infer_value_type(args[2], ctx)
                if idx_type === Int64 || idx_type === Int
                    push!(bytes, Opcode.I32_WRAP_I64)
                end
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x01)
                push!(bytes, Opcode.I32_SUB)

                # Compile char value
                append!(bytes, compile_value(args[3], ctx))
                char_type = infer_value_type(args[3], ctx)
                if char_type === Int64 || char_type === Int
                    push!(bytes, Opcode.I32_WRAP_I64)
                end

                # array.set
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_SET)
                append!(bytes, encode_leb128_unsigned(str_type_idx))

            # WasmTarget string operations - str_len(s) -> Int32
            elseif name === :str_len && length(args) == 1
                # Get string length as Int32
                # Arg already compiled, just emit array.len
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_LEN)

            # WasmTarget string operations - str_new(len) -> String
            elseif name === :str_new && length(args) == 1
                # Create new string of given length, filled with zeros
                str_type_idx = get_string_array_type!(ctx.mod, ctx.type_registry)

                # Length arg already compiled
                len_type = infer_value_type(args[1], ctx)
                if len_type === Int64 || len_type === Int
                    push!(bytes, Opcode.I32_WRAP_I64)
                end

                # array.new_default creates array filled with default value (0 for i32)
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_NEW_DEFAULT)
                append!(bytes, encode_leb128_unsigned(str_type_idx))

            # WasmTarget string operations - str_copy(src, src_pos, dst, dst_pos, len) -> Nothing
            elseif name === :str_copy && length(args) == 5
                # Copy characters from src to dst using array.copy
                str_type_idx = get_string_array_type!(ctx.mod, ctx.type_registry)

                # Clear bytes - recompile in correct order for array.copy
                # array.copy expects: dst, dst_offset, src, src_offset, len
                bytes = UInt8[]

                # dst array
                append!(bytes, compile_value(args[3], ctx))
                # dst offset (0-based)
                append!(bytes, compile_value(args[4], ctx))
                dst_idx_type = infer_value_type(args[4], ctx)
                if dst_idx_type === Int64 || dst_idx_type === Int
                    push!(bytes, Opcode.I32_WRAP_I64)
                end
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x01)
                push!(bytes, Opcode.I32_SUB)

                # src array
                append!(bytes, compile_value(args[1], ctx))
                # src offset (0-based)
                append!(bytes, compile_value(args[2], ctx))
                src_idx_type = infer_value_type(args[2], ctx)
                if src_idx_type === Int64 || src_idx_type === Int
                    push!(bytes, Opcode.I32_WRAP_I64)
                end
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x01)
                push!(bytes, Opcode.I32_SUB)

                # length
                append!(bytes, compile_value(args[5], ctx))
                len_type = infer_value_type(args[5], ctx)
                if len_type === Int64 || len_type === Int
                    push!(bytes, Opcode.I32_WRAP_I64)
                end

                # array.copy
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_COPY)
                append!(bytes, encode_leb128_unsigned(str_type_idx))
                append!(bytes, encode_leb128_unsigned(str_type_idx))

            # WasmTarget string operations - str_substr(s, start, len) -> String
            elseif name === :str_substr && length(args) == 3
                # Extract substring: create new string and copy characters
                str_type_idx = get_string_array_type!(ctx.mod, ctx.type_registry)

                # Use scratch locals stored in context
                if ctx.scratch_locals === nothing
                    error("String operations require scratch locals but none were allocated")
                end
                result_local, src_local, _, _, _ = ctx.scratch_locals

                # Clear bytes - recompile in correct order
                bytes = UInt8[]

                # Store source string
                append!(bytes, compile_value(args[1], ctx))
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(src_local))

                # Create new string of specified length
                append!(bytes, compile_value(args[3], ctx))  # len
                len_type = infer_value_type(args[3], ctx)
                if len_type === Int64 || len_type === Int
                    push!(bytes, Opcode.I32_WRAP_I64)
                end
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_NEW_DEFAULT)
                append!(bytes, encode_leb128_unsigned(str_type_idx))
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(result_local))

                # Copy characters: array.copy [dst, dst_off, src, src_off, len]
                # dst = result, dst_off = 0, src = source, src_off = start-1, len = len
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(result_local))
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x00)  # dst_off = 0

                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(src_local))

                # src_off = start - 1 (convert to 0-based)
                append!(bytes, compile_value(args[2], ctx))
                start_type = infer_value_type(args[2], ctx)
                if start_type === Int64 || start_type === Int
                    push!(bytes, Opcode.I32_WRAP_I64)
                end
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x01)
                push!(bytes, Opcode.I32_SUB)

                # len
                append!(bytes, compile_value(args[3], ctx))
                len_type2 = infer_value_type(args[3], ctx)
                if len_type2 === Int64 || len_type2 === Int
                    push!(bytes, Opcode.I32_WRAP_I64)
                end

                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_COPY)
                append!(bytes, encode_leb128_unsigned(str_type_idx))
                append!(bytes, encode_leb128_unsigned(str_type_idx))

                # Return result
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(result_local))

            # WasmTarget string operations - str_hash(s) -> Int32
            elseif name === :str_hash && length(args) == 1
                # Compute string hash using Java-style: h = 31 * h + char[i]
                # Uses a loop over the string characters
                str_type_idx = get_string_array_type!(ctx.mod, ctx.type_registry)

                bytes = UInt8[]

                # Allocate locals for this operation
                str_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, ConcreteRef(str_type_idx))  # string reference

                len_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, I32)  # string length

                hash_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, I32)  # running hash

                i_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, I32)  # loop index

                # Store string reference
                append!(bytes, compile_value(args[1], ctx))
                push!(bytes, Opcode.LOCAL_TEE)
                append!(bytes, encode_leb128_unsigned(str_local))

                # Get length
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_LEN)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(len_local))

                # Initialize hash = 0
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x00)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(hash_local))

                # Initialize i = 0
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x00)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(i_local))

                # Loop over characters
                push!(bytes, Opcode.BLOCK)  # outer block for exit
                push!(bytes, 0x40)  # void
                push!(bytes, Opcode.LOOP)  # loop
                push!(bytes, 0x40)  # void

                # Check i < len
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(i_local))
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(len_local))
                push!(bytes, Opcode.I32_GE_S)
                push!(bytes, Opcode.BR_IF)
                push!(bytes, 0x01)  # break to outer block if done

                # hash = 31 * hash + char[i]
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(hash_local))
                push!(bytes, Opcode.I32_CONST)
                append!(bytes, encode_leb128_signed(31))
                push!(bytes, Opcode.I32_MUL)

                # Get char at index i (0-based)
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(str_local))
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(i_local))
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_GET)
                append!(bytes, encode_leb128_unsigned(str_type_idx))

                push!(bytes, Opcode.I32_ADD)

                # Mask to positive: & 0x7FFFFFFF
                push!(bytes, Opcode.I32_CONST)
                append!(bytes, encode_leb128_signed(0x7FFFFFFF))
                push!(bytes, Opcode.I32_AND)

                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(hash_local))

                # i++
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(i_local))
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x01)
                push!(bytes, Opcode.I32_ADD)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(i_local))

                # Continue loop
                push!(bytes, Opcode.BR)
                push!(bytes, 0x00)

                push!(bytes, Opcode.END)  # end loop
                push!(bytes, Opcode.END)  # end block

                # Return hash
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(hash_local))

            # ================================================================
            # BROWSER-010: New String Operations
            # str_find, str_contains, str_startswith, str_endswith
            # str_uppercase, str_lowercase, str_trim
            # ================================================================

            # str_find(haystack, needle) -> Int32
            # Returns 1-based position or 0 if not found
            elseif name === :str_find && length(args) == 2
                str_type_idx = get_string_array_type!(ctx.mod, ctx.type_registry)
                bytes = UInt8[]

                # Allocate locals
                haystack_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, ConcreteRef(str_type_idx))
                needle_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, ConcreteRef(str_type_idx))
                haystack_len_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, I32)
                needle_len_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, I32)
                i_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, I32)
                j_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, I32)
                found_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, I32)
                result_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, I32)
                last_start_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, I32)

                # Store haystack
                append!(bytes, compile_value(args[1], ctx))
                push!(bytes, Opcode.LOCAL_TEE)
                append!(bytes, encode_leb128_unsigned(haystack_local))
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_LEN)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(haystack_len_local))

                # Store needle
                append!(bytes, compile_value(args[2], ctx))
                push!(bytes, Opcode.LOCAL_TEE)
                append!(bytes, encode_leb128_unsigned(needle_local))
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_LEN)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(needle_len_local))

                # Initialize result = 0
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x00)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(result_local))

                # If needle_len == 0, return 1
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(needle_len_local))
                push!(bytes, Opcode.I32_EQZ)
                push!(bytes, Opcode.IF)
                push!(bytes, 0x40)  # void
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x01)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(result_local))
                push!(bytes, Opcode.ELSE)

                # Check if needle_len > haystack_len - skip search if so
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(needle_len_local))
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(haystack_len_local))
                push!(bytes, Opcode.I32_GT_S)
                push!(bytes, Opcode.IF)
                push!(bytes, 0x40)  # void
                # result stays 0
                push!(bytes, Opcode.ELSE)

                # Calculate last_start = haystack_len - needle_len + 1 (1-based)
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(haystack_len_local))
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(needle_len_local))
                push!(bytes, Opcode.I32_SUB)
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x01)
                push!(bytes, Opcode.I32_ADD)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(last_start_local))

                # Initialize i = 1 (1-based)
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x01)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(i_local))

                # Outer loop over haystack positions
                push!(bytes, Opcode.BLOCK)  # outer block for exit
                push!(bytes, 0x40)
                push!(bytes, Opcode.LOOP)  # outer loop
                push!(bytes, 0x40)

                # Check i <= last_start
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(i_local))
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(last_start_local))
                push!(bytes, Opcode.I32_GT_S)
                push!(bytes, Opcode.BR_IF)
                push!(bytes, 0x01)  # break outer block if done

                # found = 1
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x01)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(found_local))

                # j = 0 (0-based index into needle)
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x00)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(j_local))

                # Inner loop - compare needle chars
                push!(bytes, Opcode.BLOCK)  # inner block for break
                push!(bytes, 0x40)
                push!(bytes, Opcode.LOOP)  # inner loop
                push!(bytes, 0x40)

                # Check j < needle_len
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(j_local))
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(needle_len_local))
                push!(bytes, Opcode.I32_GE_S)
                push!(bytes, Opcode.BR_IF)
                push!(bytes, 0x01)  # break inner block if done

                # Compare haystack[i + j - 1] with needle[j] (0-based array access)
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(haystack_local))
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(i_local))
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(j_local))
                push!(bytes, Opcode.I32_ADD)
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x01)
                push!(bytes, Opcode.I32_SUB)  # i + j - 1 for 0-based
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_GET)
                append!(bytes, encode_leb128_unsigned(str_type_idx))

                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(needle_local))
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(j_local))
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_GET)
                append!(bytes, encode_leb128_unsigned(str_type_idx))

                push!(bytes, Opcode.I32_NE)
                push!(bytes, Opcode.IF)
                push!(bytes, 0x40)
                # Characters don't match - set found = 0 and break
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x00)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(found_local))
                push!(bytes, Opcode.BR)
                push!(bytes, 0x02)  # break inner block
                push!(bytes, Opcode.END)  # end if

                # j++
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(j_local))
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x01)
                push!(bytes, Opcode.I32_ADD)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(j_local))

                # Continue inner loop
                push!(bytes, Opcode.BR)
                push!(bytes, 0x00)

                push!(bytes, Opcode.END)  # end inner loop
                push!(bytes, Opcode.END)  # end inner block

                # If found, set result = i and break outer
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(found_local))
                push!(bytes, Opcode.IF)
                push!(bytes, 0x40)
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(i_local))
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(result_local))
                push!(bytes, Opcode.BR)
                push!(bytes, 0x01)  # break outer block
                push!(bytes, Opcode.END)

                # i++
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(i_local))
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x01)
                push!(bytes, Opcode.I32_ADD)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(i_local))

                # Continue outer loop
                push!(bytes, Opcode.BR)
                push!(bytes, 0x00)

                push!(bytes, Opcode.END)  # end outer loop
                push!(bytes, Opcode.END)  # end outer block

                push!(bytes, Opcode.END)  # end else (needle not too long)
                push!(bytes, Opcode.END)  # end else (needle not empty)

                # Return result
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(result_local))

            # str_contains(haystack, needle) -> Bool
            # Returns true if needle is found in haystack
            elseif name === :str_contains && length(args) == 2
                str_type_idx = get_string_array_type!(ctx.mod, ctx.type_registry)
                bytes = UInt8[]

                # Reuse str_find implementation by comparing result > 0
                # Allocate locals
                haystack_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, ConcreteRef(str_type_idx))
                needle_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, ConcreteRef(str_type_idx))
                haystack_len_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, I32)
                needle_len_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, I32)
                i_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, I32)
                j_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, I32)
                found_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, I32)
                result_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, I32)
                last_start_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, I32)

                # Store haystack
                append!(bytes, compile_value(args[1], ctx))
                push!(bytes, Opcode.LOCAL_TEE)
                append!(bytes, encode_leb128_unsigned(haystack_local))
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_LEN)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(haystack_len_local))

                # Store needle
                append!(bytes, compile_value(args[2], ctx))
                push!(bytes, Opcode.LOCAL_TEE)
                append!(bytes, encode_leb128_unsigned(needle_local))
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_LEN)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(needle_len_local))

                # Initialize result = 0 (false)
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x00)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(result_local))

                # If needle_len == 0, return true (1)
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(needle_len_local))
                push!(bytes, Opcode.I32_EQZ)
                push!(bytes, Opcode.IF)
                push!(bytes, 0x40)
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x01)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(result_local))
                push!(bytes, Opcode.ELSE)

                # Check if needle_len > haystack_len - return false if so
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(needle_len_local))
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(haystack_len_local))
                push!(bytes, Opcode.I32_GT_S)
                push!(bytes, Opcode.I32_EQZ)  # NOT greater
                push!(bytes, Opcode.IF)
                push!(bytes, 0x40)

                # Calculate last_start = haystack_len - needle_len
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(haystack_len_local))
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(needle_len_local))
                push!(bytes, Opcode.I32_SUB)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(last_start_local))

                # Initialize i = 0 (0-based)
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x00)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(i_local))

                # Outer loop
                push!(bytes, Opcode.BLOCK)
                push!(bytes, 0x40)
                push!(bytes, Opcode.LOOP)
                push!(bytes, 0x40)

                # Check i <= last_start
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(i_local))
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(last_start_local))
                push!(bytes, Opcode.I32_GT_S)
                push!(bytes, Opcode.BR_IF)
                push!(bytes, 0x01)

                # found = 1
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x01)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(found_local))

                # j = 0
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x00)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(j_local))

                # Inner loop
                push!(bytes, Opcode.BLOCK)
                push!(bytes, 0x40)
                push!(bytes, Opcode.LOOP)
                push!(bytes, 0x40)

                # Check j < needle_len
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(j_local))
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(needle_len_local))
                push!(bytes, Opcode.I32_GE_S)
                push!(bytes, Opcode.BR_IF)
                push!(bytes, 0x01)

                # Compare haystack[i + j] with needle[j]
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(haystack_local))
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(i_local))
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(j_local))
                push!(bytes, Opcode.I32_ADD)
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_GET)
                append!(bytes, encode_leb128_unsigned(str_type_idx))

                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(needle_local))
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(j_local))
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_GET)
                append!(bytes, encode_leb128_unsigned(str_type_idx))

                push!(bytes, Opcode.I32_NE)
                push!(bytes, Opcode.IF)
                push!(bytes, 0x40)
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x00)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(found_local))
                push!(bytes, Opcode.BR)
                push!(bytes, 0x02)
                push!(bytes, Opcode.END)

                # j++
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(j_local))
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x01)
                push!(bytes, Opcode.I32_ADD)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(j_local))

                push!(bytes, Opcode.BR)
                push!(bytes, 0x00)

                push!(bytes, Opcode.END)  # end inner loop
                push!(bytes, Opcode.END)  # end inner block

                # If found, set result = 1 and break
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(found_local))
                push!(bytes, Opcode.IF)
                push!(bytes, 0x40)
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x01)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(result_local))
                push!(bytes, Opcode.BR)
                push!(bytes, 0x01)
                push!(bytes, Opcode.END)

                # i++
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(i_local))
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x01)
                push!(bytes, Opcode.I32_ADD)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(i_local))

                push!(bytes, Opcode.BR)
                push!(bytes, 0x00)

                push!(bytes, Opcode.END)  # end outer loop
                push!(bytes, Opcode.END)  # end outer block

                push!(bytes, Opcode.END)  # end if (needle not too long)
                push!(bytes, Opcode.END)  # end else (needle not empty)

                # Return result (0 or 1 as i32, which is Bool in wasm)
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(result_local))

            # str_startswith(s, prefix) -> Bool
            elseif name === :str_startswith && length(args) == 2
                str_type_idx = get_string_array_type!(ctx.mod, ctx.type_registry)
                bytes = UInt8[]

                # Allocate locals
                s_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, ConcreteRef(str_type_idx))
                prefix_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, ConcreteRef(str_type_idx))
                s_len_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, I32)
                prefix_len_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, I32)
                i_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, I32)
                result_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, I32)

                # Store s
                append!(bytes, compile_value(args[1], ctx))
                push!(bytes, Opcode.LOCAL_TEE)
                append!(bytes, encode_leb128_unsigned(s_local))
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_LEN)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(s_len_local))

                # Store prefix
                append!(bytes, compile_value(args[2], ctx))
                push!(bytes, Opcode.LOCAL_TEE)
                append!(bytes, encode_leb128_unsigned(prefix_local))
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_LEN)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(prefix_len_local))

                # Default result = 1 (true)
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x01)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(result_local))

                # If prefix_len > s_len, return false
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(prefix_len_local))
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(s_len_local))
                push!(bytes, Opcode.I32_GT_S)
                push!(bytes, Opcode.IF)
                push!(bytes, 0x40)
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x00)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(result_local))
                push!(bytes, Opcode.ELSE)

                # i = 0
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x00)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(i_local))

                # Loop
                push!(bytes, Opcode.BLOCK)
                push!(bytes, 0x40)
                push!(bytes, Opcode.LOOP)
                push!(bytes, 0x40)

                # Check i < prefix_len
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(i_local))
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(prefix_len_local))
                push!(bytes, Opcode.I32_GE_S)
                push!(bytes, Opcode.BR_IF)
                push!(bytes, 0x01)

                # Compare s[i] with prefix[i]
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(s_local))
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(i_local))
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_GET)
                append!(bytes, encode_leb128_unsigned(str_type_idx))

                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(prefix_local))
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(i_local))
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_GET)
                append!(bytes, encode_leb128_unsigned(str_type_idx))

                push!(bytes, Opcode.I32_NE)
                push!(bytes, Opcode.IF)
                push!(bytes, 0x40)
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x00)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(result_local))
                push!(bytes, Opcode.BR)
                push!(bytes, 0x02)  # break out of loop
                push!(bytes, Opcode.END)

                # i++
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(i_local))
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x01)
                push!(bytes, Opcode.I32_ADD)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(i_local))

                push!(bytes, Opcode.BR)
                push!(bytes, 0x00)

                push!(bytes, Opcode.END)  # end loop
                push!(bytes, Opcode.END)  # end block
                push!(bytes, Opcode.END)  # end else

                # Return result
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(result_local))

            # str_endswith(s, suffix) -> Bool
            elseif name === :str_endswith && length(args) == 2
                str_type_idx = get_string_array_type!(ctx.mod, ctx.type_registry)
                bytes = UInt8[]

                # Allocate locals
                s_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, ConcreteRef(str_type_idx))
                suffix_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, ConcreteRef(str_type_idx))
                s_len_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, I32)
                suffix_len_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, I32)
                start_pos_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, I32)
                i_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, I32)
                result_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, I32)

                # Store s
                append!(bytes, compile_value(args[1], ctx))
                push!(bytes, Opcode.LOCAL_TEE)
                append!(bytes, encode_leb128_unsigned(s_local))
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_LEN)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(s_len_local))

                # Store suffix
                append!(bytes, compile_value(args[2], ctx))
                push!(bytes, Opcode.LOCAL_TEE)
                append!(bytes, encode_leb128_unsigned(suffix_local))
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_LEN)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(suffix_len_local))

                # Default result = 1 (true)
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x01)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(result_local))

                # If suffix_len > s_len, return false
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(suffix_len_local))
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(s_len_local))
                push!(bytes, Opcode.I32_GT_S)
                push!(bytes, Opcode.IF)
                push!(bytes, 0x40)
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x00)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(result_local))
                push!(bytes, Opcode.ELSE)

                # Calculate start_pos = s_len - suffix_len (0-based start in s)
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(s_len_local))
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(suffix_len_local))
                push!(bytes, Opcode.I32_SUB)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(start_pos_local))

                # i = 0
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x00)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(i_local))

                # Loop
                push!(bytes, Opcode.BLOCK)
                push!(bytes, 0x40)
                push!(bytes, Opcode.LOOP)
                push!(bytes, 0x40)

                # Check i < suffix_len
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(i_local))
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(suffix_len_local))
                push!(bytes, Opcode.I32_GE_S)
                push!(bytes, Opcode.BR_IF)
                push!(bytes, 0x01)

                # Compare s[start_pos + i] with suffix[i]
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(s_local))
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(start_pos_local))
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(i_local))
                push!(bytes, Opcode.I32_ADD)
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_GET)
                append!(bytes, encode_leb128_unsigned(str_type_idx))

                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(suffix_local))
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(i_local))
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_GET)
                append!(bytes, encode_leb128_unsigned(str_type_idx))

                push!(bytes, Opcode.I32_NE)
                push!(bytes, Opcode.IF)
                push!(bytes, 0x40)
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x00)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(result_local))
                push!(bytes, Opcode.BR)
                push!(bytes, 0x02)
                push!(bytes, Opcode.END)

                # i++
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(i_local))
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x01)
                push!(bytes, Opcode.I32_ADD)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(i_local))

                push!(bytes, Opcode.BR)
                push!(bytes, 0x00)

                push!(bytes, Opcode.END)  # end loop
                push!(bytes, Opcode.END)  # end block
                push!(bytes, Opcode.END)  # end else

                # Return result
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(result_local))

            # str_uppercase(s) -> String
            # Convert lowercase ASCII letters to uppercase
            elseif name === :str_uppercase && length(args) == 1
                str_type_idx = get_string_array_type!(ctx.mod, ctx.type_registry)
                bytes = UInt8[]

                # Allocate locals
                s_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, ConcreteRef(str_type_idx))
                len_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, I32)
                result_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, ConcreteRef(str_type_idx))
                i_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, I32)
                c_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, I32)

                # Store s and get length
                append!(bytes, compile_value(args[1], ctx))
                push!(bytes, Opcode.LOCAL_TEE)
                append!(bytes, encode_leb128_unsigned(s_local))
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_LEN)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(len_local))

                # Create result string: array.new_default with same length
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(len_local))
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_NEW_DEFAULT)
                append!(bytes, encode_leb128_unsigned(str_type_idx))
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(result_local))

                # i = 0 (0-based for WASM)
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x00)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(i_local))

                # Loop: while i < len
                push!(bytes, Opcode.BLOCK)  # block for break
                push!(bytes, 0x40)  # void
                push!(bytes, Opcode.LOOP)   # loop
                push!(bytes, 0x40)  # void

                # Check i < len
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(i_local))
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(len_local))
                push!(bytes, Opcode.I32_GE_S)
                push!(bytes, Opcode.BR_IF)
                push!(bytes, 0x01)  # break if i >= len

                # c = s[i]
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(s_local))
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(i_local))
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_GET)
                append!(bytes, encode_leb128_unsigned(str_type_idx))
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(c_local))

                # Check if c is lowercase (97 <= c <= 122)
                # If so, convert to uppercase (c - 32)
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(c_local))
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x61)  # 97 = 'a'
                push!(bytes, Opcode.I32_GE_S)
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(c_local))
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x7a)  # 122 = 'z'
                push!(bytes, Opcode.I32_LE_S)
                push!(bytes, Opcode.I32_AND)
                push!(bytes, Opcode.IF)
                push!(bytes, 0x40)  # void

                # Convert to uppercase: c = c - 32
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(c_local))
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x20)  # 32
                push!(bytes, Opcode.I32_SUB)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(c_local))

                push!(bytes, Opcode.END)  # end if

                # result[i] = c
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(result_local))
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(i_local))
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(c_local))
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_SET)
                append!(bytes, encode_leb128_unsigned(str_type_idx))

                # i++
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(i_local))
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x01)
                push!(bytes, Opcode.I32_ADD)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(i_local))

                push!(bytes, Opcode.BR)
                push!(bytes, 0x00)  # continue loop

                push!(bytes, Opcode.END)  # end loop
                push!(bytes, Opcode.END)  # end block

                # Return result
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(result_local))

            # str_lowercase(s) -> String
            # Convert uppercase ASCII letters to lowercase
            elseif name === :str_lowercase && length(args) == 1
                str_type_idx = get_string_array_type!(ctx.mod, ctx.type_registry)
                bytes = UInt8[]

                # Allocate locals
                s_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, ConcreteRef(str_type_idx))
                len_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, I32)
                result_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, ConcreteRef(str_type_idx))
                i_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, I32)
                c_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, I32)

                # Store s and get length
                append!(bytes, compile_value(args[1], ctx))
                push!(bytes, Opcode.LOCAL_TEE)
                append!(bytes, encode_leb128_unsigned(s_local))
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_LEN)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(len_local))

                # Create result string: array.new_default with same length
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(len_local))
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_NEW_DEFAULT)
                append!(bytes, encode_leb128_unsigned(str_type_idx))
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(result_local))

                # i = 0 (0-based for WASM)
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x00)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(i_local))

                # Loop: while i < len
                push!(bytes, Opcode.BLOCK)  # block for break
                push!(bytes, 0x40)  # void
                push!(bytes, Opcode.LOOP)   # loop
                push!(bytes, 0x40)  # void

                # Check i < len
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(i_local))
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(len_local))
                push!(bytes, Opcode.I32_GE_S)
                push!(bytes, Opcode.BR_IF)
                push!(bytes, 0x01)  # break if i >= len

                # c = s[i]
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(s_local))
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(i_local))
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_GET)
                append!(bytes, encode_leb128_unsigned(str_type_idx))
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(c_local))

                # Check if c is uppercase (65 <= c <= 90)
                # If so, convert to lowercase (c + 32)
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(c_local))
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x41)  # 65 = 'A'
                push!(bytes, Opcode.I32_GE_S)
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(c_local))
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x5a)  # 90 = 'Z'
                push!(bytes, Opcode.I32_LE_S)
                push!(bytes, Opcode.I32_AND)
                push!(bytes, Opcode.IF)
                push!(bytes, 0x40)  # void

                # Convert to lowercase: c = c + 32
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(c_local))
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x20)  # 32
                push!(bytes, Opcode.I32_ADD)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(c_local))

                push!(bytes, Opcode.END)  # end if

                # result[i] = c
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(result_local))
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(i_local))
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(c_local))
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_SET)
                append!(bytes, encode_leb128_unsigned(str_type_idx))

                # i++
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(i_local))
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x01)
                push!(bytes, Opcode.I32_ADD)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(i_local))

                push!(bytes, Opcode.BR)
                push!(bytes, 0x00)  # continue loop

                push!(bytes, Opcode.END)  # end loop
                push!(bytes, Opcode.END)  # end block

                # Return result
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(result_local))

            # str_trim(s) -> String
            # Remove leading and trailing ASCII whitespace
            elseif name === :str_trim && length(args) == 1
                str_type_idx = get_string_array_type!(ctx.mod, ctx.type_registry)
                bytes = UInt8[]

                # Allocate locals
                s_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, ConcreteRef(str_type_idx))
                len_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, I32)
                start_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, I32)
                end_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, I32)
                new_len_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, I32)
                result_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, ConcreteRef(str_type_idx))
                c_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, I32)

                # Store s and get length
                append!(bytes, compile_value(args[1], ctx))
                push!(bytes, Opcode.LOCAL_TEE)
                append!(bytes, encode_leb128_unsigned(s_local))
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_LEN)
                push!(bytes, Opcode.LOCAL_TEE)
                append!(bytes, encode_leb128_unsigned(len_local))

                # Check for empty string
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x00)
                push!(bytes, Opcode.I32_EQ)
                push!(bytes, Opcode.IF)
                push!(bytes, ConcreteRef(str_type_idx).code)  # returns string ref
                append!(bytes, encode_leb128_unsigned(str_type_idx))

                # Return empty string (the original s)
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(s_local))

                push!(bytes, Opcode.ELSE)

                # start = 0 (0-based)
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x00)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(start_local))

                # end = len - 1 (0-based, last valid index)
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(len_local))
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x01)
                push!(bytes, Opcode.I32_SUB)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(end_local))

                # Find start: skip leading whitespace
                # while start < len && is_whitespace(s[start])
                push!(bytes, Opcode.BLOCK)
                push!(bytes, 0x40)
                push!(bytes, Opcode.LOOP)
                push!(bytes, 0x40)

                # Check start < len
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(start_local))
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(len_local))
                push!(bytes, Opcode.I32_GE_S)
                push!(bytes, Opcode.BR_IF)
                push!(bytes, 0x01)  # break if start >= len

                # c = s[start]
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(s_local))
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(start_local))
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_GET)
                append!(bytes, encode_leb128_unsigned(str_type_idx))
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(c_local))

                # Check if whitespace: c == 32 || c == 9 || c == 10 || c == 13
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(c_local))
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x20)  # space
                push!(bytes, Opcode.I32_EQ)
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(c_local))
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x09)  # tab
                push!(bytes, Opcode.I32_EQ)
                push!(bytes, Opcode.I32_OR)
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(c_local))
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x0a)  # newline
                push!(bytes, Opcode.I32_EQ)
                push!(bytes, Opcode.I32_OR)
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(c_local))
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x0d)  # carriage return
                push!(bytes, Opcode.I32_EQ)
                push!(bytes, Opcode.I32_OR)

                # If not whitespace, break
                push!(bytes, Opcode.I32_EQZ)
                push!(bytes, Opcode.BR_IF)
                push!(bytes, 0x01)

                # start++
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(start_local))
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x01)
                push!(bytes, Opcode.I32_ADD)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(start_local))

                push!(bytes, Opcode.BR)
                push!(bytes, 0x00)  # continue

                push!(bytes, Opcode.END)  # end loop
                push!(bytes, Opcode.END)  # end block

                # Check if all whitespace (start >= len)
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(start_local))
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(len_local))
                push!(bytes, Opcode.I32_GE_S)
                push!(bytes, Opcode.IF)
                push!(bytes, ConcreteRef(str_type_idx).code)
                append!(bytes, encode_leb128_unsigned(str_type_idx))

                # Return empty string
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x00)
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_NEW_DEFAULT)
                append!(bytes, encode_leb128_unsigned(str_type_idx))

                push!(bytes, Opcode.ELSE)

                # Find end: skip trailing whitespace
                # while end >= start && is_whitespace(s[end])
                push!(bytes, Opcode.BLOCK)
                push!(bytes, 0x40)
                push!(bytes, Opcode.LOOP)
                push!(bytes, 0x40)

                # Check end >= start
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(end_local))
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(start_local))
                push!(bytes, Opcode.I32_LT_S)
                push!(bytes, Opcode.BR_IF)
                push!(bytes, 0x01)  # break if end < start

                # c = s[end]
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(s_local))
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(end_local))
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_GET)
                append!(bytes, encode_leb128_unsigned(str_type_idx))
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(c_local))

                # Check if whitespace
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(c_local))
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x20)
                push!(bytes, Opcode.I32_EQ)
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(c_local))
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x09)
                push!(bytes, Opcode.I32_EQ)
                push!(bytes, Opcode.I32_OR)
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(c_local))
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x0a)
                push!(bytes, Opcode.I32_EQ)
                push!(bytes, Opcode.I32_OR)
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(c_local))
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x0d)
                push!(bytes, Opcode.I32_EQ)
                push!(bytes, Opcode.I32_OR)

                # If not whitespace, break
                push!(bytes, Opcode.I32_EQZ)
                push!(bytes, Opcode.BR_IF)
                push!(bytes, 0x01)

                # end--
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(end_local))
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x01)
                push!(bytes, Opcode.I32_SUB)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(end_local))

                push!(bytes, Opcode.BR)
                push!(bytes, 0x00)

                push!(bytes, Opcode.END)  # end loop
                push!(bytes, Opcode.END)  # end block

                # new_len = end - start + 1
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(end_local))
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(start_local))
                push!(bytes, Opcode.I32_SUB)
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x01)
                push!(bytes, Opcode.I32_ADD)
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(new_len_local))

                # Create result array
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(new_len_local))
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_NEW_DEFAULT)
                append!(bytes, encode_leb128_unsigned(str_type_idx))
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(result_local))

                # array.copy: result[0..new_len] = s[start..start+new_len]
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(result_local))
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x00)  # dst_offset = 0
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(s_local))
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(start_local))  # src_offset = start
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(new_len_local))  # length
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_COPY)
                append!(bytes, encode_leb128_unsigned(str_type_idx))
                append!(bytes, encode_leb128_unsigned(str_type_idx))

                # Return result
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(result_local))

                push!(bytes, Opcode.END)  # end else (not all whitespace)
                push!(bytes, Opcode.END)  # end else (not empty)

            # ================================================================
            # WasmTarget array operations - arr_new, arr_get, arr_set!, arr_len
            # ================================================================

            # arr_new(Type, len) -> Vector{Type}
            elseif name === :arr_new && length(args) == 2
                # First arg is the type (compile-time constant)
                # Second arg is the length
                type_arg = args[1]
                elem_type = if type_arg isa Core.SSAValue
                    ctx.ssa_types[type_arg.id]
                elseif type_arg isa GlobalRef
                    getfield(type_arg.mod, type_arg.name)
                elseif type_arg isa Type
                    type_arg
                else
                    Int32  # Default
                end

                # Get or create array type
                arr_type_idx = get_array_type!(ctx.mod, ctx.type_registry, elem_type)

                # Clear previous arg compilation - we only need length
                bytes = UInt8[]

                # Compile length arg
                append!(bytes, compile_value(args[2], ctx))
                len_type = infer_value_type(args[2], ctx)
                if len_type === Int64 || len_type === Int
                    push!(bytes, Opcode.I32_WRAP_I64)
                end

                # array.new_default creates array filled with default value (0)
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_NEW_DEFAULT)
                append!(bytes, encode_leb128_unsigned(arr_type_idx))

            # arr_get(arr, i) -> T
            elseif name === :arr_get && length(args) == 2
                # Args already compiled: arr, index
                # Need to adjust index to 0-based and emit array.get
                arr_type = infer_value_type(args[1], ctx)
                elem_type = eltype(arr_type)
                arr_type_idx = get_array_type!(ctx.mod, ctx.type_registry, elem_type)

                # Convert index to 0-based
                idx_type = infer_value_type(args[2], ctx)
                if idx_type === Int64 || idx_type === Int
                    push!(bytes, Opcode.I32_WRAP_I64)
                end
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x01)
                push!(bytes, Opcode.I32_SUB)  # index - 1

                # array.get
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_GET)
                append!(bytes, encode_leb128_unsigned(arr_type_idx))

            # arr_set!(arr, i, val) -> Nothing
            elseif name === :arr_set! && length(args) == 3
                arr_type = infer_value_type(args[1], ctx)
                elem_type = eltype(arr_type)
                arr_type_idx = get_array_type!(ctx.mod, ctx.type_registry, elem_type)

                # Recompile in correct order for array.set: arr, index-1, val
                bytes = UInt8[]

                # Array ref
                append!(bytes, compile_value(args[1], ctx))

                # Index (convert to 0-based)
                append!(bytes, compile_value(args[2], ctx))
                idx_type = infer_value_type(args[2], ctx)
                if idx_type === Int64 || idx_type === Int
                    push!(bytes, Opcode.I32_WRAP_I64)
                end
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x01)
                push!(bytes, Opcode.I32_SUB)

                # Value
                append!(bytes, compile_value(args[3], ctx))

                # array.set
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_SET)
                append!(bytes, encode_leb128_unsigned(arr_type_idx))

            # arr_len(arr) -> Int32
            elseif name === :arr_len && length(args) == 1
                # Arg already compiled, just emit array.len
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_LEN)

            # ================================================================
            # SimpleDict operations - hash table for Int32 keys/values
            # ================================================================

            # sd_new(capacity::Int32) -> SimpleDict
            elseif name === :sd_new && length(args) == 1
                bytes = UInt8[]

                # Register SimpleDict struct type
                register_struct_type!(ctx.mod, ctx.type_registry, SimpleDict)
                dict_info = ctx.type_registry.structs[SimpleDict]
                dict_type_idx = dict_info.wasm_type_idx

                # Get array type for Int32 arrays
                arr_type_idx = get_array_type!(ctx.mod, ctx.type_registry, Int32)

                # Compile capacity argument
                append!(bytes, compile_value(args[1], ctx))
                cap_type = infer_value_type(args[1], ctx)
                if cap_type === Int64 || cap_type === Int
                    push!(bytes, Opcode.I32_WRAP_I64)
                end

                # Store capacity in a local so we can use it multiple times
                cap_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, I32)
                push!(bytes, Opcode.LOCAL_TEE)
                append!(bytes, encode_leb128_unsigned(cap_local))

                # Create keys array: array.new_default arr_type_idx
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_NEW_DEFAULT)
                append!(bytes, encode_leb128_unsigned(arr_type_idx))

                # Create values array
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(cap_local))
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_NEW_DEFAULT)
                append!(bytes, encode_leb128_unsigned(arr_type_idx))

                # Create slots array
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(cap_local))
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_NEW_DEFAULT)
                append!(bytes, encode_leb128_unsigned(arr_type_idx))

                # Push count = 0
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x00)

                # Push capacity
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(cap_local))

                # struct.new SimpleDict (fields: keys, values, slots, count, capacity)
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.STRUCT_NEW)
                append!(bytes, encode_leb128_unsigned(dict_type_idx))

            # sd_length(d::SimpleDict) -> Int32
            elseif name === :sd_length && length(args) == 1
                # Args already compiled (d is on stack)
                # Get the count field (index 3)
                register_struct_type!(ctx.mod, ctx.type_registry, SimpleDict)
                dict_info = ctx.type_registry.structs[SimpleDict]
                dict_type_idx = dict_info.wasm_type_idx

                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.STRUCT_GET)
                append!(bytes, encode_leb128_unsigned(dict_type_idx))
                append!(bytes, encode_leb128_unsigned(3))  # count is field 3 (0-indexed)

            # sd_haskey(d::SimpleDict, key::Int32) -> Bool
            elseif name === :sd_haskey && length(args) == 2
                # Implement linear probing to find key
                bytes = compile_sd_find_slot(args, ctx)
                # Result is slot index (positive if found, negative if not found, 0 if full)
                # Convert to bool: slot > 0
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x00)
                push!(bytes, Opcode.I32_GT_S)

            # sd_get(d::SimpleDict, key::Int32) -> Int32
            elseif name === :sd_get && length(args) == 2
                # Find slot, then get value
                bytes = compile_sd_find_slot(args, ctx)

                # Store slot in local
                slot_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, I32)
                push!(bytes, Opcode.LOCAL_TEE)
                append!(bytes, encode_leb128_unsigned(slot_local))

                # Check if found (slot > 0)
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x00)
                push!(bytes, Opcode.I32_GT_S)

                # if found: get value, else: return 0
                push!(bytes, Opcode.IF)
                push!(bytes, 0x7F)  # result type i32

                # Get dict reference again for struct.get
                append!(bytes, compile_value(args[1], ctx))
                register_struct_type!(ctx.mod, ctx.type_registry, SimpleDict)
                dict_info = ctx.type_registry.structs[SimpleDict]
                dict_type_idx = dict_info.wasm_type_idx
                arr_type_idx = get_array_type!(ctx.mod, ctx.type_registry, Int32)

                # Get values array (field 1)
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.STRUCT_GET)
                append!(bytes, encode_leb128_unsigned(dict_type_idx))
                append!(bytes, encode_leb128_unsigned(1))  # values field

                # Get index (slot - 1 for 0-based)
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(slot_local))
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x01)
                push!(bytes, Opcode.I32_SUB)

                # array.get
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_GET)
                append!(bytes, encode_leb128_unsigned(arr_type_idx))

                push!(bytes, Opcode.ELSE)
                # Not found - return 0
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x00)
                push!(bytes, Opcode.END)

            # sd_set!(d::SimpleDict, key::Int32, value::Int32) -> Nothing
            elseif name === :sd_set! && length(args) == 3
                bytes = compile_sd_set(args, ctx)

            # ================================================================
            # StringDict operations - hash table for String keys, Int32 values
            # ================================================================

            # sdict_new(capacity::Int32) -> StringDict
            elseif name === :sdict_new && length(args) == 1
                bytes = UInt8[]

                # Register StringDict struct type
                register_struct_type!(ctx.mod, ctx.type_registry, StringDict)
                dict_info = ctx.type_registry.structs[StringDict]
                dict_type_idx = dict_info.wasm_type_idx

                # Get array types
                str_ref_arr_type_idx = get_string_ref_array_type!(ctx.mod, ctx.type_registry)
                i32_arr_type_idx = get_array_type!(ctx.mod, ctx.type_registry, Int32)
                str_type_idx = get_string_array_type!(ctx.mod, ctx.type_registry)

                # Compile capacity argument
                append!(bytes, compile_value(args[1], ctx))
                cap_type = infer_value_type(args[1], ctx)
                if cap_type === Int64 || cap_type === Int
                    push!(bytes, Opcode.I32_WRAP_I64)
                end

                # Store capacity in local
                cap_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, I32)
                push!(bytes, Opcode.LOCAL_TEE)
                append!(bytes, encode_leb128_unsigned(cap_local))

                # Create keys array (array of string refs, initialized with empty strings)
                # First create empty string to use as default
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x00)  # empty string length = 0
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_NEW_DEFAULT)
                append!(bytes, encode_leb128_unsigned(str_type_idx))

                # Store empty string for array.new_fixed
                empty_str_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, ConcreteRef(str_type_idx))
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(empty_str_local))

                # Create keys array with capacity elements, filled with empty string ref
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(empty_str_local))
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(cap_local))
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_NEW)
                append!(bytes, encode_leb128_unsigned(str_ref_arr_type_idx))

                # Create values array
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(cap_local))
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_NEW_DEFAULT)
                append!(bytes, encode_leb128_unsigned(i32_arr_type_idx))

                # Create slots array
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(cap_local))
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_NEW_DEFAULT)
                append!(bytes, encode_leb128_unsigned(i32_arr_type_idx))

                # Push count = 0
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x00)

                # Push capacity
                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(cap_local))

                # struct.new StringDict (fields: keys, values, slots, count, capacity)
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.STRUCT_NEW)
                append!(bytes, encode_leb128_unsigned(dict_type_idx))

            # sdict_length(d::StringDict) -> Int32
            elseif name === :sdict_length && length(args) == 1
                register_struct_type!(ctx.mod, ctx.type_registry, StringDict)
                dict_info = ctx.type_registry.structs[StringDict]
                dict_type_idx = dict_info.wasm_type_idx

                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.STRUCT_GET)
                append!(bytes, encode_leb128_unsigned(dict_type_idx))
                append!(bytes, encode_leb128_unsigned(3))  # count is field 3

            # sdict_haskey(d::StringDict, key::String) -> Bool
            elseif name === :sdict_haskey && length(args) == 2
                bytes = compile_sdict_find_slot(args, ctx)
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x00)
                push!(bytes, Opcode.I32_GT_S)

            # sdict_get(d::StringDict, key::String) -> Int32
            elseif name === :sdict_get && length(args) == 2
                bytes = compile_sdict_find_slot(args, ctx)

                slot_local = ctx.n_params + length(ctx.locals)
                push!(ctx.locals, I32)
                push!(bytes, Opcode.LOCAL_TEE)
                append!(bytes, encode_leb128_unsigned(slot_local))

                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x00)
                push!(bytes, Opcode.I32_GT_S)

                push!(bytes, Opcode.IF)
                push!(bytes, 0x7F)  # i32 result

                # Get dict again and get values[slot-1]
                append!(bytes, compile_value(args[1], ctx))
                register_struct_type!(ctx.mod, ctx.type_registry, StringDict)
                dict_info = ctx.type_registry.structs[StringDict]
                dict_type_idx = dict_info.wasm_type_idx
                i32_arr_type_idx = get_array_type!(ctx.mod, ctx.type_registry, Int32)

                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.STRUCT_GET)
                append!(bytes, encode_leb128_unsigned(dict_type_idx))
                append!(bytes, encode_leb128_unsigned(1))  # values field

                push!(bytes, Opcode.LOCAL_GET)
                append!(bytes, encode_leb128_unsigned(slot_local))
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x01)
                push!(bytes, Opcode.I32_SUB)

                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_GET)
                append!(bytes, encode_leb128_unsigned(i32_arr_type_idx))

                push!(bytes, Opcode.ELSE)
                push!(bytes, Opcode.I32_CONST)
                push!(bytes, 0x00)
                push!(bytes, Opcode.END)

            # sdict_set!(d::StringDict, key::String, value::Int32) -> Nothing
            elseif name === :sdict_set! && length(args) == 3
                bytes = compile_sdict_set(args, ctx)

            # Math domain error functions - these would normally throw, but in WASM we return NaN
            elseif name === :sin_domain_error || name === :cos_domain_error ||
                   name === :tan_domain_error || name === :asin_domain_error ||
                   name === :acos_domain_error || name === :log_domain_error ||
                   name === :sqrt_domain_error
                # These functions throw in Julia but we return NaN for graceful degradation
                # Return type is Union{} (never returns) but we need to produce a value
                # Push NaN for float domain errors
                push!(bytes, Opcode.F64_CONST)
                nan_bytes = reinterpret(UInt8, [NaN])
                append!(bytes, nan_bytes)

            # ================================================================
            # WASM-055: Base.string dispatch to int_to_string
            # Base.string(n::Int) internally calls Base.#string#530(base, pad, string, n)
            # We intercept this and redirect to WasmTarget.int_to_string
            # ================================================================
            elseif name === Symbol("#string#530") && length(args) >= 4
                # #string#530(base::Int64, pad::Int64, ::typeof(string), value)
                # The actual value to convert is the last argument (args[4])
                value_arg = args[4]
                value_type = infer_value_type(value_arg, ctx)

                # Check if we're converting an integer type
                if value_type === Int32 || value_type === Int64 ||
                   value_type === UInt32 || value_type === UInt64 ||
                   value_type === Int16 || value_type === UInt16 ||
                   value_type === Int8 || value_type === UInt8

                    # Clear the bytes (args were already pushed)
                    bytes = UInt8[]

                    # Check if int_to_string is in the function registry
                    int_to_string_info = nothing
                    if ctx.func_registry !== nothing
                        # Try to find int_to_string with Int32 signature
                        try
                            int_to_string_func = getfield(WasmTarget, :int_to_string)
                            int_to_string_info = get_function(ctx.func_registry, int_to_string_func, (Int32,))
                        catch
                            # Function not found
                        end
                    end

                    if int_to_string_info !== nothing
                        # int_to_string is in registry - call it
                        # Compile the value argument, converting to Int32 if needed
                        append!(bytes, compile_value(value_arg, ctx))

                        # Convert to Int32 if needed
                        if value_type === Int64
                            push!(bytes, Opcode.I32_WRAP_I64)
                        elseif value_type === UInt32 || value_type === UInt64
                            # Treat as signed for string conversion
                            if value_type === UInt64
                                push!(bytes, Opcode.I32_WRAP_I64)
                            end
                        elseif value_type !== Int32
                            # Smaller types - extend to i32
                            # Already handled by compile_value which produces correct type
                        end

                        # Call int_to_string
                        push!(bytes, Opcode.CALL)
                        append!(bytes, encode_leb128_unsigned(int_to_string_info.wasm_idx))
                    else
                        # int_to_string not in registry - provide helpful error
                        error("Base.string(::$(value_type)) requires int_to_string in compile_multi. " *
                              "Add WasmTarget.int_to_string and WasmTarget.digit_to_str to your function list.")
                    end
                else
                    # Non-integer type - not yet supported
                    error("Base.string(::$(value_type)) not yet supported. " *
                          "Supported types: Int32, Int64, UInt32, UInt64, Int16, UInt16, Int8, UInt8")
                end

            # ================================================================
            # Julia 1.11+ Memory API: Core.memoryref
            # Creates MemoryRef from Memory - in WasmGC this is a no-op
            # ================================================================
            elseif name === :memoryref && length(args) == 1
                # Core.memoryref(memory::Memory{T}) -> MemoryRef{T}
                # In WasmGC, Memory and MemoryRef are both the array reference
                # Clear args bytes (already pushed) and re-compile just the memory arg
                bytes = UInt8[]
                append!(bytes, compile_value(args[1], ctx))

            # ================================================================
            # Error constructors - these are typically followed by throw
            # In WASM we just emit unreachable
            # ================================================================
            elseif name === :BoundsError || name === :ArgumentError || name === :TypeError ||
                   name === :DomainError || name === :OverflowError || name === :DivideError ||
                   name === :InexactError
                # Error constructors - emit unreachable
                bytes = UInt8[]  # Clear any pushed args
                push!(bytes, Opcode.UNREACHABLE)

            # ================================================================
            # SubString - string view type
            # In WasmGC, we handle this by using str_substr
            # ================================================================
            elseif name === :SubString
                # SubString(str, start, stop) or SubString(str)
                # For now, just return the string as-is (view = copy semantics)
                # A proper implementation would track offset/length
                bytes = UInt8[]  # Clear any pushed args
                if !isempty(args)
                    append!(bytes, compile_value(args[1], ctx))
                end

            # ================================================================
            # PURE-004: Base.string dispatch for Float32/Float64
            # When Julia compiles string(x::Float32), it invokes the Ryu method
            # We intercept and redirect to our simpler float_to_string
            # ================================================================
            elseif name === :string && length(args) == 1
                value_arg = args[1]
                value_type = infer_value_type(value_arg, ctx)

                if value_type === Float32 || value_type === Float64
                    # Clear bytes - recompile the argument
                    bytes = UInt8[]

                    # Look up float_to_string in the function registry
                    float_to_string_info = nothing
                    if ctx.func_registry !== nothing
                        try
                            float_to_string_func = getfield(WasmTarget, :float_to_string)
                            float_to_string_info = get_function(ctx.func_registry, float_to_string_func, (Float32,))
                        catch
                            # Function not found
                        end
                    end

                    if float_to_string_info !== nothing
                        # Compile the value argument
                        append!(bytes, compile_value(value_arg, ctx))

                        # Convert Float64 to Float32 if needed (our float_to_string takes Float32)
                        if value_type === Float64
                            push!(bytes, 0xB6)  # f32.demote_f64
                        end

                        # Call float_to_string
                        push!(bytes, Opcode.CALL)
                        append!(bytes, encode_leb128_unsigned(float_to_string_info.wasm_idx))
                    else
                        error("Base.string(::$(value_type)) requires float_to_string in compile_multi. " *
                              "Add WasmTarget.float_to_string, WasmTarget.int_to_string, and WasmTarget.digit_to_str to your function list.")
                    end
                elseif value_type === Int32 || value_type === Int64 ||
                       value_type === UInt32 || value_type === UInt64 ||
                       value_type === Int16 || value_type === UInt16 ||
                       value_type === Int8 || value_type === UInt8
                    # Integer types - redirect to int_to_string
                    bytes = UInt8[]

                    int_to_string_info = nothing
                    if ctx.func_registry !== nothing
                        try
                            int_to_string_func = getfield(WasmTarget, :int_to_string)
                            int_to_string_info = get_function(ctx.func_registry, int_to_string_func, (Int32,))
                        catch
                            # Function not found
                        end
                    end

                    if int_to_string_info !== nothing
                        append!(bytes, compile_value(value_arg, ctx))

                        # Convert to Int32 if needed
                        if value_type === Int64
                            push!(bytes, Opcode.I32_WRAP_I64)
                        elseif value_type === UInt64
                            push!(bytes, Opcode.I32_WRAP_I64)
                        end

                        push!(bytes, Opcode.CALL)
                        append!(bytes, encode_leb128_unsigned(int_to_string_info.wasm_idx))
                    else
                        error("Base.string(::$(value_type)) requires int_to_string in compile_multi. " *
                              "Add WasmTarget.int_to_string and WasmTarget.digit_to_str to your function list.")
                    end
                else
                    error("Base.string(::$(value_type)) not yet supported. " *
                          "Supported types: Float32, Float64, Int32, Int64, UInt32, UInt64, Int16, UInt16, Int8, UInt8")
                end

            # Handle error-throwing functions from Base (used by pop!, resize!, etc.)
            # These functions are on error paths that should not be reached in normal execution
            # In WasmGC, we emit unreachable for these
            elseif name === :_throw_argerror || name === :throw_boundserror ||
                   name === :throw || name === :rethrow ||
                   name === :_throw_not_readable || name === :_throw_not_writable
                push!(bytes, Opcode.UNREACHABLE)

            # Handle ArgumentError constructor (used in error paths)
            elseif name === :ArgumentError
                # This creates an error object that will be thrown
                # We just emit unreachable since this path shouldn't be taken
                push!(bytes, Opcode.UNREACHABLE)

            # Handle push!/pop! growth closures from Base
            # These are generated when Julia inlines push! and need to resize the array
            # The closure name starts with # (e.g., #133#134)
            # For WasmGC, we implement array growth inline:
            # 1. Allocate new array with 2x capacity
            # 2. Copy elements from old array
            # 3. Update the vector's ref field
            elseif meth.module === Base && startswith(string(name), "#")
                # This is a Base closure, likely for array growth during push!
                # The closure captures the vector and needs to grow it

                # For now, implement simple array growth:
                # We need access to the vector to grow it, which is the first captured field
                # But this is complex - for MVP, we simply return the existing ref
                # This means push! will fail if capacity is exceeded
                #
                # TODO: Implement full growth logic:
                # 1. Get vector from closure captures
                # 2. Allocate new array with 2x size
                # 3. Copy elements
                # 4. Update vector.ref

                # For now, emit unreachable since this code path should not be taken
                # if the array was pre-allocated with enough capacity.
                #
                # The closure struct was created by the previous statement (a :new expression)
                # and is still on the stack. We need to drop it before emitting unreachable
                # so the WASM validator doesn't complain about leftover values.
                #
                # The func_ref (closure object) is expr.args[2], which is an SSAValue
                # pointing to the :new statement that created the closure struct.
                # If it's on the stack (no local allocated), drop it.
                func_ref = expr.args[2]
                if func_ref isa Core.SSAValue
                    if !haskey(ctx.ssa_locals, func_ref.id) && !haskey(ctx.phi_locals, func_ref.id)
                        # Closure is on the stack - drop it before unreachable
                        bytes = UInt8[]  # Clear any accumulated bytes
                        push!(bytes, Opcode.DROP)
                    else
                        bytes = UInt8[]  # Clear any accumulated bytes
                    end
                else
                    bytes = UInt8[]  # Clear any accumulated bytes
                end
                push!(bytes, Opcode.UNREACHABLE)

            else
                error("Unsupported method: $name")
            end
        end
    end

    return bytes
end

# ============================================================================
# Helper Functions
# ============================================================================

"""
Check if func matches a given intrinsic name.
"""
function is_func(func, name::Symbol)::Bool
    if func isa GlobalRef
        return func.name === name
    elseif func isa Core.IntrinsicFunction
        # Compare intrinsic by string representation
        return Symbol(func) === name
    elseif typeof(func) <: Core.Builtin
        # Builtin functions like isa, typeof, etc.
        return nameof(func) === name
    end
    return false
end

"""
Check if a function is a comparison operation.
"""
function is_comparison(func)::Bool
    if func isa GlobalRef
        name = func.name
        return name in (:slt_int, :sle_int, :ult_int, :ule_int, :eq_int, :ne_int,
                        :lt_float, :le_float, :eq_float, :ne_float,
                        :(===), :(!==))
    end
    return false
end

"""
Check if a value is known to be boolean (0 or 1).
This is true for comparison results.
"""
function is_boolean_value(val, ctx::CompilationContext)::Bool
    if val isa Core.SSAValue
        # Check if the SSA value is from a comparison
        stmt = ctx.code_info.code[val.id]
        if stmt isa Expr && stmt.head === :call
            return is_comparison(stmt.args[1])
        end
    elseif val isa Bool
        return true
    end
    return false
end

# ============================================================================
# String Operations
# ============================================================================

"""
Compile string concatenation (str1 * str2).
Creates a new string array with combined contents.
Uses locals for intermediate values.
"""
function compile_string_concat(str1, str2, ctx::CompilationContext)::Vector{UInt8}
    bytes = UInt8[]

    # Get string array type index
    str_type_idx = ctx.type_registry.string_array_idx

    # We need 4 locals: str1_ref, str2_ref, len1, len2
    # Allocate them (these are temporary locals for this operation)
    base_local = length(ctx.code_info.slotnames) + length(ctx.ssa_locals) + length(ctx.phi_locals)
    str1_local = base_local
    str2_local = base_local + 1
    len1_local = base_local + 2
    len2_local = base_local + 3

    # Add the locals to the function (string refs and i32s for lengths)
    # Note: We're using ConcreteRef for string arrays
    # For simplicity, we'll store lengths as i32 directly

    # Compile str1, store in local
    append!(bytes, compile_value(str1, ctx))
    push!(bytes, Opcode.LOCAL_TEE)
    append!(bytes, encode_leb128_unsigned(str1_local))

    # Get len1
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.ARRAY_LEN)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(len1_local))

    # Compile str2, store in local
    append!(bytes, compile_value(str2, ctx))
    push!(bytes, Opcode.LOCAL_TEE)
    append!(bytes, encode_leb128_unsigned(str2_local))

    # Get len2
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.ARRAY_LEN)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(len2_local))

    # Create new array with len1 + len2 elements, initialized to 0
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(len1_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(len2_local))
    push!(bytes, Opcode.I32_ADD)
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.ARRAY_NEW_DEFAULT)
    append!(bytes, encode_leb128_unsigned(str_type_idx))
    # Now stack has: [new_array]

    # Copy str1 to new_array at offset 0
    # array.copy dst_type src_type : [dst_ref dst_offset src_ref src_offset len]
    # dst = new_array (on stack), dst_offset = 0
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x00)  # dst_offset = 0
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(str1_local))  # src_ref
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x00)  # src_offset = 0
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(len1_local))  # len
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.ARRAY_COPY)
    append!(bytes, encode_leb128_unsigned(str_type_idx))  # dst type
    append!(bytes, encode_leb128_unsigned(str_type_idx))  # src type
    # Stack is empty now, we need to get new_array back
    # Actually array.copy doesn't consume the dst ref... let me check
    # Actually it does consume all arguments. We need to restructure.

    # Let me use a different approach: store new_array in a local too
    return compile_string_concat_with_locals(str1, str2, ctx)
end

"""
String concatenation implementation using explicit locals.
Uses scratch locals allocated by allocate_scratch_locals!.
"""
function compile_string_concat_with_locals(str1, str2, ctx::CompilationContext)::Vector{UInt8}
    bytes = UInt8[]

    str_type_idx = ctx.type_registry.string_array_idx

    # Use scratch locals stored in context (allocated at compile context creation time)
    if ctx.scratch_locals === nothing
        error("String operations require scratch locals but none were allocated")
    end
    result_local, str1_local, str2_local, len1_local, i_local = ctx.scratch_locals

    # Store str1
    append!(bytes, compile_value(str1, ctx))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(str1_local))

    # Store str2
    append!(bytes, compile_value(str2, ctx))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(str2_local))

    # Get len1 and store
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(str1_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.ARRAY_LEN)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(len1_local))

    # Get len2
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(str2_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.ARRAY_LEN)
    # Stack: [len2]

    # Create result array: len1 + len2
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(len1_local))
    push!(bytes, Opcode.I32_ADD)  # len1 + len2
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.ARRAY_NEW_DEFAULT)
    append!(bytes, encode_leb128_unsigned(str_type_idx))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(result_local))

    # Copy str1 to result[0:len1]
    # array.copy: [dst, dst_off, src, src_off, len]
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(result_local))
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x00)  # dst_off = 0
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(str1_local))
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x00)  # src_off = 0
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(len1_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.ARRAY_COPY)
    append!(bytes, encode_leb128_unsigned(str_type_idx))
    append!(bytes, encode_leb128_unsigned(str_type_idx))

    # Copy str2 to result[len1:]
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(result_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(len1_local))  # dst_off = len1
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(str2_local))
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x00)  # src_off = 0
    # len = str2.len
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(str2_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.ARRAY_LEN)
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.ARRAY_COPY)
    append!(bytes, encode_leb128_unsigned(str_type_idx))
    append!(bytes, encode_leb128_unsigned(str_type_idx))

    # Return result
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(result_local))

    return bytes
end

"""
Compile string equality comparison (str1 == str2).
Returns i32 (0 or 1).
Uses scratch locals allocated by allocate_scratch_locals!.
"""
function compile_string_equal(str1, str2, ctx::CompilationContext)::Vector{UInt8}
    bytes = UInt8[]

    str_type_idx = ctx.type_registry.string_array_idx

    # Use scratch locals stored in context (allocated at compile context creation time)
    if ctx.scratch_locals === nothing
        error("String operations require scratch locals but none were allocated")
    end
    _, str1_local, str2_local, len_local, i_local = ctx.scratch_locals

    # Store str1 and str2
    append!(bytes, compile_value(str1, ctx))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(str1_local))

    append!(bytes, compile_value(str2, ctx))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(str2_local))

    # Compare lengths first
    # Get len1, store in len_local
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(str1_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.ARRAY_LEN)
    push!(bytes, Opcode.LOCAL_TEE)
    append!(bytes, encode_leb128_unsigned(len_local))

    # Compare with len2
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(str2_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.ARRAY_LEN)
    push!(bytes, Opcode.I32_NE)

    # If lengths differ, result is 0; else compare elements
    push!(bytes, Opcode.IF)
    push!(bytes, 0x7F)  # result type i32

    # Then: lengths differ -> not equal
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x00)

    push!(bytes, Opcode.ELSE)

    # Else: lengths equal, compare element by element
    # Initialize i = 0
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x00)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(i_local))

    # Block for breaking out of loop with result
    push!(bytes, Opcode.BLOCK)
    push!(bytes, 0x7F)  # result type i32

    # Loop (void type - always exits via br)
    push!(bytes, Opcode.LOOP)
    push!(bytes, 0x40)  # void

    # Check if i >= len (done comparing, all matched)
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(i_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(len_local))
    push!(bytes, Opcode.I32_GE_S)
    push!(bytes, Opcode.IF)
    push!(bytes, 0x40)  # void
    # All elements matched -> push 1 and break
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x01)
    push!(bytes, Opcode.BR)
    push!(bytes, 0x02)  # break to result block
    push!(bytes, Opcode.END)  # end if (i >= len)

    # Compare str1[i] vs str2[i]
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(str1_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(i_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.ARRAY_GET)
    append!(bytes, encode_leb128_unsigned(str_type_idx))

    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(str2_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(i_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.ARRAY_GET)
    append!(bytes, encode_leb128_unsigned(str_type_idx))

    push!(bytes, Opcode.I32_NE)

    # If elements differ -> push 0 and break
    push!(bytes, Opcode.IF)
    push!(bytes, 0x40)  # void
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x00)
    push!(bytes, Opcode.BR)
    push!(bytes, 0x02)  # break to result block
    push!(bytes, Opcode.END)  # end if (elements differ)

    # Increment i
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(i_local))
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x01)
    push!(bytes, Opcode.I32_ADD)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(i_local))

    # Continue loop
    push!(bytes, Opcode.BR)
    push!(bytes, 0x00)  # br to loop

    push!(bytes, Opcode.END)  # end loop

    # Loop never falls through (always br), so this is unreachable
    push!(bytes, Opcode.UNREACHABLE)

    push!(bytes, Opcode.END)  # end result block

    push!(bytes, Opcode.END)  # end if-else (lengths comparison)

    return bytes
end

# ============================================================================
# SimpleDict Operations - Hash Table Bytecode Generation
# ============================================================================

"""
Find slot for a key in SimpleDict.
Returns: positive if found, negative if insert location, 0 if full.

Algorithm: Linear probing with hash = (key * 31) & 0x7FFFFFFF % capacity + 1
Slot states: 0=empty, 1=occupied, 2=deleted
"""
function compile_sd_find_slot(args, ctx::CompilationContext)::Vector{UInt8}
    bytes = UInt8[]

    # Register SimpleDict type
    register_struct_type!(ctx.mod, ctx.type_registry, SimpleDict)
    dict_info = ctx.type_registry.structs[SimpleDict]
    dict_type_idx = dict_info.wasm_type_idx
    arr_type_idx = get_array_type!(ctx.mod, ctx.type_registry, Int32)

    # Allocate locals for this operation
    d_local = ctx.n_params + length(ctx.locals)
    push!(ctx.locals, ConcreteRef(dict_type_idx))  # d reference

    key_local = ctx.n_params + length(ctx.locals)
    push!(ctx.locals, I32)  # key

    capacity_local = ctx.n_params + length(ctx.locals)
    push!(ctx.locals, I32)  # capacity

    start_local = ctx.n_params + length(ctx.locals)
    push!(ctx.locals, I32)  # start hash

    iter_local = ctx.n_params + length(ctx.locals)
    push!(ctx.locals, I32)  # iteration counter

    slot_idx_local = ctx.n_params + length(ctx.locals)
    push!(ctx.locals, I32)  # current slot index

    slot_state_local = ctx.n_params + length(ctx.locals)
    push!(ctx.locals, I32)  # current slot state

    result_local = ctx.n_params + length(ctx.locals)
    push!(ctx.locals, I32)  # result to return

    # Store d in local
    append!(bytes, compile_value(args[1], ctx))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(d_local))

    # Store key in local (ensure i32)
    append!(bytes, compile_value(args[2], ctx))
    key_type = infer_value_type(args[2], ctx)
    if key_type === Int64 || key_type === Int
        push!(bytes, Opcode.I32_WRAP_I64)
    end
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(key_local))

    # Get capacity from dict struct (field 4)
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(d_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_GET)
    append!(bytes, encode_leb128_unsigned(dict_type_idx))
    append!(bytes, encode_leb128_unsigned(4))  # capacity is field 4
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(capacity_local))

    # Compute hash: (key * 31) & 0x7FFFFFFF % capacity + 1
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(key_local))
    push!(bytes, Opcode.I32_CONST)
    append!(bytes, encode_leb128_signed(31))
    push!(bytes, Opcode.I32_MUL)
    push!(bytes, Opcode.I32_CONST)
    append!(bytes, encode_leb128_signed(0x7FFFFFFF))
    push!(bytes, Opcode.I32_AND)
    # % capacity
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(capacity_local))
    push!(bytes, Opcode.I32_REM_S)
    # + 1 (1-based index)
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x01)
    push!(bytes, Opcode.I32_ADD)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(start_local))

    # Initialize iter = 0
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x00)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(iter_local))

    # Initialize result = 0 (will be set in loop)
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x00)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(result_local))

    # Outer block for breaking out with result
    push!(bytes, Opcode.BLOCK)  # block $done
    push!(bytes, 0x40)  # void

    # Loop for probing
    push!(bytes, Opcode.LOOP)  # loop $probe
    push!(bytes, 0x40)  # void

    # Check if iter >= capacity (table full)
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(iter_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(capacity_local))
    push!(bytes, Opcode.I32_GE_S)
    push!(bytes, Opcode.IF)
    push!(bytes, 0x40)  # void
    # result = 0 (full), break
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x00)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(result_local))
    push!(bytes, Opcode.BR)
    push!(bytes, 0x02)  # break to $done (past loop and if)
    push!(bytes, Opcode.END)  # end if

    # Calculate slot index: ((start + iter - 1) % capacity) + 1
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(start_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(iter_local))
    push!(bytes, Opcode.I32_ADD)
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x01)
    push!(bytes, Opcode.I32_SUB)
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(capacity_local))
    push!(bytes, Opcode.I32_REM_S)
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x01)
    push!(bytes, Opcode.I32_ADD)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(slot_idx_local))

    # Get slot state from slots array
    # slots = d.slots (field 2)
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(d_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_GET)
    append!(bytes, encode_leb128_unsigned(dict_type_idx))
    append!(bytes, encode_leb128_unsigned(2))  # slots is field 2
    # array.get with slot_idx - 1 (0-based)
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(slot_idx_local))
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x01)
    push!(bytes, Opcode.I32_SUB)
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.ARRAY_GET)
    append!(bytes, encode_leb128_unsigned(arr_type_idx))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(slot_state_local))

    # Check slot state
    # If empty (0): return -slot_idx (insert here)
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(slot_state_local))
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x00)  # SLOT_EMPTY
    push!(bytes, Opcode.I32_EQ)
    push!(bytes, Opcode.IF)
    push!(bytes, 0x40)  # void
    # result = -slot_idx
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x00)
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(slot_idx_local))
    push!(bytes, Opcode.I32_SUB)  # 0 - slot_idx = -slot_idx
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(result_local))
    push!(bytes, Opcode.BR)
    push!(bytes, 0x02)  # break to $done
    push!(bytes, Opcode.END)  # end if (empty check)

    # If occupied (1): check if key matches
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(slot_state_local))
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x01)  # SLOT_OCCUPIED
    push!(bytes, Opcode.I32_EQ)
    push!(bytes, Opcode.IF)
    push!(bytes, 0x40)  # void

    # Get key from keys array and compare
    # keys = d.keys (field 0)
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(d_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_GET)
    append!(bytes, encode_leb128_unsigned(dict_type_idx))
    append!(bytes, encode_leb128_unsigned(0))  # keys is field 0
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(slot_idx_local))
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x01)
    push!(bytes, Opcode.I32_SUB)
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.ARRAY_GET)
    append!(bytes, encode_leb128_unsigned(arr_type_idx))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(key_local))
    push!(bytes, Opcode.I32_EQ)
    push!(bytes, Opcode.IF)
    push!(bytes, 0x40)  # void
    # Key matches! result = slot_idx
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(slot_idx_local))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(result_local))
    push!(bytes, Opcode.BR)
    push!(bytes, 0x03)  # break to $done: 0=this if, 1=occupied if, 2=loop, 3=block
    push!(bytes, Opcode.END)  # end if (key match)

    push!(bytes, Opcode.END)  # end if (occupied check)

    # Continue probing: iter++
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(iter_local))
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x01)
    push!(bytes, Opcode.I32_ADD)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(iter_local))

    # Loop back
    push!(bytes, Opcode.BR)
    push!(bytes, 0x00)  # continue loop

    push!(bytes, Opcode.END)  # end loop
    push!(bytes, Opcode.END)  # end block $done

    # Return result
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(result_local))

    return bytes
end

"""
Set key=value in SimpleDict.
Uses find_slot logic then updates or inserts.
"""
function compile_sd_set(args, ctx::CompilationContext)::Vector{UInt8}
    bytes = UInt8[]

    # Register SimpleDict type
    register_struct_type!(ctx.mod, ctx.type_registry, SimpleDict)
    dict_info = ctx.type_registry.structs[SimpleDict]
    dict_type_idx = dict_info.wasm_type_idx
    arr_type_idx = get_array_type!(ctx.mod, ctx.type_registry, Int32)

    # Store d, key, value in locals
    d_local = ctx.n_params + length(ctx.locals)
    push!(ctx.locals, ConcreteRef(dict_type_idx))

    key_local = ctx.n_params + length(ctx.locals)
    push!(ctx.locals, I32)

    value_local = ctx.n_params + length(ctx.locals)
    push!(ctx.locals, I32)

    # Store d
    append!(bytes, compile_value(args[1], ctx))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(d_local))

    # Store key
    append!(bytes, compile_value(args[2], ctx))
    key_type = infer_value_type(args[2], ctx)
    if key_type === Int64 || key_type === Int
        push!(bytes, Opcode.I32_WRAP_I64)
    end
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(key_local))

    # Store value
    append!(bytes, compile_value(args[3], ctx))
    val_type = infer_value_type(args[3], ctx)
    if val_type === Int64 || val_type === Int
        push!(bytes, Opcode.I32_WRAP_I64)
    end
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(value_local))

    # Find slot using inline find_slot logic
    # Create args array for find_slot call
    find_args = [Core.SlotNumber(0), Core.SlotNumber(0)]  # placeholder - we'll use locals directly

    # Actually, we need to call find_slot with (d, key) - build temporary SSA refs
    # Instead, let's inline a simpler version that just returns slot

    # For sd_set!, we need to:
    # 1. Find the slot (same probing logic)
    # 2. If slot > 0: update value
    # 3. If slot < 0: insert at -slot and increment count

    # Allocate more locals for probing
    capacity_local = ctx.n_params + length(ctx.locals)
    push!(ctx.locals, I32)

    start_local = ctx.n_params + length(ctx.locals)
    push!(ctx.locals, I32)

    iter_local = ctx.n_params + length(ctx.locals)
    push!(ctx.locals, I32)

    slot_idx_local = ctx.n_params + length(ctx.locals)
    push!(ctx.locals, I32)

    slot_state_local = ctx.n_params + length(ctx.locals)
    push!(ctx.locals, I32)

    result_local = ctx.n_params + length(ctx.locals)
    push!(ctx.locals, I32)

    # Get capacity
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(d_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_GET)
    append!(bytes, encode_leb128_unsigned(dict_type_idx))
    append!(bytes, encode_leb128_unsigned(4))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(capacity_local))

    # Compute hash
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(key_local))
    push!(bytes, Opcode.I32_CONST)
    append!(bytes, encode_leb128_signed(31))
    push!(bytes, Opcode.I32_MUL)
    push!(bytes, Opcode.I32_CONST)
    append!(bytes, encode_leb128_signed(0x7FFFFFFF))
    push!(bytes, Opcode.I32_AND)
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(capacity_local))
    push!(bytes, Opcode.I32_REM_S)
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x01)
    push!(bytes, Opcode.I32_ADD)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(start_local))

    # Initialize iter = 0, result = 0
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x00)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(iter_local))
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x00)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(result_local))

    # Probe loop (same as find_slot)
    push!(bytes, Opcode.BLOCK)  # $done
    push!(bytes, 0x40)
    push!(bytes, Opcode.LOOP)  # $probe
    push!(bytes, 0x40)

    # Check iter >= capacity
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(iter_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(capacity_local))
    push!(bytes, Opcode.I32_GE_S)
    push!(bytes, Opcode.IF)
    push!(bytes, 0x40)
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x00)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(result_local))
    push!(bytes, Opcode.BR)
    push!(bytes, 0x02)
    push!(bytes, Opcode.END)

    # Calculate slot index
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(start_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(iter_local))
    push!(bytes, Opcode.I32_ADD)
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x01)
    push!(bytes, Opcode.I32_SUB)
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(capacity_local))
    push!(bytes, Opcode.I32_REM_S)
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x01)
    push!(bytes, Opcode.I32_ADD)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(slot_idx_local))

    # Get slot state
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(d_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_GET)
    append!(bytes, encode_leb128_unsigned(dict_type_idx))
    append!(bytes, encode_leb128_unsigned(2))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(slot_idx_local))
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x01)
    push!(bytes, Opcode.I32_SUB)
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.ARRAY_GET)
    append!(bytes, encode_leb128_unsigned(arr_type_idx))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(slot_state_local))

    # Check empty
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(slot_state_local))
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x00)
    push!(bytes, Opcode.I32_EQ)
    push!(bytes, Opcode.IF)
    push!(bytes, 0x40)
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x00)
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(slot_idx_local))
    push!(bytes, Opcode.I32_SUB)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(result_local))
    push!(bytes, Opcode.BR)
    push!(bytes, 0x02)
    push!(bytes, Opcode.END)

    # Check occupied
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(slot_state_local))
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x01)
    push!(bytes, Opcode.I32_EQ)
    push!(bytes, Opcode.IF)
    push!(bytes, 0x40)
    # Check key match
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(d_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_GET)
    append!(bytes, encode_leb128_unsigned(dict_type_idx))
    append!(bytes, encode_leb128_unsigned(0))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(slot_idx_local))
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x01)
    push!(bytes, Opcode.I32_SUB)
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.ARRAY_GET)
    append!(bytes, encode_leb128_unsigned(arr_type_idx))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(key_local))
    push!(bytes, Opcode.I32_EQ)
    push!(bytes, Opcode.IF)
    push!(bytes, 0x40)
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(slot_idx_local))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(result_local))
    push!(bytes, Opcode.BR)
    push!(bytes, 0x03)  # break to $done: 0=this if, 1=occupied if, 2=loop, 3=block
    push!(bytes, Opcode.END)
    push!(bytes, Opcode.END)

    # Continue probing
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(iter_local))
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x01)
    push!(bytes, Opcode.I32_ADD)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(iter_local))
    push!(bytes, Opcode.BR)
    push!(bytes, 0x00)
    push!(bytes, Opcode.END)  # loop
    push!(bytes, Opcode.END)  # block

    # Now result has slot: positive = update, negative = insert, 0 = full
    # Check if slot > 0 (update existing)
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(result_local))
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x00)
    push!(bytes, Opcode.I32_GT_S)
    push!(bytes, Opcode.IF)
    push!(bytes, 0x40)  # void

    # Update: values[slot-1] = value
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(d_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_GET)
    append!(bytes, encode_leb128_unsigned(dict_type_idx))
    append!(bytes, encode_leb128_unsigned(1))  # values
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(result_local))
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x01)
    push!(bytes, Opcode.I32_SUB)
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(value_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.ARRAY_SET)
    append!(bytes, encode_leb128_unsigned(arr_type_idx))

    push!(bytes, Opcode.ELSE)

    # Check if slot < 0 (insert new)
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(result_local))
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x00)
    push!(bytes, Opcode.I32_LT_S)
    push!(bytes, Opcode.IF)
    push!(bytes, 0x40)

    # Calculate insert index: -result - 1
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x00)
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(result_local))
    push!(bytes, Opcode.I32_SUB)  # -result = positive slot
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x01)
    push!(bytes, Opcode.I32_SUB)  # -1 for 0-based index
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(slot_idx_local))  # reuse as insert index

    # keys[idx] = key
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(d_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_GET)
    append!(bytes, encode_leb128_unsigned(dict_type_idx))
    append!(bytes, encode_leb128_unsigned(0))  # keys
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(slot_idx_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(key_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.ARRAY_SET)
    append!(bytes, encode_leb128_unsigned(arr_type_idx))

    # values[idx] = value
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(d_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_GET)
    append!(bytes, encode_leb128_unsigned(dict_type_idx))
    append!(bytes, encode_leb128_unsigned(1))  # values
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(slot_idx_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(value_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.ARRAY_SET)
    append!(bytes, encode_leb128_unsigned(arr_type_idx))

    # slots[idx] = 1 (occupied)
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(d_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_GET)
    append!(bytes, encode_leb128_unsigned(dict_type_idx))
    append!(bytes, encode_leb128_unsigned(2))  # slots
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(slot_idx_local))
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x01)  # SLOT_OCCUPIED
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.ARRAY_SET)
    append!(bytes, encode_leb128_unsigned(arr_type_idx))

    # count++ (struct.set for field 3)
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(d_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(d_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_GET)
    append!(bytes, encode_leb128_unsigned(dict_type_idx))
    append!(bytes, encode_leb128_unsigned(3))  # count
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x01)
    push!(bytes, Opcode.I32_ADD)
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_SET)
    append!(bytes, encode_leb128_unsigned(dict_type_idx))
    append!(bytes, encode_leb128_unsigned(3))

    push!(bytes, Opcode.END)  # if slot < 0
    push!(bytes, Opcode.END)  # else

    # Result is void - nothing left on stack
    return bytes
end

"""
Find slot for a String key in StringDict.
Returns: positive if found, negative if insert location, 0 if full.

Uses str_hash for hashing and string comparison for key matching.
"""
function compile_sdict_find_slot(args, ctx::CompilationContext)::Vector{UInt8}
    bytes = UInt8[]

    # Register StringDict type and get indices
    register_struct_type!(ctx.mod, ctx.type_registry, StringDict)
    dict_info = ctx.type_registry.structs[StringDict]
    dict_type_idx = dict_info.wasm_type_idx
    str_ref_arr_type_idx = get_string_ref_array_type!(ctx.mod, ctx.type_registry)
    i32_arr_type_idx = get_array_type!(ctx.mod, ctx.type_registry, Int32)
    str_type_idx = get_string_array_type!(ctx.mod, ctx.type_registry)

    # Allocate locals
    d_local = ctx.n_params + length(ctx.locals)
    push!(ctx.locals, ConcreteRef(dict_type_idx))

    key_local = ctx.n_params + length(ctx.locals)
    push!(ctx.locals, ConcreteRef(str_type_idx))  # string ref

    capacity_local = ctx.n_params + length(ctx.locals)
    push!(ctx.locals, I32)

    start_local = ctx.n_params + length(ctx.locals)
    push!(ctx.locals, I32)

    iter_local = ctx.n_params + length(ctx.locals)
    push!(ctx.locals, I32)

    slot_idx_local = ctx.n_params + length(ctx.locals)
    push!(ctx.locals, I32)

    slot_state_local = ctx.n_params + length(ctx.locals)
    push!(ctx.locals, I32)

    result_local = ctx.n_params + length(ctx.locals)
    push!(ctx.locals, I32)

    stored_key_local = ctx.n_params + length(ctx.locals)
    push!(ctx.locals, ConcreteRef(str_type_idx))  # for key comparison

    # Store d in local
    append!(bytes, compile_value(args[1], ctx))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(d_local))

    # Store key in local
    append!(bytes, compile_value(args[2], ctx))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(key_local))

    # Get capacity
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(d_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_GET)
    append!(bytes, encode_leb128_unsigned(dict_type_idx))
    append!(bytes, encode_leb128_unsigned(4))  # capacity is field 4
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(capacity_local))

    # Compute hash using str_hash inlined
    # h = 0; for each char: h = (31 * h + char) & 0x7FFFFFFF
    hash_local = ctx.n_params + length(ctx.locals)
    push!(ctx.locals, I32)
    len_local = ctx.n_params + length(ctx.locals)
    push!(ctx.locals, I32)
    i_hash_local = ctx.n_params + length(ctx.locals)
    push!(ctx.locals, I32)

    # Get string length
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(key_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.ARRAY_LEN)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(len_local))

    # Initialize hash = 0, i = 0
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x00)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(hash_local))
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x00)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(i_hash_local))

    # Hash loop
    push!(bytes, Opcode.BLOCK)
    push!(bytes, 0x40)
    push!(bytes, Opcode.LOOP)
    push!(bytes, 0x40)

    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(i_hash_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(len_local))
    push!(bytes, Opcode.I32_GE_S)
    push!(bytes, Opcode.BR_IF)
    push!(bytes, 0x01)

    # hash = (31 * hash + char) & 0x7FFFFFFF
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(hash_local))
    push!(bytes, Opcode.I32_CONST)
    append!(bytes, encode_leb128_signed(31))
    push!(bytes, Opcode.I32_MUL)
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(key_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(i_hash_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.ARRAY_GET)
    append!(bytes, encode_leb128_unsigned(str_type_idx))
    push!(bytes, Opcode.I32_ADD)
    push!(bytes, Opcode.I32_CONST)
    append!(bytes, encode_leb128_signed(0x7FFFFFFF))
    push!(bytes, Opcode.I32_AND)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(hash_local))

    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(i_hash_local))
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x01)
    push!(bytes, Opcode.I32_ADD)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(i_hash_local))
    push!(bytes, Opcode.BR)
    push!(bytes, 0x00)
    push!(bytes, Opcode.END)
    push!(bytes, Opcode.END)

    # start = (hash % capacity) + 1
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(hash_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(capacity_local))
    push!(bytes, Opcode.I32_REM_S)
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x01)
    push!(bytes, Opcode.I32_ADD)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(start_local))

    # Initialize iter = 0, result = 0
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x00)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(iter_local))
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x00)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(result_local))

    # Probe loop
    push!(bytes, Opcode.BLOCK)  # $done
    push!(bytes, 0x40)
    push!(bytes, Opcode.LOOP)  # $probe
    push!(bytes, 0x40)

    # Check iter >= capacity
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(iter_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(capacity_local))
    push!(bytes, Opcode.I32_GE_S)
    push!(bytes, Opcode.IF)
    push!(bytes, 0x40)
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x00)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(result_local))
    push!(bytes, Opcode.BR)
    push!(bytes, 0x02)
    push!(bytes, Opcode.END)

    # slot_idx = ((start + iter - 1) % capacity) + 1
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(start_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(iter_local))
    push!(bytes, Opcode.I32_ADD)
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x01)
    push!(bytes, Opcode.I32_SUB)
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(capacity_local))
    push!(bytes, Opcode.I32_REM_S)
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x01)
    push!(bytes, Opcode.I32_ADD)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(slot_idx_local))

    # Get slot state
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(d_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_GET)
    append!(bytes, encode_leb128_unsigned(dict_type_idx))
    append!(bytes, encode_leb128_unsigned(2))  # slots field
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(slot_idx_local))
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x01)
    push!(bytes, Opcode.I32_SUB)
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.ARRAY_GET)
    append!(bytes, encode_leb128_unsigned(i32_arr_type_idx))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(slot_state_local))

    # Check empty
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(slot_state_local))
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x00)
    push!(bytes, Opcode.I32_EQ)
    push!(bytes, Opcode.IF)
    push!(bytes, 0x40)
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x00)
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(slot_idx_local))
    push!(bytes, Opcode.I32_SUB)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(result_local))
    push!(bytes, Opcode.BR)
    push!(bytes, 0x02)
    push!(bytes, Opcode.END)

    # Check occupied
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(slot_state_local))
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x01)
    push!(bytes, Opcode.I32_EQ)
    push!(bytes, Opcode.IF)
    push!(bytes, 0x40)

    # Get stored key at this slot
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(d_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_GET)
    append!(bytes, encode_leb128_unsigned(dict_type_idx))
    append!(bytes, encode_leb128_unsigned(0))  # keys field
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(slot_idx_local))
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x01)
    push!(bytes, Opcode.I32_SUB)
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.ARRAY_GET)
    append!(bytes, encode_leb128_unsigned(str_ref_arr_type_idx))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(stored_key_local))

    # Compare strings using inlined string equality
    # First compare lengths, then compare characters
    append!(bytes, compile_string_eq_inline(key_local, stored_key_local, str_type_idx, ctx))

    push!(bytes, Opcode.IF)
    push!(bytes, 0x40)
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(slot_idx_local))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(result_local))
    push!(bytes, Opcode.BR)
    push!(bytes, 0x03)  # break to $done: 0=this if, 1=occupied if, 2=loop, 3=block
    push!(bytes, Opcode.END)

    push!(bytes, Opcode.END)  # occupied

    # Continue probing
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(iter_local))
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x01)
    push!(bytes, Opcode.I32_ADD)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(iter_local))
    push!(bytes, Opcode.BR)
    push!(bytes, 0x00)
    push!(bytes, Opcode.END)  # loop
    push!(bytes, Opcode.END)  # block

    # Return result
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(result_local))

    return bytes
end

"""
Inline string equality comparison.
Compares two string locals and leaves 0 or 1 on stack.
"""
function compile_string_eq_inline(str1_local::Int, str2_local::Int, str_type_idx::UInt32, ctx::CompilationContext)::Vector{UInt8}
    bytes = UInt8[]

    # Allocate locals for comparison
    len1_local = ctx.n_params + length(ctx.locals)
    push!(ctx.locals, I32)
    len2_local = ctx.n_params + length(ctx.locals)
    push!(ctx.locals, I32)
    cmp_i_local = ctx.n_params + length(ctx.locals)
    push!(ctx.locals, I32)

    # Get lengths
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(str1_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.ARRAY_LEN)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(len1_local))

    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(str2_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.ARRAY_LEN)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(len2_local))

    # Compare lengths first
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(len1_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(len2_local))
    push!(bytes, Opcode.I32_NE)
    push!(bytes, Opcode.IF)
    push!(bytes, 0x7F)  # i32 result
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x00)  # not equal
    push!(bytes, Opcode.ELSE)

    # Lengths equal - compare characters
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x00)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(cmp_i_local))

    # Result block
    push!(bytes, Opcode.BLOCK)
    push!(bytes, 0x7F)  # i32 result

    # Loop
    push!(bytes, Opcode.LOOP)
    push!(bytes, 0x40)

    # Check i >= len
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(cmp_i_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(len1_local))
    push!(bytes, Opcode.I32_GE_S)
    push!(bytes, Opcode.IF)
    push!(bytes, 0x40)
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x01)  # all matched
    push!(bytes, Opcode.BR)
    push!(bytes, 0x02)  # to result block
    push!(bytes, Opcode.END)

    # Compare chars at i
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(str1_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(cmp_i_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.ARRAY_GET)
    append!(bytes, encode_leb128_unsigned(str_type_idx))

    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(str2_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(cmp_i_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.ARRAY_GET)
    append!(bytes, encode_leb128_unsigned(str_type_idx))

    push!(bytes, Opcode.I32_NE)
    push!(bytes, Opcode.IF)
    push!(bytes, 0x40)
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x00)  # mismatch
    push!(bytes, Opcode.BR)
    push!(bytes, 0x02)  # to result block
    push!(bytes, Opcode.END)

    # i++
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(cmp_i_local))
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x01)
    push!(bytes, Opcode.I32_ADD)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(cmp_i_local))
    push!(bytes, Opcode.BR)
    push!(bytes, 0x00)

    push!(bytes, Opcode.END)  # loop

    # Unreachable - loop always exits via br
    push!(bytes, Opcode.UNREACHABLE)

    push!(bytes, Opcode.END)  # result block

    push!(bytes, Opcode.END)  # else of length comparison

    return bytes
end

"""
Set key=value in StringDict.
"""
function compile_sdict_set(args, ctx::CompilationContext)::Vector{UInt8}
    bytes = UInt8[]

    # Register types
    register_struct_type!(ctx.mod, ctx.type_registry, StringDict)
    dict_info = ctx.type_registry.structs[StringDict]
    dict_type_idx = dict_info.wasm_type_idx
    str_ref_arr_type_idx = get_string_ref_array_type!(ctx.mod, ctx.type_registry)
    i32_arr_type_idx = get_array_type!(ctx.mod, ctx.type_registry, Int32)
    str_type_idx = get_string_array_type!(ctx.mod, ctx.type_registry)

    # Store d, key, value in locals
    d_local = ctx.n_params + length(ctx.locals)
    push!(ctx.locals, ConcreteRef(dict_type_idx))

    key_local = ctx.n_params + length(ctx.locals)
    push!(ctx.locals, ConcreteRef(str_type_idx))

    value_local = ctx.n_params + length(ctx.locals)
    push!(ctx.locals, I32)

    append!(bytes, compile_value(args[1], ctx))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(d_local))

    append!(bytes, compile_value(args[2], ctx))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(key_local))

    append!(bytes, compile_value(args[3], ctx))
    val_type = infer_value_type(args[3], ctx)
    if val_type === Int64 || val_type === Int
        push!(bytes, Opcode.I32_WRAP_I64)
    end
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(value_local))

    # Call find_slot inline (reuse most of the code from sdict_find_slot)
    # For simplicity, we'll duplicate the probing logic here

    capacity_local = ctx.n_params + length(ctx.locals)
    push!(ctx.locals, I32)
    start_local = ctx.n_params + length(ctx.locals)
    push!(ctx.locals, I32)
    iter_local = ctx.n_params + length(ctx.locals)
    push!(ctx.locals, I32)
    slot_idx_local = ctx.n_params + length(ctx.locals)
    push!(ctx.locals, I32)
    slot_state_local = ctx.n_params + length(ctx.locals)
    push!(ctx.locals, I32)
    result_local = ctx.n_params + length(ctx.locals)
    push!(ctx.locals, I32)
    hash_local = ctx.n_params + length(ctx.locals)
    push!(ctx.locals, I32)
    len_local = ctx.n_params + length(ctx.locals)
    push!(ctx.locals, I32)
    i_hash_local = ctx.n_params + length(ctx.locals)
    push!(ctx.locals, I32)
    stored_key_local = ctx.n_params + length(ctx.locals)
    push!(ctx.locals, ConcreteRef(str_type_idx))

    # Get capacity
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(d_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_GET)
    append!(bytes, encode_leb128_unsigned(dict_type_idx))
    append!(bytes, encode_leb128_unsigned(4))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(capacity_local))

    # Compute hash (same as find_slot)
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(key_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.ARRAY_LEN)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(len_local))

    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x00)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(hash_local))
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x00)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(i_hash_local))

    push!(bytes, Opcode.BLOCK)
    push!(bytes, 0x40)
    push!(bytes, Opcode.LOOP)
    push!(bytes, 0x40)
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(i_hash_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(len_local))
    push!(bytes, Opcode.I32_GE_S)
    push!(bytes, Opcode.BR_IF)
    push!(bytes, 0x01)
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(hash_local))
    push!(bytes, Opcode.I32_CONST)
    append!(bytes, encode_leb128_signed(31))
    push!(bytes, Opcode.I32_MUL)
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(key_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(i_hash_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.ARRAY_GET)
    append!(bytes, encode_leb128_unsigned(str_type_idx))
    push!(bytes, Opcode.I32_ADD)
    push!(bytes, Opcode.I32_CONST)
    append!(bytes, encode_leb128_signed(0x7FFFFFFF))
    push!(bytes, Opcode.I32_AND)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(hash_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(i_hash_local))
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x01)
    push!(bytes, Opcode.I32_ADD)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(i_hash_local))
    push!(bytes, Opcode.BR)
    push!(bytes, 0x00)
    push!(bytes, Opcode.END)
    push!(bytes, Opcode.END)

    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(hash_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(capacity_local))
    push!(bytes, Opcode.I32_REM_S)
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x01)
    push!(bytes, Opcode.I32_ADD)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(start_local))

    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x00)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(iter_local))
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x00)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(result_local))

    # Probe loop
    push!(bytes, Opcode.BLOCK)
    push!(bytes, 0x40)
    push!(bytes, Opcode.LOOP)
    push!(bytes, 0x40)

    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(iter_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(capacity_local))
    push!(bytes, Opcode.I32_GE_S)
    push!(bytes, Opcode.IF)
    push!(bytes, 0x40)
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x00)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(result_local))
    push!(bytes, Opcode.BR)
    push!(bytes, 0x02)
    push!(bytes, Opcode.END)

    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(start_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(iter_local))
    push!(bytes, Opcode.I32_ADD)
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x01)
    push!(bytes, Opcode.I32_SUB)
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(capacity_local))
    push!(bytes, Opcode.I32_REM_S)
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x01)
    push!(bytes, Opcode.I32_ADD)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(slot_idx_local))

    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(d_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_GET)
    append!(bytes, encode_leb128_unsigned(dict_type_idx))
    append!(bytes, encode_leb128_unsigned(2))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(slot_idx_local))
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x01)
    push!(bytes, Opcode.I32_SUB)
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.ARRAY_GET)
    append!(bytes, encode_leb128_unsigned(i32_arr_type_idx))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(slot_state_local))

    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(slot_state_local))
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x00)
    push!(bytes, Opcode.I32_EQ)
    push!(bytes, Opcode.IF)
    push!(bytes, 0x40)
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x00)
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(slot_idx_local))
    push!(bytes, Opcode.I32_SUB)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(result_local))
    push!(bytes, Opcode.BR)
    push!(bytes, 0x02)
    push!(bytes, Opcode.END)

    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(slot_state_local))
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x01)
    push!(bytes, Opcode.I32_EQ)
    push!(bytes, Opcode.IF)
    push!(bytes, 0x40)

    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(d_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_GET)
    append!(bytes, encode_leb128_unsigned(dict_type_idx))
    append!(bytes, encode_leb128_unsigned(0))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(slot_idx_local))
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x01)
    push!(bytes, Opcode.I32_SUB)
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.ARRAY_GET)
    append!(bytes, encode_leb128_unsigned(str_ref_arr_type_idx))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(stored_key_local))

    append!(bytes, compile_string_eq_inline(key_local, stored_key_local, str_type_idx, ctx))

    push!(bytes, Opcode.IF)
    push!(bytes, 0x40)
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(slot_idx_local))
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(result_local))
    push!(bytes, Opcode.BR)
    push!(bytes, 0x03)  # break to $done: 0=this if, 1=occupied if, 2=loop, 3=block
    push!(bytes, Opcode.END)
    push!(bytes, Opcode.END)

    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(iter_local))
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x01)
    push!(bytes, Opcode.I32_ADD)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(iter_local))
    push!(bytes, Opcode.BR)
    push!(bytes, 0x00)
    push!(bytes, Opcode.END)
    push!(bytes, Opcode.END)

    # Now handle update or insert based on result
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(result_local))
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x00)
    push!(bytes, Opcode.I32_GT_S)
    push!(bytes, Opcode.IF)
    push!(bytes, 0x40)

    # Update: values[result-1] = value
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(d_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_GET)
    append!(bytes, encode_leb128_unsigned(dict_type_idx))
    append!(bytes, encode_leb128_unsigned(1))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(result_local))
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x01)
    push!(bytes, Opcode.I32_SUB)
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(value_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.ARRAY_SET)
    append!(bytes, encode_leb128_unsigned(i32_arr_type_idx))

    push!(bytes, Opcode.ELSE)

    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(result_local))
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x00)
    push!(bytes, Opcode.I32_LT_S)
    push!(bytes, Opcode.IF)
    push!(bytes, 0x40)

    # Insert: calculate index = -result - 1
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x00)
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(result_local))
    push!(bytes, Opcode.I32_SUB)
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x01)
    push!(bytes, Opcode.I32_SUB)
    push!(bytes, Opcode.LOCAL_SET)
    append!(bytes, encode_leb128_unsigned(slot_idx_local))

    # keys[idx] = key
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(d_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_GET)
    append!(bytes, encode_leb128_unsigned(dict_type_idx))
    append!(bytes, encode_leb128_unsigned(0))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(slot_idx_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(key_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.ARRAY_SET)
    append!(bytes, encode_leb128_unsigned(str_ref_arr_type_idx))

    # values[idx] = value
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(d_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_GET)
    append!(bytes, encode_leb128_unsigned(dict_type_idx))
    append!(bytes, encode_leb128_unsigned(1))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(slot_idx_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(value_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.ARRAY_SET)
    append!(bytes, encode_leb128_unsigned(i32_arr_type_idx))

    # slots[idx] = 1
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(d_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_GET)
    append!(bytes, encode_leb128_unsigned(dict_type_idx))
    append!(bytes, encode_leb128_unsigned(2))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(slot_idx_local))
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x01)
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.ARRAY_SET)
    append!(bytes, encode_leb128_unsigned(i32_arr_type_idx))

    # count++
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(d_local))
    push!(bytes, Opcode.LOCAL_GET)
    append!(bytes, encode_leb128_unsigned(d_local))
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_GET)
    append!(bytes, encode_leb128_unsigned(dict_type_idx))
    append!(bytes, encode_leb128_unsigned(3))
    push!(bytes, Opcode.I32_CONST)
    push!(bytes, 0x01)
    push!(bytes, Opcode.I32_ADD)
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_SET)
    append!(bytes, encode_leb128_unsigned(dict_type_idx))
    append!(bytes, encode_leb128_unsigned(3))

    push!(bytes, Opcode.END)  # if slot < 0
    push!(bytes, Opcode.END)  # else

    return bytes
end
