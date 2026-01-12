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
    field_types::Vector{DataType}
end

"""
Registry for struct and array type mappings within a module.
"""
mutable struct TypeRegistry
    structs::Dict{DataType, StructInfo}
    arrays::Dict{Type, UInt32}  # Element type -> array type index
    string_array_idx::Union{Nothing, UInt32}  # Index of i8 array type for strings
end

TypeRegistry() = TypeRegistry(Dict{DataType, StructInfo}(), Dict{Type, UInt32}(), nothing)

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
function register_function!(registry::FunctionRegistry, name::String, func_ref, arg_types::Tuple, wasm_idx::UInt32)
    info = FunctionInfo(name, func_ref, arg_types, wasm_idx)
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
Get or create an array type for a given element type.
"""
function get_array_type!(mod::WasmModule, registry::TypeRegistry, elem_type::Type)::UInt32
    if haskey(registry.arrays, elem_type)
        return registry.arrays[elem_type]
    end

    # Create the array type
    wasm_elem_type = julia_to_wasm_type(elem_type)
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
Get a concrete Wasm type for a Julia type, using the module and registry.
This is used before CompilationContext is created.
"""
function get_concrete_wasm_type(T::Type, mod::WasmModule, registry::TypeRegistry)::WasmValType
    if is_struct_type(T)
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
    elseif T <: AbstractVector
        elem_type = eltype(T)
        if haskey(registry.arrays, elem_type)
            type_idx = registry.arrays[elem_type]
            return ConcreteRef(type_idx, true)
        else
            type_idx = get_array_type!(mod, registry, elem_type)
            return ConcreteRef(type_idx, true)
        end
    elseif T === String
        # Strings are WasmGC arrays of bytes
        type_idx = get_string_array_type!(mod, registry)
        return ConcreteRef(type_idx, true)
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

    # Register any struct/array/string types used in parameters (skip WasmGlobal)
    for (i, T) in enumerate(arg_types)
        if i in global_args
            continue  # Skip WasmGlobal
        end
        if is_struct_type(T)
            register_struct_type!(mod, type_registry, T)
        elseif T <: AbstractVector
            # Register array type for Vector parameters
            elem_type = eltype(T)
            get_array_type!(mod, type_registry, elem_type)
        elseif T === String
            # Register string array type
            get_string_array_type!(mod, type_registry)
        end
    end

    # Register return type if it's a struct/array/string
    if is_struct_type(return_type)
        register_struct_type!(mod, type_registry, return_type)
    elseif return_type <: AbstractVector
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

    # Generate function body with the function reference for self-call detection
    ctx = CompilationContext(code_info, arg_types, return_type, mod, type_registry;
                            func_idx=expected_func_idx, func_ref=f, global_args=global_args)
    body = generate_body(ctx)

    # Add function to module
    func_idx = add_function!(mod, param_types, result_types, ctx.locals, body)

    # Export the function
    add_export!(mod, func_name, 0, func_idx)

    return mod
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
    # Create shared module and registries
    mod = WasmModule()
    type_registry = TypeRegistry()
    func_registry = FunctionRegistry()

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
            if is_struct_type(T)
                register_struct_type!(mod, type_registry, T)
            elseif T <: AbstractVector
                elem_type = eltype(T)
                get_array_type!(mod, type_registry, elem_type)
            elseif T === String
                get_string_array_type!(mod, type_registry)
            end
        end

        # Register return type
        if is_struct_type(return_type)
            register_struct_type!(mod, type_registry, return_type)
        elseif return_type <: AbstractVector
            elem_type = eltype(return_type)
            get_array_type!(mod, type_registry, elem_type)
        elseif return_type === String
            get_string_array_type!(mod, type_registry)
        end

        push!(function_data, (f, arg_types, name, code_info, return_type, global_args))
    end

    # Add all required globals to the module
    for global_idx in sort(collect(keys(required_globals)))
        wasm_type, elem_type = required_globals[global_idx]
        while length(mod.globals) <= global_idx
            add_global!(mod, wasm_type, true, zero(elem_type))
        end
    end

    # Calculate function indices (accounting for imports)
    # Functions are added in order, so index = n_imports + position - 1
    n_imports = length(mod.imports)
    for (i, (f, arg_types, name, _, _, _)) in enumerate(function_data)
        func_idx = UInt32(n_imports + i - 1)
        register_function!(func_registry, name, f, arg_types, func_idx)
    end

    # Second pass: compile function bodies
    for (i, (f, arg_types, name, code_info, return_type, global_args)) in enumerate(function_data)
        func_idx = UInt32(n_imports + i - 1)

        # Generate function body
        ctx = CompilationContext(code_info, arg_types, return_type, mod, type_registry;
                                func_registry=func_registry, func_idx=func_idx, func_ref=f,
                                global_args=global_args)
        body = generate_body(ctx)

        # Get param/result types (skip WasmGlobal args)
        param_types = WasmValType[]
        for (j, T) in enumerate(arg_types)
            if !(j in global_args)
                push!(param_types, get_concrete_wasm_type(T, mod, type_registry))
            end
        end
        result_types = return_type === Nothing ? WasmValType[] : WasmValType[get_concrete_wasm_type(return_type, mod, type_registry)]

        # Add function to module
        actual_idx = add_function!(mod, param_types, result_types, ctx.locals, body)

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
Register a Julia struct type in the Wasm module.
"""
function register_struct_type!(mod::WasmModule, registry::TypeRegistry, T::DataType)
    # Already registered?
    haskey(registry.structs, T) && return registry.structs[T]

    # Get field information
    field_names = [fieldname(T, i) for i in 1:fieldcount(T)]
    field_types = [fieldtype(T, i) for i in 1:fieldcount(T)]

    # Create WasmGC field types
    wasm_fields = FieldType[]
    for ft in field_types
        # For array fields, use concrete reference to registered array type
        if ft <: AbstractVector
            elem_type = eltype(ft)
            array_type_idx = get_array_type!(mod, registry, elem_type)
            wasm_vt = ConcreteRef(array_type_idx, true)  # nullable reference
        elseif ft === String
            str_type_idx = get_string_array_type!(mod, registry)
            wasm_vt = ConcreteRef(str_type_idx, true)
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
        wasm_vt = julia_to_wasm_type(ft)
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
    # Signal substitution for Therapy.jl closures
    signal_ssa_getters::Dict{Int, UInt32}   # SSA id (from getfield) -> Wasm global index
    signal_ssa_setters::Dict{Int, UInt32}   # SSA id (from getfield) -> Wasm global index
    captured_signal_fields::Dict{Symbol, Tuple{Bool, UInt32}}  # field_name -> (is_getter, global_idx)
    # DOM bindings for Therapy.jl - emit DOM update calls after signal writes
    # Maps global_idx -> [(import_idx, [hk_arg, ...]), ...]
    dom_bindings::Dict{UInt32, Vector{Tuple{UInt32, Vector{Int32}}}}
end

function CompilationContext(code_info, arg_types::Tuple, return_type, mod::WasmModule, type_registry::TypeRegistry;
                           func_registry::Union{FunctionRegistry, Nothing}=nothing,
                           func_idx::UInt32=UInt32(0), func_ref=nothing,
                           global_args::Set{Int}=Set{Int}(),
                           captured_signal_fields::Dict{Symbol, Tuple{Bool, UInt32}}=Dict{Symbol, Tuple{Bool, UInt32}}(),
                           dom_bindings::Dict{UInt32, Vector{Tuple{UInt32, Vector{Int32}}}}=Dict{UInt32, Vector{Tuple{UInt32, Vector{Int32}}}}())
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
        Dict{Int, UInt32}(),    # signal_ssa_getters
        Dict{Int, UInt32}(),    # signal_ssa_setters
        captured_signal_fields, # captured signal field mappings
        dom_bindings            # DOM bindings for Therapy.jl
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

        push!(ctx.locals, str_ref_type)  # result/scratch ref 1
        push!(ctx.locals, str_ref_type)  # scratch ref 2
        push!(ctx.locals, str_ref_type)  # scratch ref 3
        push!(ctx.locals, I32)           # scratch i32 1 (len1)
        push!(ctx.locals, I32)           # scratch i32 2 (len2/i)
    end
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
    elseif T <: AbstractVector
        # If array is registered, return a ConcreteRef
        elem_type = eltype(T)
        if haskey(ctx.type_registry.arrays, elem_type)
            type_idx = ctx.type_registry.arrays[elem_type]
            return ConcreteRef(type_idx, true)
        else
            # Register it now
            type_idx = get_array_type!(ctx.mod, ctx.type_registry, elem_type)
            return ConcreteRef(type_idx, true)
        end
    elseif T === String
        # Strings are WasmGC arrays of bytes
        type_idx = get_string_array_type!(ctx.mod, ctx.type_registry)
        return ConcreteRef(type_idx, true)
    elseif T isa DataType && T.name.name === :MemoryRef
        # MemoryRef{T} maps to the array type for element T
        # This is Julia's internal type for array element access
        elem_type = T.parameters[1]
        if haskey(ctx.type_registry.arrays, elem_type)
            type_idx = ctx.type_registry.arrays[elem_type]
            return ConcreteRef(type_idx, true)
        else
            type_idx = get_array_type!(ctx.mod, ctx.type_registry, elem_type)
            return ConcreteRef(type_idx, true)
        end
    elseif T isa Union
        # Handle Union types by resolving to non-Nothing type
        types = Base.uniontypes(T)
        non_nothing = filter(t -> t !== Nothing, types)
        if length(non_nothing) == 1
            # Union{Nothing, T} -> use T's concrete type
            return julia_to_wasm_type_concrete(non_nothing[1], ctx)
        else
            # Multiple non-Nothing types - fall back to standard resolution
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
"""
function allocate_ssa_locals!(ctx::CompilationContext)
    code = ctx.code_info.code

    # Count uses of each SSA value
    ssa_uses = Dict{Int, Int}()
    for stmt in code
        count_ssa_uses!(stmt, ssa_uses)
    end

    # First pass: allocate locals for SSAs used more than once or with intervening ops
    needs_local_set = Set{Int}()
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

    end

    # Actually allocate the locals
    for ssa_id in sort(collect(needs_local_set))
        if !haskey(ctx.ssa_locals, ssa_id)  # Skip phi nodes already added
            wasm_type = get(ctx.ssa_types, ssa_id, Int64)
            local_idx = ctx.n_params + length(ctx.locals)
            push!(ctx.locals, julia_to_wasm_type_concrete(wasm_type, ctx))
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

    return false
end

"""
Check if a statement produces a value on the stack.
"""
function produces_stack_value(stmt)
    # Most expressions produce values
    if stmt isa Expr
        return stmt.head in (:call, :invoke, :new, :boundscheck)
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
Analyze the IR to determine types of SSA values.
Uses CodeInfo.ssavaluetypes for accurate type information.
"""
function analyze_ssa_types!(ctx::CompilationContext)
    # Use Julia's type inference results when available
    ssatypes = ctx.code_info.ssavaluetypes
    if ssatypes isa Vector
        for (i, T) in enumerate(ssatypes)
            if T !== Any && T !== Nothing
                ctx.ssa_types[i] = T
            end
        end
    end

    # Fallback: infer from calls for any missing types
    for (i, stmt) in enumerate(ctx.code_info.code)
        if !haskey(ctx.ssa_types, i)
            if stmt isa Expr && stmt.head === :call
                ctx.ssa_types[i] = infer_call_type(stmt, ctx)
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
        idx = val.n - 1
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
    elseif val isa WasmGlobal
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
Analyze the IR to find basic block boundaries.
"""
function analyze_blocks(code)
    blocks = BasicBlock[]
    block_start = 1

    for i in 1:length(code)
        stmt = code[i]
        if stmt isa Core.GotoIfNot || stmt isa Core.GotoNode || stmt isa Core.ReturnNode
            push!(blocks, BasicBlock(block_start, i, stmt))
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
Generate code using Wasm's structured control flow.
For simple if-then-else patterns, we use the `if` instruction.
"""
function generate_structured(ctx::CompilationContext, blocks::Vector{BasicBlock})::Vector{UInt8}
    bytes = UInt8[]
    code = ctx.code_info.code

    # Check for loops first
    if has_loop(ctx)
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
Generate code for a loop structure.
Wasm loop structure:
  (block \$exit
    (loop \$continue
      ... body ...
      (br_if \$exit)  ; exit condition
      (br \$continue) ; loop back
    )
  )
"""
function generate_loop_code(ctx::CompilationContext)::Vector{UInt8}
    bytes = UInt8[]
    code = ctx.code_info.code

    # Initialize phi node locals with their entry values
    for (i, stmt) in enumerate(code)
        if stmt isa Core.PhiNode
            # Find the entry value (from edge BEFORE the phi node)
            for (edge_idx, edge) in enumerate(stmt.edges)
                if edge < i  # Entry edge (from before the phi node)
                    val = stmt.values[edge_idx]
                    append!(bytes, compile_value(val, ctx))
                    local_idx = ctx.phi_locals[i]
                    push!(bytes, Opcode.LOCAL_SET)
                    append!(bytes, encode_leb128_unsigned(local_idx))
                    break
                end
            end
        end
    end

    # block $exit (for breaking out of loop)
    push!(bytes, Opcode.BLOCK)
    push!(bytes, 0x40)  # void block type

    # loop $continue
    push!(bytes, Opcode.LOOP)
    push!(bytes, 0x40)  # void block type

    # Generate loop body
    for (i, stmt) in enumerate(code)
        if stmt isa Core.PhiNode
            # Skip phi nodes in the body - they're handled via locals
            continue
        elseif stmt isa Core.GotoIfNot
            # This is the loop exit condition
            # Push condition
            append!(bytes, compile_value(stmt.cond, ctx))
            # If condition is FALSE, break out (br_if $exit with inverted condition)
            push!(bytes, Opcode.I32_EQZ)  # Invert: if NOT condition
            push!(bytes, Opcode.BR_IF)
            push!(bytes, 0x01)  # Break to outer block (depth 1)
        elseif stmt isa Core.GotoNode
            if stmt.label in ctx.loop_headers
                # This is the loop-back jump
                # First update phi locals with their iteration values
                for (j, phi_stmt) in enumerate(code)
                    if phi_stmt isa Core.PhiNode
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
                push!(bytes, 0x00)  # Branch to loop (depth 0)
            end
        elseif stmt isa Core.ReturnNode
            if isdefined(stmt, :val)
                append!(bytes, compile_value(stmt.val, ctx))
            end
            push!(bytes, Opcode.RETURN)
        elseif !(stmt === nothing)
            append!(bytes, compile_statement(stmt, i, ctx))
        end
    end

    # End loop
    push!(bytes, Opcode.END)

    # End block
    push!(bytes, Opcode.END)

    # After loop exits, return the result (last phi value typically)
    # Find the return value
    for (i, stmt) in enumerate(code)
        if stmt isa Core.ReturnNode && isdefined(stmt, :val)
            append!(bytes, compile_value(stmt.val, ctx))
            push!(bytes, Opcode.RETURN)
            break
        end
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

        # Then-branch: push the then-value
        if then_value !== nothing
            append!(bytes, compile_value(then_value, ctx))
        end

        # Else branch
        push!(bytes, Opcode.ELSE)

        # Else-branch: push the else-value
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
        # Return-based pattern (original logic)
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
            else
                append!(bytes, compile_statement(stmt, i, ctx))
            end
        end

        # Else branch
        push!(bytes, Opcode.ELSE)

        # Generate else-branch
        for i in else_start:length(code)
            stmt = code[i]
            if stmt isa Core.ReturnNode
                if isdefined(stmt, :val)
                    append!(bytes, compile_value(stmt.val, ctx))
                end
            else
                append!(bytes, compile_statement(stmt, i, ctx))
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

    # For now, handle multi-way conditionals by nesting if-else
    # This works for patterns like: if ... elseif ... else ... end

    # Count how many conditional branches we have
    conditionals = [(i, b) for (i, b) in enumerate(blocks) if b.terminator isa Core.GotoIfNot]

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
                    # Don't return here - we're in a branch
                    # But we need to drop any value on the stack
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
                            # For other calls, check type and use count
                            stmt_type = get(ctx.ssa_types, j, Nothing)
                            if stmt_type !== Nothing && stmt_type !== Any
                                is_nothing_union = stmt_type isa Union && Nothing in Base.uniontypes(stmt_type)
                                if !is_nothing_union
                                    # This call produces a value
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

            # Else branch
            push!(bytes, Opcode.ELSE)

            # Compile else-branch (else_target to end or next structure)
            for j in else_target:length(code)
                if j in compiled
                    continue
                end
                inner = code[j]
                if inner === nothing
                    push!(compiled, j)
                elseif inner isa Core.ReturnNode
                    # Don't return here either
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

                    # Drop unused values in void context (else branch)
                    if inner isa Expr && (inner.head === :call || inner.head === :invoke)
                        stmt_type = get(ctx.ssa_types, j, Nothing)
                        if stmt_type !== Nothing && stmt_type !== Any
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

            push!(bytes, Opcode.END)

            # Mark all statements up to else_target as compiled
            for j in i:else_target
                push!(compiled, j)
            end

            i = else_target + 1
            continue
        end

        # Regular statement
        append!(bytes, compile_statement(stmt, i, ctx))
        push!(compiled, i)

        # Drop unused values from statements that produce values
        if stmt isa Expr && (stmt.head === :call || stmt.head === :invoke)
            stmt_type = get(ctx.ssa_types, i, Nothing)
            if stmt_type !== Nothing && stmt_type !== Any
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
            if stmt_type !== Nothing && stmt_type !== Any
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
            if isdefined(inner, :val) && inner.val !== nothing
                # Non-void return - but we're in void handler, just return
            end
            push!(bytes, Opcode.RETURN)
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
                    if stmt_type !== Nothing && stmt_type !== Any
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
Generate nested if-else for multiple conditionals.
"""
function generate_nested_conditionals(ctx::CompilationContext, blocks, code, conditionals)::Vector{UInt8}
    bytes = UInt8[]
    result_type = julia_to_wasm_type_concrete(ctx.return_type, ctx)

    # Build a recursive if-else structure
    function gen_conditional(cond_idx::Int)::Vector{UInt8}
        inner_bytes = UInt8[]

        if cond_idx > length(conditionals)
            # No more conditionals - generate remaining code
            # Find the last block and generate it
            for block in blocks
                if block.terminator isa Core.ReturnNode && !(block.terminator isa Core.GotoIfNot)
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
        end

        # Push condition
        append!(inner_bytes, compile_value(goto_if_not.cond, ctx))

        # if block
        push!(inner_bytes, Opcode.IF)
        append!(inner_bytes, encode_block_type(result_type))

        # Then branch - find the return after this block
        then_start = block.end_idx + 1
        then_end = goto_if_not.dest - 1
        for i in then_start:min(then_end, length(code))
            stmt = code[i]
            if stmt isa Core.ReturnNode
                if isdefined(stmt, :val)
                    append!(inner_bytes, compile_value(stmt.val, ctx))
                end
                break
            else
                append!(inner_bytes, compile_statement(stmt, i, ctx))
            end
        end

        # Else branch
        push!(inner_bytes, Opcode.ELSE)

        # Recurse for next conditional or generate final else
        append!(inner_bytes, gen_conditional(cond_idx + 1))

        push!(inner_bytes, Opcode.END)

        return inner_bytes
    end

    append!(bytes, gen_conditional(1))
    push!(bytes, Opcode.RETURN)

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
        append!(bytes, compile_value(stmt.val, ctx))

        # If this SSA value needs a local, store it (and remove from stack)
        if haskey(ctx.ssa_locals, idx)
            local_idx = ctx.ssa_locals[idx]
            push!(bytes, Opcode.LOCAL_SET)  # Use SET not TEE to not leave on stack
            append!(bytes, encode_leb128_unsigned(local_idx))
        end

    elseif stmt isa GlobalRef
        # GlobalRef statement - evaluate the constant and push it
        # This handles things like Main.SLOT_EMPTY that are module-level constants
        try
            val = getfield(stmt.mod, stmt.name)
            append!(bytes, compile_value(val, ctx))

            # If this SSA value needs a local, store it
            if haskey(ctx.ssa_locals, idx)
                local_idx = ctx.ssa_locals[idx]
                push!(bytes, Opcode.LOCAL_SET)
                append!(bytes, encode_leb128_unsigned(local_idx))
            end
        catch
            # If we can't evaluate, it might be a type reference which has no runtime value
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
        end

        append!(bytes, stmt_bytes)

        # If this SSA value needs a local, store it (and remove from stack)
        # We use LOCAL_SET (not LOCAL_TEE) to avoid leaving extra values on stack
        # that would interfere with later operations. Values will be retrieved
        # via local.get when needed.
        # IMPORTANT: Only do this if the statement actually produced bytecode
        # (skipped signal statements return empty bytes)
        if haskey(ctx.ssa_locals, idx) && !isempty(stmt_bytes)
            local_idx = ctx.ssa_locals[idx]
            push!(bytes, Opcode.LOCAL_SET)
            append!(bytes, encode_leb128_unsigned(local_idx))
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

    # Get the registered struct info
    if !haskey(ctx.type_registry.structs, struct_type)
        # Register it now
        register_struct_type!(ctx.mod, ctx.type_registry, struct_type)
    end

    info = ctx.type_registry.structs[struct_type]

    # Push field values in order
    for val in field_values
        append!(bytes, compile_value(val, ctx))
    end

    # struct.new type_idx
    push!(bytes, Opcode.GC_PREFIX)
    push!(bytes, Opcode.STRUCT_NEW)
    append!(bytes, encode_leb128_unsigned(info.wasm_type_idx))

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

    if val isa Core.SSAValue
        # Check if this SSA has a local allocated
        if haskey(ctx.ssa_locals, val.id)
            local_idx = ctx.ssa_locals[val.id]
            push!(bytes, Opcode.LOCAL_GET)
            append!(bytes, encode_leb128_unsigned(local_idx))
        end
        # Otherwise, assume it's on the stack (for single-use SSAs in sequence)

    elseif val isa Core.Argument
        # Argument(2) is arg_types[1], Argument(3) is arg_types[2], etc.
        arg_idx = val.n - 1  # Convert to 1-based arg_types index

        # WasmGlobal arguments don't have locals - they're accessed via global.get/set
        # in the getfield/setfield handlers, so we skip emitting anything here
        if arg_idx in ctx.global_args
            # WasmGlobal arg - no local.get needed (handled by getfield/setfield)
            # Return empty bytes
        elseif arg_idx >= 1
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
        # GlobalRef to a constant - evaluate and compile the value
        try
            actual_val = getfield(val.mod, val.name)
            append!(bytes, compile_value(actual_val, ctx))
        catch
            # If we can't evaluate, might be a type reference (no runtime value)
        end
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
    # ONLY applies to actual getfield(Signal, :value) calls (WasmGlobal pattern)
    # For Therapy.jl closures, signal_ssa_getters maps closure field SSAs - handled in compile_invoke
    is_getfield_value = is_func(func, :getfield) && length(args) >= 2
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
    # ONLY applies to actual setfield! calls (WasmGlobal pattern), NOT closure field access
    is_setfield_call = is_func(func, :setfield!) && length(args) >= 3
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
    is_getfield = (func isa GlobalRef &&
                  ((func.mod === Core && func.name === :getfield) ||
                   (func.mod === Base && func.name === :getfield)))
    if is_getfield && length(args) >= 2
        field_ref = args[2]
        field_name = field_ref isa QuoteNode ? field_ref.value : field_ref
        if field_name === :signal
            # Skip - this is getting Signal from CompilableSignal/Setter
            return bytes
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
        push!(bytes, Opcode.SELECT)
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

        if arg_type === String || arg_type <: AbstractVector
            # For strings (i32-per-char) and arrays, length is the array length
            append!(bytes, compile_value(arg, ctx))
            push!(bytes, Opcode.GC_PREFIX)
            push!(bytes, Opcode.ARRAY_LEN)
            # array.len returns i32, extend to i64 for Julia's Int
            push!(bytes, Opcode.I64_EXTEND_I32_S)
            return bytes
        end
        # For other types, fall through to error
    end

    # Special case for getfield - struct/tuple field access
    if is_func(func, :getfield) && length(args) >= 2
        obj_arg = args[1]
        field_ref = args[2]
        obj_type = infer_value_type(obj_arg, ctx)

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

        # Handle Vector field access (:ref and :size)
        if obj_type <: AbstractVector
            field_sym = if field_ref isa QuoteNode
                field_ref.value
            else
                field_ref
            end

            if field_sym === :ref
                # :ref returns the underlying array reference
                # In WasmGC, the Vector IS the array, so just return it
                append!(bytes, compile_value(obj_arg, ctx))
                return bytes
            elseif field_sym === :size
                # :size returns a Tuple{Int64} containing the length
                # Create a tuple struct with the array length

                # Register the Tuple{Int64} type if needed
                size_tuple_type = Tuple{Int64}
                if !haskey(ctx.type_registry.structs, size_tuple_type)
                    register_tuple_type!(ctx.mod, ctx.type_registry, size_tuple_type)
                end

                # Emit: array.len -> extend to i64 -> struct.new for tuple
                append!(bytes, compile_value(obj_arg, ctx))
                push!(bytes, Opcode.GC_PREFIX)
                push!(bytes, Opcode.ARRAY_LEN)
                # Convert i32 to i64 for Julia compatibility
                push!(bytes, Opcode.I64_EXTEND_I32_S)

                # Create a tuple struct with this value
                if haskey(ctx.type_registry.structs, size_tuple_type)
                    info = ctx.type_registry.structs[size_tuple_type]
                    push!(bytes, Opcode.GC_PREFIX)
                    push!(bytes, Opcode.STRUCT_NEW)
                    append!(bytes, encode_leb128_unsigned(info.wasm_type_idx))
                end
                return bytes
            end
        end

        # Handle struct field access by name
        if is_struct_type(obj_type) && haskey(ctx.type_registry.structs, obj_type)
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
                    # Dynamic index - not yet supported
                    nothing
                else
                    nothing
                end

                if field_idx !== nothing && field_idx >= 1 && field_idx <= length(info.field_names)
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

        # Extract element type from MemoryRef{T}
        elem_type = Int32  # default
        if ref_type isa DataType && ref_type.name.name === :MemoryRef
            elem_type = ref_type.parameters[1]
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

    # Special case for memoryrefnew - create offset reference
    # memoryrefnew(base_ref, index, boundscheck) -> MemoryRef at offset
    # In WasmGC, this translates to keeping the array ref and converting index to i32
    if is_func(func, :memoryrefnew) && length(args) >= 2
        base_ref = args[1]
        index = args[2]

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

    # Special case for setfield! - mutable struct field assignment
    # Also handles WasmGlobal (:value -> global.set)
    if is_func(func, :setfield!) && length(args) >= 3
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
    end

    # Push arguments onto the stack (normal case)
    for arg in args
        append!(bytes, compile_value(arg, ctx))
    end

    # Determine argument type for opcode selection
    arg_type = length(args) > 0 ? infer_value_type(args[1], ctx) : Int64
    is_32bit = arg_type === Int32 || arg_type === UInt32 || arg_type === Bool

    # Match intrinsics by name
    if is_func(func, :add_int)
        push!(bytes, is_32bit ? Opcode.I32_ADD : Opcode.I64_ADD)

    elseif is_func(func, :sub_int)
        push!(bytes, is_32bit ? Opcode.I32_SUB : Opcode.I64_SUB)

    elseif is_func(func, :mul_int)
        push!(bytes, is_32bit ? Opcode.I32_MUL : Opcode.I64_MUL)

    elseif is_func(func, :sdiv_int) || is_func(func, :checked_sdiv_int)
        push!(bytes, is_32bit ? Opcode.I32_DIV_S : Opcode.I64_DIV_S)

    elseif is_func(func, :udiv_int) || is_func(func, :checked_udiv_int)
        push!(bytes, is_32bit ? Opcode.I32_DIV_U : Opcode.I64_DIV_U)

    elseif is_func(func, :srem_int) || is_func(func, :checked_srem_int)
        push!(bytes, is_32bit ? Opcode.I32_REM_S : Opcode.I64_REM_S)

    elseif is_func(func, :urem_int) || is_func(func, :checked_urem_int)
        push!(bytes, is_32bit ? Opcode.I32_REM_U : Opcode.I64_REM_U)

    # Bitcast (no-op for same-size types)
    elseif is_func(func, :bitcast)
        # Bitcast between same-size types is a no-op in Wasm
        # The value is already on the stack
        # Just leave it there

    elseif is_func(func, :neg_int)
        # neg x = 0 - x
        # We need to push 0 first, then the value is already on stack
        # But value was already pushed, so we need: push 0, swap, sub
        # Actually easier: use (0 - x) pattern
        # Re-emit: i64.const 0, then value, then sub
        # But value already on stack... let's use a different approach
        # Wasm doesn't have neg, so we compute (0 - x)
        # Pop the value we just pushed, push 0, push value, sub
        if is_32bit
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

    # Comparison operations
    elseif is_func(func, :slt_int)  # signed less than
        push!(bytes, is_32bit ? Opcode.I32_LT_S : Opcode.I64_LT_S)

    elseif is_func(func, :sle_int)  # signed less or equal
        push!(bytes, is_32bit ? Opcode.I32_LE_S : Opcode.I64_LE_S)

    elseif is_func(func, :ult_int)  # unsigned less than
        push!(bytes, is_32bit ? Opcode.I32_LT_U : Opcode.I64_LT_U)

    elseif is_func(func, :ule_int)  # unsigned less or equal
        push!(bytes, is_32bit ? Opcode.I32_LE_U : Opcode.I64_LE_U)

    elseif is_func(func, :eq_int)
        push!(bytes, is_32bit ? Opcode.I32_EQ : Opcode.I64_EQ)

    elseif is_func(func, :ne_int)
        push!(bytes, is_32bit ? Opcode.I32_NE : Opcode.I64_NE)

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

    # Identity comparison (=== for integers is same as ==)
    elseif is_func(func, :(===))
        push!(bytes, is_32bit ? Opcode.I32_EQ : Opcode.I64_EQ)

    elseif is_func(func, :(!==))
        push!(bytes, is_32bit ? Opcode.I32_NE : Opcode.I64_NE)

    # Bitwise operations
    elseif is_func(func, :and_int)
        push!(bytes, is_32bit ? Opcode.I32_AND : Opcode.I64_AND)

    elseif is_func(func, :or_int)
        push!(bytes, is_32bit ? Opcode.I32_OR : Opcode.I64_OR)

    elseif is_func(func, :xor_int)
        push!(bytes, is_32bit ? Opcode.I32_XOR : Opcode.I64_XOR)

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
        if is_32bit && length(args) >= 2
            shift_type = infer_value_type(args[2], ctx)
            if shift_type === Int64 || shift_type === UInt64
                # Truncate i64 shift amount to i32
                push!(bytes, Opcode.I32_WRAP_I64)
            end
        end
        push!(bytes, is_32bit ? Opcode.I32_SHL : Opcode.I64_SHL)

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
        if is_32bit && length(args) >= 2
            shift_type = infer_value_type(args[2], ctx)
            if shift_type === Int64 || shift_type === UInt64
                # Truncate i64 shift amount to i32
                push!(bytes, Opcode.I32_WRAP_I64)
            end
        end
        push!(bytes, is_32bit ? Opcode.I32_SHR_U : Opcode.I64_SHR_U)

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

    # Type conversions
    elseif is_func(func, :sext_int)  # Sign extend i32 to i64
        push!(bytes, Opcode.I64_EXTEND_I32_S)

    elseif is_func(func, :zext_int)  # Zero extend i32 to i64
        push!(bytes, Opcode.I64_EXTEND_I32_U)

    elseif is_func(func, :trunc_int)  # Truncate i64 to i32
        push!(bytes, Opcode.I32_WRAP_I64)

    elseif is_func(func, :sitofp)  # Signed int to float
        # sitofp(TargetType, value) - first arg is target type, second is value
        # Need to check: target float type (first arg) and source int type (second arg)
        target_type = args[1]  # Float32 or Float64
        source_type = length(args) >= 2 ? infer_value_type(args[2], ctx) : Int64
        source_is_32bit = source_type === Int32 || source_type === UInt32

        if target_type === Float32
            push!(bytes, source_is_32bit ? Opcode.F32_CONVERT_I32_S : Opcode.F32_CONVERT_I64_S)
        else  # Float64
            push!(bytes, source_is_32bit ? Opcode.F64_CONVERT_I32_S : Opcode.F64_CONVERT_I64_S)
        end

    elseif is_func(func, :uitofp)  # Unsigned int to float
        target_type = args[1]
        source_type = length(args) >= 2 ? infer_value_type(args[2], ctx) : Int64
        source_is_32bit = source_type === Int32 || source_type === UInt32

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

    # Push arguments (for non-signal calls)
    for arg in args
        append!(bytes, compile_value(arg, ctx))
    end

    arg_type = length(args) > 0 ? infer_value_type(args[1], ctx) : Int64
    is_32bit = arg_type === Int32 || arg_type === UInt32

    mi = expr.args[1]
    if mi isa Core.MethodInstance
        meth = mi.def
        if meth isa Method
            name = meth.name

            # Check if this is a self-recursive call
            # The second argument of invoke is the function reference
            func_ref = expr.args[2]
            is_self_call = false
            if ctx.func_ref !== nothing && func_ref isa GlobalRef
                # Check if this GlobalRef refers to the same function
                try
                    called_func = getfield(func_ref.mod, func_ref.name)
                    is_self_call = called_func === ctx.func_ref
                catch
                    is_self_call = false
                end
            end

            # Check for cross-function call within the module first
            cross_call_handled = false
            if ctx.func_registry !== nothing && func_ref isa GlobalRef && !is_self_call
                # Try to find this function in our registry
                called_func = try
                    getfield(func_ref.mod, func_ref.name)
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
            elseif name === :throw_boundserror || name === :throw
                # Error throwing functions - emit unreachable
                # Clear the stack first (arguments were pushed but not needed)
                bytes = UInt8[]  # Reset - don't need the pushed args
                push!(bytes, Opcode.UNREACHABLE)
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

                # Use scratch locals
                n_locals_before_scratch = length(ctx.locals) - 5
                scratch_base = ctx.n_params + n_locals_before_scratch
                result_local = scratch_base      # ref for result
                src_local = scratch_base + 1     # ref for source string

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

    # Use scratch locals allocated at the end of ctx.locals
    # The scratch locals are: ref1, ref2, ref3, i32_1, i32_2
    # They start at index (n_params + n_other_locals)
    n_locals_before_scratch = length(ctx.locals) - 5  # 5 scratch locals
    scratch_base = ctx.n_params + n_locals_before_scratch

    result_local = scratch_base      # ref for result
    str1_local = scratch_base + 1    # ref for str1
    str2_local = scratch_base + 2    # ref for str2
    len1_local = scratch_base + 3    # i32 for len1
    i_local = scratch_base + 4       # i32 for len2/index

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

    # Use scratch locals allocated at the end of ctx.locals
    n_locals_before_scratch = length(ctx.locals) - 5  # 5 scratch locals
    scratch_base = ctx.n_params + n_locals_before_scratch

    # Use scratch locals: ref1, ref2, i32_1 (len), i32_2 (i)
    str1_local = scratch_base + 1    # ref for str1
    str2_local = scratch_base + 2    # ref for str2
    len_local = scratch_base + 3     # i32 for len
    i_local = scratch_base + 4       # i32 for loop index

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
