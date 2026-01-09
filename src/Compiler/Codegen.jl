# Code Generation - Julia IR to Wasm instructions
# Maps Julia SSA statements to WebAssembly bytecode

export compile_function

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
Get or create the string array type (array of i8 for UTF-8 bytes).
"""
function get_string_array_type!(mod::WasmModule, registry::TypeRegistry)::UInt32
    if registry.string_array_idx === nothing
        # Create an i8 array type for strings (immutable is fine for strings)
        registry.string_array_idx = add_array_type!(mod, I32, false)  # Use i32 for now (i8 needs packed types)
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

    # Register any struct types used in parameters
    for T in arg_types
        if is_struct_type(T)
            register_struct_type!(mod, type_registry, T)
        elseif T <: AbstractVector
            # Register array type for Vector parameters
            elem_type = eltype(T)
            get_array_type!(mod, type_registry, elem_type)
        end
    end

    # Register return type if it's a struct
    if is_struct_type(return_type)
        register_struct_type!(mod, type_registry, return_type)
    elseif return_type <: AbstractVector
        elem_type = eltype(return_type)
        get_array_type!(mod, type_registry, elem_type)
    end

    # Determine Wasm types for parameters and return (using concrete types for GC refs)
    param_types = WasmValType[get_concrete_wasm_type(T, mod, type_registry) for T in arg_types]
    result_types = return_type === Nothing ? WasmValType[] : WasmValType[get_concrete_wasm_type(return_type, mod, type_registry)]

    # For single-function modules, the function index is 0
    # This allows recursive calls to work
    expected_func_idx = UInt32(0)

    # Generate function body with the function reference for self-call detection
    ctx = CompilationContext(code_info, arg_types, return_type, mod, type_registry;
                            func_idx=expected_func_idx, func_ref=f)
    body = generate_body(ctx)

    # Add function to module
    func_idx = add_function!(mod, param_types, result_types, ctx.locals, body)

    # Export the function
    add_export!(mod, func_name, 0, func_idx)

    return mod
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

    # Arrays have special handling - not user structs
    T <: AbstractArray && return false

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
        wasm_vt = julia_to_wasm_type(ft)
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
    func_idx::UInt32             # Index of the function being compiled (for recursion)
    func_ref::Any                # Reference to original function (for self-call detection)
end

function CompilationContext(code_info, arg_types::Tuple, return_type, mod::WasmModule, type_registry::TypeRegistry;
                           func_idx::UInt32=UInt32(0), func_ref=nothing)
    ctx = CompilationContext(
        code_info,
        arg_types,
        return_type,
        length(arg_types),
        WasmValType[],
        Dict{Int, Type}(),
        Dict{Int, Int}(),
        Dict{Int, Int}(),
        Set{Int}(),
        mod,
        type_registry,
        func_idx,
        func_ref
    )
    # Analyze SSA types and allocate locals for multi-use SSAs
    analyze_ssa_types!(ctx)
    analyze_control_flow!(ctx)  # Find loops and phi nodes
    allocate_ssa_locals!(ctx)
    return ctx
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
    else
        # Use the standard conversion for non-struct types
        return julia_to_wasm_type(T)
    end
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
    end
    return Int64
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
"""
function is_simple_conditional(blocks::Vector{BasicBlock}, code)
    # Simple pattern has 2-3 blocks with specific structure
    if length(blocks) < 2
        return false
    end

    # First block should end with GotoIfNot
    return blocks[1].terminator isa Core.GotoIfNot
end

"""
Generate code for a simple if-then-else pattern.
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

    # Determine result type for the if block
    result_type = julia_to_wasm_type(ctx.return_type)

    # Start if block (condition is on stack)
    # if (result type) ... else ... end
    push!(bytes, Opcode.IF)
    push!(bytes, UInt8(result_type))  # Block type = result type

    # Find the then-branch (statements between GotoIfNot and target)
    # and else-branch (statements at and after target)
    then_start = first_block.end_idx + 1
    else_start = target_label

    # Generate then-branch (executed when condition is TRUE, i.e., NOT jumping)
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

    return bytes
end

"""
Generate code for more complex control flow patterns.
Uses nested blocks with br instructions.
"""
function generate_complex_flow(ctx::CompilationContext, blocks::Vector{BasicBlock}, code)::Vector{UInt8}
    bytes = UInt8[]

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
Generate nested if-else for multiple conditionals.
"""
function generate_nested_conditionals(ctx::CompilationContext, blocks, code, conditionals)::Vector{UInt8}
    bytes = UInt8[]
    result_type = julia_to_wasm_type(ctx.return_type)

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
        push!(inner_bytes, UInt8(result_type))

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

    elseif stmt isa Expr
        if stmt.head === :call
            append!(bytes, compile_call(stmt, idx, ctx))
        elseif stmt.head === :invoke
            append!(bytes, compile_invoke(stmt, idx, ctx))
        elseif stmt.head === :new
            # Struct construction: %new(Type, args...)
            append!(bytes, compile_new(stmt, idx, ctx))
        elseif stmt.head === :boundscheck
            # Bounds check - we can skip this as Wasm has its own bounds checking
            # This is a no-op that produces a Bool (we push false since we're not doing checks)
            push!(bytes, Opcode.I32_CONST)
            push!(bytes, 0x00)  # false = no bounds checking
        end

        # If this SSA value needs a local, store it (and remove from stack)
        # We use LOCAL_SET (not LOCAL_TEE) to avoid leaving extra values on stack
        # that would interfere with later operations. Values will be retrieved
        # via local.get when needed.
        if haskey(ctx.ssa_locals, idx)
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
        local_idx = val.n - 2  # Argument(2) -> local 0
        if local_idx >= 0
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

    # Special case for getfield - struct/tuple field access
    if is_func(func, :getfield) && length(args) >= 2
        obj_arg = args[1]
        field_ref = args[2]
        obj_type = infer_value_type(obj_arg, ctx)

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
    elseif is_func(func, :shl_int)
        push!(bytes, is_32bit ? Opcode.I32_SHL : Opcode.I64_SHL)

    elseif is_func(func, :ashr_int)  # arithmetic shift right
        push!(bytes, is_32bit ? Opcode.I32_SHR_S : Opcode.I64_SHR_S)

    elseif is_func(func, :lshr_int)  # logical shift right
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

    # Push arguments
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

            if is_self_call
                # Self-recursive call - emit call instruction
                push!(bytes, Opcode.CALL)
                append!(bytes, encode_leb128_unsigned(ctx.func_idx))
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
