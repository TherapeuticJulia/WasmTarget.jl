"""
    SubsetCompiler

A simple Julia subset compiler that goes directly from AST to WASM.
No type inference required - all types must be explicitly annotated.

This is designed to be compilable to WASM itself for browser execution.
"""
module SubsetCompiler

using JuliaSyntax
using JuliaSyntax: SyntaxNode, GreenNode, Kind, kind, children, sourcetext, head, span
using JuliaSyntax: @K_str

export compile_subset, SubsetError, TypedExpr

# ============================================================================
# Types
# ============================================================================

"""Errors during subset compilation"""
struct SubsetError <: Exception
    msg::String
    node::Union{SyntaxNode, Nothing}
end

Base.showerror(io::IO, e::SubsetError) = print(io, "SubsetError: ", e.msg)

"""Supported types in the subset"""
@enum SubsetType begin
    T_I32
    T_I64
    T_F32
    T_F64
    T_Bool
    T_Void  # for functions with no return
end

"""Map Julia type names to SubsetType"""
const TYPE_MAP = Dict{String, SubsetType}(
    "Int32" => T_I32,
    "Int64" => T_I64,
    "Float32" => T_F32,
    "Float64" => T_F64,
    "Bool" => T_Bool,
    "Nothing" => T_Void,
)

"""WASM type codes"""
const WASM_TYPE = Dict{SubsetType, UInt8}(
    T_I32 => 0x7f,
    T_I64 => 0x7e,
    T_F32 => 0x7d,
    T_F64 => 0x7c,
    T_Bool => 0x7f,  # i32
)

"""A typed expression - AST node with resolved type"""
struct TypedExpr
    kind::Symbol          # :literal, :var, :binop, :call, :if, :while, :return, etc.
    typ::SubsetType       # Result type
    children::Vector{Any} # Child expressions or data
    span::UnitRange{Int}  # Source location
end

"""Function signature"""
struct FunctionSig
    name::String
    params::Vector{Tuple{String, SubsetType}}  # (name, type) pairs
    ret_type::SubsetType
end

"""Compilation context"""
mutable struct CompileCtx
    locals::Dict{String, Tuple{Int, SubsetType}}  # var -> (local_idx, type)
    functions::Dict{String, FunctionSig}          # name -> signature
    next_local::Int
    errors::Vector{SubsetError}
end

CompileCtx() = CompileCtx(Dict(), Dict(), 0, SubsetError[])

# ============================================================================
# Parsing Helpers
# ============================================================================

"""Get the kind of a syntax node as a symbol for easier matching"""
function node_kind(node::SyntaxNode)::Symbol
    k = kind(node)
    # Map JuliaSyntax kinds to our symbols
    if k == K"Integer"
        return :integer
    elseif k == K"Float"
        return :float
    elseif k == K"true" || k == K"false"
        return :bool
    elseif k == K"Identifier"
        return :identifier
    elseif k == K"call"
        return :call
    elseif k == K"function"
        return :function
    elseif k == K"="
        return :assign
    elseif k == K"::"
        return :typed
    elseif k == K"if"
        return :if
    elseif k == K"while"
        return :while
    elseif k == K"for"
        return :for
    elseif k == K"return"
        return :return
    elseif k == K"block"
        return :block
    elseif k == K"+"
        return :add
    elseif k == K"-"
        return :sub
    elseif k == K"*"
        return :mul
    elseif k == K"/"
        return :div
    elseif k == K"%"
        return :rem
    elseif k == K"<"
        return :lt
    elseif k == K"<="
        return :le
    elseif k == K">"
        return :gt
    elseif k == K">="
        return :ge
    elseif k == K"=="
        return :eq
    elseif k == K"!="
        return :ne
    elseif k == K"&&"
        return :and
    elseif k == K"||"
        return :or
    elseif k == K"!"
        return :not
    elseif k == K"toplevel"
        return :toplevel
    else
        return :unknown
    end
end

"""Get the text content of a node"""
function node_text(node::SyntaxNode)::String
    sourcetext(node)
end

"""Get children of a node, filtering trivia"""
function node_children(node::SyntaxNode)
    cs = children(node)
    cs === nothing ? SyntaxNode[] : collect(cs)
end

# ============================================================================
# Type Resolution
# ============================================================================

"""Parse a type annotation node into SubsetType"""
function resolve_type(node::SyntaxNode)::SubsetType
    text = strip(node_text(node))
    if haskey(TYPE_MAP, text)
        return TYPE_MAP[text]
    else
        throw(SubsetError("Unsupported type: $text", node))
    end
end

"""Infer type from a literal"""
function literal_type(node::SyntaxNode)::SubsetType
    k = node_kind(node)
    if k == :integer
        return T_I64  # Default to Int64
    elseif k == :float
        return T_F64  # Default to Float64
    elseif k == :bool
        return T_Bool
    else
        throw(SubsetError("Cannot infer type from: $k", node))
    end
end

# ============================================================================
# AST Analysis
# ============================================================================

"""Analyze a function definition, extract signature"""
function analyze_function(node::SyntaxNode, ctx::CompileCtx)::FunctionSig
    cs = node_children(node)

    # Function structure: call(name, params...) [:: rettype] block
    if length(cs) < 2
        throw(SubsetError("Invalid function definition", node))
    end

    # First child is the signature (call node with name and typed params)
    sig_node = cs[1]

    # Check for return type annotation
    ret_type = T_Void
    body_idx = 2

    if length(cs) >= 2 && node_kind(cs[2]) == :typed
        # Has return type annotation
        # The typed node wraps the signature and return type
        throw(SubsetError("TODO: Handle return type annotation", node))
    end

    # Parse signature
    if node_kind(sig_node) == :call
        sig_children = node_children(sig_node)
        if isempty(sig_children)
            throw(SubsetError("Function missing name", node))
        end

        name = node_text(sig_children[1])
        params = Tuple{String, SubsetType}[]

        for i in 2:length(sig_children)
            param = sig_children[i]
            if node_kind(param) == :typed
                # param::Type
                pc = node_children(param)
                if length(pc) >= 2
                    pname = node_text(pc[1])
                    ptype = resolve_type(pc[2])
                    push!(params, (pname, ptype))
                end
            else
                throw(SubsetError("Parameter must have type annotation: $(node_text(param))", param))
            end
        end

        return FunctionSig(name, params, ret_type)
    elseif node_kind(sig_node) == :typed
        # function name(args)::RetType ... end
        typed_children = node_children(sig_node)
        if length(typed_children) >= 2
            call_node = typed_children[1]
            ret_type = resolve_type(typed_children[2])

            if node_kind(call_node) == :call
                call_children = node_children(call_node)
                name = node_text(call_children[1])
                params = Tuple{String, SubsetType}[]

                for i in 2:length(call_children)
                    param = call_children[i]
                    if node_kind(param) == :typed
                        pc = node_children(param)
                        if length(pc) >= 2
                            pname = node_text(pc[1])
                            ptype = resolve_type(pc[2])
                            push!(params, (pname, ptype))
                        end
                    else
                        throw(SubsetError("Parameter must have type annotation", param))
                    end
                end

                return FunctionSig(name, params, ret_type)
            end
        end
        throw(SubsetError("Invalid function signature", sig_node))
    else
        throw(SubsetError("Expected function signature, got: $(node_kind(sig_node))", sig_node))
    end
end

"""Type-check and transform an expression"""
function analyze_expr(node::SyntaxNode, ctx::CompileCtx)::TypedExpr
    k = node_kind(node)

    if k == :integer
        # Parse the integer, check if it fits in i32 or needs i64
        text = node_text(node)
        val = parse(Int64, text)
        typ = T_I64
        return TypedExpr(:literal, typ, [val], 1:span(node))

    elseif k == :float
        text = node_text(node)
        val = parse(Float64, text)
        return TypedExpr(:literal, T_F64, [val], 1:span(node))

    elseif k == :bool
        val = node_text(node) == "true"
        return TypedExpr(:literal, T_Bool, [val], 1:span(node))

    elseif k == :identifier
        name = node_text(node)
        if haskey(ctx.locals, name)
            idx, typ = ctx.locals[name]
            return TypedExpr(:var, typ, [name, idx], 1:span(node))
        else
            throw(SubsetError("Undefined variable: $name", node))
        end

    elseif k in (:add, :sub, :mul, :div, :rem)
        cs = node_children(node)
        if length(cs) != 2
            throw(SubsetError("Binary op requires 2 operands", node))
        end
        left = analyze_expr(cs[1], ctx)
        right = analyze_expr(cs[2], ctx)
        if left.typ != right.typ
            throw(SubsetError("Type mismatch: $(left.typ) vs $(right.typ)", node))
        end
        return TypedExpr(k, left.typ, [left, right], 1:span(node))

    elseif k in (:lt, :le, :gt, :ge, :eq, :ne)
        cs = node_children(node)
        if length(cs) != 2
            throw(SubsetError("Comparison requires 2 operands", node))
        end
        left = analyze_expr(cs[1], ctx)
        right = analyze_expr(cs[2], ctx)
        if left.typ != right.typ
            throw(SubsetError("Type mismatch in comparison", node))
        end
        return TypedExpr(k, T_Bool, [left, right], 1:span(node))

    elseif k == :call
        cs = node_children(node)
        if isempty(cs)
            throw(SubsetError("Empty call", node))
        end
        fname = node_text(cs[1])

        # Type conversion calls
        if fname in ("Int32", "Int64", "Float32", "Float64")
            if length(cs) != 2
                throw(SubsetError("Type conversion requires 1 argument", node))
            end
            arg = analyze_expr(cs[2], ctx)
            target = TYPE_MAP[fname]
            return TypedExpr(:convert, target, [arg, target], 1:span(node))
        end

        # User function calls
        if !haskey(ctx.functions, fname)
            throw(SubsetError("Unknown function: $fname", node))
        end
        sig = ctx.functions[fname]
        args = TypedExpr[]
        for i in 2:length(cs)
            push!(args, analyze_expr(cs[i], ctx))
        end
        if length(args) != length(sig.params)
            throw(SubsetError("Argument count mismatch for $fname", node))
        end
        for (i, (arg, (_, ptype))) in enumerate(zip(args, sig.params))
            if arg.typ != ptype
                throw(SubsetError("Type mismatch for argument $i of $fname", node))
            end
        end
        return TypedExpr(:call, sig.ret_type, [fname, args], 1:span(node))

    elseif k == :return
        cs = node_children(node)
        if isempty(cs)
            return TypedExpr(:return, T_Void, [], 1:span(node))
        end
        val = analyze_expr(cs[1], ctx)
        return TypedExpr(:return, val.typ, [val], 1:span(node))

    else
        throw(SubsetError("Unsupported expression: $k", node))
    end
end

# ============================================================================
# WASM Code Generation
# ============================================================================

"""LEB128 encoding for unsigned integers"""
function leb128_unsigned(value::Integer)::Vector{UInt8}
    result = UInt8[]
    while true
        byte = UInt8(value & 0x7f)
        value >>= 7
        if value != 0
            byte |= 0x80
        end
        push!(result, byte)
        if value == 0
            break
        end
    end
    result
end

"""LEB128 encoding for signed integers"""
function leb128_signed(value::Integer)::Vector{UInt8}
    result = UInt8[]
    more = true
    while more
        byte = UInt8(value & 0x7f)
        value >>= 7
        if (value == 0 && (byte & 0x40) == 0) || (value == -1 && (byte & 0x40) != 0)
            more = false
        else
            byte |= 0x80
        end
        push!(result, byte)
    end
    result
end

"""Generate WASM for a typed expression"""
function codegen_expr(expr::TypedExpr, ctx::CompileCtx)::Vector{UInt8}
    bytes = UInt8[]

    if expr.kind == :literal
        val = expr.children[1]
        if expr.typ == T_I32
            push!(bytes, 0x41)  # i32.const
            append!(bytes, leb128_signed(Int32(val)))
        elseif expr.typ == T_I64
            push!(bytes, 0x42)  # i64.const
            append!(bytes, leb128_signed(Int64(val)))
        elseif expr.typ == T_F32
            push!(bytes, 0x43)  # f32.const
            append!(bytes, reinterpret(UInt8, [Float32(val)]))
        elseif expr.typ == T_F64
            push!(bytes, 0x44)  # f64.const
            append!(bytes, reinterpret(UInt8, [Float64(val)]))
        elseif expr.typ == T_Bool
            push!(bytes, 0x41)  # i32.const
            push!(bytes, val ? 0x01 : 0x00)
        end

    elseif expr.kind == :var
        idx = expr.children[2]
        push!(bytes, 0x20)  # local.get
        append!(bytes, leb128_unsigned(idx))

    elseif expr.kind in (:add, :sub, :mul, :div, :rem)
        left, right = expr.children
        append!(bytes, codegen_expr(left, ctx))
        append!(bytes, codegen_expr(right, ctx))

        # Opcode depends on type
        op_map = if expr.typ == T_I32
            Dict(:add => 0x6a, :sub => 0x6b, :mul => 0x6c, :div => 0x6d, :rem => 0x6f)
        elseif expr.typ == T_I64
            Dict(:add => 0x7c, :sub => 0x7d, :mul => 0x7e, :div => 0x7f, :rem => 0x81)
        elseif expr.typ == T_F32
            Dict(:add => 0x92, :sub => 0x93, :mul => 0x94, :div => 0x95)
        elseif expr.typ == T_F64
            Dict(:add => 0xa0, :sub => 0xa1, :mul => 0xa2, :div => 0xa3)
        else
            Dict{Symbol, UInt8}()
        end

        if haskey(op_map, expr.kind)
            push!(bytes, op_map[expr.kind])
        else
            error("No opcode for $(expr.kind) with type $(expr.typ)")
        end

    elseif expr.kind in (:lt, :le, :gt, :ge, :eq, :ne)
        left, right = expr.children
        append!(bytes, codegen_expr(left, ctx))
        append!(bytes, codegen_expr(right, ctx))

        cmp_type = left.typ
        op_map = if cmp_type == T_I32
            Dict(:lt => 0x48, :le => 0x4c, :gt => 0x4a, :ge => 0x4e, :eq => 0x46, :ne => 0x47)
        elseif cmp_type == T_I64
            Dict(:lt => 0x53, :le => 0x57, :gt => 0x55, :ge => 0x59, :eq => 0x51, :ne => 0x52)
        elseif cmp_type == T_F32
            Dict(:lt => 0x5d, :le => 0x5f, :gt => 0x5e, :ge => 0x60, :eq => 0x5b, :ne => 0x5c)
        elseif cmp_type == T_F64
            Dict(:lt => 0x63, :le => 0x65, :gt => 0x64, :ge => 0x66, :eq => 0x61, :ne => 0x62)
        else
            Dict{Symbol, UInt8}()
        end
        push!(bytes, op_map[expr.kind])

    elseif expr.kind == :return
        if !isempty(expr.children)
            append!(bytes, codegen_expr(expr.children[1], ctx))
        end
        push!(bytes, 0x0f)  # return

    elseif expr.kind == :call
        fname = expr.children[1]
        args = expr.children[2]
        for arg in args
            append!(bytes, codegen_expr(arg, ctx))
        end
        # TODO: resolve function index
        push!(bytes, 0x10)  # call
        push!(bytes, 0x00)  # function index (placeholder)

    elseif expr.kind == :convert
        arg = expr.children[1]
        target = expr.children[2]
        append!(bytes, codegen_expr(arg, ctx))

        # Type conversion opcodes
        src = arg.typ
        if src == target
            # No-op
        elseif src == T_I64 && target == T_I32
            push!(bytes, 0xa7)  # i32.wrap_i64
        elseif src == T_I32 && target == T_I64
            push!(bytes, 0xac)  # i64.extend_i32_s
        elseif src == T_F64 && target == T_F32
            push!(bytes, 0xb6)  # f32.demote_f64
        elseif src == T_F32 && target == T_F64
            push!(bytes, 0xbb)  # f64.promote_f32
        elseif src == T_I32 && target == T_F32
            push!(bytes, 0xb2)  # f32.convert_i32_s
        elseif src == T_I32 && target == T_F64
            push!(bytes, 0xb7)  # f64.convert_i32_s
        elseif src == T_I64 && target == T_F32
            push!(bytes, 0xb4)  # f32.convert_i64_s
        elseif src == T_I64 && target == T_F64
            push!(bytes, 0xb9)  # f64.convert_i64_s
        elseif src == T_F32 && target == T_I32
            push!(bytes, 0xa8)  # i32.trunc_f32_s
        elseif src == T_F64 && target == T_I32
            push!(bytes, 0xaa)  # i32.trunc_f64_s
        elseif src == T_F32 && target == T_I64
            push!(bytes, 0xae)  # i64.trunc_f32_s
        elseif src == T_F64 && target == T_I64
            push!(bytes, 0xb0)  # i64.trunc_f64_s
        else
            error("Unsupported conversion: $src -> $target")
        end
    end

    bytes
end

# ============================================================================
# Public API
# ============================================================================

"""
    compile_subset(source::String) -> Vector{UInt8}

Compile Julia subset source code to WASM bytecode.
"""
function compile_subset(source::String)::Vector{UInt8}
    # Parse with JuliaSyntax
    tree = JuliaSyntax.parseall(SyntaxNode, source)

    ctx = CompileCtx()

    # First pass: collect function signatures
    for node in node_children(tree)
        if node_kind(node) == :function
            sig = analyze_function(node, ctx)
            ctx.functions[sig.name] = sig
        end
    end

    # TODO: Second pass: type-check and generate code
    # TODO: Build WASM module with sections

    # For now, return empty module
    UInt8[0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]
end

end # module
