# ExprEval.jl - Tree-walking evaluator for Julia Expr ASTs
#
# This provides a concrete ExprNode representation that WasmTarget can compile
# to WasmGC, plus a host-side conversion from Julia Expr to ExprNode arrays.
#
# Design:
# - ExprNode is a flat struct with index-based child references (no Vector{Any})
# - Nodes are stored in a flat Vector{ExprNode} array
# - eval_node() walks the tree recursively using indices
# - expr_to_nodes() converts Julia Expr to the flat representation (host-side only)

export ExprNode, EvalEnv, eval_node, expr_to_nodes, eval_expr

"""
Flat AST node representation for WasmGC compilation.
Children are referenced by 1-based index into the nodes array.
"""
mutable struct ExprNode
    tag::Int32       # 0=literal, 1=add, 2=sub, 3=mul, 4=assign, 5=block, 6=varref, 7=ifnode
    value::Int64     # For literals: the value. For varref/assign: variable slot index
    child1::Int32    # Index into node array (1-based, 0 = no child)
    child2::Int32    # Index into node array (1-based, 0 = no child)
    child3::Int32    # Index into node array (1-based, 0 = no child)
end

"""
Variable environment for the Expr evaluator: fixed-size slot storage for variable bindings.
"""
mutable struct EvalEnv
    slots::Vector{Int64}
end

"""
    eval_node(nodes::Vector{ExprNode}, idx::Int32, env::EvalEnv) -> Int64

Evaluate the ExprNode at index `idx` in the `nodes` array.
This function is designed to compile to WasmGC via WasmTarget.
"""
function eval_node(nodes::Vector{ExprNode}, idx::Int32, env::EvalEnv)::Int64
    node = nodes[idx]
    tag = node.tag

    if tag == Int32(0)
        # Literal: return the value directly
        return node.value
    elseif tag == Int32(1)
        # Add: left + right
        left = eval_node(nodes, node.child1, env)
        right = eval_node(nodes, node.child2, env)
        return left + right
    elseif tag == Int32(2)
        # Sub: left - right
        left = eval_node(nodes, node.child1, env)
        right = eval_node(nodes, node.child2, env)
        return left - right
    elseif tag == Int32(3)
        # Mul: left * right
        left = eval_node(nodes, node.child1, env)
        right = eval_node(nodes, node.child2, env)
        return left * right
    elseif tag == Int32(4)
        # Assign: evaluate value, store in slot
        val = eval_node(nodes, node.child1, env)
        slot = Int(node.value)
        env.slots[slot] = val
        return val
    elseif tag == Int32(5)
        # Block: evaluate children in sequence, return last
        result = Int64(0)
        if node.child1 != Int32(0)
            result = eval_node(nodes, node.child1, env)
        end
        if node.child2 != Int32(0)
            result = eval_node(nodes, node.child2, env)
        end
        if node.child3 != Int32(0)
            result = eval_node(nodes, node.child3, env)
        end
        return result
    elseif tag == Int32(6)
        # Varref: read from slot
        slot = Int(node.value)
        return env.slots[slot]
    elseif tag == Int32(7)
        # If: condition, then-branch, else-branch
        cond = eval_node(nodes, node.child1, env)
        if cond != Int64(0)
            return eval_node(nodes, node.child2, env)
        else
            return eval_node(nodes, node.child3, env)
        end
    else
        return Int64(-1)
    end
end

# ============================================================================
# Host-side conversion: Expr -> ExprNode array
# (This part runs in Julia, NOT compiled to WasmGC)
# ============================================================================

"""
    expr_to_nodes(expr) -> (nodes::Vector{ExprNode}, root_idx::Int32, num_slots::Int32)

Convert a Julia Expr (or literal) to a flat ExprNode array.
Returns the node array, root index, and number of variable slots needed.
"""
function expr_to_nodes(expr)
    nodes = ExprNode[]
    var_map = Dict{Symbol, Int32}()
    next_slot = Ref(Int32(1))

    function get_slot(s::Symbol)::Int32
        if haskey(var_map, s)
            return var_map[s]
        else
            slot = next_slot[]
            next_slot[] += Int32(1)
            var_map[s] = slot
            return slot
        end
    end

    function convert_expr(e)::Int32
        if e isa Integer
            push!(nodes, ExprNode(Int32(0), Int64(e), Int32(0), Int32(0), Int32(0)))
            return Int32(length(nodes))
        elseif e isa Bool
            push!(nodes, ExprNode(Int32(0), e ? Int64(1) : Int64(0), Int32(0), Int32(0), Int32(0)))
            return Int32(length(nodes))
        elseif e isa Symbol
            slot = get_slot(e)
            push!(nodes, ExprNode(Int32(6), Int64(slot), Int32(0), Int32(0), Int32(0)))
            return Int32(length(nodes))
        elseif e isa Expr
            if e.head == :call
                op = e.args[1]
                if op == :+ || op == :(+)
                    left = convert_expr(e.args[2])
                    right = convert_expr(e.args[3])
                    push!(nodes, ExprNode(Int32(1), Int64(0), left, right, Int32(0)))
                    return Int32(length(nodes))
                elseif op == :- || op == :(-)
                    left = convert_expr(e.args[2])
                    right = convert_expr(e.args[3])
                    push!(nodes, ExprNode(Int32(2), Int64(0), left, right, Int32(0)))
                    return Int32(length(nodes))
                elseif op == :* || op == :(*)
                    left = convert_expr(e.args[2])
                    right = convert_expr(e.args[3])
                    push!(nodes, ExprNode(Int32(3), Int64(0), left, right, Int32(0)))
                    return Int32(length(nodes))
                end
            elseif e.head == :(=)
                lhs = e.args[1]::Symbol
                slot = get_slot(lhs)
                rhs_idx = convert_expr(e.args[2])
                push!(nodes, ExprNode(Int32(4), Int64(slot), rhs_idx, Int32(0), Int32(0)))
                return Int32(length(nodes))
            elseif e.head == :block
                # Filter out LineNumberNodes
                stmts = filter(a -> !(a isa LineNumberNode), e.args)
                c1 = length(stmts) >= 1 ? convert_expr(stmts[1]) : Int32(0)
                c2 = length(stmts) >= 2 ? convert_expr(stmts[2]) : Int32(0)
                c3 = length(stmts) >= 3 ? convert_expr(stmts[3]) : Int32(0)
                push!(nodes, ExprNode(Int32(5), Int64(0), c1, c2, c3))
                return Int32(length(nodes))
            elseif e.head == :if
                cond_idx = convert_expr(e.args[1])
                then_idx = convert_expr(e.args[2])
                else_idx = length(e.args) >= 3 ? convert_expr(e.args[3]) : Int32(0)
                push!(nodes, ExprNode(Int32(7), Int64(0), cond_idx, then_idx, else_idx))
                return Int32(length(nodes))
            end
        end
        # Fallback: literal 0
        push!(nodes, ExprNode(Int32(0), Int64(0), Int32(0), Int32(0), Int32(0)))
        return Int32(length(nodes))
    end

    root_idx = convert_expr(expr)
    num_slots = max(Int32(1), next_slot[] - Int32(1))
    return nodes, root_idx, num_slots
end

"""
    eval_expr(expr) -> Int64

Evaluate a Julia Expr by converting to ExprNode array and walking the tree.
This is the user-facing API that combines conversion + evaluation.
"""
function eval_expr(expr)::Int64
    nodes, root_idx, num_slots = expr_to_nodes(expr)
    env = EvalEnv(zeros(Int64, Int(num_slots)))
    return eval_node(nodes, root_idx, env)
end
