# Julia IR Handling
# Interface to Julia's typed intermediate representation

export get_typed_ir, IRStatement

"""
    get_typed_ir(f, arg_types)

Get Julia's typed IR (SSA form) for a function with given argument types.
Returns the CodeInfo object from code_typed.
"""
function get_typed_ir(f, arg_types::Tuple)
    # Get the typed IR using Julia's introspection
    results = Base.code_typed(f, arg_types; optimize=true)

    if isempty(results)
        error("No method found for $f with types $arg_types")
    end

    # Return the first (and usually only) result
    code_info, return_type = results[1]
    return code_info, return_type
end

"""
Extract the return type from IR.
"""
function get_return_type(code_info)
    # The return type is in the slottypes or can be inferred from the last statement
    return code_info.rettype
end

"""
Get parameter types from code_info.
"""
function get_param_types(code_info)
    # First slot is the function itself, rest are parameters
    # slottypes[1] is typeof(f), slottypes[2:end] are param types
    if hasfield(typeof(code_info), :slottypes) && code_info.slottypes !== nothing
        return code_info.slottypes[2:end]
    else
        return []
    end
end
