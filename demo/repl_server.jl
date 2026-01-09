# WasmTarget.jl REPL Server
# Honest Julia-to-Wasm compilation - no fallbacks, no cheating

using Sockets
using JSON

push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using WasmTarget

const PORT = 8081

# REPL state
mutable struct REPLState
    mod::Module
    functions::Dict{Symbol, Tuple{Any, Tuple, Vector{UInt8}}}  # name -> (func, arg_types, wasm_bytes)
    variables::Dict{Symbol, Tuple{Any, Type}}   # name -> (value, type)
end

REPLState() = REPLState(Module(), Dict{Symbol, Tuple{Any, Tuple, Vector{UInt8}}}(), Dict{Symbol, Tuple{Any, Type}}())

const repl_state = REPLState()

"""
Map Julia types to Wasm-compatible types.
"""
function normalize_type(T::Type)
    if T === Int || T === Int64
        Int32  # Use Int32 for Wasm compatibility
    elseif T === Float64
        Float64
    elseif T === Float32
        Float32
    elseif T === Bool
        Int32
    else
        T
    end
end

"""
Resolve a type expression to a Julia type.
"""
function resolve_type(type_expr)
    # Handle common types directly
    if type_expr === :Int32
        return Int32
    elseif type_expr === :Int64
        return Int64
    elseif type_expr === :Int
        return Int32  # Normalize to Int32 for Wasm
    elseif type_expr === :Float32
        return Float32
    elseif type_expr === :Float64
        return Float64
    elseif type_expr === :Bool
        return Int32
    else
        # Try to evaluate in Main
        try
            return Core.eval(Main, type_expr)
        catch
            return nothing
        end
    end
end

"""
Extract argument types from a function definition.
Returns nothing if types are not annotated.
"""
function extract_arg_types(func_expr)
    # Get the signature part
    if func_expr.head == :(=) && func_expr.args[1] isa Expr && func_expr.args[1].head == :call
        sig = func_expr.args[1]
    elseif func_expr.head == :function
        sig = func_expr.args[1]
    else
        return nothing, nothing
    end

    func_name = sig.args[1]
    args = sig.args[2:end]

    types = Type[]
    for arg in args
        if arg isa Expr && arg.head == :(::)
            # Has type annotation: a::Int32
            type_expr = arg.args[2]
            T = resolve_type(type_expr)
            if T === nothing
                return func_name, nothing
            end
            push!(types, normalize_type(T))
        else
            # No type annotation - can't compile to Wasm
            return func_name, nothing
        end
    end

    return func_name, tuple(types...)  # Return tuple instance, not Tuple type
end

"""
Evaluate a REPL expression.
Returns: (type, data) where type is "result", "wasm", or "error"
"""
function eval_repl(code::String)
    code = strip(code)
    isempty(code) && return ("result", "")

    try
        expr = Meta.parse(code)

        if expr isa Expr && (expr.head == :(=) || expr.head == :function)
            lhs = expr.args[1]

            if lhs isa Expr && lhs.head == :call
                # Function definition
                return handle_function_def(expr)
            elseif lhs isa Symbol
                # Variable assignment
                return handle_assignment(expr)
            end
        elseif expr isa Expr && expr.head == :call
            # Function call
            return handle_function_call(expr)
        else
            # Simple expression - evaluate for display only
            # But be honest: this runs in Julia, not Wasm
            result = Core.eval(repl_state.mod, expr)
            return ("result", repr(result) * "  # evaluated in Julia (not Wasm)")
        end
    catch e
        return ("error", sprint(showerror, e))
    end
end

"""
Handle function definition with type annotations.
"""
function handle_function_def(expr)
    func_name, arg_types = extract_arg_types(expr)

    if arg_types === nothing
        return ("error", "Functions require type annotations for Wasm compilation.\nExample: add(a::Int32, b::Int32) = a + b")
    end

    try
        # Define the function in Main so types resolve correctly
        Core.eval(Main, expr)
        func = getfield(Main, func_name)

        # Compile to Wasm using invokelatest to handle world age
        wasm_bytes = Base.invokelatest(WasmTarget.compile, func, arg_types)

        # Store function and pre-compiled Wasm bytes
        repl_state.functions[func_name] = (func, arg_types, collect(wasm_bytes))

        return ("result", "$func_name (compiled to $(length(wasm_bytes)) bytes of Wasm)")
    catch e
        # Include full stack trace for debugging
        io = IOBuffer()
        showerror(io, e, catch_backtrace())
        return ("error", "Wasm compilation failed: " * String(take!(io)))
    end
end

"""
Handle variable assignment.
"""
function handle_assignment(expr)
    var_name = expr.args[1]
    val_expr = expr.args[2]

    try
        val = Core.eval(repl_state.mod, val_expr)
        T = normalize_type(typeof(val))
        val = convert(T, val)

        Core.eval(repl_state.mod, :($var_name = $val))
        repl_state.variables[var_name] = (val, T)

        return ("result", repr(val))
    catch e
        return ("error", sprint(showerror, e))
    end
end

"""
Handle function call - send pre-compiled Wasm to browser for execution.
"""
function handle_function_call(expr)
    func_name = expr.args[1]
    arg_exprs = expr.args[2:end]

    if !haskey(repl_state.functions, func_name)
        return ("error", "Function `$func_name` not defined.\nDefine with types: $func_name(a::Int32, b::Int32) = ...")
    end

    func, expected_types, wasm_bytes = repl_state.functions[func_name]

    try
        # Evaluate arguments
        args = []
        for (i, arg_expr) in enumerate(arg_exprs)
            val = if arg_expr isa Symbol && haskey(repl_state.variables, arg_expr)
                repl_state.variables[arg_expr][1]
            elseif arg_expr isa Number
                arg_expr
            else
                Core.eval(Main, arg_expr)
            end

            # Convert to expected type
            expected_type = expected_types[i]
            push!(args, convert(expected_type, val))
        end

        # Send pre-compiled Wasm bytes to browser for execution
        return ("wasm", Dict(
            "bytes" => wasm_bytes,
            "args" => args,
            "func_name" => string(func_name),
            "size" => length(wasm_bytes)
        ))
    catch e
        return ("error", sprint(showerror, e))
    end
end

"""
Reset REPL state.
"""
function reset_repl!()
    repl_state.mod = Module()
    empty!(repl_state.functions)
    empty!(repl_state.variables)
end

"""
Handle HTTP request.
"""
function handle_request(client)
    try
        # Read headers
        request = ""
        content_length = 0
        while true
            line = readline(client)
            request *= line * "\n"
            if startswith(lowercase(line), "content-length:")
                content_length = parse(Int, strip(split(line, ":")[2]))
            end
            if line == "" || line == "\r"
                break
            end
        end

        # CORS preflight
        if startswith(request, "OPTIONS")
            write(client, "HTTP/1.1 200 OK\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: POST, GET, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type\r\nContent-Length: 0\r\n\r\n")
            return
        end

        body = content_length > 0 ? String(read(client, content_length)) : ""

        response_data = if startswith(request, "POST /eval")
            data = JSON.parse(body)
            type, result = eval_repl(get(data, "code", ""))
            if type == "wasm"
                Dict("type" => "wasm", "wasm" => result["bytes"], "args" => result["args"],
                     "func_name" => result["func_name"], "size" => result["size"])
            elseif type == "error"
                Dict("type" => "error", "message" => result)
            else
                Dict("type" => "result", "value" => result)
            end
        elseif startswith(request, "POST /reset")
            reset_repl!()
            Dict("type" => "result", "value" => "REPL state cleared")
        elseif startswith(request, "GET /health")
            Dict("status" => "ok", "version" => "0.3.0", "mode" => "wasm-only")
        else
            Dict("type" => "error", "message" => "Unknown endpoint")
        end

        json_response = JSON.json(response_data)
        write(client, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: $(length(json_response))\r\n\r\n$json_response")
    catch e
        error_json = JSON.json(Dict("type" => "error", "message" => sprint(showerror, e)))
        write(client, "HTTP/1.1 500 Error\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: $(length(error_json))\r\n\r\n$error_json")
    finally
        close(client)
    end
end

function main()
    server = listen(PORT)
    println("═══════════════════════════════════════════════════════════")
    println("  WasmTarget.jl REPL Server (Wasm-only mode)")
    println("  http://localhost:$PORT")
    println("═══════════════════════════════════════════════════════════")
    println("  Functions MUST have type annotations to compile to Wasm")
    println("  Example: add(a::Int32, b::Int32) = a + b")
    println("═══════════════════════════════════════════════════════════")

    while true
        client = accept(server)
        @async handle_request(client)
    end
end

main()
