# WasmTarget.jl Live Compilation Server
# Receives Julia code, compiles to Wasm, returns bytes

using Sockets
using JSON

# Add parent directory to load path
push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using WasmTarget

const PORT = 8081

# Store compiled functions for reuse
const compiled_cache = Dict{String, Vector{UInt8}}()

"""
Compile a Julia expression to Wasm and return the bytes.
"""
function compile_expr(code::String, arg_types_str::String)
    try
        # Parse the argument types
        arg_types = eval(Meta.parse("($arg_types_str)"))

        # Create a unique module for this compilation
        mod = Module()

        # Evaluate the function definition in the module
        func_expr = Meta.parse(code)
        Core.eval(mod, func_expr)

        # Get the function name
        if func_expr.head == :(=) && func_expr.args[1] isa Expr && func_expr.args[1].head == :call
            func_name = func_expr.args[1].args[1]
        elseif func_expr.head == :function
            func_name = func_expr.args[1].args[1]
        else
            return Dict("error" => "Could not determine function name")
        end

        # Get the function from the module
        func = getfield(mod, func_name)

        # Compile to Wasm
        wasm_bytes = WasmTarget.compile(func, arg_types)

        return Dict(
            "success" => true,
            "name" => string(func_name),
            "bytes" => collect(wasm_bytes),
            "size" => length(wasm_bytes)
        )
    catch e
        return Dict("error" => sprint(showerror, e))
    end
end

"""
Handle incoming HTTP request.
"""
function handle_request(client)
    try
        # Read the HTTP request
        request = ""
        while true
            line = readline(client)
            request *= line * "\n"
            if line == "" || line == "\r"
                break
            end
        end

        # Check for CORS preflight
        if startswith(request, "OPTIONS")
            response = """HTTP/1.1 200 OK\r
Access-Control-Allow-Origin: *\r
Access-Control-Allow-Methods: POST, GET, OPTIONS\r
Access-Control-Allow-Headers: Content-Type\r
Content-Length: 0\r
\r
"""
            write(client, response)
            return
        end

        # Parse content length
        content_length = 0
        for line in split(request, "\n")
            if startswith(lowercase(line), "content-length:")
                content_length = parse(Int, strip(split(line, ":")[2]))
            end
        end

        # Read body
        body = ""
        if content_length > 0
            body = String(read(client, content_length))
        end

        # Handle the request
        result = if startswith(request, "GET /health")
            Dict("status" => "ok", "version" => "0.1.0")
        elseif startswith(request, "POST /compile")
            data = JSON.parse(body)
            code = get(data, "code", "")
            types = get(data, "types", "")
            compile_expr(code, types)
        elseif startswith(request, "GET /")
            Dict("message" => "WasmTarget.jl Compilation Server", "endpoints" => ["/health", "/compile"])
        else
            Dict("error" => "Unknown endpoint")
        end

        # Send response
        json_response = JSON.json(result)
        response = """HTTP/1.1 200 OK\r
Content-Type: application/json\r
Access-Control-Allow-Origin: *\r
Content-Length: $(length(json_response))\r
\r
$json_response"""

        write(client, response)
    catch e
        error_response = JSON.json(Dict("error" => sprint(showerror, e)))
        response = """HTTP/1.1 500 Internal Server Error\r
Content-Type: application/json\r
Access-Control-Allow-Origin: *\r
Content-Length: $(length(error_response))\r
\r
$error_response"""
        write(client, response)
    finally
        close(client)
    end
end

function main()
    server = listen(PORT)
    println("WasmTarget.jl Compilation Server running on http://localhost:$PORT")
    println("Endpoints:")
    println("  GET  /health  - Health check")
    println("  POST /compile - Compile Julia code to Wasm")
    println("    Body: {\"code\": \"f(x::Int32) = x + 1\", \"types\": \"Int32\"}")
    println()
    println("Press Ctrl+C to stop")

    while true
        client = accept(server)
        @async handle_request(client)
    end
end

main()
