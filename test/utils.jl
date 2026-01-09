# Test Utilities - Node.js Wasm Execution Harness
# This is the "Ground Truth" verification engine for TDD

using Test
import JSON

# ============================================================================
# Node.js Detection
# ============================================================================

"""
Check if Node.js is available and get the command.
Returns a tuple of (command, needs_experimental_flag).
Requires Node.js v20+ for WasmGC support.
- v20-22: WasmGC is experimental (needs --experimental-wasm-gc flag)
- v23+: WasmGC is stable (no flag needed)
"""
function detect_node()
    try
        version_str = read(`node --version`, String)
        # Parse version (format: v20.x.x)
        m = match(r"v(\d+)\.", version_str)
        if m !== nothing
            major_version = parse(Int, m.captures[1])
            if major_version >= 23
                # WasmGC is stable in v23+
                return (`node`, false)
            elseif major_version >= 20
                # WasmGC is experimental in v20-22
                return (`node`, true)
            else
                @warn "Node.js version $version_str found, but v20+ required for WasmGC"
                return (nothing, false)
            end
        end
        return (nothing, false)
    catch
        @warn "Node.js not found. Wasm execution tests will be skipped."
        return (nothing, false)
    end
end

const (NODE_CMD, NEEDS_EXPERIMENTAL_FLAG) = detect_node()

# ============================================================================
# Wasm Execution
# ============================================================================

"""
    run_wasm(wasm_bytes::Vector{UInt8}, func_name::String, args...) -> Any

Execute a WebAssembly function in Node.js and return the result.

# Arguments
- `wasm_bytes`: The compiled WebAssembly binary
- `func_name`: Name of the exported function to call
- `args...`: Arguments to pass to the function

# Returns
The result of the function call, parsed from JSON.
Returns `nothing` if Node.js is not available.
"""
function run_wasm(wasm_bytes::Vector{UInt8}, func_name::String, args...)
    if NODE_CMD === nothing
        @warn "Node.js not available. Skipping Wasm execution."
        return nothing
    end

    dir = mktempdir()
    wasm_path = joinpath(dir, "module.wasm")
    js_path = joinpath(dir, "loader.mjs")

    # Write the Wasm binary
    write(wasm_path, wasm_bytes)

    # Convert Julia args to JS args
    # Handle BigInt for 64-bit integers
    js_args = join(map(arg -> format_js_arg(arg), args), ", ")

    # Generate the loader script
    loader_script = """
import fs from 'fs';

const bytes = fs.readFileSync('$(escape_string(wasm_path))');

async function run() {
    try {
        const wasmModule = await WebAssembly.instantiate(bytes, {});
        const func = wasmModule.instance.exports['$func_name'];

        if (typeof func !== 'function') {
            console.error('Export "$func_name" is not a function');
            process.exit(1);
        }

        const result = func($js_args);

        // Handle BigInt serialization for JSON
        const serialized = JSON.stringify(result, (key, value) => {
            if (typeof value === 'bigint') {
                // Return as string with marker for parsing
                return { __bigint__: value.toString() };
            }
            return value;
        });

        console.log(serialized);
    } catch (e) {
        console.error('Wasm execution error:', e.message);
        process.exit(1);
    }
}

run();
"""

    open(js_path, "w") do io
        print(io, loader_script)
    end

    # Run Node.js (with experimental flag if needed for older versions)
    try
        node_cmd = NEEDS_EXPERIMENTAL_FLAG ? `$NODE_CMD --experimental-wasm-gc $js_path` : `$NODE_CMD $js_path`
        output = read(pipeline(node_cmd; stderr=stderr), String)
        output = strip(output)

        if isempty(output)
            return nothing
        end

        # Parse the JSON result
        result = JSON.parse(output)

        # Handle BigInt unmarshaling
        return unmarshal_result(result)
    catch e
        if e isa ProcessFailedException
            error("Wasm execution failed. Check stderr for details.")
        end
        rethrow()
    end
end

"""
Format a Julia argument for JavaScript code.
"""
function format_js_arg(arg)
    if arg isa Int64 || arg isa Int
        # Use BigInt with string argument to preserve precision
        # BigInt(number) loses precision for large numbers, but BigInt("string") doesn't
        return "BigInt(\"$(arg)\")"
    elseif arg isa Int32
        return string(arg)
    elseif arg isa Float64 || arg isa Float32
        return string(arg)
    else
        return repr(arg)
    end
end

"""
Unmarshal a JSON result, handling BigInt markers.
"""
function unmarshal_result(result)
    if result isa Dict && haskey(result, "__bigint__")
        return parse(Int64, result["__bigint__"])
    elseif result isa Vector
        return [unmarshal_result(r) for r in result]
    elseif result isa Dict
        return Dict(k => unmarshal_result(v) for (k, v) in result)
    else
        return result
    end
end

# ============================================================================
# Wasm Execution with Imports
# ============================================================================

"""
    run_wasm_with_imports(wasm_bytes, func_name, imports, args...) -> Any

Execute a WebAssembly function with JavaScript imports.

# Arguments
- `wasm_bytes`: The compiled WebAssembly binary
- `func_name`: Name of the exported function to call
- `imports`: Dict of module_name => Dict of field_name => JS function code
- `args...`: Arguments to pass to the function

# Example
```julia
imports = Dict("env" => Dict("log" => "(x) => console.log(x)"))
run_wasm_with_imports(bytes, "main", imports, Int32(42))
```
"""
function run_wasm_with_imports(wasm_bytes::Vector{UInt8}, func_name::String,
                               imports::Dict, args...)
    if NODE_CMD === nothing
        @warn "Node.js not available. Skipping Wasm execution."
        return nothing
    end

    dir = mktempdir()
    wasm_path = joinpath(dir, "module.wasm")
    js_path = joinpath(dir, "loader.mjs")

    # Write the Wasm binary
    write(wasm_path, wasm_bytes)

    # Convert Julia args to JS args
    js_args = join(map(arg -> format_js_arg(arg), args), ", ")

    # Build imports object
    imports_js = build_imports_js(imports)

    # Generate the loader script
    loader_script = """
import fs from 'fs';

const bytes = fs.readFileSync('$(escape_string(wasm_path))');

$imports_js

async function run() {
    try {
        const wasmModule = await WebAssembly.instantiate(bytes, importObject);
        const func = wasmModule.instance.exports['$func_name'];

        if (typeof func !== 'function') {
            console.error('Export "$func_name" is not a function');
            process.exit(1);
        }

        const result = func($js_args);

        // Handle BigInt serialization for JSON
        const serialized = JSON.stringify(result, (key, value) => {
            if (typeof value === 'bigint') {
                return { __bigint__: value.toString() };
            }
            return value;
        });

        console.log(serialized);
    } catch (e) {
        console.error('Wasm execution error:', e.message);
        process.exit(1);
    }
}

run();
"""

    open(js_path, "w") do io
        print(io, loader_script)
    end

    # Run Node.js
    try
        node_cmd = NEEDS_EXPERIMENTAL_FLAG ? `$NODE_CMD --experimental-wasm-gc $js_path` : `$NODE_CMD $js_path`
        output = read(pipeline(node_cmd; stderr=stderr), String)
        output = strip(output)

        if isempty(output)
            return nothing
        end

        result = JSON.parse(output)
        return unmarshal_result(result)
    catch e
        if e isa ProcessFailedException
            error("Wasm execution failed. Check stderr for details.")
        end
        rethrow()
    end
end

"""
Build JavaScript code for import object.
"""
function build_imports_js(imports::Dict)
    parts = String[]
    push!(parts, "const importObject = {")
    for (mod_name, fields) in imports
        push!(parts, "  \"$mod_name\": {")
        for (field_name, func_code) in fields
            push!(parts, "    \"$field_name\": $func_code,")
        end
        push!(parts, "  },")
    end
    push!(parts, "};")
    return join(parts, "\n")
end

# ============================================================================
# TDD Test Macros
# ============================================================================

"""
    @test_compile func_call

Test that compiling and running a Julia function in Wasm produces
the same result as running it natively in Julia.

# Example
```julia
my_add(a, b) = a + b
@test_compile my_add(1, 2)
```
"""
macro test_compile(func_call)
    quote
        # 1. Run in Julia (Ground Truth)
        expected = $(esc(func_call))

        # 2. Extract function and args
        f = $(esc(func_call.args[1]))
        args = ($(esc.(func_call.args[2:end])...),)
        arg_types = map(typeof, args)

        # 3. Compile to Wasm
        wasm_bytes = WasmTarget.compile(f, Tuple(arg_types))

        # 4. Run in Node
        actual = run_wasm(wasm_bytes, string(nameof(f)), args...)

        # 5. Verify
        if actual !== nothing
            @test actual == expected
        else
            @warn "Skipped Wasm verification (Node.js not available)"
        end
    end
end

"""
    @test_wasm_output wasm_bytes func_name args... expected

Test that running a Wasm binary produces the expected output.
Useful for testing hand-crafted Wasm binaries.
"""
macro test_wasm_output(wasm_bytes, func_name, args, expected)
    quote
        actual = run_wasm($(esc(wasm_bytes)), $(esc(func_name)), $(esc(args))...)
        if actual !== nothing
            @test actual == $(esc(expected))
        else
            @warn "Skipped Wasm verification (Node.js not available)"
        end
    end
end

# ============================================================================
# Debug Utilities
# ============================================================================

"""
    dump_wasm(wasm_bytes::Vector{UInt8}, path::String)

Write Wasm bytes to a file for debugging with external tools.
"""
function dump_wasm(wasm_bytes::Vector{UInt8}, path::String)
    write(path, wasm_bytes)
    println("Wrote $(length(wasm_bytes)) bytes to $path")
end

"""
    hexdump(bytes::Vector{UInt8})

Print bytes as hex for debugging.
"""
function hexdump(bytes::Vector{UInt8}; columns=16)
    for (i, b) in enumerate(bytes)
        print(string(b, base=16, pad=2), " ")
        if i % columns == 0
            println()
        end
    end
    if length(bytes) % columns != 0
        println()
    end
end
