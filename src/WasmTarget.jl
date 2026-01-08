module WasmTarget

# Builder - Low-level Wasm binary emitter
include("Builder/Types.jl")
include("Builder/Writer.jl")
include("Builder/Instructions.jl")

# Compiler - Julia IR to Wasm translation
include("Compiler/IR.jl")
include("Compiler/Codegen.jl")

# Runtime - Intrinsics and stdlib mapping
include("Runtime/Intrinsics.jl")

# Main API
export compile, WasmModule

"""
    compile(f, arg_types) -> Vector{UInt8}

Compile a Julia function `f` with the given argument types to WebAssembly bytes.
"""
function compile end

end # module
