# WasmTarget.jl

A Julia-to-WebAssembly compiler targeting the WasmGC (Garbage Collection) proposal. This package compiles Julia functions directly to WebAssembly binaries that can run in modern browsers and Node.js environments with WasmGC support.

## Features

- **Direct Compilation**: Compile Julia functions to WebAssembly without intermediate languages
- **WasmGC Support**: Uses WebAssembly GC proposal for structs, arrays, and reference types
- **Type Support**: Integers (32/64-bit), floats, booleans, strings, structs, tuples, and arrays
- **Control Flow**: Full support for loops, recursion, branches, and phi nodes
- **Multi-Function Modules**: Compile multiple functions into a single module with cross-function calls
- **Multiple Dispatch**: Same function name with different type signatures dispatches correctly
- **JS Interop**: `externref` support for holding JavaScript objects, import JS functions
- **Tables**: Function reference tables for indirect calls and dynamic dispatch
- **Linear Memory**: Memory sections with load/store operations and data initialization
- **Globals**: Mutable and immutable global variables, exportable to JS
- **String Operations**: String concatenation (`*`) and equality comparison (`==`)
- **Result Type Patterns**: Error handling via custom result structs with control flow

## Requirements

- Julia 1.9+
- Node.js 20+ for testing (v23+ recommended for stable WasmGC support)

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/your-username/WasmTarget.jl")
```

## Quick Start

### Single Function

```julia
using WasmTarget

# Define a function
@noinline function add(a::Int32, b::Int32)::Int32
    return a + b
end

# Compile to WebAssembly
wasm_bytes = compile(add, (Int32, Int32))

# Write to file
write("add.wasm", wasm_bytes)
```

### Multiple Functions

```julia
using WasmTarget

@noinline function helper(x::Int32)::Int32
    return x * Int32(2)
end

@noinline function use_helper(x::Int32)::Int32
    return helper(x) + Int32(1)
end

# Compile multiple functions into one module
wasm_bytes = compile_multi([
    (helper, (Int32,)),
    (use_helper, (Int32,)),
])
```

### Running in JavaScript

```javascript
const fs = require('fs');
const bytes = fs.readFileSync('add.wasm');

WebAssembly.instantiate(bytes).then(mod => {
    console.log(mod.instance.exports.add(5, 3)); // Output: 8
});
```

## Supported Types

| Julia Type | WebAssembly Type |
|------------|------------------|
| `Int32`, `UInt32` | `i32` |
| `Int64`, `UInt64`, `Int` | `i64` |
| `Float32` | `f32` |
| `Float64` | `f64` |
| `Bool` | `i32` (0 or 1) |
| `String` | WasmGC array of i32 |
| User structs | WasmGC struct |
| `Tuple{...}` | WasmGC struct |
| `Vector{T}` | WasmGC array |
| `JSValue` | `externref` |

## Advanced Usage

### Structs

```julia
mutable struct Point
    x::Int32
    y::Int32
end

@noinline function point_sum(p::Point)::Int32
    return p.x + p.y
end

wasm_bytes = compile(point_sum, (Point,))
```

### JavaScript Interop

```julia
using WasmTarget

# JSValue represents a JavaScript object (externref)
@noinline function process_js(obj::JSValue)::JSValue
    return obj
end

wasm_bytes = compile(process_js, (JSValue,))
```

### Custom Exports

```julia
# Give a custom export name
wasm_bytes = compile_multi([
    (my_function, (Int32,), "customName"),
])
```

### Low-Level API

For advanced use cases, you can use the Builder API directly:

```julia
using WasmTarget
using WasmTarget: WasmModule, add_import!, add_function!, add_export!,
                  add_global!, add_global_export!, add_table!, add_memory!,
                  add_data_segment!, to_bytes, I32, FuncRef, Opcode

mod = WasmModule()

# Add imports
add_import!(mod, "env", "log", [I32], [])

# Add globals for state
count_idx = add_global!(mod, I32, true, 0)  # mutable i32 initialized to 0
add_global_export!(mod, "count", count_idx)

# Add tables for function references
table_idx = add_table!(mod, FuncRef, 4)  # table of 4 funcrefs

# Add linear memory
mem_idx = add_memory!(mod, 1)  # 1 page (64KB)
add_data_segment!(mod, 0, 0, "Hello, World!")  # initialize with string

# Build module
bytes = to_bytes(mod)
```

## Architecture

```
src/
  WasmTarget.jl          # Main entry point, compile() API
  Builder/
    Types.jl             # Wasm type definitions (NumType, RefType, etc.)
    Writer.jl            # Binary serialization to .wasm format
    Instructions.jl      # Module building, function/type management
  Compiler/
    IR.jl                # Julia IR extraction via code_typed
    Codegen.jl           # IR to Wasm bytecode translation
  Runtime/
    Intrinsics.jl        # Julia intrinsic to Wasm opcode mapping
```

## Limitations

- No closures (use macros for compile-time code generation instead)
- No exceptions (use Result-type patterns - now supported!)
- No async/await (use callbacks)
- Limited Base Julia coverage (focused on core primitives)
- String indexing not supported (Julia's UTF-8 IR is too complex)
- Array resize operations (`push!`, `pop!`) not supported (WasmGC arrays are fixed-size)

## Examples

See the `examples/` directory for practical demos:

- **counter_demo.jl**: Interactive counter showing the Therapy.jl reactive pattern
  - Wasm globals for state management
  - Exported functions as event handlers
  - JS imports for DOM manipulation

Run the counter demo:
```bash
julia --project=. examples/counter_demo.jl
cd examples && python3 -m http.server 8080
# Open http://localhost:8080/counter.html
```

## Testing

```bash
julia --project=. test/runtests.jl
```

Tests require Node.js 20+ for WasmGC execution.

## Related Projects

This package is designed as the foundation for **Therapy.jl**, a reactive web framework inspired by Leptos (Rust) and SolidJS, bringing Julia to the browser with fine-grained reactivity.

## License

MIT License
