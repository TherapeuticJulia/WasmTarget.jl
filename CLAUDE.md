# CLAUDE.md - Project Context and Development Guide

This document provides comprehensive context for AI assistants working on WasmTarget.jl, including its purpose, architecture, development history, and future direction.

## Project Vision

WasmTarget.jl is a Julia-to-WebAssembly compiler targeting WasmGC. It serves as the foundation for **Therapy.jl**, a planned reactive web framework that will bring Julia to the browser with a developer experience similar to React/SolidJS but with Julia's power and Leptos/Rust's performance characteristics.

### Long-Term Goals

1. **WasmTarget.jl** (this package): Core compiler from Julia to WasmGC
2. **Therapy.jl** (planned): Reactive web framework built on WasmTarget.jl
   - Fine-grained reactivity (SolidJS-inspired)
   - Full-stack capabilities (Leptos-inspired)
   - Julia macros for compile-time optimization (avoiding runtime closures)
3. **Reactive Notebooks** (planned): Pluto.jl alternative built as Therapy.jl components

### Design Philosophy

- **Leverage Julia's strengths**: Use macros and multiple dispatch instead of closures
- **Compile-time over runtime**: Move as much work as possible to compile time
- **Direct WebAssembly**: No intermediate languages, direct Julia IR to Wasm bytecode
- **WasmGC-first**: Use WebAssembly GC proposal for memory management

## Architecture

### Source Tree

```
src/
  WasmTarget.jl              # Entry point: compile(), compile_multi()

  Builder/                   # Low-level Wasm binary construction
    Types.jl                 # Type definitions (NumType, RefType, JSValue, etc.)
    Writer.jl                # Binary serialization (LEB128, sections)
    Instructions.jl          # Module building (WasmModule, add_function!, etc.)

  Compiler/                  # Julia to Wasm translation
    IR.jl                    # Julia IR extraction via Base.code_typed()
    Codegen.jl               # Main compilation logic (~2000 lines)
                             # - TypeRegistry: struct/array type mappings
                             # - FunctionRegistry: multi-function module support
                             # - CompilationContext: per-function state
                             # - Basic block analysis and control flow
                             # - SSA value tracking and local allocation
                             # - Statement compilation (calls, returns, etc.)

  Runtime/
    Intrinsics.jl            # Julia intrinsic to Wasm opcode mapping

test/
  runtests.jl                # Comprehensive test suite (~1350 lines)
  utils.jl                   # Node.js execution harness
```

### Key Data Structures

#### TypeRegistry (Codegen.jl)
Maps Julia types to WasmGC type indices:
- `structs`: Dict{DataType, StructInfo} - user structs and tuples
- `arrays`: Dict{Type, UInt32} - array element type to type index
- `string_array_idx`: String type index (i32 array)

#### FunctionRegistry (Codegen.jl)
Enables cross-function calls in multi-function modules:
- `functions`: Dict{String, FunctionInfo} - name to info
- `by_ref`: Dict{Any, Vector{FunctionInfo}} - function reference to overloads

#### CompilationContext (Codegen.jl)
Per-function compilation state:
- `code_info`: Julia IR from code_typed
- `ssa_types`: SSA value type inference
- `ssa_locals`: Multi-use SSA values get Wasm locals
- `phi_locals`: Phi node allocations
- `loop_headers`: Backward jump targets
- `func_registry`: For cross-function call resolution

### Compilation Flow

1. `compile(f, arg_types)` or `compile_multi(functions)`
2. Get Julia IR via `Base.code_typed(f, arg_types)`
3. Analyze control flow (loops, branches, phi nodes)
4. Allocate locals for SSA values that need them
5. Generate Wasm bytecode for each statement
6. Serialize to binary format

## Type Mappings

| Julia | Wasm | Notes |
|-------|------|-------|
| Int32, UInt32 | i32 | |
| Int64, UInt64, Int | i64 | |
| Float32 | f32 | |
| Float64 | f64 | |
| Bool | i32 | 0 or 1 |
| String | WasmGC array (i32) | One element per character |
| User struct | WasmGC struct | Fields map directly |
| Tuple{...} | WasmGC struct | Immutable |
| Vector{T} | WasmGC array | Mutable |
| JSValue | externref | For JS object references |

## Development History

### Completed Features (as of current state)

1. **Basic Types**: i32, i64, f32, f64, bool
2. **Arithmetic**: add, sub, mul, div, rem, and, or, xor, shifts
3. **Comparisons**: eq, ne, lt, le, gt, ge (signed and unsigned)
4. **Control Flow**: if/else, loops (while, for), branches
5. **Structs**: WasmGC struct creation, field access
6. **Tuples**: As immutable WasmGC structs
7. **Arrays**: WasmGC arrays with get/set/length
8. **Strings**: As i32 arrays, sizeof/length, concatenation, equality
9. **Recursion**: Self-recursive functions
10. **Multi-Function Modules**: compile_multi() API
11. **Cross-Function Calls**: Functions calling other functions in same module
12. **Multiple Dispatch**: Same function name, different type signatures
13. **JS Interop**: externref (JSValue type), imports with RefTypes
14. **Wasm Globals**: Mutable/immutable globals, exported to JS
15. **Callbacks**: Exported functions work as JS event handlers
16. **Tables**: funcref and externref tables with element segments
17. **Indirect Calls**: call_indirect for dynamic dispatch
18. **Linear Memory**: Memory sections with load/store operations
19. **Data Segments**: Initialize memory with constant data (strings, bytes)
20. **Result Type Patterns**: Error handling with custom result structs and control flow

### Therapy.jl Pattern Demo

The counter example in `/tmp/counter.wasm` demonstrates the core pattern:

```julia
# Globals serve as reactive state (like Signals)
global_count = add_global!(mod, I32, true, 0)

# Exported functions serve as event handlers
# increment() modifies global, JS calls it on click
add_export!(mod, "increment", 0, func_idx)

# JS imports for DOM manipulation
add_import!(mod, "dom", "set_text_content", [ExternRef, I32], [])
```

This pattern enables Therapy.jl to:
- Store reactive state in Wasm globals
- Generate event handlers as exported functions (via macros)
- Call DOM APIs through imports

### Test Coverage

~205 tests organized in phases:
- Phase 1-3: Infrastructure, builder, compiler basics
- Phase 4-6: Control flow, integers, type conversions
- Phase 7-9: Structs, tuples, arrays
- Phase 10-12: JS imports, loops, recursion
- Phase 13-15: Struct access, floats, strings
- Phase 16-17: Multi-function modules, JS interop
- Phase 18: Tables, memory, and data segments

## Future Work (Roadmap for Therapy.jl)

### Near-Term (WasmTarget.jl)

1. **More String Operations**: Concatenation, comparison
2. **Better Error Messages**: Source location tracking
3. **funcref Support**: ref.func and function reference passing

### Medium-Term (Therapy.jl Foundation)

1. **DOM Binding Layer**: Typed wrappers for DOM APIs
2. **Signal Primitives**: Reactive state management
3. **Component Macros**: @component, @effect, @html
4. **Event Handling**: onclick, oninput, etc.

### Long-Term (Full Therapy.jl)

1. **Server Functions**: RPC to Julia backend
2. **SSR Support**: Server-side rendering
3. **Hydration**: Client-side takeover of SSR HTML
4. **Routing**: Client-side navigation
5. **Reactive Notebooks**: Pluto alternative

## Key Decisions and Rationale

### Why WasmGC instead of linear memory?
- Automatic garbage collection
- Direct struct/array support
- Better integration with JS GC
- Simpler code generation

### Why no closures?
- Julia IR for closures is complex (environment capture)
- Macros can achieve same goals at compile time
- Better performance (no runtime environment lookup)

### Why primitive type for JSValue?
- Empty structs are optimized away by Julia
- Primitive type preserves the value in IR
- 64-bit matches pointer size

### Why i32 arrays for strings?
- WasmGC packed arrays (i8) need special handling
- i32 is simpler and works for Unicode codepoints
- Can optimize later if needed

## Common Issues and Solutions

### "Unsupported method: X"
The function calls a Julia method not yet implemented in `compile_invoke` or `compile_call`. Add handling for it in Codegen.jl.

### Functions defined in @testset don't work
Functions inside @testset become closures. Define test functions at module level.

### Type mismatch errors
Check `get_concrete_wasm_type` returns correct WasmGC type. May need to register types in TypeRegistry.

### Cross-function calls fail
Ensure both functions are in same `compile_multi` call and function references match.

## Testing Commands

```bash
# Run all tests
julia --project=. test/runtests.jl

# Run specific test (edit runtests.jl to focus)
julia --project=. -e 'using WasmTarget; ...'

# Test wasm manually with Node.js
node -e 'const bytes = require("fs").readFileSync("test.wasm"); WebAssembly.instantiate(bytes).then(m => console.log(m.instance.exports))'
```

## Key Files to Read

1. `src/Compiler/Codegen.jl` - Main compilation logic
2. `src/Builder/Types.jl` - Type definitions and mappings
3. `test/runtests.jl` - Examples of all supported features
4. `test/utils.jl` - Node.js execution harness

## Contact and Context

This project is being developed as part of a larger vision to bring Julia to the web with a modern reactive framework. The design prioritizes Julia's unique strengths (multiple dispatch, macros, type inference) over copying patterns from other languages that might not fit Julia well.
