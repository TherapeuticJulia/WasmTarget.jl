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
| WasmGlobal{T, IDX} | global (phantom param) | Compiles to global.get/set |

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
21. **WasmGlobal{T, IDX}**: Type-safe global variable handles with compile-time indices
22. **Array Access in Loops**: Conditionals inside loops with array operations (dart2wasm patterns)
23. **SimpleDict**: Hash table with Int32 keys/values, linear probing (sd_new, sd_get, sd_set!, sd_haskey, sd_length)
24. **StringDict**: Hash table with String keys/Int32 values (sdict_new, sdict_get, sdict_set!, sdict_haskey, sdict_length)
25. **str_hash**: String hashing function for hash tables
26. **Primitive Types**: Custom primitive types (e.g., JuliaSyntax.Kind 16-bit)
27. **Union Type Discrimination**: `isa` operator for Union{Nothing, T} patterns
28. **Recursive Struct Registration**: Self-referential types like GreenNode
29. **Nested Struct Types**: Automatically register nested struct field types
30. **JuliaSyntax.jl Support**: 29+ functions from JuliaSyntax compile and run correctly
31. **Multi-edge OR Patterns**: `a || b || c` with 3+ conditions producing a phi node

### WasmGlobal - Direct Julia Compilation with Wasm Globals

The `WasmGlobal{T, IDX}` type enables directly compiling Julia functions that use Wasm global variables:

```julia
using WasmTarget

# Define global types with compile-time indices
const Counter = WasmGlobal{Int32, 0}   # Global index 0
const Flag = WasmGlobal{Int32, 1}      # Global index 1

# Write pure Julia functions - no Wasm knowledge required
function increment(g::Counter)::Int32
    g[] = g[] + Int32(1)
    return g[]
end

function toggle(f::Flag)::Int32
    f[] = f[] == Int32(0) ? Int32(1) : Int32(0)
    return f[]
end

# Compile to Wasm - globals auto-created, phantom params removed
bytes = compile_multi([
    (increment, (Counter,)),
    (toggle, (Flag,)),
])
```

Key features:
- **Type-level index**: `IDX` is a type parameter, known at compile time
- **Phantom parameter**: WasmGlobal args don't become Wasm function parameters
- **Auto-created globals**: Compiler automatically adds globals to the module
- **Julia semantics**: `g[]` and `g[] = x` work in Julia for testing
- **Multiple globals**: Use different type aliases for different indices
- **Multi-function sharing**: Functions in same `compile_multi` share globals

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

~259 tests organized in phases:
- Phase 1-3: Infrastructure, builder, compiler basics
- Phase 4-6: Control flow, integers, type conversions
- Phase 7-9: Structs, tuples, arrays
- Phase 10-12: JS imports, loops, recursion
- Phase 13-15: Struct access, floats, strings
- Phase 16-17: Multi-function modules, JS interop
- Phase 18: Tables, memory, and data segments
- Phase 19: SimpleDict (Int32 keys)
- Phase 20: StringDict (String keys)

## dart2wasm Parallel Architecture

WasmTarget.jl follows the same architecture as dart2wasm:

**dart2wasm pipeline:**
```
Dart source → CFE (parser) → Kernel IR → dart2wasm → WASM + .mjs runtime
```

**WasmTarget.jl pipeline:**
```
Julia source → Julia compiler → IR (Base.code_typed) → WasmTarget.jl → WASM
```

### Key Lessons from dart2wasm

1. **Use native compiler infrastructure**: dart2wasm reuses Dart's CFE for parsing/analysis. We reuse Julia's compiler via `Base.code_typed()`.

2. **Control flow compilation**: dart2wasm uses block-based control flow with `br_if` for complex patterns. We use nested if/else (simpler but works for most cases).
   - **Solved**: Julia's `||` and `&&` operators now compile natively. The compiler handles:
     - Simple `||` patterns (forward GotoNode to merge points)
     - Simple `&&` patterns (multiple GotoIfNot to same else target)
     - Combined `&&`/`||` patterns with PhiNode (boolean merge)

3. **WasmGC-first**: Both dart2wasm and WasmTarget.jl target WasmGC for automatic memory management and direct struct/array support.

4. **JS runtime companion**: dart2wasm outputs a `.mjs` file alongside WASM. We'll need similar for DOM bindings.

### Two Compilation Paths

1. **Build-time path (Therapy.jl apps)**: Compile user code at build time
   - Parse/analyze using Julia's compiler
   - Generate WASM via WasmTarget.jl
   - Ship pre-compiled WASM to browser

2. **Runtime path (Browser REPL)**: Compile Julia tools to WASM so the compiler runs in-browser
   - JuliaSyntax.jl in WASM for parsing
   - JuliaLowering.jl in WASM for lowering
   - WasmTarget.jl in WASM for codegen
   - Result: Like Rust Playground - write Julia, compile & execute entirely in browser, no server

## Roadmap: Path to dart2wasm Parity

The goal is full dart2wasm parity: being able to compile arbitrary Julia code to WASM for browser execution. Unlike dart2wasm (where the Dart compiler runs natively), our ultimate goal is to have WasmTarget.jl itself run in the browser as WASM - enabling a fully client-side Julia REPL like Rust Playground.

### Current Status (as of Jan 2026)

| Feature | dart2wasm | WasmTarget.jl | WebAssemblyCompiler.jl | WasmCompiler.jl |
|---------|-----------|---------------|------------------------|-----------------|
| Basic types, structs | ✅ | ✅ | ✅ | ✅ |
| 1D Arrays (Vector) | ✅ | ✅ | ✅ | ❌ |
| **Multi-dim Arrays** | ✅ | 🚧 HIGH PRIORITY | ❌ | ❌ |
| Control flow | ✅ | ✅ | ✅ | ✅ |
| JS interop (externref) | ✅ | ✅ | ✅ | ❌ |
| `\|\|`/`&&` operators | ✅ | ✅ | ✅ | ✅ |
| Try/catch exceptions | ✅ | ✅ | ❌ | ⚠️ WIP |
| Closures | ✅ | ✅ | ❌ | ❌ |
| Union{Nothing,T} | ✅ | ✅ | ❌ | ❌ |
| Multiple dispatch | ✅ | ⚠️ partial | ❌ | ❌ |
| Standard library | ✅ | ⚠️ Dict-like done | ✅ Dict | ❌ |
| JuliaSyntax.jl | N/A | ✅ 29+ functions | N/A | N/A |

**WasmTarget.jl advantages**: Exceptions, Closures, Union types, Multi-dim arrays (coming)
**Estimated parity: ~70%**

### Phase 1: Control Flow Completeness ✅ COMPLETE

Goal: Handle ALL Julia IR control flow patterns natively.

1. **Short-circuit operators** ✅ COMPLETE
   - `||` operator: Forward GotoNode to merge points
   - `&&` operator: Multiple GotoIfNot to same else target (block/br_if pattern)
   - Combined `&&`/`||`: PhiNode boolean merge

2. **Exception handling** ✅ COMPLETE
   - `try`/`catch` → WASM `try_table` with `catch_all`
   - `throw()` → WASM `throw` instruction with tag
   - Exception tag (void signature) auto-created in module
   - Handles if-statements inside try body
   - Multiple throw points supported

3. **All GotoIfNot patterns** ✅ COMPLETE
   - Every Julia IR pattern now compiles correctly

**Success criteria**: Can compile any Julia function without control flow errors. ✅ ACHIEVED

### Phase 2: Language Feature Completeness (Closures ✅ COMPLETE)

Goal: Support all Julia language features that dart2wasm supports for Dart.

1. **Closures** ✅ COMPLETE
   - Closure struct generation (WasmGC structs with captured fields)
   - Captured variable handling (struct.get for field access)
   - Closure creation via %new expression
   - Closure passing between functions
   - Julia inlines closure bodies when type is known at compile time
   - Multi-field closures supported (multiple captured variables)

2. **Array Access in Loops** ✅ COMPLETE
   - Conditionals inside loops with array element access
   - dart2wasm-style nested block/br patterns for inner conditionals
   - Dead code elimination for `@inbounds boundscheck(false)` patterns
   - MemoryRef types handled as virtual stack pairs (no local allocation)
   - Phi node value flow for inner conditional merge points
   - Correct loop phi vs inner conditional phi initialization

3. **Full multiple dispatch**: Runtime method lookup
   - Method tables
   - Dynamic dispatch via call_indirect

4. **Abstract types and unions**: Type hierarchy support
   - Union type handling
   - Abstract type dispatch

5. **Generators/iterators**: If needed for standard library

**Success criteria**: Can compile JuliaSyntax.jl's code patterns.

### Phase 3: Standard Library (In Progress)

Goal: Core Julia functionality available in WASM.

1. **String operations**: ✅ str_hash complete
2. **Collections**: ✅ SimpleDict + StringDict (hash tables with Int32/String keys)
3. **Math functions**: Comprehensive math library
4. **I/O abstractions**: Print, string formatting

**Success criteria**: Common Julia code "just works".

### Phase 4: Runtime & DOM

Goal: Browser integration matching dart2wasm's Flutter Web capabilities.

1. **DOM binding layer**: Typed wrappers for DOM APIs
2. **Event handling**: onclick, oninput, etc.
3. **JS runtime companion**: `.mjs` file like dart2wasm
4. **Async support**: Promises, event loop integration

**Success criteria**: Can build interactive web apps.

### Phase 5: Self-Hosting (REPL)

Goal: Julia compiler runs in the browser.

1. **Compile JuliaSyntax.jl → WASM**: Parser in browser
2. **Compile JuliaLowering.jl → WASM**: Lowering in browser
3. **Compile WasmTarget.jl → WASM**: Codegen in browser
4. **REPL UI**: Interactive Julia in browser

**Success criteria**: Type Julia code in browser, see results.

### Phase 6: Browser REPL for WasmTarget.jl Docs

Goal: Interactive documentation with live Julia-to-WASM compilation.

**Architecture**: Fully client-side (like Rust Playground)
- **Compiler in browser**: WasmTarget.jl itself compiled to WASM (from Phase 5)
- **No Julia server required**: Everything runs in the browser
- **UI built with Therapy.jl**: Homepage features a big Julia REPL terminal

This is the same architecture as Rust Playground (https://play.rust-lang.org/) where the compiler runs in the browser via WASM, not on a server.

1. **Self-hosted compiler**: Phase 5 enables WasmTarget.jl to run in browser as WASM
2. **REPL UI**: Terminal-style interface built with Therapy.jl, CodeMirror for editing
3. **Interactive execution**: Write Julia code → compile via WASM compiler → run result in browser
4. **Documenter.jl integration**: Homepage is the REPL, docs surround it

**Success criteria**: Users can write Julia code in browser, compile and execute it entirely client-side (no server roundtrip).

## TherapeuticJulia Ecosystem

WasmTarget.jl is part of the TherapeuticJulia ecosystem (sibling packages):

```
TherapeuticJulia/
├── WasmTarget.jl    # This package: Julia → WASM compiler (foundation)
├── Therapy.jl       # Reactive web framework (Leptos-style, uses WasmTarget)
└── Sessions.jl      # VSCode+Pluto hybrid IDE (uses Therapy.jl)
```

**Dependency hierarchy:**
1. **WasmTarget.jl** - Most fundamental. No dependencies on siblings.
2. **Therapy.jl** - Depends on WasmTarget.jl for compiling event handlers to WASM
3. **Sessions.jl** - Depends on Therapy.jl for UI components

**Important:** Therapy.jl is a **separate, already-functional package** with:
- Fine-grained reactivity (signals, effects, memos)
- JSX-style components (Div, Button, Span, etc.)
- SSR with hydration
- File-path routing
- Tailwind CSS integration
- Direct IR compilation to WASM via WasmTarget.jl

WasmTarget.jl's job is to provide the WASM compilation foundation. Therapy.jl handles the reactive framework concerns.

## Future Work (Post-Parity)

Once dart2wasm parity is achieved:

1. **Browser REPL**: Interactive Julia in WasmTarget.jl docs
2. **Sessions.jl maturity**: Full notebook IDE in browser
3. **Julia Package Ecosystem**: Compile popular packages to WASM
4. **Performance optimization**: Match dart2wasm's performance
5. **Developer tooling**: Source maps, debugging, hot reload

## Key Decisions and Rationale

### Why WasmGC instead of linear memory?
- Automatic garbage collection
- Direct struct/array support
- Better integration with JS GC
- Simpler code generation

### Closure Support
- Closures ARE supported (Phase 2 complete)
- Closure structs with captured variables compile to WasmGC structs
- Julia inlines closure bodies when type is known at compile time
- Works seamlessly with Therapy.jl event handlers

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
