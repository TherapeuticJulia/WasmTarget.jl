# WasmTarget.jl - Supported Julia Subset

This document defines what Julia code WasmTarget.jl can compile to WebAssembly.
**We use Julia's full type inference** - no explicit type annotations required.

## Comparison to Other Julia-to-WASM Compilers

| Feature | WasmTarget.jl | WebAssemblyCompiler.jl | WasmCompiler.jl | StaticCompiler.jl |
|---------|---------------|------------------------|-----------------|-------------------|
| **Memory model** | WasmGC | WasmGC | WasmGC | Linear memory |
| **Type inference** | ✅ Julia's | ✅ Julia's | ✅ Julia's | ✅ Julia's |
| **GC** | ✅ Browser | ✅ Browser | ✅ Browser | ❌ None |
| **Structs** | ✅ | ✅ | ✅ | ✅ |
| **Tuples** | ✅ | ✅ | ✅ | ✅ |
| **1D Arrays** | ✅ Vector{T} | ✅ Vector{T}/Any | ❌ | ⚠️ StaticTools |
| **Multi-dim Arrays** | 🚧 Planned | ❌ | ❌ | ❌ |
| **Strings** | ✅ | ✅ | ⚠️ Limited | ❌ |
| **Exceptions** | ✅ try/catch | ❌ | ⚠️ WIP | ❌ |
| **Closures** | ✅ | ❌ | ❌ | ❌ |
| **Union{Nothing,T}** | ✅ | ❌ | ❌ | ❌ |
| **Hash tables** | ✅ Dict-like | ✅ Dict | ❌ | ❌ |
| **Recursion** | ✅ | ✅ | ✅ | ✅ |
| **Mutual recursion** | ✅ | ✅ | ✅ | ❌ |
| **@goto/@label** | ❌ | ❌ | ✅ | ❌ |

### Key Advantages of WasmTarget.jl
1. **Exceptions** - Full try/catch/throw support (others don't have this)
2. **Closures** - Captured variables work (others don't support this)
3. **Union types** - `Union{Nothing, T}` with `isa` operator
4. **Multi-dim arrays** - Coming soon (NO other Julia-to-WASM compiler has this)

## How It Works

```julia
# Write normal Julia code - NO explicit types needed
function fibonacci(n)
    if n <= 1
        return n
    end
    return fibonacci(n - 1) + fibonacci(n - 2)
end

# Compile by specifying argument types at compile time
# Julia's compiler infers everything else
bytes = compile(fibonacci, (Int32,))
```

Julia's `Base.code_typed()` does full type inference, giving us typed IR.
We then generate WASM from that IR.

## Supported Types

### Primitives
| Julia Type | WASM Type | Notes |
|------------|-----------|-------|
| `Int32` | i32 | |
| `UInt32` | i32 | |
| `Int64` / `Int` | i64 | |
| `UInt64` | i64 | |
| `Float32` | f32 | |
| `Float64` | f64 | |
| `Bool` | i32 | 0 or 1 |

### Compound Types
| Julia Type | WASM Type | Notes |
|------------|-----------|-------|
| `String` | GC array (i32) | Immutable, supports ==, sizeof, length |
| `Vector{T}` | GC array | Mutable, T must be concrete |
| `struct Foo ... end` | GC struct | User-defined |
| `Tuple{A,B,...}` | GC struct | Immutable |
| `Union{Nothing,T}` | Tagged union | `isa` operator supported |

### Special Types
| Julia Type | WASM Type | Notes |
|------------|-----------|-------|
| `JSValue` | externref | JavaScript object reference |
| `WasmGlobal{T,IDX}` | global | Compile-time global access |

## Supported Language Features

### Control Flow
- ✅ `if`/`elseif`/`else`
- ✅ `while` loops
- ✅ `for` loops (ranges)
- ✅ `&&` and `||` (short-circuit)
- ✅ `try`/`catch`/`throw`
- ✅ Recursion

### Functions
- ✅ Regular functions
- ✅ Multiple functions calling each other (`compile_multi`)
- ✅ Closures (captured variables)
- ✅ Multiple dispatch (same name, different types)

### Operators
- ✅ Arithmetic: `+`, `-`, `*`, `/`, `%`, `^`
- ✅ Comparison: `==`, `!=`, `<`, `<=`, `>`, `>=`
- ✅ Logical: `&&`, `||`, `!`
- ✅ Bitwise: `&`, `|`, `⊻`, `~`, `<<`, `>>`

### Data Structures
- ✅ Struct creation and field access
- ✅ Tuple creation and indexing
- ✅ Array creation, indexing, mutation
- ✅ String operations (equality, length, concatenation)
- ✅ SimpleDict / StringDict (hash tables)

## NOT Yet Supported (Roadmap)

### HIGH PRIORITY (differentiators)
- 🚧 **Multi-dimensional arrays** (`Matrix`, `Array{T,N}`) - NO other Julia-WASM compiler has this!
- 🚧 **Full dynamic dispatch** - Runtime method lookup

### MEDIUM PRIORITY
- ❌ Full `Dict` (use SimpleDict/StringDict for now)
- ❌ `@goto` / `@label` (WasmCompiler.jl has this)
- ❌ Varargs and keyword arguments

### LOW PRIORITY
- ❌ `@generated` functions
- ❌ FFI / `ccall`
- ❌ I/O operations
- ❌ Metaprogramming at runtime

## Comparison Test Suite

Every feature is tested against base Julia to ensure identical behavior:

```julia
# Test runs in BOTH Julia and WASM, compares results
function test_add()
    source = "add(x, y) = x + y"

    # Run in Julia
    julia_result = run_julia(source, "add", 3, 4)

    # Compile to WASM, run in Node.js
    wasm_result = run_wasm(source, "add", 3, 4)

    # Must match!
    @test julia_result == wasm_result
end
```

Current test coverage: **48 comparison tests** across arithmetic, comparisons, control flow, type conversions, and edge cases.

## Browser REPL Architecture

For the interactive browser REPL, we need type inference to run somewhere:

**Option A: Server-assisted** (simpler, full Julia support)
```
Browser: Julia source → Server: code_typed() → Browser: WASM codegen → Execute
```

**Option B: Full client-side** (harder, requires compiling type inferencer)
```
Browser: Julia source → JuliaSyntax.jl (WASM) → JuliaLowering.jl (WASM) →
         Type inference (WASM) → WasmTarget.jl (WASM) → Execute
```

**Option C: Hybrid** (practical middle ground)
```
Browser: Julia source + type hints → Simple inferencer (WASM) →
         WasmTarget.jl (WASM) → Execute
```

The goal is maximum Julia compatibility while being practical about browser constraints.
