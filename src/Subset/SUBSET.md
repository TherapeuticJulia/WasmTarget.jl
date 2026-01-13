# Julia Subset for Browser REPL

This document defines the Julia subset supported by WasmTarget.jl's browser REPL.
The goal is a minimal but useful subset that can be compiled directly from AST to WASM
without full type inference.

## Design Principles

1. **Explicit types required** - Function signatures must have type annotations
2. **No runtime dispatch** - Types known at compile time
3. **Subset of Base Julia** - Every program in our subset runs identically in real Julia
4. **Progressive expansion** - Start minimal, add features as needed

## Supported Types

### Primitives
| Type | WASM | Notes |
|------|------|-------|
| `Int32` | i32 | 32-bit signed integer |
| `Int64` | i64 | 64-bit signed integer |
| `Float32` | f32 | 32-bit float |
| `Float64` | f64 | 64-bit float |
| `Bool` | i32 | 0 or 1 |

### Compound Types (Phase 2)
| Type | WASM | Notes |
|------|------|-------|
| `String` | GC array (i32) | Immutable |
| `Vector{T}` | GC array | Mutable, T must be concrete |
| `struct` | GC struct | User-defined, concrete fields |
| `Tuple{...}` | GC struct | Immutable |

## Supported Syntax

### Expressions
```julia
# Literals
42              # Int64 (default) or Int32 if annotated
3.14            # Float64 (default) or Float32 if annotated
true, false     # Bool
"hello"         # String (Phase 2)

# Arithmetic
x + y, x - y, x * y, x / y, x % y
x ^ y           # Power (integer only initially)

# Comparisons
x == y, x != y, x < y, x <= y, x > y, x >= y

# Logical
x && y, x || y, !x

# Bitwise
x & y, x | y, x ⊻ y, ~x, x << n, x >> n
```

### Statements
```julia
# Variable declaration (type inferred from RHS or explicit)
x = 42
y::Int32 = 10

# If/else
if condition
    ...
elseif condition
    ...
else
    ...
end

# While loop
while condition
    ...
end

# For loop (ranges only)
for i in 1:10
    ...
end

# Return
return value
```

### Functions
```julia
# REQUIRED: All parameters and return type must be annotated
function add(x::Int32, y::Int32)::Int32
    return x + y
end

# Short form OK
f(x::Int32)::Int32 = x * 2
```

### NOT Supported (Phase 1)
- Multiple dispatch (only one method per function name)
- Abstract types (`Any`, `Number`, `Integer`, etc.)
- Union types (`Union{Int, Nothing}`)
- Closures and anonymous functions
- Macros
- Modules/imports
- Exceptions (try/catch)
- Mutable structs (Phase 2)
- Arrays (Phase 2)
- Strings (Phase 2)

## Compilation Strategy

```
Julia source (subset)
    ↓ JuliaSyntax.jl (parse)
SyntaxNode (AST)
    ↓ SubsetCompiler (type check + codegen)
WASM bytecode
```

No type inference needed because:
1. All function signatures are fully typed
2. Local variable types inferred from initialization or annotation
3. No dynamic dispatch - function calls resolved at compile time

## Testing Strategy

Every feature has dual tests:
1. **Base Julia test** - Run in real Julia, capture result
2. **WASM test** - Compile with WasmTarget, run in Node.js, compare

```julia
# Test macro that runs both
@subset_test "addition" begin
    function add(x::Int32, y::Int32)::Int32
        return x + y
    end
end args=(Int32(3), Int32(4)) expected=Int32(7)
```

## Comparison to Other Subsets

| Feature | Our Subset | AssemblyScript | Dart (subset) |
|---------|------------|----------------|---------------|
| Static types | Required | Required | Optional |
| Generics | No (Phase 2) | Yes | Yes |
| Classes/Structs | Phase 2 | Yes | Yes |
| Closures | No | Yes | Yes |
| GC | WasmGC | Linear memory | WasmGC |

## Roadmap

### Phase 1: Primitives + Control Flow
- [x] Int32, Int64, Float32, Float64, Bool
- [ ] Arithmetic, comparisons, logical ops
- [ ] If/else, while, for (range)
- [ ] Functions with type annotations
- [ ] AST-to-WASM compiler

### Phase 2: Compound Types
- [ ] Strings
- [ ] Arrays (Vector{T})
- [ ] Structs
- [ ] Tuples

### Phase 3: Browser REPL
- [ ] Compile subset compiler to WASM
- [ ] Terminal UI with Therapy.jl
- [ ] Live compilation and execution
