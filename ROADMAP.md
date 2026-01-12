# WasmTarget.jl Roadmap

## Vision

Build a complete Julia-to-WebAssembly compiler that can compile JuliaSyntax.jl and JuliaLowering.jl to WASM, enabling a fully functional Julia REPL in the browser.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Browser Julia REPL                            │
├─────────────────────────────────────────────────────────────────┤
│  JuliaSyntax.jl (WASM)  │  JuliaLowering.jl (WASM)  │  IR Interp │
├─────────────────────────────────────────────────────────────────┤
│                    WasmTarget.jl Compiler                        │
├─────────────────────────────────────────────────────────────────┤
│                    WasmGC Runtime                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Core Language Support (Current)

### Completed
- [x] Basic types: Int32, Int64, Float32, Float64, Bool
- [x] Arithmetic and comparison operations
- [x] Control flow: if/else, loops, branches
- [x] Structs (WasmGC struct types)
- [x] Tuples (as immutable structs)
- [x] Arrays (WasmGC arrays with get/set/length)
- [x] Strings (as i32 arrays)
- [x] Multi-function modules
- [x] Cross-function calls
- [x] Multiple dispatch (same name, different signatures)
- [x] Recursion
- [x] Mutable structs (setfield!)
- [x] SimpleDict (hash table for Int32 keys/values)

### In Progress
- [ ] Char type support (map to i32)
- [ ] Iterator protocol basics

### Remaining
- [ ] Union types (simple cases)
- [ ] Nothing/missing handling
- [ ] Basic exception simulation (Result types)

---

## Phase 2: JuliaSyntax.jl Prerequisites

### Analysis Required
1. **Identify JuliaSyntax.jl dependencies**
   - What types does it use? (String, Char, Vector, Dict, etc.)
   - What control flow patterns?
   - What Base functions?

2. **Tokenizer subset**
   - Character iteration
   - String operations (peek, advance, substring)
   - Token struct creation
   - Simple state machine

### Implementation Tasks

#### 2.1 Char Support
- [ ] Map Julia Char to i32 in WASM
- [ ] Char comparisons
- [ ] Char arithmetic (for ranges)
- [ ] String indexing returns Char
- [ ] isdigit, isalpha, isspace intrinsics

#### 2.2 Enhanced String Support
- [ ] String iteration (for c in str)
- [ ] String slicing (str[i:j])
- [ ] String search (findfirst, findnext)
- [ ] Character classification functions
- [ ] String building/concatenation optimization

#### 2.3 Iterator Protocol
- [ ] iterate(collection) -> (item, state) or nothing
- [ ] iterate(collection, state) -> (item, state) or nothing
- [ ] for loop compilation using iterate
- [ ] Support for String, Vector, range iterators

#### 2.4 Vector Operations
- [ ] push!/pop! (requires resizable arrays or linked structure)
- [ ] resize!
- [ ] Vector concatenation
- [ ] Vector comprehensions (simple cases)

#### 2.5 Dict{K,V} Generalization
- [ ] Extend SimpleDict to generic key types
- [ ] String keys
- [ ] Symbol keys (as interned strings)
- [ ] Proper hashing for different types

---

## Phase 3: JuliaSyntax.jl Compilation

### 3.1 Tokenizer (ANALYZED - HIGHLY FEASIBLE)

**Architecture (from JuliaSyntax.jl analysis):**
```
tokenize(input)
    → Lexer(IOBuffer(input))  # Iterator creation
    → Base.iterate(lexer)      # Standard iterator protocol
    → next_token(lexer)        # Main dispatch
        → _next_token(lexer, char)  # Char-based dispatcher
            → lex_whitespace/lex_digit/lex_string/etc.
            → emit(lexer, kind)     # Token emission
```

**Required Types:**
- `RawToken` struct (kind::UInt16, startbyte::Int, endbyte::Int, suffix::Bool)
- `StringState` struct (triplestr, raw, delim, paren_depth)
- `Lexer` mutable struct with 4-char lookahead buffer
- `Vector{StringState}` for interpolation state stack

**Required Base Functions:**
| Function | Status | Notes |
|----------|--------|-------|
| read(io, Char) | Need I/O layer | Can use byte buffer abstraction |
| position(io) | Need I/O layer | Track position locally |
| eof(io) | Need I/O layer | Check buffer bounds |
| isvalid(Char) | Need to add | Unicode validation |
| Unicode.category_code | Need tables | Start with ASCII subset |
| push!/pop!/last | Supported | For interpolation stack |
| if/while/for | Supported | Standard control flow |

**Implementation Strategy:**
1. **Phase 3.1a: ASCII-Only Tokenizer** (2-3 days)
   - Strip Unicode handling
   - Use simplified character tests
   - Basic tokens: keywords, identifiers, operators, literals

2. **Phase 3.1b: I/O Abstraction Layer** (1-2 days)
   - Create Wasm-compatible byte buffer type
   - Implement read, position, eof
   - Refactor Lexer to use abstractions

3. **Phase 3.1c: Full Unicode Support** (3-5 days)
   - Add Unicode category tables
   - Support raw strings, interpolations

**Test Strategy:**
- Simple expressions: `"1 + 2"`, `"x = 1"`
- Keywords: `"function f() end"`, `"if x else y end"`
- Strings: `"hello"`, `"""multiline"""`
- Compare output with native JuliaSyntax tokenizer

### 3.2 Parser
Target: `JuliaSyntax.parseall(source)`

**Architecture:**
- Pratt parser for expression precedence
- Recursive descent for statements
- Green tree (lossless) construction

**Required features:**
- Recursive descent parsing
- AST node creation
- Error handling via token emission (no exceptions)
- Operator precedence tables

**Challenges:**
- Expression precedence handling
- Nested data structures (AST nodes)
- Error recovery without exceptions

### 3.3 AST Representation
- GreenNode / SyntaxNode compilation
- Tree traversal (children iteration)
- Source location tracking (byte spans)

---

## Phase 4: JuliaLowering.jl Compilation

### Prerequisites
- Full JuliaSyntax.jl working
- More complex type handling
- Scope analysis data structures

### Components
1. **Macro expansion** (subset)
2. **Desugaring** (for loops, comprehensions, etc.)
3. **Scope analysis**
4. **IR generation**

---

## Phase 5: IR Interpreter

### Design Options

#### Option A: Tree-walking interpreter in WASM
- Compile a simple interpreter loop
- Execute AST/IR nodes directly
- Simpler but slower

#### Option B: Bytecode interpreter in WASM
- Compile IR to simple bytecode
- Stack-based VM in WASM
- More complex but faster

#### Option C: Hybrid
- Use WASM for hot paths
- Interpret cold paths

### Implementation
1. Define IR format (reuse Julia's or simplified version)
2. Implement dispatch on IR node types
3. Handle variable bindings (environment)
4. Basic operations (arithmetic, control flow)
5. Function calls

---

## Phase 6: Browser REPL

### Components
1. **Input handling** - CodeMirror or similar editor
2. **Compilation pipeline** - JuliaSyntax → JuliaLowering → IR
3. **Execution** - IR interpreter
4. **Output display** - Results, errors, graphics

### Integration
- WASM module loading
- JS ↔ WASM communication
- DOM manipulation for output
- History/state management

---

## Technical Challenges & Solutions

### Challenge 1: Closures
**Problem:** Julia uses closures extensively; they're complex to compile.
**Solution:**
- Avoid closures in compilable code where possible
- Use macros to expand closure patterns at compile time
- Implement simple closure conversion for necessary cases

### Challenge 2: Dynamic Dispatch
**Problem:** Julia's multiple dispatch is runtime-resolved.
**Solution:**
- Compile monomorphic call sites directly
- Use type inference to resolve most dispatch statically
- Fall back to dispatch tables for truly dynamic calls

### Challenge 3: Exceptions
**Problem:** WASM has limited exception support.
**Solution:**
- Use Result types for expected errors
- Error codes for recoverable errors
- Trap for unrecoverable errors

### Challenge 4: Memory Management
**Problem:** Julia has a GC; WASM GC is different.
**Solution:**
- Use WasmGC for all allocations
- Let WasmGC handle collection
- Avoid manual memory management patterns

### Challenge 5: Recursion Limits
**Problem:** Deep recursion in parser can overflow stack.
**Solution:**
- Convert tail-recursive functions
- Use explicit stack for deeply nested structures
- Set reasonable limits

---

## Testing Strategy

### Unit Tests
- Each feature tested in isolation
- Compare WASM output to native Julia
- Property-based testing for numeric operations

### Integration Tests
- Compile and run JuliaSyntax functions
- Compare tokenizer/parser output
- Round-trip tests (parse → unparse → parse)

### End-to-End Tests
- Full REPL interaction tests
- Complex expression evaluation
- Error handling tests

---

## Milestones

### Milestone 1: Tokenizer Demo (Target: Phase 2)
- Compile JuliaSyntax tokenizer to WASM
- Demo: tokenize simple Julia code in browser
- Validates string/char/iterator support

### Milestone 2: Parser Demo (Target: Phase 3)
- Compile full parser to WASM
- Demo: parse Julia code, display AST
- Validates recursive structures, complex control flow

### Milestone 3: Lowering Demo (Target: Phase 4)
- Compile JuliaLowering to WASM
- Demo: show IR for Julia code
- Validates complex analysis passes

### Milestone 4: Calculator REPL (Target: Phase 5)
- Simple expression evaluation
- Demo: `1 + 2 * 3` returns `7`
- Validates interpreter basics

### Milestone 5: Full REPL (Target: Phase 6)
- Variable bindings
- Function definitions
- Control flow
- Demo: interactive Julia programming in browser

---

## File Structure

```
WasmTarget.jl/
├── src/
│   ├── WasmTarget.jl           # Main module
│   ├── Builder/                # WASM binary construction
│   │   ├── Types.jl
│   │   ├── Writer.jl
│   │   └── Instructions.jl
│   ├── Compiler/               # Julia → WASM translation
│   │   ├── IR.jl
│   │   └── Codegen.jl
│   └── Runtime/                # Runtime support
│       ├── Intrinsics.jl
│       ├── StringOps.jl
│       ├── ArrayOps.jl
│       └── SimpleDict.jl
├── test/
│   ├── runtests.jl
│   └── utils.jl
├── docs/                       # Documentation site
├── CLAUDE.md                   # AI assistant context
└── ROADMAP.md                  # This file
```

---

## Current Status

**Date:** 2026-01-12

**Completed:**
- Core compiler infrastructure
- Basic types and operations (Int32, Int64, Float32, Float64, Bool)
- Structs (mutable and immutable), tuples, arrays, strings
- Multi-function modules with cross-function calls
- SimpleDict implementation (hash table for Int32 keys/values)
- Char type support (map to i32, comparisons work)
- Count leading/trailing zeros intrinsics (ctlz_int, cttz_int)
- JuliaSyntax.jl tokenizer analysis (HIGHLY FEASIBLE)

**Next Steps:**
1. Create I/O abstraction layer (ByteBuffer type)
2. Add character classification functions (isdigit, isalpha, isspace)
3. Implement iterator protocol for strings
4. Create minimal tokenizer compilation test
5. Compile ASCII-only tokenizer subset

**Key Findings from JuliaSyntax.jl Analysis:**
- Tokenizer uses simple if/else dispatch on characters
- ~1300 lines, ~35 lexing functions
- No closures, no recursion (except tail calls)
- Main blockers: I/O abstraction, Unicode tables
- Strategy: Start with ASCII-only subset

**Blockers:**
- Need I/O abstraction layer (read, position, eof)
- Unicode category_code requires lookup tables (defer to Phase 3.1c)

---

## Notes for Future Sessions

When resuming work:
1. Read this ROADMAP.md for context
2. Check CLAUDE.md for technical details
3. Run `julia --project=. test/runtests.jl` to verify state
4. Check git log for recent changes
5. Continue from "Next Steps" above
