# Wasm Types - Value types, Reference types, and composite types
# Reference: https://webassembly.github.io/spec/core/binary/types.html

export ValType, NumType, RefType, ConcreteRef, FuncType, StructType, ArrayType, FieldType, CompositeType, WasmValType, JSValue, WasmGlobal

# ============================================================================
# Value Types (Section 5.3.1)
# ============================================================================

"""
    ValType

Abstract type for all WebAssembly value types.
"""
abstract type ValType end

"""
    NumType <: ValType

Numeric types in WebAssembly.
"""
@enum NumType::UInt8 begin
    I32 = 0x7F  # i32
    I64 = 0x7E  # i64
    F32 = 0x7D  # f32
    F64 = 0x7C  # f64
end

"""
    RefType <: ValType

Reference types in WebAssembly (including WasmGC extensions).
"""
@enum RefType::UInt8 begin
    FuncRef = 0x70    # funcref
    ExternRef = 0x6F  # externref
    AnyRef = 0x6E     # anyref (WasmGC)
    EqRef = 0x6D      # eqref (WasmGC)
    I31Ref = 0x6C     # i31ref (WasmGC)
    StructRef = 0x6B  # structref (WasmGC)
    ArrayRef = 0x6A   # arrayref (WasmGC)
    ExnRef = 0x69     # exnref (exception handling)
end

"""
    ConcreteRef

A concrete reference type with a type index, e.g., `(ref null \$typeidx)`.
Used for locals and parameters that hold instances of specific struct/array types.
"""
struct ConcreteRef
    type_idx::UInt32
    nullable::Bool
end

ConcreteRef(type_idx::UInt32) = ConcreteRef(type_idx, true)  # Default nullable

"""
    WasmValType

Union type for all Wasm value types (numeric, reference, packed, concrete refs).
"""
const WasmValType = Union{NumType, RefType, ConcreteRef, UInt8}

# ============================================================================
# Function Types (Section 5.3.6)
# ============================================================================

"""
    FuncType

A function type describing the signature of a WebAssembly function.
Supports both numeric types and reference types (for WasmGC).
"""
struct FuncType
    params::Vector{WasmValType}
    results::Vector{WasmValType}
end

# Convenience constructor for NumType-only signatures
FuncType(params::Vector{NumType}, results::Vector{NumType}) =
    FuncType(WasmValType[p for p in params], WasmValType[r for r in results])

# ============================================================================
# WasmGC Types
# Reference: https://github.com/WebAssembly/gc/blob/main/proposals/gc/Overview.md
# ============================================================================

"""
    FieldType

A field in a WasmGC struct type.
"""
struct FieldType
    valtype::WasmValType  # The type of the field
    mutable_::Bool        # Whether the field is mutable
end

FieldType(valtype::WasmValType) = FieldType(valtype, true)  # Default to mutable

"""
    StructType

A WasmGC struct type with named fields.
"""
struct StructType
    fields::Vector{FieldType}
end

"""
    ArrayType

A WasmGC array type with element type.
"""
struct ArrayType
    elem::FieldType  # Element type with mutability
end

ArrayType(valtype::WasmValType) = ArrayType(FieldType(valtype, true))

"""
    CompositeType

Union of all composite types in WasmGC.
"""
const CompositeType = Union{FuncType, StructType, ArrayType}

"""
    HeapType

Represents a heap type - either an abstract type or a concrete type index.
"""
struct HeapType
    # If index >= 0, it's a concrete type index
    # If index < 0, it's an abstract type encoded as negative
    index::Int32
end

# Abstract heap types (encoded as negative values internally)
const HEAP_FUNC = HeapType(-1)      # func
const HEAP_EXTERN = HeapType(-2)    # extern
const HEAP_ANY = HeapType(-3)       # any
const HEAP_EQ = HeapType(-4)        # eq
const HEAP_I31 = HeapType(-5)       # i31
const HEAP_STRUCT = HeapType(-6)    # struct
const HEAP_ARRAY = HeapType(-7)     # array
const HEAP_NONE = HeapType(-8)      # none
const HEAP_NOEXTERN = HeapType(-9)  # noextern
const HEAP_NOFUNC = HeapType(-10)   # nofunc

HeapType(idx::Integer) = HeapType(Int32(idx))

"""
    RefTypeGC

A reference type in WasmGC with nullability.
"""
struct RefTypeGC
    nullable::Bool
    heaptype::HeapType
end

RefTypeGC(ht::HeapType) = RefTypeGC(true, ht)  # Default to nullable

# ============================================================================
# Limits (for memories and tables)
# ============================================================================

struct Limits
    min::UInt32
    max::Union{Nothing, UInt32}
end

Limits(min::Integer) = Limits(UInt32(min), nothing)
Limits(min::Integer, max::Integer) = Limits(UInt32(min), UInt32(max))

# ============================================================================
# JS Interop Types
# ============================================================================

"""
    JSValue

A Julia type representing a JavaScript value held as an externref.
Used for DOM elements, JS objects, and other JS values.

This is a primitive type to prevent Julia from optimizing it away.
"""
primitive type JSValue 64 end

# ============================================================================
# WasmGlobal - Handle for Wasm Global Variables
# ============================================================================

"""
    WasmGlobal{T, IDX}

A handle to a WebAssembly global variable at index `IDX`. When compiled to Wasm:
- `global[]` (getindex) → `global.get IDX`
- `global[] = x` (setindex!) → `global.set IDX, x`

The index is a type parameter so it's known at compile time, which is required
because Wasm's `global.get` and `global.set` instructions take immediate indices.

This is a general-purpose abstraction for any Julia code that needs to
interact with Wasm global variables. Use cases include:
- Stateful applications
- Game engines
- Reactive frameworks
- Any code needing mutable Wasm state

# Type Parameters
- `T`: The type of value stored in the global (Int32, Float64, etc.)
- `IDX`: The Wasm global index (0-based), must be an Int literal

# Example
```julia
# Define types for specific globals (index is compile-time constant)
const Counter = WasmGlobal{Int32, 0}   # Global index 0
const Flag = WasmGlobal{Int32, 1}      # Global index 1

# Functions that use globals - index is known from the type
function increment(g::Counter)::Int32
    g[] = g[] + Int32(1)
    return g[]
end

function toggle(g::Flag)::Int32
    g[] = g[] == Int32(0) ? Int32(1) : Int32(0)
    return g[]
end

# Create instances (value is for Julia-side testing)
counter = Counter(0)
flag = Flag(1)

# Compile to Wasm - global index extracted from type
wasm_bytes = compile(increment, (Counter,))
```
"""
mutable struct WasmGlobal{T, IDX}
    value::T
end

# Constructor with zero initial value
WasmGlobal{T, IDX}() where {T, IDX} = WasmGlobal{T, IDX}(zero(T))

# Get the global index from the type
global_index(::Type{WasmGlobal{T, IDX}}) where {T, IDX} = IDX
global_index(g::WasmGlobal{T, IDX}) where {T, IDX} = IDX

# Get the element type
global_eltype(::Type{WasmGlobal{T, IDX}}) where {T, IDX} = T

# Accessor methods - work in Julia (for testing) and compile to Wasm global ops
function Base.getindex(g::WasmGlobal{T, IDX})::T where {T, IDX}
    return g.value
end

function Base.setindex!(g::WasmGlobal{T, IDX}, v::T)::T where {T, IDX}
    g.value = v
    return v
end

# ============================================================================
# Helper functions
# ============================================================================

"""
Convert a Julia type to a Wasm value type (NumType or RefType).
"""
function julia_to_wasm_type(::Type{T})::WasmValType where T
    if T === Int32 || T === UInt32
        return I32
    elseif T === Int64 || T === UInt64 || T === Int
        return I64
    elseif T === Float32
        return F32
    elseif T === Float64
        return F64
    elseif T === Bool
        # Bool is represented as i32 (0 or 1)
        return I32
    elseif T === Char
        # Char is represented as i32 (Unicode codepoint)
        return I32
    elseif T === UInt8 || T === Int8 || T === UInt16 || T === Int16
        # Smaller integers also use i32
        return I32
    elseif T === Int128 || T === UInt128
        # 128-bit integers are represented as WasmGC structs with two i64 fields
        return StructRef
    elseif T === Nothing
        # Nothing has no Wasm representation - handled specially
        # Return I32 as a placeholder (functions returning Nothing don't actually return)
        return I32
    elseif T === Any
        # Any can hold any value - map to externref for JS interop
        # This handles Julia 1.12 closure types that have Any fields
        return ExternRef
    elseif T === JSValue
        # JS values are held as externref
        return ExternRef
    elseif T === String || T === Symbol || T <: AbstractString
        # Strings and Symbols are represented as WasmGC byte arrays
        # Symbol is stored as its name string
        # AbstractString maps to String representation (concrete in WasmGC)
        return ArrayRef
    elseif T <: Tuple
        # Tuples map to WasmGC structs
        return StructRef
    elseif T <: AbstractArray
        # Arrays map to WasmGC arrays
        return ArrayRef
    elseif T <: WasmGlobal
        # WasmGlobal is passed as a WasmGC struct (holds just value since idx is in type)
        return StructRef
    elseif T isa Union
        # Handle Union types by finding a common Wasm type
        return resolve_union_type(T)
    elseif isconcretetype(T) && isstructtype(T)
        # User-defined structs map to WasmGC structs
        return StructRef
    elseif T isa UnionAll && isstructtype(T)
        # Parametric struct type without concrete parameters (e.g., SyntaxGraph)
        # Use AnyRef since we can't know the specific type parameter
        return AnyRef
    elseif T <: Function
        # Abstract Function types (non-closure) map to externref
        return ExternRef
    elseif T <: Type
        # Type{X} is a singleton type (the only value is X itself)
        # Represent as i32 constant tag — used for dispatch, not actual data
        return I32
    elseif isprimitivetype(T)
        # Custom primitive types (e.g., JuliaSyntax.Kind) - map by size
        sz = sizeof(T)
        if sz <= 4
            return I32
        elseif sz <= 8
            return I64
        else
            error("Primitive type too large for Wasm: $T ($sz bytes)")
        end
    else
        error("Unsupported Julia type for Wasm: $T")
    end
end

"""
Resolve a Union type to a common Wasm type.

Strategy:
- Union{Nothing, T} -> type of T (Nothing is "no value")
- Union{T1, T2, ...} where all are numeric -> widest numeric type
- Otherwise error
"""
function resolve_union_type(T::Union)::WasmValType
    # Get the union types
    types = Base.uniontypes(T)

    # Filter out Nothing
    non_nothing = filter(t -> t !== Nothing, types)

    if isempty(non_nothing)
        # Union of just Nothing - shouldn't happen but handle it
        return I32
    elseif length(non_nothing) == 1
        # Union{Nothing, T} -> T
        return julia_to_wasm_type(non_nothing[1])
    else
        # Multiple non-Nothing types - find common numeric type
        return find_common_wasm_type(non_nothing)
    end
end

"""
Find a common Wasm type for a list of Julia types.
For numeric types, returns the widest type.
"""
function find_common_wasm_type(types::Vector)::WasmValType
    # Check if all are numeric
    if all(t -> t <: Number, types)
        # Prefer i64 over i32, f64 over f32
        has_i64 = any(t -> t === Int64 || t === UInt64 || t === Int, types)
        has_f64 = any(t -> t === Float64, types)
        has_f32 = any(t -> t === Float32, types)
        has_float = has_f64 || has_f32

        if has_float
            return has_f64 ? F64 : F32
        elseif has_i64
            return I64
        else
            return I32
        end
    end

    # Check if all are reference types (strings, arrays, structs)
    if all(t -> t === String || t === Symbol || t <: AbstractArray || (isconcretetype(t) && isstructtype(t)), types)
        # Use generic reference type
        return StructRef
    end

    # Heterogeneous union (mix of primitives, strings, structs, etc.)
    # Use externref as the universal boxed value type (same as Any)
    return ExternRef
end

"""
Get the element type from a WasmGlobal type.
"""
function wasm_global_element_type(::Type{WasmGlobal{T, IDX}}) where {T, IDX}
    return T
end
