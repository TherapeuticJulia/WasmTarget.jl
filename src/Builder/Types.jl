# Wasm Types - Value types, Reference types, and composite types
# Reference: https://webassembly.github.io/spec/core/binary/types.html

export ValType, NumType, RefType, FuncType

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
end

# ============================================================================
# Function Types (Section 5.3.6)
# ============================================================================

"""
    FuncType

A function type describing the signature of a WebAssembly function.
"""
struct FuncType
    params::Vector{NumType}
    results::Vector{NumType}
end

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
# Helper functions
# ============================================================================

"""
Convert a Julia type to a Wasm NumType.
"""
function julia_to_wasm_type(::Type{T}) where T
    if T === Int32 || T === UInt32
        return I32
    elseif T === Int64 || T === UInt64 || T === Int
        return I64
    elseif T === Float32
        return F32
    elseif T === Float64
        return F64
    else
        error("Unsupported Julia type for Wasm: $T")
    end
end
