# Runtime Intrinsics - Mapping Julia Core.Intrinsics to Wasm
# Reference: Julia's Core.Intrinsics module

export IntrinsicMapping, get_wasm_opcode

"""
Mapping from Julia intrinsics to Wasm opcodes.
"""
const INTRINSIC_MAPPING = Dict{Symbol, Tuple{UInt8, UInt8}}(
    # Integer arithmetic (i32, i64 variants)
    :add_int => (Opcode.I32_ADD, Opcode.I64_ADD),
    :sub_int => (Opcode.I32_SUB, Opcode.I64_SUB),
    :mul_int => (Opcode.I32_MUL, Opcode.I64_MUL),
    :sdiv_int => (Opcode.I32_DIV_S, Opcode.I64_DIV_S),
    :udiv_int => (Opcode.I32_DIV_U, Opcode.I64_DIV_U),
    :srem_int => (Opcode.I32_REM_S, Opcode.I64_REM_S),
    :urem_int => (Opcode.I32_REM_U, Opcode.I64_REM_U),

    # Bitwise operations
    :and_int => (Opcode.I32_AND, Opcode.I64_AND),
    :or_int => (Opcode.I32_OR, Opcode.I64_OR),
    :xor_int => (Opcode.I32_XOR, Opcode.I64_XOR),
    :shl_int => (Opcode.I32_SHL, Opcode.I64_SHL),
    :ashr_int => (Opcode.I32_SHR_S, Opcode.I64_SHR_S),
    :lshr_int => (Opcode.I32_SHR_U, Opcode.I64_SHR_U),

    # Comparisons
    :eq_int => (Opcode.I32_EQ, Opcode.I64_EQ),
    :ne_int => (Opcode.I32_NE, Opcode.I64_NE),
    :slt_int => (Opcode.I32_LT_S, Opcode.I64_LT_S),
    :ult_int => (Opcode.I32_LT_U, Opcode.I64_LT_U),
    :sle_int => (Opcode.I32_LE_S, Opcode.I64_LE_S),
    :ule_int => (Opcode.I32_LE_U, Opcode.I64_LE_U),
)

"""
Mapping from Julia floating point operations to Wasm opcodes.
"""
const FLOAT_INTRINSIC_MAPPING = Dict{Symbol, Tuple{UInt8, UInt8}}(
    # Float arithmetic (f32, f64 variants)
    :add_float => (Opcode.F32_ADD, Opcode.F64_ADD),
    :sub_float => (Opcode.F32_SUB, Opcode.F64_SUB),
    :mul_float => (Opcode.F32_MUL, Opcode.F64_MUL),
    :div_float => (Opcode.F32_DIV, Opcode.F64_DIV),

    # Float unary
    :neg_float => (Opcode.F32_NEG, Opcode.F64_NEG),
    :abs_float => (Opcode.F32_ABS, Opcode.F64_ABS),
    :sqrt_float => (Opcode.F32_SQRT, Opcode.F64_SQRT),
    :ceil_float => (Opcode.F32_CEIL, Opcode.F64_CEIL),
    :floor_float => (Opcode.F32_FLOOR, Opcode.F64_FLOOR),
    :trunc_float => (Opcode.F32_TRUNC, Opcode.F64_TRUNC),
)

"""
    get_wasm_opcode(intrinsic::Symbol, is_64bit::Bool) -> UInt8

Get the Wasm opcode for a Julia intrinsic.
"""
function get_wasm_opcode(intrinsic::Symbol, is_64bit::Bool)::UInt8
    if haskey(INTRINSIC_MAPPING, intrinsic)
        opcodes = INTRINSIC_MAPPING[intrinsic]
        return is_64bit ? opcodes[2] : opcodes[1]
    elseif haskey(FLOAT_INTRINSIC_MAPPING, intrinsic)
        opcodes = FLOAT_INTRINSIC_MAPPING[intrinsic]
        return is_64bit ? opcodes[2] : opcodes[1]
    else
        error("Unknown intrinsic: $intrinsic")
    end
end

"""
Check if an intrinsic is supported.
"""
function is_supported_intrinsic(intrinsic::Symbol)::Bool
    return haskey(INTRINSIC_MAPPING, intrinsic) || haskey(FLOAT_INTRINSIC_MAPPING, intrinsic)
end
