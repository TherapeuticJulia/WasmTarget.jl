# WebAssembly Instructions and Opcodes
# Reference: https://webassembly.github.io/spec/core/binary/instructions.html

export Opcode, WasmModule, add_function!, add_export!, to_bytes

# ============================================================================
# Opcodes (Section 5.4)
# ============================================================================

module Opcode
    # Control instructions
    const UNREACHABLE = 0x00
    const NOP = 0x01
    const BLOCK = 0x02
    const LOOP = 0x03
    const IF = 0x04
    const ELSE = 0x05
    const END = 0x0B
    const BR = 0x0C
    const BR_IF = 0x0D
    const BR_TABLE = 0x0E
    const RETURN = 0x0F
    const CALL = 0x10
    const CALL_INDIRECT = 0x11

    # Parametric instructions
    const DROP = 0x1A
    const SELECT = 0x1B

    # Variable instructions
    const LOCAL_GET = 0x20
    const LOCAL_SET = 0x21
    const LOCAL_TEE = 0x22
    const GLOBAL_GET = 0x23
    const GLOBAL_SET = 0x24

    # Memory instructions
    const I32_LOAD = 0x28
    const I64_LOAD = 0x29
    const F32_LOAD = 0x2A
    const F64_LOAD = 0x2B
    const I32_STORE = 0x36
    const I64_STORE = 0x37
    const F32_STORE = 0x38
    const F64_STORE = 0x39
    const MEMORY_SIZE = 0x3F
    const MEMORY_GROW = 0x40

    # Numeric instructions - Constants
    const I32_CONST = 0x41
    const I64_CONST = 0x42
    const F32_CONST = 0x43
    const F64_CONST = 0x44

    # Numeric instructions - i32 operations
    const I32_EQZ = 0x45
    const I32_EQ = 0x46
    const I32_NE = 0x47
    const I32_LT_S = 0x48
    const I32_LT_U = 0x49
    const I32_GT_S = 0x4A
    const I32_GT_U = 0x4B
    const I32_LE_S = 0x4C
    const I32_LE_U = 0x4D
    const I32_GE_S = 0x4E
    const I32_GE_U = 0x4F

    # Numeric instructions - i64 comparisons
    const I64_EQZ = 0x50
    const I64_EQ = 0x51
    const I64_NE = 0x52
    const I64_LT_S = 0x53
    const I64_LT_U = 0x54
    const I64_GT_S = 0x55
    const I64_GT_U = 0x56
    const I64_LE_S = 0x57
    const I64_LE_U = 0x58
    const I64_GE_S = 0x59
    const I64_GE_U = 0x5A

    # Numeric instructions - i32 arithmetic
    const I32_CLZ = 0x67
    const I32_CTZ = 0x68
    const I32_POPCNT = 0x69
    const I32_ADD = 0x6A
    const I32_SUB = 0x6B
    const I32_MUL = 0x6C
    const I32_DIV_S = 0x6D
    const I32_DIV_U = 0x6E
    const I32_REM_S = 0x6F
    const I32_REM_U = 0x70
    const I32_AND = 0x71
    const I32_OR = 0x72
    const I32_XOR = 0x73
    const I32_SHL = 0x74
    const I32_SHR_S = 0x75
    const I32_SHR_U = 0x76
    const I32_ROTL = 0x77
    const I32_ROTR = 0x78

    # Numeric instructions - i64 arithmetic
    const I64_CLZ = 0x79
    const I64_CTZ = 0x7A
    const I64_POPCNT = 0x7B
    const I64_ADD = 0x7C
    const I64_SUB = 0x7D
    const I64_MUL = 0x7E
    const I64_DIV_S = 0x7F
    const I64_DIV_U = 0x80
    const I64_REM_S = 0x81
    const I64_REM_U = 0x82
    const I64_AND = 0x83
    const I64_OR = 0x84
    const I64_XOR = 0x85
    const I64_SHL = 0x86
    const I64_SHR_S = 0x87
    const I64_SHR_U = 0x88
    const I64_ROTL = 0x89
    const I64_ROTR = 0x8A

    # Numeric instructions - f32 operations
    const F32_ABS = 0x8B
    const F32_NEG = 0x8C
    const F32_CEIL = 0x8D
    const F32_FLOOR = 0x8E
    const F32_TRUNC = 0x8F
    const F32_NEAREST = 0x90
    const F32_SQRT = 0x91
    const F32_ADD = 0x92
    const F32_SUB = 0x93
    const F32_MUL = 0x94
    const F32_DIV = 0x95
    const F32_MIN = 0x96
    const F32_MAX = 0x97
    const F32_COPYSIGN = 0x98

    # Numeric instructions - f64 operations
    const F64_ABS = 0x99
    const F64_NEG = 0x9A
    const F64_CEIL = 0x9B
    const F64_FLOOR = 0x9C
    const F64_TRUNC = 0x9D
    const F64_NEAREST = 0x9E
    const F64_SQRT = 0x9F
    const F64_ADD = 0xA0
    const F64_SUB = 0xA1
    const F64_MUL = 0xA2
    const F64_DIV = 0xA3
    const F64_MIN = 0xA4
    const F64_MAX = 0xA5
    const F64_COPYSIGN = 0xA6

    # Conversion operations
    const I32_WRAP_I64 = 0xA7
    const I64_EXTEND_I32_S = 0xAC
    const I64_EXTEND_I32_U = 0xAD
    const F32_CONVERT_I32_S = 0xB2
    const F32_CONVERT_I32_U = 0xB3
    const F64_CONVERT_I32_S = 0xB7
    const F64_CONVERT_I32_U = 0xB8
    const F64_CONVERT_I64_S = 0xB9
    const F64_CONVERT_I64_U = 0xBA
end

# ============================================================================
# WasmModule - High-level module builder
# ============================================================================

"""
Represents a WebAssembly function definition.
"""
struct WasmFunction
    type_idx::UInt32
    locals::Vector{NumType}
    body::Vector{UInt8}
end

"""
Represents an export entry.
"""
struct WasmExport
    name::String
    kind::UInt8  # 0=func, 1=table, 2=memory, 3=global
    idx::UInt32
end

"""
    WasmModule

A WebAssembly module builder. Use this to construct modules programmatically.
"""
mutable struct WasmModule
    types::Vector{FuncType}
    functions::Vector{WasmFunction}
    exports::Vector{WasmExport}
end

WasmModule() = WasmModule(FuncType[], WasmFunction[], WasmExport[])

# ============================================================================
# Module Building API
# ============================================================================

"""
    add_type!(mod, func_type) -> type_idx

Add a function type to the module and return its index.
"""
function add_type!(mod::WasmModule, ft::FuncType)::UInt32
    # Check if type already exists
    for (i, existing) in enumerate(mod.types)
        if existing.params == ft.params && existing.results == ft.results
            return UInt32(i - 1)
        end
    end
    push!(mod.types, ft)
    return UInt32(length(mod.types) - 1)
end

"""
    add_function!(mod, params, results, locals, body) -> func_idx

Add a function to the module and return its index.
"""
function add_function!(mod::WasmModule,
                       params::Vector{NumType},
                       results::Vector{NumType},
                       locals::Vector{NumType},
                       body::Vector{UInt8})::UInt32
    ft = FuncType(params, results)
    type_idx = add_type!(mod, ft)
    push!(mod.functions, WasmFunction(type_idx, locals, body))
    return UInt32(length(mod.functions) - 1)
end

"""
    add_export!(mod, name, kind, idx)

Add an export entry to the module.
- kind: 0=func, 1=table, 2=memory, 3=global
"""
function add_export!(mod::WasmModule, name::String, kind::Integer, idx::Integer)
    push!(mod.exports, WasmExport(name, UInt8(kind), UInt32(idx)))
    return mod
end

# ============================================================================
# Binary Serialization
# ============================================================================

const WASM_MAGIC = UInt8[0x00, 0x61, 0x73, 0x6D]  # \0asm
const WASM_VERSION = UInt8[0x01, 0x00, 0x00, 0x00]  # version 1

# Section IDs
const SECTION_TYPE = 0x01
const SECTION_FUNCTION = 0x03
const SECTION_EXPORT = 0x07
const SECTION_CODE = 0x0A

"""
    to_bytes(mod::WasmModule) -> Vector{UInt8}

Serialize a WasmModule to binary format.
"""
function to_bytes(mod::WasmModule)::Vector{UInt8}
    w = WasmWriter()

    # Magic number and version
    write_bytes!(w, WASM_MAGIC...)
    write_bytes!(w, WASM_VERSION...)

    # Type section
    if !isempty(mod.types)
        write_section!(w, SECTION_TYPE) do section
            write_u32!(section, length(mod.types))
            for ft in mod.types
                write_byte!(section, 0x60)  # functype
                write_vec!(section, [UInt8(p) for p in ft.params])
                write_vec!(section, [UInt8(r) for r in ft.results])
            end
        end
    end

    # Function section (type indices)
    if !isempty(mod.functions)
        write_section!(w, SECTION_FUNCTION) do section
            write_u32!(section, length(mod.functions))
            for func in mod.functions
                write_u32!(section, func.type_idx)
            end
        end
    end

    # Export section
    if !isempty(mod.exports)
        write_section!(w, SECTION_EXPORT) do section
            write_u32!(section, length(mod.exports))
            for exp in mod.exports
                write_name!(section, exp.name)
                write_byte!(section, exp.kind)
                write_u32!(section, exp.idx)
            end
        end
    end

    # Code section
    if !isempty(mod.functions)
        write_section!(w, SECTION_CODE) do section
            write_u32!(section, length(mod.functions))
            for func in mod.functions
                # Write function body with locals
                body_writer = WasmWriter()

                # Locals count (compressed format)
                if isempty(func.locals)
                    write_u32!(body_writer, 0)
                else
                    # Group consecutive locals of same type
                    local_groups = group_locals(func.locals)
                    write_u32!(body_writer, length(local_groups))
                    for (count, type) in local_groups
                        write_u32!(body_writer, count)
                        write_byte!(body_writer, UInt8(type))
                    end
                end

                # Body instructions
                append!(body_writer.buffer, func.body)

                # Write body size then body
                write_u32!(section, length(body_writer.buffer))
                append!(section.buffer, body_writer.buffer)
            end
        end
    end

    return bytes(w)
end

"""
Write a section with automatic size calculation.
"""
function write_section!(f::Function, w::WasmWriter, section_id::UInt8)
    section = WasmWriter()
    f(section)

    write_byte!(w, section_id)
    write_u32!(w, length(section.buffer))
    append!(w.buffer, section.buffer)
end

"""
Group consecutive locals of the same type.
"""
function group_locals(locals::Vector{NumType})
    isempty(locals) && return Tuple{Int, NumType}[]

    groups = Tuple{Int, NumType}[]
    current_type = locals[1]
    count = 1

    for i in 2:length(locals)
        if locals[i] == current_type
            count += 1
        else
            push!(groups, (count, current_type))
            current_type = locals[i]
            count = 1
        end
    end
    push!(groups, (count, current_type))

    return groups
end
