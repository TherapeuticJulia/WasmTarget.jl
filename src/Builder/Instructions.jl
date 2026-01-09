# WebAssembly Instructions and Opcodes
# Reference: https://webassembly.github.io/spec/core/binary/instructions.html

export Opcode, WasmModule, WasmImport, WasmTable, WasmMemory, WasmDataSegment, add_function!, add_import!, add_export!, add_struct_type!, add_array_type!, add_table!, add_table_export!, add_elem_segment!, add_memory!, add_memory_export!, add_data_segment!, to_bytes

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

    # Numeric instructions - f32 comparisons
    const F32_EQ = 0x5B
    const F32_NE = 0x5C
    const F32_LT = 0x5D
    const F32_GT = 0x5E
    const F32_LE = 0x5F
    const F32_GE = 0x60

    # Numeric instructions - f64 comparisons
    const F64_EQ = 0x61
    const F64_NE = 0x62
    const F64_LT = 0x63
    const F64_GT = 0x64
    const F64_LE = 0x65
    const F64_GE = 0x66

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
    const F32_CONVERT_I64_S = 0xB4
    const F32_CONVERT_I64_U = 0xB5
    const F64_CONVERT_I32_S = 0xB7
    const F64_CONVERT_I32_U = 0xB8
    const F64_CONVERT_I64_S = 0xB9
    const F64_CONVERT_I64_U = 0xBA

    # Float to int conversions
    const I32_TRUNC_F32_S = 0xA8
    const I32_TRUNC_F32_U = 0xA9
    const I32_TRUNC_F64_S = 0xAA
    const I32_TRUNC_F64_U = 0xAB
    const I64_TRUNC_F32_S = 0xAE
    const I64_TRUNC_F32_U = 0xAF
    const I64_TRUNC_F64_S = 0xB0
    const I64_TRUNC_F64_U = 0xB1

    # Reinterpret operations
    const I32_REINTERPRET_F32 = 0xBC
    const I64_REINTERPRET_F64 = 0xBD
    const F32_REINTERPRET_I32 = 0xBE
    const F64_REINTERPRET_I64 = 0xBF

    # ========================================================================
    # WasmGC Instructions (0xFB prefix)
    # Reference: https://github.com/WebAssembly/gc/blob/main/proposals/gc/Overview.md
    # ========================================================================
    const GC_PREFIX = 0xFB

    # Struct operations
    const STRUCT_NEW = 0x00       # struct.new $t : [field types] -> [(ref $t)]
    const STRUCT_NEW_DEFAULT = 0x01  # struct.new_default $t : [] -> [(ref $t)]
    const STRUCT_GET = 0x02       # struct.get $t $i : [(ref null $t)] -> [field type]
    const STRUCT_GET_S = 0x03     # struct.get_s $t $i (packed signed)
    const STRUCT_GET_U = 0x04     # struct.get_u $t $i (packed unsigned)
    const STRUCT_SET = 0x05       # struct.set $t $i : [(ref null $t) value] -> []

    # Array operations
    const ARRAY_NEW = 0x06        # array.new $t : [elem init, len] -> [(ref $t)]
    const ARRAY_NEW_DEFAULT = 0x07  # array.new_default $t : [len] -> [(ref $t)]
    const ARRAY_NEW_FIXED = 0x08  # array.new_fixed $t $n : [elem...] -> [(ref $t)]
    const ARRAY_NEW_DATA = 0x09   # array.new_data $t $d : [offset, len] -> [(ref $t)]
    const ARRAY_NEW_ELEM = 0x0A   # array.new_elem $t $e
    const ARRAY_GET = 0x0B        # array.get $t : [(ref null $t) i32] -> [elem type]
    const ARRAY_GET_S = 0x0C      # array.get_s (packed signed)
    const ARRAY_GET_U = 0x0D      # array.get_u (packed unsigned)
    const ARRAY_SET = 0x0E        # array.set $t : [(ref null $t) i32 value] -> []
    const ARRAY_LEN = 0x0F        # array.len : [(ref null array)] -> [i32]
    const ARRAY_FILL = 0x10       # array.fill $t : [(ref null $t) i32 value i32] -> []
    const ARRAY_COPY = 0x11       # array.copy $t1 $t2

    # Reference type operations
    const REF_NULL = 0xD0         # ref.null $t : [] -> [(ref null $t)]
    const REF_IS_NULL = 0xD1      # ref.is_null : [(ref null $t)] -> [i32]
    const REF_FUNC = 0xD2         # ref.func $f : [] -> [(ref $f)]
    const REF_EQ = 0xD3           # ref.eq : [(eqref) (eqref)] -> [i32]
    const REF_AS_NON_NULL = 0xD4  # ref.as_non_null : [(ref null $t)] -> [(ref $t)]

    # GC casting operations (0xFB prefix)
    const REF_CAST = 0x17         # ref.cast (ref null? $t) : [(ref null? $ht)] -> [(ref null? $t)]
    const REF_TEST = 0x14         # ref.test (ref null? $t) : [(ref null? $ht)] -> [i32]
    const BR_ON_CAST = 0x18       # br_on_cast
    const BR_ON_CAST_FAIL = 0x19  # br_on_cast_fail

    # i31 operations (0xFB prefix)
    const REF_I31 = 0x1C          # ref.i31 : [i32] -> [(ref i31)]
    const I31_GET_S = 0x1D        # i31.get_s : [(ref null i31)] -> [i32]
    const I31_GET_U = 0x1E        # i31.get_u : [(ref null i31)] -> [i32]

    # any/extern conversions (0xFB prefix)
    const ANY_CONVERT_EXTERN = 0x1A  # any.convert_extern
    const EXTERN_CONVERT_ANY = 0x1B  # extern.convert_any
end

# ============================================================================
# WasmModule - High-level module builder
# ============================================================================

"""
Represents a WebAssembly function definition.
"""
struct WasmFunction
    type_idx::UInt32
    locals::Vector{WasmValType}
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
Represents an import entry.
"""
struct WasmImport
    module_name::String
    field_name::String
    kind::UInt8  # 0=func, 1=table, 2=memory, 3=global
    type_idx::UInt32  # For functions, the type index
end

"""
    WasmGlobalDef

Internal representation of a WebAssembly global variable definition.
"""
struct WasmGlobalDef
    valtype::WasmValType     # Type of the global
    mutable_::Bool           # Whether the global is mutable
    init::Vector{UInt8}      # Initialization expression (bytecode)
end

"""
    WasmTable

A WebAssembly table for holding references (funcref, externref).
"""
struct WasmTable
    reftype::RefType         # funcref (0x70) or externref (0x6F)
    min::UInt32              # Minimum size
    max::Union{UInt32, Nothing}  # Maximum size (nothing = no max)
end

"""
    WasmElemSegment

An element segment for initializing tables with function references.
"""
struct WasmElemSegment
    table_idx::UInt32        # Which table to initialize
    offset::UInt32           # Offset in table (constant)
    func_indices::Vector{UInt32}  # Function indices to place in table
end

"""
    WasmMemory

A WebAssembly linear memory (in pages of 64KB).
"""
struct WasmMemory
    min::UInt32              # Minimum size in pages
    max::Union{UInt32, Nothing}  # Maximum size in pages (nothing = no max)
end

"""
    WasmDataSegment

A data segment for initializing linear memory with constant data.
"""
struct WasmDataSegment
    memory_idx::UInt32       # Which memory to initialize
    offset::UInt32           # Offset in memory (constant)
    data::Vector{UInt8}      # The data to initialize with
end

"""
    WasmModule

A WebAssembly module builder. Use this to construct modules programmatically.
"""
mutable struct WasmModule
    types::Vector{CompositeType}  # Can contain FuncType, StructType, ArrayType
    imports::Vector{WasmImport}   # Imported functions/tables/etc
    functions::Vector{WasmFunction}
    tables::Vector{WasmTable}     # Tables for funcref/externref
    memories::Vector{WasmMemory}  # Linear memories
    globals::Vector{WasmGlobalDef}   # Global variables
    exports::Vector{WasmExport}
    elem_segments::Vector{WasmElemSegment}  # Element segments for table init
    data_segments::Vector{WasmDataSegment}  # Data segments for memory init
end

WasmModule() = WasmModule(CompositeType[], WasmImport[], WasmFunction[], WasmTable[], WasmMemory[], WasmGlobalDef[], WasmExport[], WasmElemSegment[], WasmDataSegment[])

# ============================================================================
# Module Building API
# ============================================================================

"""
    add_type!(mod, composite_type) -> type_idx

Add a composite type (FuncType, StructType, or ArrayType) to the module and return its index.
"""
function add_type!(mod::WasmModule, ct::CompositeType)::UInt32
    # Check if type already exists
    for (i, existing) in enumerate(mod.types)
        if types_equal(existing, ct)
            return UInt32(i - 1)
        end
    end
    push!(mod.types, ct)
    return UInt32(length(mod.types) - 1)
end

function types_equal(a::FuncType, b::FuncType)
    a.params == b.params && a.results == b.results
end

function types_equal(a::StructType, b::StructType)
    length(a.fields) == length(b.fields) &&
    all(fields_equal(af, bf) for (af, bf) in zip(a.fields, b.fields))
end

function types_equal(a::ArrayType, b::ArrayType)
    fields_equal(a.elem, b.elem)
end

types_equal(a::CompositeType, b::CompositeType) = false  # Different types

function fields_equal(a::FieldType, b::FieldType)
    a.valtype == b.valtype && a.mutable_ == b.mutable_
end

"""
    add_struct_type!(mod, fields) -> type_idx

Add a struct type to the module and return its index.
"""
function add_struct_type!(mod::WasmModule, fields::Vector{FieldType})::UInt32
    add_type!(mod, StructType(fields))
end

"""
    add_array_type!(mod, elem_type, mutable_=true) -> type_idx

Add an array type to the module and return its index.
"""
function add_array_type!(mod::WasmModule, elem_type::WasmValType, mutable_::Bool=true)::UInt32
    add_type!(mod, ArrayType(FieldType(elem_type, mutable_)))
end

"""
    add_import!(mod, module_name, field_name, params, results) -> func_idx

Add an imported function to the module and return its function index.
Imported functions come before local functions in the function index space.
"""
function add_import!(mod::WasmModule,
                     module_name::String,
                     field_name::String,
                     params::Vector{NumType},
                     results::Vector{NumType})::UInt32
    ft = FuncType(params, results)
    type_idx = add_type!(mod, ft)
    push!(mod.imports, WasmImport(module_name, field_name, 0x00, type_idx))
    return UInt32(length(mod.imports) - 1)  # Import function indices
end

# Overload for WasmValType (supports RefType, externref, etc.)
function add_import!(mod::WasmModule,
                     module_name::String,
                     field_name::String,
                     params::Vector{<:WasmValType},
                     results::Vector{<:WasmValType})::UInt32
    # Convert to WasmValType vectors
    param_vec = WasmValType[p for p in params]
    result_vec = WasmValType[r for r in results]
    ft = FuncType(param_vec, result_vec)
    type_idx = add_type!(mod, ft)
    push!(mod.imports, WasmImport(module_name, field_name, 0x00, type_idx))
    return UInt32(length(mod.imports) - 1)
end

"""
    num_imported_funcs(mod) -> Int

Return the number of imported functions (affects function index space).
"""
function num_imported_funcs(mod::WasmModule)::Int
    count(imp -> imp.kind == 0x00, mod.imports)
end

"""
    add_function!(mod, params, results, locals, body) -> func_idx

Add a function to the module and return its index.
Note: Local function indices start after imported functions.
Params and results can be NumType or WasmValType vectors.
"""
function add_function!(mod::WasmModule,
                       params::Vector{<:WasmValType},
                       results::Vector{<:WasmValType},
                       locals::Vector{<:WasmValType},
                       body::Vector{UInt8})::UInt32
    ft = FuncType(WasmValType[p for p in params], WasmValType[r for r in results])
    type_idx = add_type!(mod, ft)
    push!(mod.functions, WasmFunction(type_idx, WasmValType[l for l in locals], body))
    # Function index = number of imported functions + local function index
    return UInt32(num_imported_funcs(mod) + length(mod.functions) - 1)
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

"""
    add_global!(mod, valtype, mutable, init_value) -> global_idx

Add a global variable to the module and return its index.
The init_value should be a constant of the appropriate type.
"""
function add_global!(mod::WasmModule, valtype::WasmValType, mutable_::Bool, init_value)::UInt32
    # Generate initialization expression
    init = UInt8[]
    if valtype == I32
        push!(init, Opcode.I32_CONST)
        append!(init, encode_leb128_signed(Int32(init_value)))
    elseif valtype == I64
        push!(init, Opcode.I64_CONST)
        append!(init, encode_leb128_signed(Int64(init_value)))
    elseif valtype == F32
        push!(init, Opcode.F32_CONST)
        append!(init, reinterpret(UInt8, [Float32(init_value)]))
    elseif valtype == F64
        push!(init, Opcode.F64_CONST)
        append!(init, reinterpret(UInt8, [Float64(init_value)]))
    elseif valtype == ExternRef
        # externref initialized to null
        push!(init, Opcode.REF_NULL)
        push!(init, 0x6F)  # externref heap type
    else
        error("Unsupported global type: $valtype")
    end
    push!(init, Opcode.END)

    push!(mod.globals, WasmGlobalDef(valtype, mutable_, init))
    return UInt32(length(mod.globals) - 1)
end

"""
    add_global_export!(mod, name, global_idx)

Export a global variable.
"""
function add_global_export!(mod::WasmModule, name::String, global_idx::Integer)
    add_export!(mod, name, 3, global_idx)  # kind 3 = global
end

"""
    add_table!(mod, reftype, min, max=nothing) -> table_idx

Add a table to the module. Tables hold references (funcref or externref).
"""
function add_table!(mod::WasmModule, reftype::RefType, min::Integer, max::Union{Integer, Nothing}=nothing)::UInt32
    max_val = max === nothing ? nothing : UInt32(max)
    push!(mod.tables, WasmTable(reftype, UInt32(min), max_val))
    return UInt32(length(mod.tables) - 1)
end

"""
    add_table_export!(mod, name, table_idx)

Export a table.
"""
function add_table_export!(mod::WasmModule, name::String, table_idx::Integer)
    add_export!(mod, name, 1, table_idx)  # kind 1 = table
end

"""
    add_elem_segment!(mod, table_idx, offset, func_indices)

Add an element segment to initialize a table with function references.
"""
function add_elem_segment!(mod::WasmModule, table_idx::Integer, offset::Integer, func_indices::Vector{<:Integer})
    push!(mod.elem_segments, WasmElemSegment(UInt32(table_idx), UInt32(offset), UInt32[f for f in func_indices]))
    return mod
end

"""
    add_memory!(mod, min, max=nothing) -> memory_idx

Add a linear memory to the module. Size is in pages (64KB each).
"""
function add_memory!(mod::WasmModule, min::Integer, max::Union{Integer, Nothing}=nothing)::UInt32
    max_val = max === nothing ? nothing : UInt32(max)
    push!(mod.memories, WasmMemory(UInt32(min), max_val))
    return UInt32(length(mod.memories) - 1)
end

"""
    add_memory_export!(mod, name, memory_idx)

Export a memory.
"""
function add_memory_export!(mod::WasmModule, name::String, memory_idx::Integer)
    add_export!(mod, name, 2, memory_idx)  # kind 2 = memory
end

"""
    add_data_segment!(mod, memory_idx, offset, data)

Add a data segment to initialize linear memory with constant data.
Data can be a Vector{UInt8} or a String.
"""
function add_data_segment!(mod::WasmModule, memory_idx::Integer, offset::Integer, data::Vector{UInt8})
    push!(mod.data_segments, WasmDataSegment(UInt32(memory_idx), UInt32(offset), data))
    return mod
end

function add_data_segment!(mod::WasmModule, memory_idx::Integer, offset::Integer, data::String)
    add_data_segment!(mod, memory_idx, offset, Vector{UInt8}(codeunits(data)))
end

# ============================================================================
# Binary Serialization
# ============================================================================

const WASM_MAGIC = UInt8[0x00, 0x61, 0x73, 0x6D]  # \0asm
const WASM_VERSION = UInt8[0x01, 0x00, 0x00, 0x00]  # version 1

# Section IDs
const SECTION_TYPE = 0x01
const SECTION_IMPORT = 0x02
const SECTION_FUNCTION = 0x03
const SECTION_TABLE = 0x04
const SECTION_MEMORY = 0x05
const SECTION_GLOBAL = 0x06
const SECTION_EXPORT = 0x07
const SECTION_ELEMENT = 0x09
const SECTION_CODE = 0x0A
const SECTION_DATA = 0x0B

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
            for ct in mod.types
                write_composite_type!(section, ct)
            end
        end
    end

    # Import section
    if !isempty(mod.imports)
        write_section!(w, SECTION_IMPORT) do section
            write_u32!(section, length(mod.imports))
            for imp in mod.imports
                write_name!(section, imp.module_name)
                write_name!(section, imp.field_name)
                write_byte!(section, imp.kind)
                write_u32!(section, imp.type_idx)
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

    # Table section
    if !isempty(mod.tables)
        write_section!(w, SECTION_TABLE) do section
            write_u32!(section, length(mod.tables))
            for table in mod.tables
                write_valtype!(section, table.reftype)
                # Limits: 0x00 = min only, 0x01 = min and max
                if table.max === nothing
                    write_byte!(section, 0x00)
                    write_u32!(section, table.min)
                else
                    write_byte!(section, 0x01)
                    write_u32!(section, table.min)
                    write_u32!(section, table.max)
                end
            end
        end
    end

    # Memory section
    if !isempty(mod.memories)
        write_section!(w, SECTION_MEMORY) do section
            write_u32!(section, length(mod.memories))
            for mem in mod.memories
                # Limits: 0x00 = min only, 0x01 = min and max
                if mem.max === nothing
                    write_byte!(section, 0x00)
                    write_u32!(section, mem.min)
                else
                    write_byte!(section, 0x01)
                    write_u32!(section, mem.min)
                    write_u32!(section, mem.max)
                end
            end
        end
    end

    # Global section
    if !isempty(mod.globals)
        write_section!(w, SECTION_GLOBAL) do section
            write_u32!(section, length(mod.globals))
            for g in mod.globals
                # Global type: valtype + mutability
                write_valtype!(section, g.valtype)
                write_byte!(section, g.mutable_ ? 0x01 : 0x00)
                # Init expression (already includes END byte)
                append!(section.buffer, g.init)
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

    # Element section
    if !isempty(mod.elem_segments)
        write_section!(w, SECTION_ELEMENT) do section
            write_u32!(section, length(mod.elem_segments))
            for elem in mod.elem_segments
                # Element segment kind 0: active, table index 0, funcref
                # Binary format: flags (0) + offset expr + vec(funcidx)
                write_byte!(section, 0x00)  # flags: active segment, table 0
                # Offset expression (i32.const offset)
                push!(section.buffer, Opcode.I32_CONST)
                append!(section.buffer, encode_leb128_signed(Int32(elem.offset)))
                push!(section.buffer, Opcode.END)
                # Vector of function indices
                write_u32!(section, length(elem.func_indices))
                for func_idx in elem.func_indices
                    write_u32!(section, func_idx)
                end
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
                    for (count, valtype) in local_groups
                        write_u32!(body_writer, count)
                        write_valtype!(body_writer, valtype)
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

    # Data section
    if !isempty(mod.data_segments)
        write_section!(w, SECTION_DATA) do section
            write_u32!(section, length(mod.data_segments))
            for data in mod.data_segments
                # Active data segment (mode 0): memory index 0
                write_byte!(section, 0x00)
                # Offset expression (i32.const offset)
                push!(section.buffer, Opcode.I32_CONST)
                append!(section.buffer, encode_leb128_signed(Int32(data.offset)))
                push!(section.buffer, Opcode.END)
                # Data bytes
                write_u32!(section, length(data.data))
                append!(section.buffer, data.data)
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
function group_locals(locals::Vector{<:WasmValType})
    isempty(locals) && return Tuple{Int, WasmValType}[]

    groups = Tuple{Int, WasmValType}[]
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

# ============================================================================
# Composite Type Serialization (WasmGC)
# ============================================================================

# Type constructors for binary encoding
const FUNCTYPE_BYTE = 0x60
const STRUCTTYPE_BYTE = 0x5F
const ARRAYTYPE_BYTE = 0x5E

"""
Write a composite type to the type section.
"""
function write_composite_type!(w::WasmWriter, ft::FuncType)
    write_byte!(w, FUNCTYPE_BYTE)
    # Write params as a vector of valtypes
    write_u32!(w, length(ft.params))
    for p in ft.params
        write_valtype!(w, p)
    end
    # Write results as a vector of valtypes
    write_u32!(w, length(ft.results))
    for r in ft.results
        write_valtype!(w, r)
    end
end

function write_composite_type!(w::WasmWriter, st::StructType)
    write_byte!(w, STRUCTTYPE_BYTE)
    write_u32!(w, length(st.fields))
    for field in st.fields
        write_field_type!(w, field)
    end
end

function write_composite_type!(w::WasmWriter, at::ArrayType)
    write_byte!(w, ARRAYTYPE_BYTE)
    write_field_type!(w, at.elem)
end

"""
Write a field type (valtype + mutability).
"""
function write_field_type!(w::WasmWriter, ft::FieldType)
    write_valtype!(w, ft.valtype)
    write_byte!(w, ft.mutable_ ? 0x01 : 0x00)
end

"""
Write a value type (NumType, RefType, or packed type).
"""
function write_valtype!(w::WasmWriter, vt::NumType)
    write_byte!(w, UInt8(vt))
end

function write_valtype!(w::WasmWriter, vt::RefType)
    write_byte!(w, UInt8(vt))
end

function write_valtype!(w::WasmWriter, vt::UInt8)
    write_byte!(w, vt)
end

function write_valtype!(w::WasmWriter, vt::ConcreteRef)
    # Concrete reference type: (ref null $typeidx) or (ref $typeidx)
    # Binary format: 0x63 (nullable) or 0x64 (non-nullable) followed by heap type index
    if vt.nullable
        write_byte!(w, 0x63)  # ref null
    else
        write_byte!(w, 0x64)  # ref
    end
    # Heap type index is a signed LEB128 (s33)
    write_i32!(w, Int32(vt.type_idx))
end
