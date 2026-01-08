# Binary serialization for WebAssembly modules
# Reference: https://webassembly.github.io/spec/core/binary/conventions.html

export encode_leb128_unsigned, encode_leb128_signed, WasmWriter

# ============================================================================
# LEB128 Encoding (Section 5.2.2)
# ============================================================================

"""
    encode_leb128_unsigned(value) -> Vector{UInt8}

Encode an unsigned integer using LEB128 (Little Endian Base 128).
"""
function encode_leb128_unsigned(value::Integer)::Vector{UInt8}
    result = UInt8[]
    value = unsigned(value)
    while true
        byte = UInt8(value & 0x7F)
        value >>= 7
        if value != 0
            byte |= 0x80  # Set continuation bit
        end
        push!(result, byte)
        value == 0 && break
    end
    return result
end

"""
    encode_leb128_signed(value) -> Vector{UInt8}

Encode a signed integer using LEB128 (Little Endian Base 128).
"""
function encode_leb128_signed(value::Integer)::Vector{UInt8}
    result = UInt8[]
    more = true
    while more
        byte = UInt8(value & 0x7F)
        value >>= 7
        # Check if we need more bytes
        if (value == 0 && (byte & 0x40) == 0) || (value == -1 && (byte & 0x40) != 0)
            more = false
        else
            byte |= 0x80  # Set continuation bit
        end
        push!(result, byte)
    end
    return result
end

# ============================================================================
# WasmWriter - Binary stream builder
# ============================================================================

"""
    WasmWriter

A mutable buffer for building WebAssembly binary data.
"""
mutable struct WasmWriter
    buffer::Vector{UInt8}
end

WasmWriter() = WasmWriter(UInt8[])

Base.length(w::WasmWriter) = length(w.buffer)
bytes(w::WasmWriter) = w.buffer

"""
Write raw bytes to the buffer.
"""
function write_bytes!(w::WasmWriter, data::Vector{UInt8})
    append!(w.buffer, data)
    return w
end

function write_bytes!(w::WasmWriter, data::UInt8...)
    append!(w.buffer, data)
    return w
end

"""
Write a single byte.
"""
function write_byte!(w::WasmWriter, b::UInt8)
    push!(w.buffer, b)
    return w
end

"""
Write an unsigned LEB128 integer.
"""
function write_u32!(w::WasmWriter, value::Integer)
    append!(w.buffer, encode_leb128_unsigned(value))
    return w
end

"""
Write a signed LEB128 integer.
"""
function write_i32!(w::WasmWriter, value::Integer)
    append!(w.buffer, encode_leb128_signed(value))
    return w
end

function write_i64!(w::WasmWriter, value::Integer)
    append!(w.buffer, encode_leb128_signed(value))
    return w
end

"""
Write a 32-bit float (little endian).
"""
function write_f32!(w::WasmWriter, value::Float32)
    append!(w.buffer, reinterpret(UInt8, [value]))
    return w
end

"""
Write a 64-bit float (little endian).
"""
function write_f64!(w::WasmWriter, value::Float64)
    append!(w.buffer, reinterpret(UInt8, [value]))
    return w
end

"""
Write a vector with its length prefix (LEB128).
"""
function write_vec!(w::WasmWriter, items::Vector)
    write_u32!(w, length(items))
    for item in items
        write_item!(w, item)
    end
    return w
end

# Generic item writer - override for specific types
write_item!(w::WasmWriter, b::UInt8) = write_byte!(w, b)
write_item!(w::WasmWriter, n::NumType) = write_byte!(w, UInt8(n))

"""
Write a name (UTF-8 string with length prefix).
"""
function write_name!(w::WasmWriter, name::String)
    name_bytes = Vector{UInt8}(name)
    write_u32!(w, length(name_bytes))
    append!(w.buffer, name_bytes)
    return w
end
