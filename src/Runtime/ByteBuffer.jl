# ByteBuffer - A simple byte buffer for WASM-compatible I/O
#
# This provides an I/O abstraction that can be compiled to WASM.
# Used by the tokenizer to read characters from a string.
#
# Design:
# - Stores string as a byte array
# - Tracks current position
# - Provides read, position, eof operations
# - All operations are WASM-compilable

export ByteBuffer, bb_new, bb_read, bb_peek, bb_position, bb_eof, bb_seek, bb_remaining
export is_ascii_digit, is_ascii_alpha, is_ascii_alphanum, is_ascii_space
export is_ascii_newline, is_ascii_hex, is_identifier_start, is_identifier_char, is_operator_char
export EOF_CHAR

"""
ByteBuffer - A simple in-memory byte buffer for reading characters.

Designed to be WASM-compilable. Replaces IOBuffer for tokenizer use.
"""
mutable struct ByteBuffer
    data::Vector{Int32}    # Character codepoints (stored as i32)
    pos::Int32             # Current position (1-indexed)
    len::Int32             # Total length
end

"""
    bb_new(s::String)::ByteBuffer

Create a new ByteBuffer from a string.
"""
@noinline function bb_new(data::Vector{Int32})::ByteBuffer
    len = arr_len(data)
    return ByteBuffer(data, Int32(1), len)
end

"""
    bb_eof(buf::ByteBuffer)::Bool

Check if at end of buffer.
"""
@noinline function bb_eof(buf::ByteBuffer)::Bool
    return buf.pos > buf.len
end

"""
    bb_position(buf::ByteBuffer)::Int32

Get current position (1-indexed).
"""
@noinline function bb_position(buf::ByteBuffer)::Int32
    return buf.pos
end

"""
    bb_peek(buf::ByteBuffer)::Int32

Peek at current character without advancing. Returns -1 at EOF.
"""
@noinline function bb_peek(buf::ByteBuffer)::Int32
    if buf.pos > buf.len
        return Int32(-1)  # EOF marker
    end
    return arr_get(buf.data, buf.pos)
end

"""
    bb_read(buf::ByteBuffer)::Int32

Read current character and advance position. Returns -1 at EOF.
"""
@noinline function bb_read(buf::ByteBuffer)::Int32
    if buf.pos > buf.len
        return Int32(-1)  # EOF marker
    end
    c = arr_get(buf.data, buf.pos)
    buf.pos = buf.pos + Int32(1)
    return c
end

"""
    bb_seek(buf::ByteBuffer, pos::Int32)::Nothing

Seek to a specific position.
"""
@noinline function bb_seek(buf::ByteBuffer, pos::Int32)::Nothing
    buf.pos = pos
    return nothing
end

"""
    bb_remaining(buf::ByteBuffer)::Int32

Get number of remaining characters.
"""
@noinline function bb_remaining(buf::ByteBuffer)::Int32
    if buf.pos > buf.len
        return Int32(0)
    end
    return buf.len - buf.pos + Int32(1)
end

# Character classification functions (ASCII subset)
# These are needed by the tokenizer

const EOF_CHAR = Int32(-1)

"""
    is_ascii_digit(c::Int32)::Bool

Check if character is ASCII digit (0-9).
"""
@noinline function is_ascii_digit(c::Int32)::Bool
    return c >= Int32(48) && c <= Int32(57)  # '0' = 48, '9' = 57
end

"""
    is_ascii_alpha(c::Int32)::Bool

Check if character is ASCII letter (a-z, A-Z).
"""
@noinline function is_ascii_alpha(c::Int32)::Bool
    lower = c >= Int32(97) && c <= Int32(122)  # 'a' = 97, 'z' = 122
    upper = c >= Int32(65) && c <= Int32(90)   # 'A' = 65, 'Z' = 90
    return lower || upper
end

"""
    is_ascii_alphanum(c::Int32)::Bool

Check if character is ASCII alphanumeric.
"""
@noinline function is_ascii_alphanum(c::Int32)::Bool
    return is_ascii_digit(c) || is_ascii_alpha(c)
end

"""
    is_ascii_space(c::Int32)::Bool

Check if character is ASCII whitespace (space, tab, newline, etc.).
"""
@noinline function is_ascii_space(c::Int32)::Bool
    return c == Int32(32) ||   # space
           c == Int32(9) ||    # tab
           c == Int32(10) ||   # newline
           c == Int32(13) ||   # carriage return
           c == Int32(12) ||   # form feed
           c == Int32(11)      # vertical tab
end

"""
    is_ascii_newline(c::Int32)::Bool

Check if character is newline.
"""
@noinline function is_ascii_newline(c::Int32)::Bool
    return c == Int32(10) || c == Int32(13)  # LF or CR
end

"""
    is_ascii_hex(c::Int32)::Bool

Check if character is hex digit (0-9, a-f, A-F).
"""
@noinline function is_ascii_hex(c::Int32)::Bool
    digit = c >= Int32(48) && c <= Int32(57)   # 0-9
    lower = c >= Int32(97) && c <= Int32(102)  # a-f
    upper = c >= Int32(65) && c <= Int32(70)   # A-F
    return digit || lower || upper
end

"""
    is_identifier_start(c::Int32)::Bool

Check if character can start an identifier (letter or underscore).
"""
@noinline function is_identifier_start(c::Int32)::Bool
    return is_ascii_alpha(c) || c == Int32(95)  # '_' = 95
end

"""
    is_identifier_char(c::Int32)::Bool

Check if character can continue an identifier (alphanumeric or underscore).
"""
@noinline function is_identifier_char(c::Int32)::Bool
    return is_ascii_alphanum(c) || c == Int32(95)
end

"""
    is_operator_char(c::Int32)::Bool

Check if character is an operator character.
"""
@noinline function is_operator_char(c::Int32)::Bool
    return c == Int32(43) ||   # +
           c == Int32(45) ||   # -
           c == Int32(42) ||   # *
           c == Int32(47) ||   # /
           c == Int32(37) ||   # %
           c == Int32(94) ||   # ^
           c == Int32(38) ||   # &
           c == Int32(124) ||  # |
           c == Int32(60) ||   # <
           c == Int32(62) ||   # >
           c == Int32(61) ||   # =
           c == Int32(33) ||   # !
           c == Int32(126) ||  # ~
           c == Int32(58) ||   # :
           c == Int32(36)      # $
end
