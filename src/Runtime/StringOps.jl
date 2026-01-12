# String Operations - Intrinsic string functions that compile to WASM
# These functions are recognized by the compiler and translated directly to WASM array operations.
#
# In WASM, strings are stored as i32 arrays (one element per character).
# These functions compile to direct array operations.

export str_char, str_setchar!, str_len, str_new, str_copy, str_substr

"""
    str_char(s::String, i::Int)::Int32

Get the character (as Int32 codepoint) at 1-based index `i` in string `s`.
Compiles to WASM `array.get` instruction.

Note: In WASM, strings are i32 arrays. This function assumes single-byte ASCII
characters for the Julia fallback.

# Example
```julia
s = "hello"
c = str_char(s, 1)  # Returns Int32('h') = 104
```
"""
@noinline function str_char(s::String, i::Int)::Int32
    # Julia fallback - return codeunit as Int32
    # Use inferencebarrier to prevent constant folding
    return Base.inferencebarrier(Int32(codeunit(s, i)))::Int32
end

# Type-stable version for Int32 index
@noinline function str_char(s::String, i::Int32)::Int32
    # Use inferencebarrier to prevent constant folding
    return Base.inferencebarrier(Int32(codeunit(s, Int(i))))::Int32
end

"""
    str_setchar!(s::String, i::Int, c::Int32)::Nothing

Set the character at 1-based index `i` in string `s` to character `c`.
Compiles to WASM `array.set` instruction.

Note: In Julia, strings are immutable. This function only works in WASM
where strings are mutable i32 arrays.
"""
# Global to prevent optimization of str_setchar!
const _SETCHAR_SINK = Ref{Int32}(0)

@noinline function str_setchar!(s::String, i::Int, c::Int32)::Nothing
    # Julia fallback - no-op (strings are immutable in Julia)
    # But use _SETCHAR_SINK to prevent compiler from optimizing away the call
    _SETCHAR_SINK[] = c
    return nothing
end

# Int32 index version
@noinline function str_setchar!(s::String, i::Int32, c::Int32)::Nothing
    # Use _SETCHAR_SINK to prevent optimization
    _SETCHAR_SINK[] = c
    return nothing
end

"""
    str_len(s::String)::Int32

Get the length of string `s` as Int32.
Compiles to WASM `array.len` instruction.

# Example
```julia
s = "hello"
len = str_len(s)  # Returns Int32(5)
```
"""
@noinline function str_len(s::String)::Int32
    # Use inferencebarrier to prevent constant folding
    return Base.inferencebarrier(Int32(length(s)))::Int32
end

"""
    str_new(len::Int32)::String

Create a new string of the given length, filled with null characters.
Compiles to WASM `array.new_default` instruction.

# Example
```julia
s = str_new(Int32(10))  # Creates empty string of length 10
```
"""
@noinline function str_new(len::Int32)::String
    # Julia fallback - create string of nulls
    # Use inferencebarrier to prevent constant folding
    return Base.inferencebarrier(repeat("\0", len))::String
end

# Also support Int64 version
@noinline function str_new(len::Int)::String
    return Base.inferencebarrier(repeat("\0", len))::String
end

"""
    str_copy(src::String, src_pos::Int32, dst::String, dst_pos::Int32, len::Int32)::Nothing

Copy `len` characters from `src` starting at `src_pos` to `dst` starting at `dst_pos`.
Compiles to WASM `array.copy` instruction.

Note: Positions are 1-based (Julia convention), but internally converted to 0-based for WASM.
"""
# Global to prevent optimization of str_copy
const _STRCOPY_SINK = Ref{Int32}(0)

@noinline function str_copy(src::String, src_pos::Int32, dst::String, dst_pos::Int32, len::Int32)::Nothing
    # Julia fallback - no-op (strings are immutable in Julia)
    # Use _STRCOPY_SINK to prevent optimization
    _STRCOPY_SINK[] = len
    return nothing
end

"""
    str_substr(s::String, start::Int32, len::Int32)::String

Extract a substring of `len` characters starting at 1-based index `start`.
Returns a new string containing the extracted characters.

Compiles to WASM operations: str_new + str_copy.

# Example
```julia
s = "hello world"
sub = str_substr(s, Int32(7), Int32(5))  # Returns "world"
```
"""
@noinline function str_substr(s::String, start::Int32, len::Int32)::String
    # Julia fallback - use SubString
    return Base.inferencebarrier(String(SubString(s, start, start + len - 1)))::String
end

# Int64 versions for convenience
@noinline function str_substr(s::String, start::Int, len::Int)::String
    return str_substr(s, Int32(start), Int32(len))
end

"""
    str_eq(a::String, b::String)::Bool

Check if two strings are equal character by character.
Returns true if all characters match and lengths are equal.

# Example
```julia
str_eq("hello", "hello")  # Returns true
str_eq("hello", "world")  # Returns false
```
"""
@noinline function str_eq(a::String, b::String)::Bool
    return Base.inferencebarrier(a == b)::Bool
end
