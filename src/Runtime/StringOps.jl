# String Operations - Intrinsic string functions that compile to WASM
# These functions are recognized by the compiler and translated directly to WASM array operations.
#
# In WASM, strings are stored as i32 arrays (one element per character).
# These functions compile to direct array operations.

export str_char, str_setchar!, str_len, str_new, str_copy, str_substr, str_eq, str_hash,
       str_contains, str_find, str_uppercase, str_lowercase, str_trim, str_startswith, str_endswith,
       digit_to_str, int_to_string, float_to_string

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

"""
    str_hash(s::String)::Int32

Compute a hash value for the string.
Uses Java-style hash: h = 31 * h + char[i] for each character.
Result is masked to positive Int32 range.

This is the same algorithm used by SimpleDict for Int32 keys,
extended to work with strings.

# Example
```julia
h = str_hash("hello")  # Returns consistent Int32 hash value
```
"""
@noinline function str_hash(s::String)::Int32
    # Julia fallback - compute hash
    h = Int32(0)
    for c in s
        h = Int32(31) * h + Int32(UInt8(c))
        h = h & Int32(0x7FFFFFFF)  # Keep positive
    end
    return Base.inferencebarrier(h)::Int32
end

# =============================================================================
# BROWSER-010: New String Operations (expanded Julia subset for dart2wasm parity)
# =============================================================================

"""
    str_find(haystack::String, needle::String)::Int32

Find the first occurrence of `needle` in `haystack`.
Returns 1-based index of first match, or Int32(0) if not found.

Compiles to a WASM loop with character comparisons.

# Example
```julia
str_find("hello world", "world")  # Returns Int32(7)
str_find("hello world", "xyz")    # Returns Int32(0)
```
"""
@noinline function str_find(haystack::String, needle::String)::Int32
    haystack_len = str_len(haystack)
    needle_len = str_len(needle)

    # Empty needle always found at position 1
    if needle_len == Int32(0)
        return Int32(1)
    end

    # Needle longer than haystack - not found
    if needle_len > haystack_len
        return Int32(0)
    end

    # Search for needle
    i = Int32(1)
    last_start = haystack_len - needle_len + Int32(1)
    while i <= last_start
        # Check if needle matches at position i
        found = true
        j = Int32(1)
        while j <= needle_len
            if str_char(haystack, i + j - Int32(1)) != str_char(needle, j)
                found = false
                break
            end
            j = j + Int32(1)
        end
        if found
            return i
        end
        i = i + Int32(1)
    end

    return Int32(0)
end

"""
    str_contains(haystack::String, needle::String)::Bool

Check if `haystack` contains `needle`.
Returns true if `needle` is found anywhere in `haystack`.

# Example
```julia
str_contains("hello world", "world")  # Returns true
str_contains("hello world", "xyz")    # Returns false
```
"""
@noinline function str_contains(haystack::String, needle::String)::Bool
    return Base.inferencebarrier(str_find(haystack, needle) > Int32(0))::Bool
end

"""
    str_startswith(s::String, prefix::String)::Bool

Check if string `s` starts with `prefix`.

# Example
```julia
str_startswith("hello world", "hello")  # Returns true
str_startswith("hello world", "world")  # Returns false
```
"""
@noinline function str_startswith(s::String, prefix::String)::Bool
    s_len = str_len(s)
    prefix_len = str_len(prefix)

    if prefix_len > s_len
        return Base.inferencebarrier(false)::Bool
    end

    i = Int32(1)
    while i <= prefix_len
        if str_char(s, i) != str_char(prefix, i)
            return Base.inferencebarrier(false)::Bool
        end
        i = i + Int32(1)
    end

    return Base.inferencebarrier(true)::Bool
end

"""
    str_endswith(s::String, suffix::String)::Bool

Check if string `s` ends with `suffix`.

# Example
```julia
str_endswith("hello world", "world")  # Returns true
str_endswith("hello world", "hello")  # Returns false
```
"""
@noinline function str_endswith(s::String, suffix::String)::Bool
    s_len = str_len(s)
    suffix_len = str_len(suffix)

    if suffix_len > s_len
        return Base.inferencebarrier(false)::Bool
    end

    start_pos = s_len - suffix_len + Int32(1)
    i = Int32(1)
    while i <= suffix_len
        if str_char(s, start_pos + i - Int32(1)) != str_char(suffix, i)
            return Base.inferencebarrier(false)::Bool
        end
        i = i + Int32(1)
    end

    return Base.inferencebarrier(true)::Bool
end

# ASCII character code constants
const _CHAR_A_UPPER = Int32(65)   # 'A'
const _CHAR_Z_UPPER = Int32(90)   # 'Z'
const _CHAR_A_LOWER = Int32(97)   # 'a'
const _CHAR_Z_LOWER = Int32(122)  # 'z'
const _CHAR_CASE_DIFF = Int32(32) # Difference between upper and lower case

"""
    str_uppercase(s::String)::String

Convert all lowercase ASCII letters in the string to uppercase.
Non-ASCII characters are unchanged.

Returns a new string with uppercase letters.

# Example
```julia
str_uppercase("Hello World")  # Returns "HELLO WORLD"
str_uppercase("abc123")       # Returns "ABC123"
```
"""
@noinline function str_uppercase(s::String)::String
    len = str_len(s)
    result = str_new(len)

    i = Int32(1)
    while i <= len
        c = str_char(s, i)
        # Check if lowercase letter
        if c >= _CHAR_A_LOWER && c <= _CHAR_Z_LOWER
            c = c - _CHAR_CASE_DIFF  # Convert to uppercase
        end
        str_setchar!(result, i, c)
        i = i + Int32(1)
    end

    return Base.inferencebarrier(result)::String
end

"""
    str_lowercase(s::String)::String

Convert all uppercase ASCII letters in the string to lowercase.
Non-ASCII characters are unchanged.

Returns a new string with lowercase letters.

# Example
```julia
str_lowercase("Hello World")  # Returns "hello world"
str_lowercase("ABC123")       # Returns "abc123"
```
"""
@noinline function str_lowercase(s::String)::String
    len = str_len(s)
    result = str_new(len)

    i = Int32(1)
    while i <= len
        c = str_char(s, i)
        # Check if uppercase letter
        if c >= _CHAR_A_UPPER && c <= _CHAR_Z_UPPER
            c = c + _CHAR_CASE_DIFF  # Convert to lowercase
        end
        str_setchar!(result, i, c)
        i = i + Int32(1)
    end

    return Base.inferencebarrier(result)::String
end

# ASCII whitespace characters
const _CHAR_SPACE = Int32(32)  # ' '
const _CHAR_TAB = Int32(9)     # '\t'
const _CHAR_NEWLINE = Int32(10) # '\n'
const _CHAR_CR = Int32(13)     # '\r'

"""
    _is_whitespace(c::Int32)::Bool

Internal helper to check if a character is ASCII whitespace.
"""
@inline function _is_whitespace(c::Int32)::Bool
    return c == _CHAR_SPACE || c == _CHAR_TAB || c == _CHAR_NEWLINE || c == _CHAR_CR
end

"""
    str_trim(s::String)::String

Remove leading and trailing ASCII whitespace from the string.
Whitespace includes: space, tab, newline, carriage return.

Returns a new string with whitespace trimmed.

# Example
```julia
str_trim("  hello  ")     # Returns "hello"
str_trim("\\t\\nhello\\n")  # Returns "hello"
```
"""
@noinline function str_trim(s::String)::String
    len = str_len(s)

    # Handle empty string
    if len == Int32(0)
        return Base.inferencebarrier(s)::String
    end

    # Find start (skip leading whitespace)
    start_pos = Int32(1)
    while start_pos <= len && _is_whitespace(str_char(s, start_pos))
        start_pos = start_pos + Int32(1)
    end

    # All whitespace
    if start_pos > len
        return Base.inferencebarrier("")::String
    end

    # Find end (skip trailing whitespace)
    end_pos = len
    while end_pos >= start_pos && _is_whitespace(str_char(s, end_pos))
        end_pos = end_pos - Int32(1)
    end

    # Extract substring
    new_len = end_pos - start_pos + Int32(1)
    return str_substr(s, start_pos, new_len)
end

# =============================================================================
# WASM-054: Integer to String Conversion
# =============================================================================

"""
    digit_to_str(d::Int32)::String

Convert a single digit (0-9) to its string representation.
This is a helper function for int_to_string.

Works in both Julia and compiles to WASM.

# Example
```julia
digit_to_str(Int32(5))  # Returns "5"
digit_to_str(Int32(0))  # Returns "0"
```
"""
@noinline function digit_to_str(d::Int32)::String
    if d == Int32(0)
        return "0"
    elseif d == Int32(1)
        return "1"
    elseif d == Int32(2)
        return "2"
    elseif d == Int32(3)
        return "3"
    elseif d == Int32(4)
        return "4"
    elseif d == Int32(5)
        return "5"
    elseif d == Int32(6)
        return "6"
    elseif d == Int32(7)
        return "7"
    elseif d == Int32(8)
        return "8"
    else  # d == 9
        return "9"
    end
end

"""
    int_to_string(n::Int32)::String

Convert an Int32 to its string representation.
Handles positive, negative, and zero values.

This function is designed to work in both Julia and compile to WASM.
It uses string concatenation which compiles to WASM operations.

# Example
```julia
int_to_string(Int32(12345))  # Returns "12345"
int_to_string(Int32(-42))    # Returns "-42"
int_to_string(Int32(0))      # Returns "0"
```
"""
@noinline function int_to_string(n::Int32)::String
    # Handle zero special case
    if n == Int32(0)
        return "0"
    end

    # Handle negative numbers
    negative = n < Int32(0)
    if negative
        n = -n
    end

    # Build string from least significant digit
    # We prepend each digit to build the string
    result = ""
    while n > Int32(0)
        digit = n % Int32(10)
        result = digit_to_str(digit) * result  # prepend digit
        n = n ÷ Int32(10)
    end

    # Add sign if negative
    if negative
        result = "-" * result
    end

    return result
end

"""
    float_to_string(f::Float32)::String

Convert a Float32 to its string representation.
Handles positive, negative, and zero values.
Shows one decimal place for simplicity.

This function is designed to work in both Julia and compile to WASM.

# Example
```julia
float_to_string(Float32(3.14))   # Returns "3.1"
float_to_string(Float32(-2.5))   # Returns "-2.5"
float_to_string(Float32(0.0))    # Returns "0.0"
```
"""
@noinline function float_to_string(f::Float32)::String
    # Handle negative
    negative = f < Float32(0.0)
    if negative
        f = -f
    end

    # Get integer part
    int_part = Int32(floor(f))

    # Get fractional part (one decimal place)
    frac_part = Int32(round((f - Float32(int_part)) * Float32(10.0)))

    # Handle rounding that causes frac_part to be 10
    if frac_part >= Int32(10)
        frac_part = Int32(0)
        int_part = int_part + Int32(1)
    end

    int_str = int_to_string(int_part)
    frac_str = int_to_string(frac_part)

    if negative
        return "-" * int_str * "." * frac_str
    end
    return int_str * "." * frac_str
end

"""
    float_to_string(f::Float64)::String

Convert a Float64 to its string representation.
Converts to Float32 internally for simplicity.
"""
@noinline function float_to_string(f::Float64)::String
    return float_to_string(Float32(f))
end
