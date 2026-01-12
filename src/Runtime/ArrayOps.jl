# Array Operations - Intrinsic array functions that compile to WASM
# These functions are recognized by the compiler and translated directly to WasmGC array operations.
#
# This follows the same pattern as StringOps.jl - we provide Julia fallbacks
# that work in Julia, and the compiler replaces them with WasmGC operations.

export arr_new, arr_get, arr_set!, arr_len, arr_fill!

# Global sinks to prevent optimization (same pattern as StringOps.jl)
const _ARR_SET_SINK = Ref{Int32}(0)
const _ARR_FILL_SINK = Ref{Int32}(0)

"""
    arr_new(::Type{T}, len::Int32)::Vector{T}

Create a new array of type T with the given length, filled with zeros/default.
Compiles to WASM `array.new_default` instruction.

# Example
```julia
arr = arr_new(Int32, Int32(10))  # Creates Int32 array of length 10
```
"""
@noinline function arr_new(::Type{Int32}, len::Int32)::Vector{Int32}
    return Base.inferencebarrier(zeros(Int32, len))::Vector{Int32}
end

@noinline function arr_new(::Type{Int64}, len::Int32)::Vector{Int64}
    return Base.inferencebarrier(zeros(Int64, len))::Vector{Int64}
end

@noinline function arr_new(::Type{Float32}, len::Int32)::Vector{Float32}
    return Base.inferencebarrier(zeros(Float32, len))::Vector{Float32}
end

@noinline function arr_new(::Type{Float64}, len::Int32)::Vector{Float64}
    return Base.inferencebarrier(zeros(Float64, len))::Vector{Float64}
end

# Int64 length versions
@noinline function arr_new(::Type{T}, len::Int) where T
    return arr_new(T, Int32(len))
end

"""
    arr_get(arr::Vector{T}, i::Int32)::T

Get the element at 1-based index `i` in array `arr`.
Compiles to WASM `array.get` instruction.

# Example
```julia
arr = arr_new(Int32, Int32(5))
val = arr_get(arr, Int32(1))  # Gets first element
```
"""
@noinline function arr_get(arr::Vector{Int32}, i::Int32)::Int32
    return Base.inferencebarrier(arr[i])::Int32
end

@noinline function arr_get(arr::Vector{Int32}, i::Int)::Int32
    return Base.inferencebarrier(arr[i])::Int32
end

@noinline function arr_get(arr::Vector{Int64}, i::Int32)::Int64
    return Base.inferencebarrier(arr[i])::Int64
end

@noinline function arr_get(arr::Vector{Int64}, i::Int)::Int64
    return Base.inferencebarrier(arr[i])::Int64
end

@noinline function arr_get(arr::Vector{Float32}, i::Int32)::Float32
    return Base.inferencebarrier(arr[i])::Float32
end

@noinline function arr_get(arr::Vector{Float64}, i::Int32)::Float64
    return Base.inferencebarrier(arr[i])::Float64
end

"""
    arr_set!(arr::Vector{T}, i::Int32, val::T)::Nothing

Set the element at 1-based index `i` in array `arr` to `val`.
Compiles to WASM `array.set` instruction.

# Example
```julia
arr = arr_new(Int32, Int32(5))
arr_set!(arr, Int32(1), Int32(42))  # Sets first element to 42
```
"""
@noinline function arr_set!(arr::Vector{Int32}, i::Int32, val::Int32)::Nothing
    arr[i] = val
    _ARR_SET_SINK[] = val  # Prevent optimization
    return nothing
end

@noinline function arr_set!(arr::Vector{Int32}, i::Int, val::Int32)::Nothing
    arr[i] = val
    _ARR_SET_SINK[] = val
    return nothing
end

@noinline function arr_set!(arr::Vector{Int64}, i::Int32, val::Int64)::Nothing
    arr[i] = val
    _ARR_SET_SINK[] = Int32(val & 0xFFFFFFFF)
    return nothing
end

@noinline function arr_set!(arr::Vector{Float32}, i::Int32, val::Float32)::Nothing
    arr[i] = val
    _ARR_SET_SINK[] = Int32(0)
    return nothing
end

@noinline function arr_set!(arr::Vector{Float64}, i::Int32, val::Float64)::Nothing
    arr[i] = val
    _ARR_SET_SINK[] = Int32(0)
    return nothing
end

"""
    arr_len(arr::Vector{T})::Int32

Get the length of array `arr` as Int32.
Compiles to WASM `array.len` instruction.

# Example
```julia
arr = arr_new(Int32, Int32(5))
len = arr_len(arr)  # Returns Int32(5)
```
"""
@noinline function arr_len(arr::Vector{T})::Int32 where T
    return Base.inferencebarrier(Int32(length(arr)))::Int32
end

"""
    arr_fill!(arr::Vector{T}, val::T)::Nothing

Fill the entire array with a value. Uses a loop internally.
Compiles to a WASM loop with array.set instructions.

# Example
```julia
arr = arr_new(Int32, Int32(5))
arr_fill!(arr, Int32(42))  # All elements now 42
```
"""
@noinline function arr_fill!(arr::Vector{Int32}, val::Int32)::Nothing
    len = arr_len(arr)
    i = Int32(1)
    while i <= len
        arr_set!(arr, i, val)
        i = i + Int32(1)
    end
    _ARR_FILL_SINK[] = val
    return nothing
end
