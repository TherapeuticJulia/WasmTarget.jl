using WasmTarget
using Test

include("utils.jl")

# Recursive test functions (must be at module level for proper GlobalRef resolution)
@noinline function test_factorial_rec(n::Int32)::Int32
    if n <= Int32(1)
        return Int32(1)
    else
        return n * test_factorial_rec(n - Int32(1))
    end
end

@noinline function test_fib(n::Int32)::Int32
    if n <= Int32(1)
        return n
    else
        return test_fib(n - Int32(1)) + test_fib(n - Int32(2))
    end
end

@noinline function test_sum_rec(n::Int32)::Int32
    if n <= Int32(0)
        return Int32(0)
    else
        return n + test_sum_rec(n - Int32(1))
    end
end

# Mutual recursion test functions (BROWSER-013)
@noinline function is_even_mutual(n::Int32)::Int32
    if n == Int32(0)
        return Int32(1)  # true
    else
        return is_odd_mutual(n - Int32(1))
    end
end

@noinline function is_odd_mutual(n::Int32)::Int32
    if n == Int32(0)
        return Int32(0)  # false
    else
        return is_even_mutual(n - Int32(1))
    end
end

# Deep recursion test function (BROWSER-013)
@noinline function deep_recursion_test(n::Int32, depth::Int32)::Int32
    if depth <= Int32(0)
        return n
    else
        return deep_recursion_test(n + Int32(1), depth - Int32(1))
    end
end

# Complex while loop condition test (BROWSER-013)
@noinline function complex_while_test(n::Int32)::Int32
    result::Int32 = Int32(0)
    i::Int32 = Int32(0)
    @inbounds while i < n && result < Int32(100)
        result = result + i
        i = i + Int32(1)
    end
    return result
end

# Nested conditional test function (BROWSER-013)
@noinline function nested_cond_test(a::Int32, b::Int32)::Int32
    if a > Int32(0)
        if b > Int32(0)
            return a + b
        else
            return a - b
        end
    else
        if b > Int32(0)
            return b - a
        else
            return a * b
        end
    end
end

# Multi-branch if-elseif-else test (BROWSER-013)
@noinline function classify_number_test(n::Int32)::Int32
    if n < Int32(0)
        return Int32(-1)  # negative
    elseif n == Int32(0)
        return Int32(0)   # zero
    else
        return Int32(1)   # positive
    end
end

# Struct for testing compiled struct field access
mutable struct TestPoint2D
    x::Int32
    y::Int32
end

# Function that creates a struct and accesses its fields
# Uses inferencebarrier to prevent Julia optimizer from eliminating the struct
@noinline function test_point_sum(x::Int32, y::Int32)::Int32
    p = Base.inferencebarrier(TestPoint2D(x, y))::TestPoint2D
    return p.x + p.y
end

@noinline function test_point_diff(x::Int32, y::Int32)::Int32
    p = Base.inferencebarrier(TestPoint2D(x, y))::TestPoint2D
    return p.x - p.y
end

# Float operations test
@noinline function test_float_add(a::Float64, b::Float64)::Float64
    return a + b
end

@noinline function test_float_mul(a::Float64, b::Float64)::Float64
    return a * b
end

# Branching test
@noinline function test_branch(a::Int32, b::Int32)::Int32
    sum = a + b
    if sum > Int32(100)
        return sum - Int32(50)
    else
        return sum * Int32(2)
    end
end

# Cross-function call test functions (must be at module level)
@noinline function cross_helper_double(x::Int32)::Int32
    return x * Int32(2)
end

@noinline function cross_use_helper(x::Int32)::Int32
    return cross_helper_double(x) + Int32(1)
end

# Multiple dispatch test functions
@noinline function dispatch_process(x::Int32)::Int32
    return x * Int32(2)
end

@noinline function dispatch_process(x::Int64)::Int64
    return x * Int64(3)
end

@noinline function dispatch_use_i32(x::Int32)::Int32
    return dispatch_process(x) + Int32(1)
end

@noinline function dispatch_use_i64(x::Int64)::Int64
    return dispatch_process(x) + Int64(1)
end

@testset "WasmTarget.jl" begin

    # ========================================================================
    # Phase 1: Infrastructure Tests - Verify the test harness works
    # ========================================================================
    @testset "Phase 1: Test Harness Infrastructure" begin

        @testset "LEB128 Encoding" begin
            # Test unsigned LEB128
            @test WasmTarget.encode_leb128_unsigned(0) == [0x00]
            @test WasmTarget.encode_leb128_unsigned(1) == [0x01]
            @test WasmTarget.encode_leb128_unsigned(127) == [0x7F]
            @test WasmTarget.encode_leb128_unsigned(128) == [0x80, 0x01]
            @test WasmTarget.encode_leb128_unsigned(255) == [0xFF, 0x01]
            @test WasmTarget.encode_leb128_unsigned(624485) == [0xE5, 0x8E, 0x26]

            # Test signed LEB128
            @test WasmTarget.encode_leb128_signed(0) == [0x00]
            @test WasmTarget.encode_leb128_signed(1) == [0x01]
            @test WasmTarget.encode_leb128_signed(-1) == [0x7F]
            @test WasmTarget.encode_leb128_signed(63) == [0x3F]
            @test WasmTarget.encode_leb128_signed(-64) == [0x40]
            @test WasmTarget.encode_leb128_signed(64) == [0xC0, 0x00]
            @test WasmTarget.encode_leb128_signed(-65) == [0xBF, 0x7F]
        end

        @testset "Hardcoded Wasm Binary - i32.add" begin
            # Hand-assembled Wasm binary that exports an i32.add function
            # This tests that our Node.js harness can execute Wasm
            #
            # WAT equivalent:
            # (module
            #   (func (export "add") (param i32 i32) (result i32)
            #     local.get 0
            #     local.get 1
            #     i32.add))

            hardcoded_wasm = UInt8[
                # Magic number and version
                0x00, 0x61, 0x73, 0x6D,  # \0asm
                0x01, 0x00, 0x00, 0x00,  # version 1

                # Type section (section id 1)
                0x01,                    # section id
                0x07,                    # section size (7 bytes)
                0x01,                    # num types
                0x60,                    # func type
                0x02,                    # num params
                0x7F, 0x7F,              # i32, i32
                0x01,                    # num results
                0x7F,                    # i32

                # Function section (section id 3)
                0x03,                    # section id
                0x02,                    # section size
                0x01,                    # num functions
                0x00,                    # type index 0

                # Export section (section id 7)
                0x07,                    # section id
                0x07,                    # section size
                0x01,                    # num exports
                0x03,                    # name length
                0x61, 0x64, 0x64,        # "add"
                0x00,                    # export kind (function)
                0x00,                    # function index

                # Code section (section id 10)
                0x0A,                    # section id
                0x09,                    # section size
                0x01,                    # num functions
                0x07,                    # function body size
                0x00,                    # num locals
                0x20, 0x00,              # local.get 0
                0x20, 0x01,              # local.get 1
                0x6A,                    # i32.add
                0x0B,                    # end
            ]

            # Test that the harness can execute this binary
            if NODE_CMD !== nothing
                result = run_wasm(hardcoded_wasm, "add", Int32(2), Int32(3))
                @test result == 5

                result = run_wasm(hardcoded_wasm, "add", Int32(100), Int32(-50))
                @test result == 50
            else
                @warn "Skipping Wasm execution tests (Node.js not available)"
            end
        end

        @testset "Hardcoded Wasm Binary - i64.add" begin
            # Hand-assembled Wasm binary for i64 addition
            # WAT: (func (export "add64") (param i64 i64) (result i64) ...)

            hardcoded_wasm_i64 = UInt8[
                # Magic and version
                0x00, 0x61, 0x73, 0x6D,
                0x01, 0x00, 0x00, 0x00,

                # Type section
                0x01,
                0x07,
                0x01,
                0x60,
                0x02,
                0x7E, 0x7E,              # i64, i64
                0x01,
                0x7E,                    # i64

                # Function section
                0x03,
                0x02,
                0x01,
                0x00,

                # Export section
                0x07,
                0x09,                    # section size
                0x01,
                0x05,                    # name length
                0x61, 0x64, 0x64, 0x36, 0x34,  # "add64"
                0x00,
                0x00,

                # Code section
                0x0A,
                0x09,
                0x01,
                0x07,
                0x00,
                0x20, 0x00,
                0x20, 0x01,
                0x7C,                    # i64.add
                0x0B,
            ]

            if NODE_CMD !== nothing
                result = run_wasm(hardcoded_wasm_i64, "add64", Int64(10), Int64(20))
                @test result == 30

                # Test with large numbers that would overflow JS Number
                large_a = Int64(9007199254740993)  # 2^53 + 1
                large_b = Int64(1)
                result = run_wasm(hardcoded_wasm_i64, "add64", large_a, large_b)
                @test result == large_a + large_b
            else
                @warn "Skipping Wasm execution tests (Node.js not available)"
            end
        end
    end

    # ========================================================================
    # Phase 2: Wasm Builder Tests
    # ========================================================================
    @testset "Phase 2: Wasm Builder" begin

        @testset "WasmModule - i32.add generation" begin
            mod = WasmTarget.WasmModule()

            # Create a function: (param i32 i32) (result i32) -> local.get 0, local.get 1, i32.add
            body = UInt8[
                WasmTarget.Opcode.LOCAL_GET, 0x00,
                WasmTarget.Opcode.LOCAL_GET, 0x01,
                WasmTarget.Opcode.I32_ADD,
                WasmTarget.Opcode.END,
            ]

            func_idx = WasmTarget.add_function!(
                mod,
                [WasmTarget.I32, WasmTarget.I32],
                [WasmTarget.I32],
                WasmTarget.NumType[],
                body
            )

            WasmTarget.add_export!(mod, "add", 0, func_idx)

            wasm_bytes = WasmTarget.to_bytes(mod)

            # Verify we can execute it
            if NODE_CMD !== nothing
                result = run_wasm(wasm_bytes, "add", Int32(7), Int32(8))
                @test result == 15
            end
        end

        @testset "WasmModule - i64.add generation" begin
            mod = WasmTarget.WasmModule()

            body = UInt8[
                WasmTarget.Opcode.LOCAL_GET, 0x00,
                WasmTarget.Opcode.LOCAL_GET, 0x01,
                WasmTarget.Opcode.I64_ADD,
                WasmTarget.Opcode.END,
            ]

            func_idx = WasmTarget.add_function!(
                mod,
                [WasmTarget.I64, WasmTarget.I64],
                [WasmTarget.I64],
                WasmTarget.NumType[],
                body
            )

            WasmTarget.add_export!(mod, "add64", 0, func_idx)

            wasm_bytes = WasmTarget.to_bytes(mod)

            if NODE_CMD !== nothing
                result = run_wasm(wasm_bytes, "add64", Int64(100), Int64(200))
                @test result == 300
            end
        end

        @testset "WasmModule - Multiple functions" begin
            mod = WasmTarget.WasmModule()

            # Add function
            add_body = UInt8[
                WasmTarget.Opcode.LOCAL_GET, 0x00,
                WasmTarget.Opcode.LOCAL_GET, 0x01,
                WasmTarget.Opcode.I32_ADD,
                WasmTarget.Opcode.END,
            ]
            add_idx = WasmTarget.add_function!(
                mod, [WasmTarget.I32, WasmTarget.I32], [WasmTarget.I32],
                WasmTarget.NumType[], add_body
            )

            # Subtract function
            sub_body = UInt8[
                WasmTarget.Opcode.LOCAL_GET, 0x00,
                WasmTarget.Opcode.LOCAL_GET, 0x01,
                WasmTarget.Opcode.I32_SUB,
                WasmTarget.Opcode.END,
            ]
            sub_idx = WasmTarget.add_function!(
                mod, [WasmTarget.I32, WasmTarget.I32], [WasmTarget.I32],
                WasmTarget.NumType[], sub_body
            )

            WasmTarget.add_export!(mod, "add", 0, add_idx)
            WasmTarget.add_export!(mod, "sub", 0, sub_idx)

            wasm_bytes = WasmTarget.to_bytes(mod)

            if NODE_CMD !== nothing
                @test run_wasm(wasm_bytes, "add", Int32(10), Int32(5)) == 15
                @test run_wasm(wasm_bytes, "sub", Int32(10), Int32(5)) == 5
            end
        end
    end

    # ========================================================================
    # Phase 3: Compiler Tests - Julia IR to Wasm
    # ========================================================================
    @testset "Phase 3: Julia Compiler" begin

        @testset "Simple Int64 addition" begin
            # Define a simple function
            simple_add(a, b) = a + b

            if NODE_CMD !== nothing
                # Compile and run
                wasm_bytes = WasmTarget.compile(simple_add, (Int64, Int64))

                # Debug: dump the bytes
                # dump_wasm(wasm_bytes, "/tmp/simple_add.wasm")

                result = run_wasm(wasm_bytes, "simple_add", Int64(5), Int64(7))
                @test result == 12
            end
        end

        @testset "TDD Macro - @test_compile" begin
            my_add(x, y) = x + y

            if NODE_CMD !== nothing
                @test_compile my_add(Int64(10), Int64(20))
                @test_compile my_add(Int64(-5), Int64(5))
                @test_compile my_add(Int64(0), Int64(0))
            end
        end

    end

    # ========================================================================
    # Phase 4: Control Flow and Comparisons
    # ========================================================================
    @testset "Phase 4: Control Flow" begin

        @testset "Comparisons - returning Bool as i32" begin
            is_positive(x) = x > 0
            is_negative(x) = x < 0
            is_zero(x) = x == 0
            is_not_zero(x) = x != 0
            is_lte(x, y) = x <= y
            is_gte(x, y) = x >= y

            if NODE_CMD !== nothing
                # Test is_positive
                @test_compile is_positive(Int64(5))
                @test_compile is_positive(Int64(-5))
                @test_compile is_positive(Int64(0))

                # Test is_negative
                @test_compile is_negative(Int64(5))
                @test_compile is_negative(Int64(-5))

                # Test is_zero
                @test_compile is_zero(Int64(0))
                @test_compile is_zero(Int64(1))

                # Test is_not_zero
                @test_compile is_not_zero(Int64(0))
                @test_compile is_not_zero(Int64(42))

                # Test is_lte and is_gte
                @test_compile is_lte(Int64(3), Int64(5))
                @test_compile is_lte(Int64(5), Int64(5))
                @test_compile is_lte(Int64(7), Int64(5))
                @test_compile is_gte(Int64(7), Int64(5))
                @test_compile is_gte(Int64(5), Int64(5))
            end
        end

        @testset "Simple conditional - ternary" begin
            # x < 0 ? -x : x  (absolute value)
            my_abs(x) = x < 0 ? -x : x

            if NODE_CMD !== nothing
                @test_compile my_abs(Int64(5))
                @test_compile my_abs(Int64(-5))
                @test_compile my_abs(Int64(0))
            end
        end

        @testset "Max/Min functions" begin
            my_max(a, b) = a > b ? a : b
            my_min(a, b) = a < b ? a : b

            if NODE_CMD !== nothing
                @test_compile my_max(Int64(10), Int64(20))
                @test_compile my_max(Int64(20), Int64(10))
                @test_compile my_max(Int64(5), Int64(5))

                @test_compile my_min(Int64(10), Int64(20))
                @test_compile my_min(Int64(20), Int64(10))
            end
        end

        @testset "If-else blocks" begin
            # TODO: Multi-branch if-elseif-else patterns require better SSA/stack management
            # The basic two-branch if-else works (tested in ternary and max/min)
            # But multi-branch patterns generate invalid stack states
            @test_skip "Multi-branch if-else needs stack management improvements"
        end

        @testset "Nested conditionals" begin
            # TODO: Same issue as if-else blocks - multi-branch patterns
            @test_skip "Multi-branch conditionals need stack management improvements"
        end

    end

    # ========================================================================
    # Phase 5: More Integer Operations
    # ========================================================================
    @testset "Phase 5: Integer Operations" begin

        @testset "Subtraction and Multiplication" begin
            my_sub(a, b) = a - b
            my_mul(a, b) = a * b

            if NODE_CMD !== nothing
                @test_compile my_sub(Int64(10), Int64(3))
                @test_compile my_sub(Int64(3), Int64(10))
                @test_compile my_mul(Int64(6), Int64(7))
                @test_compile my_mul(Int64(-3), Int64(4))
            end
        end

        @testset "Division and Remainder" begin
            my_div(a, b) = a ÷ b  # Integer division
            my_rem(a, b) = a % b  # Remainder

            if NODE_CMD !== nothing
                @test_compile my_div(Int64(10), Int64(3))
                @test_compile my_div(Int64(20), Int64(4))
                @test_compile my_rem(Int64(10), Int64(3))
                @test_compile my_rem(Int64(20), Int64(4))
            end
        end

        @testset "Negation" begin
            my_neg(x) = -x

            if NODE_CMD !== nothing
                @test_compile my_neg(Int64(5))
                @test_compile my_neg(Int64(-5))
                @test_compile my_neg(Int64(0))
            end
        end

        @testset "Bitwise operations" begin
            my_and(a, b) = a & b
            my_or(a, b) = a | b
            my_xor(a, b) = a ⊻ b
            my_not(x) = ~x

            if NODE_CMD !== nothing
                @test_compile my_and(Int64(0b1100), Int64(0b1010))
                @test_compile my_or(Int64(0b1100), Int64(0b1010))
                @test_compile my_xor(Int64(0b1100), Int64(0b1010))
                @test_compile my_not(Int64(0))
            end
        end

        @testset "Shift operations" begin
            # TODO: Shift operations with multi-statement IR require proper SSA local handling
            # Currently skipped - tracked as future work for local variable management
            # The issue is that SSA values on the stack may not be in the order expected
            # by Wasm's shift instructions when there are intermediate computations.
            @test_skip "Shifts need SSA local handling"
        end

    end

    # ========================================================================
    # Phase 6: Type Conversions
    # ========================================================================
    @testset "Phase 6: Type Conversions" begin

        @testset "Int32 to Int64" begin
            widen32(x::Int32) = Int64(x)

            if NODE_CMD !== nothing
                @test_compile widen32(Int32(42))
                @test_compile widen32(Int32(-42))
                @test_compile widen32(Int32(0))
            end
        end

        @testset "Int64 to Int32 (truncate)" begin
            narrow64(x::Int64) = Int32(x % Int32)

            if NODE_CMD !== nothing
                @test_compile narrow64(Int64(42))
                @test_compile narrow64(Int64(-42))
            end
        end

        @testset "Int to Float" begin
            int_to_f64(x::Int64) = Float64(x)
            int_to_f32(x::Int32) = Float32(x)

            if NODE_CMD !== nothing
                @test_compile int_to_f64(Int64(42))
                @test_compile int_to_f64(Int64(-42))
                @test_compile int_to_f32(Int32(42))
            end
        end

        @testset "Float arithmetic" begin
            add_f64(a::Float64, b::Float64) = a + b
            mul_f64(a::Float64, b::Float64) = a * b
            sub_f64(a::Float64, b::Float64) = a - b
            div_f64(a::Float64, b::Float64) = a / b

            if NODE_CMD !== nothing
                @test_compile add_f64(1.5, 2.5)
                @test_compile mul_f64(3.0, 4.0)
                @test_compile sub_f64(10.0, 3.0)
                @test_compile div_f64(10.0, 4.0)
            end
        end

    end

    # ========================================================================
    # Phase 7: WasmGC Structs
    # ========================================================================
    @testset "Phase 7: WasmGC Structs" begin

        @testset "Builder: Struct type creation" begin
            using WasmTarget: WasmModule, add_struct_type!, FieldType, I32, I64, to_bytes

            # Create a module with a struct type
            mod = WasmModule()

            # Add a struct type with two i32 fields
            fields = [FieldType(I32, true), FieldType(I32, true)]
            type_idx = add_struct_type!(mod, fields)

            @test type_idx == 0

            # Verify it can be serialized without error
            bytes = to_bytes(mod)
            @test length(bytes) > 8  # At least magic + version

            # Check magic number
            @test bytes[1:4] == UInt8[0x00, 0x61, 0x73, 0x6D]
        end

        @testset "Builder: Struct with mixed fields" begin
            using WasmTarget: WasmModule, add_struct_type!, FieldType, I32, I64, F64, to_bytes

            mod = WasmModule()

            # Struct with i32, i64, f64 fields
            fields = [FieldType(I32, true), FieldType(I64, false), FieldType(F64, true)]
            type_idx = add_struct_type!(mod, fields)

            @test type_idx == 0

            bytes = to_bytes(mod)
            @test length(bytes) > 8
        end

        @testset "Builder: Struct type deduplication" begin
            using WasmTarget: WasmModule, add_struct_type!, FieldType, I32, to_bytes

            mod = WasmModule()

            # Add same struct type twice
            fields = [FieldType(I32, true)]
            type_idx1 = add_struct_type!(mod, fields)
            type_idx2 = add_struct_type!(mod, fields)

            @test type_idx1 == type_idx2  # Should be deduplicated
        end

        @testset "Hand-crafted: Struct creation and field access" begin
            if NODE_CMD !== nothing
                using WasmTarget: WasmModule, add_struct_type!, add_function!, add_export!
                using WasmTarget: FieldType, I32, to_bytes, Opcode, encode_leb128_unsigned, encode_leb128_signed

                # Create a module that:
                # 1. Defines a struct type { i32, i32 }
                # 2. Has a function that creates a struct and reads field 0

                mod = WasmModule()

                # Add struct type: { field0: i32, field1: i32 }
                struct_type_idx = add_struct_type!(mod, [FieldType(I32, true), FieldType(I32, true)])

                # Function: () -> i32
                # Creates struct with values (42, 99), returns field 0
                body = UInt8[]

                # Push field values for struct.new (i32.const uses signed LEB128!)
                push!(body, Opcode.I32_CONST)
                append!(body, encode_leb128_signed(42))  # field 0 value
                push!(body, Opcode.I32_CONST)
                append!(body, encode_leb128_signed(99))  # field 1 value

                # struct.new $type
                push!(body, Opcode.GC_PREFIX)
                push!(body, Opcode.STRUCT_NEW)
                append!(body, encode_leb128_unsigned(struct_type_idx))

                # struct.get $type $field
                push!(body, Opcode.GC_PREFIX)
                push!(body, Opcode.STRUCT_GET)
                append!(body, encode_leb128_unsigned(struct_type_idx))
                append!(body, encode_leb128_unsigned(0))  # field index

                # End function
                push!(body, Opcode.END)

                func_idx = add_function!(mod, NumType[], NumType[I32], NumType[], body)
                add_export!(mod, "get_field0", 0, func_idx)

                wasm_bytes = to_bytes(mod)
                result = run_wasm(wasm_bytes, "get_field0")

                @test result == 42
            end
        end

        @testset "Hand-crafted: Struct field 1 access" begin
            if NODE_CMD !== nothing
                using WasmTarget: WasmModule, add_struct_type!, add_function!, add_export!
                using WasmTarget: FieldType, I32, to_bytes, Opcode, encode_leb128_unsigned, encode_leb128_signed

                mod = WasmModule()
                struct_type_idx = add_struct_type!(mod, [FieldType(I32, true), FieldType(I32, true)])

                body = UInt8[]

                # Create struct with (42, 99) - use signed LEB128 for i32.const
                push!(body, Opcode.I32_CONST)
                append!(body, encode_leb128_signed(42))
                push!(body, Opcode.I32_CONST)
                append!(body, encode_leb128_signed(99))

                push!(body, Opcode.GC_PREFIX)
                push!(body, Opcode.STRUCT_NEW)
                append!(body, encode_leb128_unsigned(struct_type_idx))

                # Get field 1
                push!(body, Opcode.GC_PREFIX)
                push!(body, Opcode.STRUCT_GET)
                append!(body, encode_leb128_unsigned(struct_type_idx))
                append!(body, encode_leb128_unsigned(1))  # field 1

                push!(body, Opcode.END)

                func_idx = add_function!(mod, NumType[], NumType[I32], NumType[], body)
                add_export!(mod, "get_field1", 0, func_idx)

                wasm_bytes = to_bytes(mod)
                result = run_wasm(wasm_bytes, "get_field1")

                @test result == 99
            end
        end

        @testset "Hand-crafted: Struct with parameters" begin
            if NODE_CMD !== nothing
                using WasmTarget: WasmModule, add_struct_type!, add_function!, add_export!
                using WasmTarget: FieldType, I32, to_bytes, Opcode, encode_leb128_unsigned

                # Function: (a: i32, b: i32) -> i32
                # Creates struct(a, b), returns field y (b)
                mod = WasmModule()
                struct_type_idx = add_struct_type!(mod, [FieldType(I32, true), FieldType(I32, true)])

                body = UInt8[]

                # Push function args for struct
                push!(body, Opcode.LOCAL_GET)
                push!(body, 0x00)  # arg a
                push!(body, Opcode.LOCAL_GET)
                push!(body, 0x01)  # arg b

                # struct.new
                push!(body, Opcode.GC_PREFIX)
                push!(body, Opcode.STRUCT_NEW)
                append!(body, encode_leb128_unsigned(struct_type_idx))

                # struct.get field 1 (y)
                push!(body, Opcode.GC_PREFIX)
                push!(body, Opcode.STRUCT_GET)
                append!(body, encode_leb128_unsigned(struct_type_idx))
                append!(body, encode_leb128_unsigned(1))

                push!(body, Opcode.END)

                func_idx = add_function!(mod, NumType[I32, I32], NumType[I32], NumType[], body)
                add_export!(mod, "create_and_get_y", 0, func_idx)

                wasm_bytes = to_bytes(mod)

                @test run_wasm(wasm_bytes, "create_and_get_y", Int32(10), Int32(20)) == 20
                @test run_wasm(wasm_bytes, "create_and_get_y", Int32(100), Int32(200)) == 200
            end
        end

    end

    # ========================================================================
    # Phase 8: Tuples
    # ========================================================================
    @testset "Phase 8: Tuples" begin

        @testset "Hand-crafted: Tuple creation and access" begin
            if NODE_CMD !== nothing
                using WasmTarget: WasmModule, add_struct_type!, add_function!, add_export!
                using WasmTarget: FieldType, I32, to_bytes, Opcode, encode_leb128_unsigned

                # Function: (a: i32, b: i32) -> i32
                # Creates tuple (a, b), returns first element
                mod = WasmModule()

                # Tuple is represented as struct { field0: i32, field1: i32 }
                tuple_type_idx = add_struct_type!(mod, [FieldType(I32, false), FieldType(I32, false)])

                body = UInt8[]

                # Push tuple elements
                push!(body, Opcode.LOCAL_GET)
                push!(body, 0x00)
                push!(body, Opcode.LOCAL_GET)
                push!(body, 0x01)

                # struct.new
                push!(body, Opcode.GC_PREFIX)
                push!(body, Opcode.STRUCT_NEW)
                append!(body, encode_leb128_unsigned(tuple_type_idx))

                # Get element 0
                push!(body, Opcode.GC_PREFIX)
                push!(body, Opcode.STRUCT_GET)
                append!(body, encode_leb128_unsigned(tuple_type_idx))
                append!(body, encode_leb128_unsigned(0))

                push!(body, Opcode.END)

                func_idx = add_function!(mod, NumType[I32, I32], NumType[I32], NumType[], body)
                add_export!(mod, "tuple_first", 0, func_idx)

                wasm_bytes = to_bytes(mod)
                @test run_wasm(wasm_bytes, "tuple_first", Int32(10), Int32(20)) == 10
            end
        end

        @testset "Hand-crafted: Tuple second element" begin
            if NODE_CMD !== nothing
                using WasmTarget: WasmModule, add_struct_type!, add_function!, add_export!
                using WasmTarget: FieldType, I32, to_bytes, Opcode, encode_leb128_unsigned

                mod = WasmModule()
                tuple_type_idx = add_struct_type!(mod, [FieldType(I32, false), FieldType(I32, false)])

                body = UInt8[]

                push!(body, Opcode.LOCAL_GET)
                push!(body, 0x00)
                push!(body, Opcode.LOCAL_GET)
                push!(body, 0x01)
                push!(body, Opcode.GC_PREFIX)
                push!(body, Opcode.STRUCT_NEW)
                append!(body, encode_leb128_unsigned(tuple_type_idx))

                # Get element 1 (second)
                push!(body, Opcode.GC_PREFIX)
                push!(body, Opcode.STRUCT_GET)
                append!(body, encode_leb128_unsigned(tuple_type_idx))
                append!(body, encode_leb128_unsigned(1))

                push!(body, Opcode.END)

                func_idx = add_function!(mod, NumType[I32, I32], NumType[I32], NumType[], body)
                add_export!(mod, "tuple_second", 0, func_idx)

                wasm_bytes = to_bytes(mod)
                @test run_wasm(wasm_bytes, "tuple_second", Int32(10), Int32(20)) == 20
            end
        end

        @testset "Hand-crafted: 3-element tuple" begin
            if NODE_CMD !== nothing
                using WasmTarget: WasmModule, add_struct_type!, add_function!, add_export!
                using WasmTarget: FieldType, I32, I64, to_bytes, Opcode, encode_leb128_unsigned, encode_leb128_signed

                mod = WasmModule()
                # Tuple{Int32, Int32, Int32}
                tuple_type_idx = add_struct_type!(mod, [
                    FieldType(I32, false),
                    FieldType(I32, false),
                    FieldType(I32, false)
                ])

                body = UInt8[]

                # Create tuple (10, 20, 30), return third element
                push!(body, Opcode.I32_CONST)
                append!(body, encode_leb128_signed(10))
                push!(body, Opcode.I32_CONST)
                append!(body, encode_leb128_signed(20))
                push!(body, Opcode.I32_CONST)
                append!(body, encode_leb128_signed(30))

                push!(body, Opcode.GC_PREFIX)
                push!(body, Opcode.STRUCT_NEW)
                append!(body, encode_leb128_unsigned(tuple_type_idx))

                push!(body, Opcode.GC_PREFIX)
                push!(body, Opcode.STRUCT_GET)
                append!(body, encode_leb128_unsigned(tuple_type_idx))
                append!(body, encode_leb128_unsigned(2))  # third element

                push!(body, Opcode.END)

                func_idx = add_function!(mod, NumType[], NumType[I32], NumType[], body)
                add_export!(mod, "tuple_third", 0, func_idx)

                wasm_bytes = to_bytes(mod)
                @test run_wasm(wasm_bytes, "tuple_third") == 30
            end
        end

    end

    # ========================================================================
    # Phase 9: WasmGC Arrays
    # ========================================================================
    @testset "Phase 9: WasmGC Arrays" begin

        @testset "Builder: Array type creation" begin
            using WasmTarget: WasmModule, add_array_type!, I32, to_bytes

            mod = WasmModule()
            arr_type_idx = add_array_type!(mod, I32, true)

            @test arr_type_idx == 0
            bytes = to_bytes(mod)
            @test length(bytes) > 8
            @test bytes[1:4] == UInt8[0x00, 0x61, 0x73, 0x6D]
        end

        @testset "Hand-crafted: Array length" begin
            if NODE_CMD !== nothing
                using WasmTarget: WasmModule, add_array_type!, add_function!, add_export!
                using WasmTarget: I32, to_bytes, Opcode, encode_leb128_unsigned, encode_leb128_signed

                # Function: () -> i32
                # Creates array of length 5, returns the length
                mod = WasmModule()
                arr_type_idx = add_array_type!(mod, I32, true)

                body = UInt8[]

                # Create array with init value 0 and length 5
                push!(body, Opcode.I32_CONST)
                append!(body, encode_leb128_signed(0))
                push!(body, Opcode.I32_CONST)
                append!(body, encode_leb128_signed(5))

                push!(body, Opcode.GC_PREFIX)
                push!(body, Opcode.ARRAY_NEW)
                append!(body, encode_leb128_unsigned(arr_type_idx))

                # Get array length
                push!(body, Opcode.GC_PREFIX)
                push!(body, Opcode.ARRAY_LEN)

                push!(body, Opcode.END)

                func_idx = add_function!(mod, NumType[], NumType[I32], NumType[], body)
                add_export!(mod, "arr_len", 0, func_idx)

                wasm_bytes = to_bytes(mod)
                @test run_wasm(wasm_bytes, "arr_len") == 5
            end
        end

        @testset "Hand-crafted: Array get element" begin
            if NODE_CMD !== nothing
                using WasmTarget: WasmModule, add_array_type!, add_function!, add_export!
                using WasmTarget: I32, to_bytes, Opcode, encode_leb128_unsigned, encode_leb128_signed

                # Create array with init value 42, get element at index 0
                mod = WasmModule()
                arr_type_idx = add_array_type!(mod, I32, true)

                body = UInt8[]

                # Create array with init value 42 and length 3
                push!(body, Opcode.I32_CONST)
                append!(body, encode_leb128_signed(42))  # all elements will be 42
                push!(body, Opcode.I32_CONST)
                append!(body, encode_leb128_signed(3))

                push!(body, Opcode.GC_PREFIX)
                push!(body, Opcode.ARRAY_NEW)
                append!(body, encode_leb128_unsigned(arr_type_idx))

                # Get element at index 1
                push!(body, Opcode.I32_CONST)
                append!(body, encode_leb128_signed(1))
                push!(body, Opcode.GC_PREFIX)
                push!(body, Opcode.ARRAY_GET)
                append!(body, encode_leb128_unsigned(arr_type_idx))

                push!(body, Opcode.END)

                func_idx = add_function!(mod, NumType[], NumType[I32], NumType[], body)
                add_export!(mod, "arr_get", 0, func_idx)

                wasm_bytes = to_bytes(mod)
                @test run_wasm(wasm_bytes, "arr_get") == 42
            end
        end

        @testset "Hand-crafted: Array new_fixed" begin
            if NODE_CMD !== nothing
                using WasmTarget: WasmModule, add_array_type!, add_function!, add_export!
                using WasmTarget: I32, to_bytes, Opcode, encode_leb128_unsigned, encode_leb128_signed

                # Create array with fixed elements [10, 20, 30], get middle element
                mod = WasmModule()
                arr_type_idx = add_array_type!(mod, I32, true)

                body = UInt8[]

                # Push elements for array.new_fixed
                push!(body, Opcode.I32_CONST)
                append!(body, encode_leb128_signed(10))
                push!(body, Opcode.I32_CONST)
                append!(body, encode_leb128_signed(20))
                push!(body, Opcode.I32_CONST)
                append!(body, encode_leb128_signed(30))

                # array.new_fixed $type $count
                push!(body, Opcode.GC_PREFIX)
                push!(body, Opcode.ARRAY_NEW_FIXED)
                append!(body, encode_leb128_unsigned(arr_type_idx))
                append!(body, encode_leb128_unsigned(3))  # count

                # Get element at index 1 (should be 20)
                push!(body, Opcode.I32_CONST)
                append!(body, encode_leb128_signed(1))
                push!(body, Opcode.GC_PREFIX)
                push!(body, Opcode.ARRAY_GET)
                append!(body, encode_leb128_unsigned(arr_type_idx))

                push!(body, Opcode.END)

                func_idx = add_function!(mod, NumType[], NumType[I32], NumType[], body)
                add_export!(mod, "arr_fixed_get", 0, func_idx)

                wasm_bytes = to_bytes(mod)
                @test run_wasm(wasm_bytes, "arr_fixed_get") == 20
            end
        end

    end

    # ========================================================================
    # Phase 10: JavaScript Imports
    # ========================================================================
    @testset "Phase 10: JavaScript Imports" begin

        @testset "Builder: Add import function" begin
            using WasmTarget: WasmModule, add_import!, add_function!, add_export!
            using WasmTarget: I32, to_bytes

            mod = WasmModule()
            # Import a function: env.log_i32(i32) -> void
            import_idx = add_import!(mod, "env", "log_i32", NumType[I32], NumType[])
            @test import_idx == 0

            # Add a local function that calls the import
            body = UInt8[
                0x20, 0x00,  # local.get 0
                0x10, 0x00,  # call 0 (the imported function)
                0x0B         # end
            ]
            func_idx = add_function!(mod, NumType[I32], NumType[], NumType[], body)
            # func_idx should be 1 (after the imported function)
            @test func_idx == 1

            add_export!(mod, "test", 0, func_idx)

            bytes = to_bytes(mod)
            @test length(bytes) > 8
            @test bytes[1:4] == UInt8[0x00, 0x61, 0x73, 0x6D]
        end

        @testset "Execute: Import and call JavaScript function" begin
            if NODE_CMD !== nothing
                using WasmTarget: WasmModule, add_import!, add_function!, add_export!
                using WasmTarget: I32, to_bytes, Opcode, encode_leb128_unsigned, encode_leb128_signed

                mod = WasmModule()

                # Import: env.double(i32) -> i32
                import_idx = add_import!(mod, "env", "double_it", NumType[I32], NumType[I32])

                # Local function: (param i32) -> i32
                # Calls the imported double_it function
                body = UInt8[]
                push!(body, Opcode.LOCAL_GET)
                append!(body, encode_leb128_unsigned(0))
                push!(body, Opcode.CALL)
                append!(body, encode_leb128_unsigned(0))  # call import at index 0
                push!(body, Opcode.END)

                func_idx = add_function!(mod, NumType[I32], NumType[I32], NumType[], body)
                add_export!(mod, "call_double", 0, func_idx)

                wasm_bytes = to_bytes(mod)

                # Run with imports
                result = run_wasm_with_imports(wasm_bytes, "call_double",
                    Dict("env" => Dict("double_it" => "(x) => x * 2")),
                    Int32(21))
                @test result == 42
            end
        end

    end

    @testset "Phase 11: Loops" begin

        @testset "Simple while loop - sum 1 to n" begin
            @noinline function simple_sum(n::Int32)::Int32
                total::Int32 = Int32(0)
                i::Int32 = Int32(1)
                @inbounds while i <= n
                    total = total + i
                    i = i + Int32(1)
                end
                return total
            end

            wasm_bytes = WasmTarget.compile(simple_sum, (Int32,))
            @test length(wasm_bytes) > 0

            # Test execution
            @test run_wasm(wasm_bytes, "simple_sum", Int32(5)) == 15
            @test run_wasm(wasm_bytes, "simple_sum", Int32(10)) == 55
            @test run_wasm(wasm_bytes, "simple_sum", Int32(100)) == 5050
        end

        @testset "Factorial loop" begin
            @noinline function factorial_loop(n::Int32)::Int32
                result::Int32 = Int32(1)
                i::Int32 = Int32(1)
                @inbounds while i <= n
                    result = result * i
                    i = i + Int32(1)
                end
                return result
            end

            wasm_bytes = WasmTarget.compile(factorial_loop, (Int32,))
            @test length(wasm_bytes) > 0

            @test run_wasm(wasm_bytes, "factorial_loop", Int32(1)) == 1
            @test run_wasm(wasm_bytes, "factorial_loop", Int32(5)) == 120
            @test run_wasm(wasm_bytes, "factorial_loop", Int32(6)) == 720
        end

        @testset "Count down loop" begin
            @noinline function count_down(n::Int32)::Int32
                total::Int32 = Int32(0)
                i::Int32 = n
                @inbounds while i > Int32(0)
                    total = total + i
                    i = i - Int32(1)
                end
                return total
            end

            wasm_bytes = WasmTarget.compile(count_down, (Int32,))
            @test length(wasm_bytes) > 0

            @test run_wasm(wasm_bytes, "count_down", Int32(5)) == 15
            @test run_wasm(wasm_bytes, "count_down", Int32(10)) == 55
        end

    end

    # Note: Recursive functions must be defined at module level (not inside @testset)
    # to avoid closure capture which is not yet supported in the Wasm compiler

    @testset "Phase 12: Recursion" begin

        @testset "Recursive factorial" begin
            wasm_bytes = WasmTarget.compile(test_factorial_rec, (Int32,))
            @test length(wasm_bytes) > 0

            @test run_wasm(wasm_bytes, "test_factorial_rec", Int32(1)) == 1
            @test run_wasm(wasm_bytes, "test_factorial_rec", Int32(5)) == 120
            @test run_wasm(wasm_bytes, "test_factorial_rec", Int32(6)) == 720
            @test run_wasm(wasm_bytes, "test_factorial_rec", Int32(10)) == 3628800
        end

        @testset "Recursive fibonacci" begin
            wasm_bytes = WasmTarget.compile(test_fib, (Int32,))
            @test length(wasm_bytes) > 0

            @test run_wasm(wasm_bytes, "test_fib", Int32(0)) == 0
            @test run_wasm(wasm_bytes, "test_fib", Int32(1)) == 1
            @test run_wasm(wasm_bytes, "test_fib", Int32(5)) == 5
            @test run_wasm(wasm_bytes, "test_fib", Int32(10)) == 55
        end

        @testset "Recursive sum" begin
            wasm_bytes = WasmTarget.compile(test_sum_rec, (Int32,))
            @test length(wasm_bytes) > 0

            @test run_wasm(wasm_bytes, "test_sum_rec", Int32(0)) == 0
            @test run_wasm(wasm_bytes, "test_sum_rec", Int32(5)) == 15
            @test run_wasm(wasm_bytes, "test_sum_rec", Int32(100)) == 5050
        end

    end

    # ========================================================================
    # Phase 13: Compiled Struct Field Access
    # ========================================================================
    @testset "Phase 13: Compiled Struct Access" begin

        @testset "Struct creation and field sum" begin
            wasm_bytes = WasmTarget.compile(test_point_sum, (Int32, Int32))
            @test length(wasm_bytes) > 0

            @test run_wasm(wasm_bytes, "test_point_sum", Int32(10), Int32(20)) == 30
            @test run_wasm(wasm_bytes, "test_point_sum", Int32(100), Int32(200)) == 300
            @test run_wasm(wasm_bytes, "test_point_sum", Int32(-5), Int32(15)) == 10
        end

        @testset "Struct creation and field difference" begin
            wasm_bytes = WasmTarget.compile(test_point_diff, (Int32, Int32))
            @test length(wasm_bytes) > 0

            @test run_wasm(wasm_bytes, "test_point_diff", Int32(30), Int32(10)) == 20
            @test run_wasm(wasm_bytes, "test_point_diff", Int32(100), Int32(50)) == 50
            @test run_wasm(wasm_bytes, "test_point_diff", Int32(5), Int32(10)) == -5
        end

    end

    # ========================================================================
    # Phase 14: Float Operations and Branching
    # ========================================================================
    @testset "Phase 14: Float Operations" begin

        @testset "Float addition" begin
            wasm_bytes = WasmTarget.compile(test_float_add, (Float64, Float64))
            @test length(wasm_bytes) > 0

            @test run_wasm(wasm_bytes, "test_float_add", 1.5, 2.5) ≈ 4.0
            @test run_wasm(wasm_bytes, "test_float_add", -1.0, 1.0) ≈ 0.0
            @test run_wasm(wasm_bytes, "test_float_add", 100.5, 200.5) ≈ 301.0
        end

        @testset "Float multiplication" begin
            wasm_bytes = WasmTarget.compile(test_float_mul, (Float64, Float64))
            @test length(wasm_bytes) > 0

            @test run_wasm(wasm_bytes, "test_float_mul", 2.0, 3.0) ≈ 6.0
            @test run_wasm(wasm_bytes, "test_float_mul", -2.0, 4.0) ≈ -8.0
            @test run_wasm(wasm_bytes, "test_float_mul", 0.5, 0.5) ≈ 0.25
        end

        @testset "Integer branching" begin
            wasm_bytes = WasmTarget.compile(test_branch, (Int32, Int32))
            @test length(wasm_bytes) > 0

            # sum = 110 > 100, so return 110 - 50 = 60
            @test run_wasm(wasm_bytes, "test_branch", Int32(60), Int32(50)) == 60
            # sum = 50 <= 100, so return 50 * 2 = 100
            @test run_wasm(wasm_bytes, "test_branch", Int32(30), Int32(20)) == 100
            # sum = 101 > 100, so return 101 - 50 = 51
            @test run_wasm(wasm_bytes, "test_branch", Int32(100), Int32(1)) == 51
        end

    end

    # ========================================================================
    # Phase 15: Strings
    # ========================================================================
    @testset "Phase 15: Strings" begin

        # String sizeof - returns byte length of string
        @noinline function str_sizeof(s::String)::Int64
            return sizeof(s)
        end

        @testset "String sizeof compilation" begin
            wasm_bytes = WasmTarget.compile(str_sizeof, (String,))
            @test length(wasm_bytes) > 0

            # Validate the module
            @test validate_wasm(wasm_bytes)
        end

        # String length - returns character count
        @noinline function str_length(s::String)::Int64
            return length(s)
        end

        @testset "String length compilation" begin
            wasm_bytes = WasmTarget.compile(str_length, (String,))
            @test length(wasm_bytes) > 0

            # Validate the module
            @test validate_wasm(wasm_bytes)
        end

        # String literal - returns a constant string
        @noinline function str_literal()::String
            return "hello"
        end

        @testset "String literal compilation" begin
            wasm_bytes = WasmTarget.compile(str_literal, ())
            @test length(wasm_bytes) > 0

            # Validate the module
            @test validate_wasm(wasm_bytes)
        end

        # String concatenation
        @noinline function str_concat(a::String, b::String)::String
            return a * b
        end

        @testset "String concatenation" begin
            wasm_bytes = WasmTarget.compile(str_concat, (String, String))
            @test length(wasm_bytes) > 0
            @test validate_wasm(wasm_bytes)
        end

        # String equality
        @noinline function str_equal(a::String, b::String)::Bool
            return a == b
        end

        @testset "String equality" begin
            wasm_bytes = WasmTarget.compile(str_equal, (String, String))
            @test length(wasm_bytes) > 0
            @test validate_wasm(wasm_bytes)
        end

        # String hashing for dict keys
        @testset "String hash" begin
            function test_str_hash()::Int32
                return str_hash("hello")
            end

            wasm_bytes = WasmTarget.compile(test_str_hash, ())
            @test length(wasm_bytes) > 0
            @test validate_wasm(wasm_bytes)
            # Verify hash matches Julia's fallback
            @test run_wasm(wasm_bytes, "test_str_hash") == str_hash("hello")
        end

        @testset "String hash consistency" begin
            function test_hash_diff()::Int32
                h1 = str_hash("hello")
                h2 = str_hash("world")
                if h1 == h2
                    return Int32(0)
                else
                    return Int32(1)
                end
            end

            wasm_bytes = WasmTarget.compile(test_hash_diff, ())
            @test length(wasm_bytes) > 0
            @test validate_wasm(wasm_bytes)
            @test run_wasm(wasm_bytes, "test_hash_diff") == 1  # Different strings have different hashes
        end

        # ======================================================================
        # BROWSER-010: New String Operations
        # ======================================================================

        @testset "str_find - basic search" begin
            function test_str_find_basic()::Int32
                return str_find("hello world", "world")
            end

            wasm_bytes = WasmTarget.compile(test_str_find_basic, ())
            @test length(wasm_bytes) > 0
            @test validate_wasm(wasm_bytes)
            @test run_wasm(wasm_bytes, "test_str_find_basic") == 7  # "world" starts at position 7
        end

        @testset "str_find - not found" begin
            function test_str_find_notfound()::Int32
                return str_find("hello world", "xyz")
            end

            wasm_bytes = WasmTarget.compile(test_str_find_notfound, ())
            @test length(wasm_bytes) > 0
            @test validate_wasm(wasm_bytes)
            @test run_wasm(wasm_bytes, "test_str_find_notfound") == 0  # Not found returns 0
        end

        @testset "str_contains - found" begin
            function test_str_contains_found()::Int32
                if str_contains("hello world", "world")
                    return Int32(1)
                else
                    return Int32(0)
                end
            end

            wasm_bytes = WasmTarget.compile(test_str_contains_found, ())
            @test length(wasm_bytes) > 0
            @test validate_wasm(wasm_bytes)
            @test run_wasm(wasm_bytes, "test_str_contains_found") == 1
        end

        @testset "str_contains - not found" begin
            function test_str_contains_notfound()::Int32
                if str_contains("hello world", "xyz")
                    return Int32(1)
                else
                    return Int32(0)
                end
            end

            wasm_bytes = WasmTarget.compile(test_str_contains_notfound, ())
            @test length(wasm_bytes) > 0
            @test validate_wasm(wasm_bytes)
            @test run_wasm(wasm_bytes, "test_str_contains_notfound") == 0
        end

        @testset "str_startswith - true case" begin
            function test_str_startswith_true()::Int32
                if str_startswith("hello world", "hello")
                    return Int32(1)
                else
                    return Int32(0)
                end
            end

            wasm_bytes = WasmTarget.compile(test_str_startswith_true, ())
            @test length(wasm_bytes) > 0
            @test validate_wasm(wasm_bytes)
            @test run_wasm(wasm_bytes, "test_str_startswith_true") == 1
        end

        @testset "str_startswith - false case" begin
            function test_str_startswith_false()::Int32
                if str_startswith("hello world", "world")
                    return Int32(1)
                else
                    return Int32(0)
                end
            end

            wasm_bytes = WasmTarget.compile(test_str_startswith_false, ())
            @test length(wasm_bytes) > 0
            @test validate_wasm(wasm_bytes)
            @test run_wasm(wasm_bytes, "test_str_startswith_false") == 0
        end

        @testset "str_endswith - true case" begin
            function test_str_endswith_true()::Int32
                if str_endswith("hello world", "world")
                    return Int32(1)
                else
                    return Int32(0)
                end
            end

            wasm_bytes = WasmTarget.compile(test_str_endswith_true, ())
            @test length(wasm_bytes) > 0
            @test validate_wasm(wasm_bytes)
            @test run_wasm(wasm_bytes, "test_str_endswith_true") == 1
        end

        @testset "str_endswith - false case" begin
            function test_str_endswith_false()::Int32
                if str_endswith("hello world", "hello")
                    return Int32(1)
                else
                    return Int32(0)
                end
            end

            wasm_bytes = WasmTarget.compile(test_str_endswith_false, ())
            @test length(wasm_bytes) > 0
            @test validate_wasm(wasm_bytes)
            @test run_wasm(wasm_bytes, "test_str_endswith_false") == 0
        end

        # ========================================================================
        # BROWSER-010: str_uppercase, str_lowercase, str_trim
        # ========================================================================

        @testset "str_uppercase - basic" begin
            function test_str_uppercase()::Int32
                result = str_uppercase("hello")
                # Check first char is 'H' (72)
                return str_char(result, Int32(1))
            end

            wasm_bytes = WasmTarget.compile(test_str_uppercase, ())
            @test length(wasm_bytes) > 0
            @test validate_wasm(wasm_bytes)
            @test run_wasm(wasm_bytes, "test_str_uppercase") == 72  # 'H'
        end

        @testset "str_uppercase - mixed case" begin
            function test_str_uppercase_mixed()::Int32
                result = str_uppercase("HeLLo WoRLD")
                # Check length is preserved
                len = str_len(result)
                # Check some characters
                first = str_char(result, Int32(1))  # 'H' = 72
                fifth = str_char(result, Int32(5))  # 'O' = 79
                space = str_char(result, Int32(6))  # ' ' = 32
                last = str_char(result, Int32(11)) # 'D' = 68
                # Return sum as verification
                return first + fifth + space + last
            end

            wasm_bytes = WasmTarget.compile(test_str_uppercase_mixed, ())
            @test length(wasm_bytes) > 0
            @test validate_wasm(wasm_bytes)
            @test run_wasm(wasm_bytes, "test_str_uppercase_mixed") == 72 + 79 + 32 + 68  # 251
        end

        @testset "str_lowercase - basic" begin
            function test_str_lowercase()::Int32
                result = str_lowercase("HELLO")
                # Check first char is 'h' (104)
                return str_char(result, Int32(1))
            end

            wasm_bytes = WasmTarget.compile(test_str_lowercase, ())
            @test length(wasm_bytes) > 0
            @test validate_wasm(wasm_bytes)
            @test run_wasm(wasm_bytes, "test_str_lowercase") == 104  # 'h'
        end

        @testset "str_lowercase - mixed case" begin
            function test_str_lowercase_mixed()::Int32
                result = str_lowercase("HeLLo WoRLD")
                # Check length is preserved
                len = str_len(result)
                # Check some characters
                first = str_char(result, Int32(1))  # 'h' = 104
                fifth = str_char(result, Int32(5))  # 'o' = 111
                space = str_char(result, Int32(6))  # ' ' = 32
                last = str_char(result, Int32(11)) # 'd' = 100
                # Return sum as verification
                return first + fifth + space + last
            end

            wasm_bytes = WasmTarget.compile(test_str_lowercase_mixed, ())
            @test length(wasm_bytes) > 0
            @test validate_wasm(wasm_bytes)
            @test run_wasm(wasm_bytes, "test_str_lowercase_mixed") == 104 + 111 + 32 + 100  # 347
        end

        @testset "str_trim - leading and trailing spaces" begin
            function test_str_trim_both()::Int32
                result = str_trim("  hello  ")
                # Length should be 5
                return str_len(result)
            end

            wasm_bytes = WasmTarget.compile(test_str_trim_both, ())
            @test length(wasm_bytes) > 0
            @test validate_wasm(wasm_bytes)
            @test run_wasm(wasm_bytes, "test_str_trim_both") == 5
        end

        @testset "str_trim - content preserved" begin
            function test_str_trim_content()::Int32
                result = str_trim("  hello  ")
                # First char should be 'h' (104)
                return str_char(result, Int32(1))
            end

            wasm_bytes = WasmTarget.compile(test_str_trim_content, ())
            @test length(wasm_bytes) > 0
            @test validate_wasm(wasm_bytes)
            @test run_wasm(wasm_bytes, "test_str_trim_content") == 104  # 'h'
        end

        @testset "str_trim - no whitespace" begin
            function test_str_trim_no_ws()::Int32
                result = str_trim("hello")
                # Length should remain 5
                return str_len(result)
            end

            wasm_bytes = WasmTarget.compile(test_str_trim_no_ws, ())
            @test length(wasm_bytes) > 0
            @test validate_wasm(wasm_bytes)
            @test run_wasm(wasm_bytes, "test_str_trim_no_ws") == 5
        end

        @testset "str_trim - all whitespace" begin
            function test_str_trim_all_ws()::Int32
                result = str_trim("   ")
                # Length should be 0
                return str_len(result)
            end

            wasm_bytes = WasmTarget.compile(test_str_trim_all_ws, ())
            @test length(wasm_bytes) > 0
            @test validate_wasm(wasm_bytes)
            @test run_wasm(wasm_bytes, "test_str_trim_all_ws") == 0
        end

        @testset "str_trim - tabs and newlines" begin
            function test_str_trim_special()::Int32
                # "\thello\n" - tab at start, newline at end
                s = str_new(Int32(7))
                str_setchar!(s, Int32(1), Int32(9))   # tab
                str_setchar!(s, Int32(2), Int32(104)) # h
                str_setchar!(s, Int32(3), Int32(101)) # e
                str_setchar!(s, Int32(4), Int32(108)) # l
                str_setchar!(s, Int32(5), Int32(108)) # l
                str_setchar!(s, Int32(6), Int32(111)) # o
                str_setchar!(s, Int32(7), Int32(10))  # newline
                result = str_trim(s)
                # Length should be 5
                return str_len(result)
            end

            wasm_bytes = WasmTarget.compile(test_str_trim_special, ())
            @test length(wasm_bytes) > 0
            @test validate_wasm(wasm_bytes)
            @test run_wasm(wasm_bytes, "test_str_trim_special") == 5
        end

        # BROWSER-010: Dedicated tests for str_char and str_substr

        @testset "str_char - get character at index" begin
            function test_str_char_basic()::Int32
                s = "hello"
                return str_char(s, Int32(1))  # 'h' = 104
            end

            wasm_bytes = WasmTarget.compile(test_str_char_basic, ())
            @test length(wasm_bytes) > 0
            @test validate_wasm(wasm_bytes)
            @test run_wasm(wasm_bytes, "test_str_char_basic") == 104  # 'h'
        end

        @testset "str_char - multiple positions" begin
            function test_str_char_multi()::Int32
                s = "hello"
                # Sum first and last character: 'h'(104) + 'o'(111) = 215
                return str_char(s, Int32(1)) + str_char(s, Int32(5))
            end

            wasm_bytes = WasmTarget.compile(test_str_char_multi, ())
            @test length(wasm_bytes) > 0
            @test validate_wasm(wasm_bytes)
            @test run_wasm(wasm_bytes, "test_str_char_multi") == 215
        end

        @testset "str_substr - extract substring" begin
            function test_str_substr_basic()::Int32
                s = "hello world"
                sub = str_substr(s, Int32(7), Int32(5))  # "world"
                return str_len(sub)
            end

            wasm_bytes = WasmTarget.compile(test_str_substr_basic, ())
            @test length(wasm_bytes) > 0
            @test validate_wasm(wasm_bytes)
            @test run_wasm(wasm_bytes, "test_str_substr_basic") == 5
        end

        @testset "str_substr - verify content" begin
            function test_str_substr_content()::Int32
                s = "hello world"
                sub = str_substr(s, Int32(7), Int32(5))  # "world"
                # Return first char of "world" = 'w' = 119
                return str_char(sub, Int32(1))
            end

            wasm_bytes = WasmTarget.compile(test_str_substr_content, ())
            @test length(wasm_bytes) > 0
            @test validate_wasm(wasm_bytes)
            @test run_wasm(wasm_bytes, "test_str_substr_content") == 119  # 'w'
        end

        @testset "str_char - character comparison for tokenizer" begin
            # This test verifies the pattern used in tokenizer
            function test_char_comparison()::Int32
                s = "hello"
                c = str_char(s, Int32(1))
                # Compare character to ASCII code
                if c == Int32(104)  # 'h'
                    return Int32(1)
                else
                    return Int32(0)
                end
            end

            wasm_bytes = WasmTarget.compile(test_char_comparison, ())
            @test length(wasm_bytes) > 0
            @test validate_wasm(wasm_bytes)
            @test run_wasm(wasm_bytes, "test_char_comparison") == 1
        end

    end

    # ========================================================================
    # Phase 16: Multi-Function Modules
    # ========================================================================
    @testset "Phase 16: Multi-Function Modules" begin

        @noinline function multi_add(a::Int32, b::Int32)::Int32
            return a + b
        end

        @noinline function multi_sub(a::Int32, b::Int32)::Int32
            return a - b
        end

        @noinline function multi_mul(a::Int32, b::Int32)::Int32
            return a * b
        end

        @testset "Multiple functions in one module" begin
            wasm_bytes = WasmTarget.compile_multi([
                (multi_add, (Int32, Int32)),
                (multi_sub, (Int32, Int32)),
                (multi_mul, (Int32, Int32)),
            ])
            @test length(wasm_bytes) > 0
            @test validate_wasm(wasm_bytes)

            # Test each function works correctly
            @test run_wasm(wasm_bytes, "multi_add", Int32(5), Int32(3)) == 8
            @test run_wasm(wasm_bytes, "multi_sub", Int32(10), Int32(4)) == 6
            @test run_wasm(wasm_bytes, "multi_mul", Int32(6), Int32(7)) == 42
        end

        @testset "Cross-function calls" begin
            # Uses module-level functions: cross_helper_double, cross_use_helper
            wasm_bytes = WasmTarget.compile_multi([
                (cross_helper_double, (Int32,)),
                (cross_use_helper, (Int32,)),
            ])
            @test length(wasm_bytes) > 0
            @test validate_wasm(wasm_bytes)

            # Test helper directly
            @test run_wasm(wasm_bytes, "cross_helper_double", Int32(5)) == 10

            # Test function that calls another function
            @test run_wasm(wasm_bytes, "cross_use_helper", Int32(5)) == 11   # 5*2 + 1
            @test run_wasm(wasm_bytes, "cross_use_helper", Int32(10)) == 21  # 10*2 + 1
        end

        @testset "Multiple dispatch" begin
            # Same function (dispatch_process) with different type signatures
            wasm_bytes = WasmTarget.compile_multi([
                (dispatch_process, (Int32,), "process_i32"),
                (dispatch_process, (Int64,), "process_i64"),
                (dispatch_use_i32, (Int32,)),
                (dispatch_use_i64, (Int64,)),
            ])
            @test length(wasm_bytes) > 0
            @test validate_wasm(wasm_bytes)

            # Test direct calls to each dispatch variant
            @test run_wasm(wasm_bytes, "process_i32", Int32(5)) == 10   # 5*2
            @test run_wasm(wasm_bytes, "process_i64", Int64(5)) == 15   # 5*3

            # Test calls through dispatching functions
            @test run_wasm(wasm_bytes, "dispatch_use_i32", Int32(5)) == 11  # 5*2 + 1
            @test run_wasm(wasm_bytes, "dispatch_use_i64", Int64(5)) == 16  # 5*3 + 1
        end

        # Result type pattern test
        mutable struct ResultType
            success::Bool
            value::Int32
        end

        @noinline function result_try_div(a::Int32, b::Int32)::ResultType
            if b == Int32(0)
                return ResultType(false, Int32(0))
            else
                return ResultType(true, a ÷ b)
            end
        end

        @noinline function result_get_value(r::ResultType)::Int32
            return r.value
        end

        @noinline function result_is_success(r::ResultType)::Bool
            return r.success
        end

        @testset "Result type pattern" begin
            wasm_bytes = WasmTarget.compile_multi([
                (result_try_div, (Int32, Int32)),
                (result_get_value, (ResultType,)),
                (result_is_success, (ResultType,))
            ])
            @test length(wasm_bytes) > 0
            @test validate_wasm(wasm_bytes)
        end

    end

    # ========================================================================
    # Phase 17: JS Interop (externref)
    # ========================================================================
    @testset "Phase 17: JS Interop" begin

        @testset "externref pass-through" begin
            @noinline function jsval_passthrough(x::JSValue)::JSValue
                return x
            end

            wasm_bytes = WasmTarget.compile(jsval_passthrough, (JSValue,))
            @test length(wasm_bytes) > 0
            @test validate_wasm(wasm_bytes)
        end

        @testset "Wasm globals" begin
            # Test global variable creation and export
            mod = WasmTarget.WasmModule()

            # Add mutable i32 global
            global_idx = WasmTarget.add_global!(mod, WasmTarget.I32, true, 0)
            @test global_idx == 0

            # Export it
            WasmTarget.add_global_export!(mod, "counter", global_idx)

            # Serialize and validate
            bytes = WasmTarget.to_bytes(mod)
            @test length(bytes) > 0
            @test validate_wasm(bytes)
        end

    end

    # ========================================================================
    # Phase 18: Tables and Indirect Calls
    # ========================================================================
    @testset "Phase 18: Tables" begin

        @testset "Basic table creation" begin
            mod = WasmTarget.WasmModule()

            # Add a funcref table with 4 slots
            table_idx = WasmTarget.add_table!(mod, WasmTarget.FuncRef, 4)
            @test table_idx == 0

            # Add some functions to populate the table
            func1_idx = WasmTarget.add_function!(
                mod,
                [WasmTarget.I32],
                [WasmTarget.I32],
                WasmTarget.WasmValType[],
                UInt8[
                    WasmTarget.Opcode.LOCAL_GET, 0x00,  # get param
                    WasmTarget.Opcode.I32_CONST, 0x02,  # push 2
                    WasmTarget.Opcode.I32_MUL,          # multiply
                    WasmTarget.Opcode.END
                ]
            )

            func2_idx = WasmTarget.add_function!(
                mod,
                [WasmTarget.I32],
                [WasmTarget.I32],
                WasmTarget.WasmValType[],
                UInt8[
                    WasmTarget.Opcode.LOCAL_GET, 0x00,  # get param
                    WasmTarget.Opcode.I32_CONST, 0x03,  # push 3
                    WasmTarget.Opcode.I32_MUL,          # multiply
                    WasmTarget.Opcode.END
                ]
            )

            # Export them for testing
            WasmTarget.add_export!(mod, "double", 0, func1_idx)
            WasmTarget.add_export!(mod, "triple", 0, func2_idx)

            bytes = WasmTarget.to_bytes(mod)
            @test length(bytes) > 0
            @test validate_wasm(bytes)

            # Test the functions work
            @test run_wasm(bytes, "double", Int32(5)) == 10
            @test run_wasm(bytes, "triple", Int32(5)) == 15
        end

        @testset "Table with element segment" begin
            mod = WasmTarget.WasmModule()

            # Add funcref table
            table_idx = WasmTarget.add_table!(mod, WasmTarget.FuncRef, 4)

            # Add two functions with same signature
            func_double = WasmTarget.add_function!(
                mod,
                [WasmTarget.I32],
                [WasmTarget.I32],
                WasmTarget.WasmValType[],
                UInt8[
                    WasmTarget.Opcode.LOCAL_GET, 0x00,
                    WasmTarget.Opcode.I32_CONST, 0x02,
                    WasmTarget.Opcode.I32_MUL,
                    WasmTarget.Opcode.END
                ]
            )

            func_triple = WasmTarget.add_function!(
                mod,
                [WasmTarget.I32],
                [WasmTarget.I32],
                WasmTarget.WasmValType[],
                UInt8[
                    WasmTarget.Opcode.LOCAL_GET, 0x00,
                    WasmTarget.Opcode.I32_CONST, 0x03,
                    WasmTarget.Opcode.I32_MUL,
                    WasmTarget.Opcode.END
                ]
            )

            # Initialize table with element segment
            WasmTarget.add_elem_segment!(mod, 0, 0, [func_double, func_triple])

            # Export table for JS inspection
            WasmTarget.add_table_export!(mod, "funcs", table_idx)

            bytes = WasmTarget.to_bytes(mod)
            @test length(bytes) > 0
            @test validate_wasm(bytes)
        end

        @testset "Table with limits" begin
            mod = WasmTarget.WasmModule()

            # Table with both min and max
            table_idx = WasmTarget.add_table!(mod, WasmTarget.FuncRef, 2, 10)
            @test table_idx == 0

            bytes = WasmTarget.to_bytes(mod)
            @test length(bytes) > 0
            @test validate_wasm(bytes)
        end

        @testset "externref table" begin
            mod = WasmTarget.WasmModule()

            # Table for holding JS objects
            table_idx = WasmTarget.add_table!(mod, WasmTarget.ExternRef, 8)
            WasmTarget.add_table_export!(mod, "objects", table_idx)

            bytes = WasmTarget.to_bytes(mod)
            @test length(bytes) > 0
            @test validate_wasm(bytes)
        end

        @testset "call_indirect" begin
            mod = WasmTarget.WasmModule()

            # Add function type for i32 -> i32
            type_idx = WasmTarget.add_type!(mod, WasmTarget.FuncType(
                [WasmTarget.I32],
                [WasmTarget.I32]
            ))

            # Add funcref table
            table_idx = WasmTarget.add_table!(mod, WasmTarget.FuncRef, 4)

            # Add two functions with the same signature
            func_double = WasmTarget.add_function!(
                mod,
                [WasmTarget.I32],
                [WasmTarget.I32],
                WasmTarget.WasmValType[],
                UInt8[
                    WasmTarget.Opcode.LOCAL_GET, 0x00,
                    WasmTarget.Opcode.I32_CONST, 0x02,
                    WasmTarget.Opcode.I32_MUL,
                    WasmTarget.Opcode.END
                ]
            )

            func_triple = WasmTarget.add_function!(
                mod,
                [WasmTarget.I32],
                [WasmTarget.I32],
                WasmTarget.WasmValType[],
                UInt8[
                    WasmTarget.Opcode.LOCAL_GET, 0x00,
                    WasmTarget.Opcode.I32_CONST, 0x03,
                    WasmTarget.Opcode.I32_MUL,
                    WasmTarget.Opcode.END
                ]
            )

            # Initialize table: [func_double, func_triple]
            WasmTarget.add_elem_segment!(mod, 0, 0, [func_double, func_triple])

            # Add a dispatcher function that takes (value, index) and calls indirectly
            # call_indirect format: call_indirect type_idx table_idx
            dispatcher = WasmTarget.add_function!(
                mod,
                [WasmTarget.I32, WasmTarget.I32],  # value, table_index
                [WasmTarget.I32],
                WasmTarget.WasmValType[],
                UInt8[
                    WasmTarget.Opcode.LOCAL_GET, 0x00,  # push value
                    WasmTarget.Opcode.LOCAL_GET, 0x01,  # push table index
                    WasmTarget.Opcode.CALL_INDIRECT,
                    type_idx % UInt8,                   # type index
                    0x00,                               # table index
                    WasmTarget.Opcode.END
                ]
            )

            WasmTarget.add_export!(mod, "dispatch", 0, dispatcher)

            bytes = WasmTarget.to_bytes(mod)
            @test length(bytes) > 0
            @test validate_wasm(bytes)

            # dispatch(5, 0) should call func_double(5) = 10
            @test run_wasm(bytes, "dispatch", Int32(5), Int32(0)) == 10
            # dispatch(5, 1) should call func_triple(5) = 15
            @test run_wasm(bytes, "dispatch", Int32(5), Int32(1)) == 15
        end

        @testset "Linear memory" begin
            mod = WasmTarget.WasmModule()

            # Add memory with 1 page (64KB)
            mem_idx = WasmTarget.add_memory!(mod, 1)
            @test mem_idx == 0

            # Export the memory
            WasmTarget.add_memory_export!(mod, "memory", mem_idx)

            # Add a function that uses memory operations
            func_idx = WasmTarget.add_function!(
                mod,
                [WasmTarget.I32, WasmTarget.I32],  # address, value
                WasmTarget.WasmValType[],
                WasmTarget.WasmValType[],
                UInt8[
                    WasmTarget.Opcode.LOCAL_GET, 0x00,  # address
                    WasmTarget.Opcode.LOCAL_GET, 0x01,  # value
                    WasmTarget.Opcode.I32_STORE, 0x02, 0x00,  # store (align=4, offset=0)
                    WasmTarget.Opcode.END
                ]
            )
            WasmTarget.add_export!(mod, "store", 0, func_idx)

            # Add a load function
            load_idx = WasmTarget.add_function!(
                mod,
                [WasmTarget.I32],      # address
                [WasmTarget.I32],      # result
                WasmTarget.WasmValType[],
                UInt8[
                    WasmTarget.Opcode.LOCAL_GET, 0x00,  # address
                    WasmTarget.Opcode.I32_LOAD, 0x02, 0x00,  # load (align=4, offset=0)
                    WasmTarget.Opcode.END
                ]
            )
            WasmTarget.add_export!(mod, "load", 0, load_idx)

            bytes = WasmTarget.to_bytes(mod)
            @test length(bytes) > 0
            @test validate_wasm(bytes)

            # Test memory operations via Node.js
            js_code = """
            const bytes = Buffer.from([$(join(bytes, ","))]);
            WebAssembly.instantiate(bytes).then(result => {
                const { store, load, memory } = result.instance.exports;
                store(0, 42);
                console.log(load(0));
            });
            """
            result = read(`node -e $js_code`, String)
            @test strip(result) == "42"
        end

        @testset "Memory with max limit" begin
            mod = WasmTarget.WasmModule()

            # Add memory with min 1 page, max 10 pages
            mem_idx = WasmTarget.add_memory!(mod, 1, 10)
            @test mem_idx == 0

            bytes = WasmTarget.to_bytes(mod)
            @test length(bytes) > 0
            @test validate_wasm(bytes)
        end

        @testset "Data segment with string" begin
            mod = WasmTarget.WasmModule()

            # Add memory
            mem_idx = WasmTarget.add_memory!(mod, 1)
            WasmTarget.add_memory_export!(mod, "memory", mem_idx)

            # Initialize memory with "Hello"
            WasmTarget.add_data_segment!(mod, 0, 0, "Hello")

            # Add a function to read the first byte
            func_idx = WasmTarget.add_function!(
                mod,
                WasmTarget.WasmValType[],
                [WasmTarget.I32],
                WasmTarget.WasmValType[],
                UInt8[
                    WasmTarget.Opcode.I32_CONST, 0x00,  # address 0
                    WasmTarget.Opcode.I32_LOAD, 0x00, 0x00,  # load (unaligned)
                    WasmTarget.Opcode.END
                ]
            )
            WasmTarget.add_export!(mod, "read_first", 0, func_idx)

            bytes = WasmTarget.to_bytes(mod)
            @test length(bytes) > 0
            @test validate_wasm(bytes)

            # Test via Node.js - "Hello" as little-endian i32 is 'H' + 'e'<<8 + 'l'<<16 + 'l'<<24
            # = 0x48 + 0x65<<8 + 0x6c<<16 + 0x6c<<24 = 0x6c6c6548
            expected = Int32('H') | (Int32('e') << 8) | (Int32('l') << 16) | (Int32('l') << 24)
            @test run_wasm(bytes, "read_first") == expected
        end

        @testset "Data segment with raw bytes" begin
            mod = WasmTarget.WasmModule()

            mem_idx = WasmTarget.add_memory!(mod, 1)

            # Initialize with raw bytes [1, 2, 3, 4] at offset 16 (multiple of 4 for alignment)
            WasmTarget.add_data_segment!(mod, 0, 16, UInt8[1, 2, 3, 4])

            # Function to load i32 from offset 16
            # Note: i32.const uses signed LEB128, 16 = 0x10 fits in single byte
            func_idx = WasmTarget.add_function!(
                mod,
                WasmTarget.WasmValType[],
                [WasmTarget.I32],
                WasmTarget.WasmValType[],
                UInt8[
                    WasmTarget.Opcode.I32_CONST, 0x10,    # 16
                    WasmTarget.Opcode.I32_LOAD, 0x02, 0x00,  # align=4, offset=0
                    WasmTarget.Opcode.END
                ]
            )
            WasmTarget.add_export!(mod, "read_data", 0, func_idx)

            bytes = WasmTarget.to_bytes(mod)
            @test length(bytes) > 0
            @test validate_wasm(bytes)

            # Little-endian: [1, 2, 3, 4] = 0x04030201
            expected = Int32(1) | (Int32(2) << 8) | (Int32(3) << 16) | (Int32(4) << 24)
            @test run_wasm(bytes, "read_data") == expected
        end

    end

    # ================================================================
    # Phase 19: SimpleDict (Hash Table) Support
    # ================================================================

    @testset "Phase 19: SimpleDict operations" begin

        @testset "sd_new creates dictionary" begin
            # Simple function that creates dict and returns its length (should be 0)
            function test_dict_new()::Int32
                d = sd_new(Int32(8))
                return sd_length(d)
            end

            bytes = compile(test_dict_new, ())
            @test length(bytes) > 0
            @test validate_wasm(bytes)
            @test run_wasm(bytes, "test_dict_new") == 0
        end

        @testset "sd_set! and sd_get" begin
            # Set a key-value pair and retrieve it
            function test_dict_set_get()::Int32
                d = sd_new(Int32(8))
                sd_set!(d, Int32(5), Int32(42))
                return sd_get(d, Int32(5))
            end

            bytes = compile(test_dict_set_get, ())
            @test length(bytes) > 0
            @test validate_wasm(bytes)
            @test run_wasm(bytes, "test_dict_set_get") == 42
        end

        @testset "sd_haskey" begin
            # Check if key exists
            function test_dict_haskey()::Int32
                d = sd_new(Int32(8))
                sd_set!(d, Int32(10), Int32(100))
                if sd_haskey(d, Int32(10))
                    return Int32(1)
                else
                    return Int32(0)
                end
            end

            bytes = compile(test_dict_haskey, ())
            @test length(bytes) > 0
            @test validate_wasm(bytes)
            @test run_wasm(bytes, "test_dict_haskey") == 1
        end

        @testset "sd_haskey returns false for missing key" begin
            function test_dict_haskey_missing()::Int32
                d = sd_new(Int32(8))
                sd_set!(d, Int32(10), Int32(100))
                if sd_haskey(d, Int32(99))
                    return Int32(1)
                else
                    return Int32(0)
                end
            end

            bytes = compile(test_dict_haskey_missing, ())
            @test length(bytes) > 0
            @test validate_wasm(bytes)
            @test run_wasm(bytes, "test_dict_haskey_missing") == 0
        end

        @testset "sd_length increases with inserts" begin
            function test_dict_length()::Int32
                d = sd_new(Int32(8))
                sd_set!(d, Int32(1), Int32(10))
                sd_set!(d, Int32(2), Int32(20))
                sd_set!(d, Int32(3), Int32(30))
                return sd_length(d)
            end

            bytes = compile(test_dict_length, ())
            @test length(bytes) > 0
            @test validate_wasm(bytes)
            @test run_wasm(bytes, "test_dict_length") == 3
        end

        @testset "sd_set! updates existing key" begin
            function test_dict_update()::Int32
                d = sd_new(Int32(8))
                sd_set!(d, Int32(5), Int32(10))
                sd_set!(d, Int32(5), Int32(99))  # Update same key
                return sd_get(d, Int32(5))
            end

            bytes = compile(test_dict_update, ())
            @test length(bytes) > 0
            @test validate_wasm(bytes)
            @test run_wasm(bytes, "test_dict_update") == 99
        end

        @testset "sd_get returns 0 for missing key" begin
            function test_dict_get_missing()::Int32
                d = sd_new(Int32(8))
                sd_set!(d, Int32(5), Int32(42))
                return sd_get(d, Int32(99))  # Key doesn't exist
            end

            bytes = compile(test_dict_get_missing, ())
            @test length(bytes) > 0
            @test validate_wasm(bytes)
            @test run_wasm(bytes, "test_dict_get_missing") == 0
        end

        @testset "Multiple keys with linear probing" begin
            # Test that hash collisions are handled
            function test_dict_collisions()::Int32
                d = sd_new(Int32(4))  # Small capacity to force collisions
                sd_set!(d, Int32(1), Int32(11))
                sd_set!(d, Int32(5), Int32(55))  # May collide with key 1
                sd_set!(d, Int32(9), Int32(99))  # May collide with previous
                # Verify all keys are retrievable
                return sd_get(d, Int32(1)) + sd_get(d, Int32(5)) + sd_get(d, Int32(9))
            end

            bytes = compile(test_dict_collisions, ())
            @test length(bytes) > 0
            @test validate_wasm(bytes)
            @test run_wasm(bytes, "test_dict_collisions") == (11 + 55 + 99)
        end

    end

    # ================================================================
    # Phase 20: StringDict (String-keyed Hash Table) Support
    # ================================================================

    @testset "Phase 20: StringDict operations" begin

        @testset "sdict_new creates dictionary" begin
            # Simple function that creates dict and returns its length (should be 0)
            function test_sdict_new()::Int32
                d = sdict_new(Int32(8))
                return sdict_length(d)
            end

            bytes = compile(test_sdict_new, ())
            @test length(bytes) > 0
            @test validate_wasm(bytes)
            @test run_wasm(bytes, "test_sdict_new") == 0
        end

        @testset "sdict_set! and sdict_get" begin
            # Set a key-value pair and retrieve it
            function test_sdict_set_get()::Int32
                d = sdict_new(Int32(8))
                sdict_set!(d, "hello", Int32(42))
                return sdict_get(d, "hello")
            end

            bytes = compile(test_sdict_set_get, ())
            @test length(bytes) > 0
            @test validate_wasm(bytes)
            @test run_wasm(bytes, "test_sdict_set_get") == 42
        end

        @testset "sdict_haskey" begin
            # Check if key exists
            function test_sdict_haskey()::Int32
                d = sdict_new(Int32(8))
                sdict_set!(d, "test", Int32(100))
                if sdict_haskey(d, "test")
                    return Int32(1)
                else
                    return Int32(0)
                end
            end

            bytes = compile(test_sdict_haskey, ())
            @test length(bytes) > 0
            @test validate_wasm(bytes)
            @test run_wasm(bytes, "test_sdict_haskey") == 1
        end

        @testset "sdict_haskey returns false for missing key" begin
            function test_sdict_haskey_missing()::Int32
                d = sdict_new(Int32(8))
                sdict_set!(d, "exists", Int32(100))
                if sdict_haskey(d, "missing")
                    return Int32(1)
                else
                    return Int32(0)
                end
            end

            bytes = compile(test_sdict_haskey_missing, ())
            @test length(bytes) > 0
            @test validate_wasm(bytes)
            @test run_wasm(bytes, "test_sdict_haskey_missing") == 0
        end

        @testset "sdict_length increases with inserts" begin
            function test_sdict_length()::Int32
                d = sdict_new(Int32(8))
                sdict_set!(d, "one", Int32(10))
                sdict_set!(d, "two", Int32(20))
                sdict_set!(d, "three", Int32(30))
                return sdict_length(d)
            end

            bytes = compile(test_sdict_length, ())
            @test length(bytes) > 0
            @test validate_wasm(bytes)
            @test run_wasm(bytes, "test_sdict_length") == 3
        end

        @testset "sdict_set! updates existing key" begin
            function test_sdict_update()::Int32
                d = sdict_new(Int32(8))
                sdict_set!(d, "key", Int32(10))
                sdict_set!(d, "key", Int32(99))  # Update same key
                return sdict_get(d, "key")
            end

            bytes = compile(test_sdict_update, ())
            @test length(bytes) > 0
            @test validate_wasm(bytes)
            @test run_wasm(bytes, "test_sdict_update") == 99
        end

        @testset "sdict_get returns 0 for missing key" begin
            function test_sdict_get_missing()::Int32
                d = sdict_new(Int32(8))
                sdict_set!(d, "exists", Int32(42))
                return sdict_get(d, "nothere")  # Key doesn't exist
            end

            bytes = compile(test_sdict_get_missing, ())
            @test length(bytes) > 0
            @test validate_wasm(bytes)
            @test run_wasm(bytes, "test_sdict_get_missing") == 0
        end

        @testset "Multiple string keys" begin
            # Test with multiple string keys
            function test_sdict_multi()::Int32
                d = sdict_new(Int32(8))
                sdict_set!(d, "apple", Int32(1))
                sdict_set!(d, "banana", Int32(2))
                sdict_set!(d, "cherry", Int32(3))
                # Verify all keys are retrievable
                return sdict_get(d, "apple") + sdict_get(d, "banana") + sdict_get(d, "cherry")
            end

            bytes = compile(test_sdict_multi, ())
            @test length(bytes) > 0
            @test validate_wasm(bytes)
            @test run_wasm(bytes, "test_sdict_multi") == (1 + 2 + 3)
        end

    end

    # ========================================================================
    # Phase 21: Multi-dimensional Arrays (Matrix)
    # ========================================================================
    @testset "Phase 21: Multi-dimensional Arrays (Matrix)" begin

        @testset "Matrix type compiles" begin
            # Test that functions accepting Matrix compile correctly
            function test_matrix_accept(m::Matrix{Int32})::Int32
                return Int32(1)  # Just accept and return
            end

            bytes = compile(test_matrix_accept, (Matrix{Int32},))
            @test length(bytes) > 0
            @test validate_wasm(bytes)
        end

        @testset "Matrix .size field access compiles" begin
            # Test accessing the .size field of a Matrix
            function test_matrix_get_rows(m::Matrix{Int32})::Int64
                return m.size[1]
            end

            function test_matrix_get_cols(m::Matrix{Int32})::Int64
                return m.size[2]
            end

            bytes_rows = compile(test_matrix_get_rows, (Matrix{Int32},))
            @test length(bytes_rows) > 0
            @test validate_wasm(bytes_rows)

            bytes_cols = compile(test_matrix_get_cols, (Matrix{Int32},))
            @test length(bytes_cols) > 0
            @test validate_wasm(bytes_cols)
        end

        @testset "Matrix .ref field access compiles" begin
            # Test accessing the .ref field (underlying MemoryRef)
            function test_matrix_ref(m::Matrix{Int32})::Int64
                ref = m.ref
                return Int64(1)  # Just access ref
            end

            bytes = compile(test_matrix_ref, (Matrix{Int32},))
            @test length(bytes) > 0
            @test validate_wasm(bytes)
        end

        @testset "Matrix Float64 compiles" begin
            # Test Matrix with different element types
            function test_matrix_f64_rows(m::Matrix{Float64})::Int64
                return m.size[1]
            end

            bytes = compile(test_matrix_f64_rows, (Matrix{Float64},))
            @test length(bytes) > 0
            @test validate_wasm(bytes)
        end

        @testset "Matrix compile_multi" begin
            # Test multiple Matrix functions together
            function mat_rows(m::Matrix{Int32})::Int64
                return m.size[1]
            end

            function mat_cols(m::Matrix{Int32})::Int64
                return m.size[2]
            end

            function mat_total(m::Matrix{Int32})::Int64
                return m.size[1] * m.size[2]
            end

            bytes = compile_multi([
                (mat_rows, (Matrix{Int32},)),
                (mat_cols, (Matrix{Int32},)),
                (mat_total, (Matrix{Int32},)),
            ])
            @test length(bytes) > 0
            @test validate_wasm(bytes)
        end

    end

    # ========================================================================
    # Phase 22: Math Functions (WASM-native)
    # ========================================================================
    @testset "Phase 22: Math Functions (WASM-native)" begin

        @testset "sqrt (via llvm intrinsic)" begin
            if NODE_CMD !== nothing
                # Use the raw llvm intrinsic to avoid domain checking
                function test_sqrt_fast(x::Float64)::Float64
                    return Base.Math.sqrt_llvm(x)
                end

                bytes = compile(test_sqrt_fast, (Float64,))
                @test length(bytes) > 0
                @test validate_wasm(bytes)
                @test run_wasm(bytes, "test_sqrt_fast", Float64[4.0]) ≈ 2.0
                @test run_wasm(bytes, "test_sqrt_fast", Float64[9.0]) ≈ 3.0
                @test run_wasm(bytes, "test_sqrt_fast", Float64[2.0]) ≈ sqrt(2.0)
            end
        end

        @testset "abs" begin
            if NODE_CMD !== nothing
                function test_abs(x::Float64)::Float64
                    return abs(x)
                end

                bytes = compile(test_abs, (Float64,))
                @test length(bytes) > 0
                @test validate_wasm(bytes)
                @test run_wasm(bytes, "test_abs", Float64[-5.0]) ≈ 5.0
                @test run_wasm(bytes, "test_abs", Float64[3.0]) ≈ 3.0
                @test run_wasm(bytes, "test_abs", Float64[-0.0]) ≈ 0.0
            end
        end

        @testset "floor" begin
            if NODE_CMD !== nothing
                function test_floor(x::Float64)::Float64
                    return floor(x)
                end

                bytes = compile(test_floor, (Float64,))
                @test length(bytes) > 0
                @test validate_wasm(bytes)
                @test run_wasm(bytes, "test_floor", Float64[3.7]) ≈ 3.0
                @test run_wasm(bytes, "test_floor", Float64[-2.3]) ≈ -3.0
                @test run_wasm(bytes, "test_floor", Float64[5.0]) ≈ 5.0
            end
        end

        @testset "ceil" begin
            if NODE_CMD !== nothing
                function test_ceil(x::Float64)::Float64
                    return ceil(x)
                end

                bytes = compile(test_ceil, (Float64,))
                @test length(bytes) > 0
                @test validate_wasm(bytes)
                @test run_wasm(bytes, "test_ceil", Float64[3.2]) ≈ 4.0
                @test run_wasm(bytes, "test_ceil", Float64[-2.7]) ≈ -2.0
                @test run_wasm(bytes, "test_ceil", Float64[5.0]) ≈ 5.0
            end
        end

        @testset "round" begin
            if NODE_CMD !== nothing
                function test_round(x::Float64)::Float64
                    return round(x)
                end

                bytes = compile(test_round, (Float64,))
                @test length(bytes) > 0
                @test validate_wasm(bytes)
                @test run_wasm(bytes, "test_round", Float64[3.2]) ≈ 3.0
                @test run_wasm(bytes, "test_round", Float64[3.7]) ≈ 4.0
                @test run_wasm(bytes, "test_round", Float64[-2.5]) ≈ -2.0  # Round to even
            end
        end

        @testset "trunc" begin
            if NODE_CMD !== nothing
                function test_trunc(x::Float64)::Float64
                    return trunc(x)
                end

                bytes = compile(test_trunc, (Float64,))
                @test length(bytes) > 0
                @test validate_wasm(bytes)
                @test run_wasm(bytes, "test_trunc", Float64[3.7]) ≈ 3.0
                @test run_wasm(bytes, "test_trunc", Float64[-3.7]) ≈ -3.0
                @test run_wasm(bytes, "test_trunc", Float64[5.0]) ≈ 5.0
            end
        end

        @testset "Float32 variants" begin
            if NODE_CMD !== nothing
                function test_abs_f32(x::Float32)::Float32
                    return abs(x)
                end

                function test_floor_f32(x::Float32)::Float32
                    return floor(x)
                end

                bytes_abs = compile(test_abs_f32, (Float32,))
                @test length(bytes_abs) > 0
                @test validate_wasm(bytes_abs)

                bytes_floor = compile(test_floor_f32, (Float32,))
                @test length(bytes_floor) > 0
                @test validate_wasm(bytes_floor)
            end
        end

    end

    # ========================================================================
    # Phase 23: Void Control Flow Tests
    # Tests for complex control flow in void-returning functions (event handlers)
    # Covers: nested &&/||, sequential ifs, early returns
    # ========================================================================
    @testset "Phase 23: Void Control Flow" begin

        # Test helper: a mutable struct to track side effects
        mutable struct VoidTestState
            value::Int32
        end

        # ----------------------------------------------------------------
        # Test 1: Simple nested && operator (a && b && c pattern)
        # ----------------------------------------------------------------
        @testset "Nested && (triple condition)" begin
            @noinline function void_nested_and(state::VoidTestState, a::Int32, b::Int32, c::Int32)::Nothing
                if a > Int32(0) && b > Int32(0) && c > Int32(0)
                    state.value = Int32(1)
                end
                return nothing
            end

            bytes = compile(void_nested_and, (VoidTestState, Int32, Int32, Int32))
            @test length(bytes) > 0
            @test validate_wasm(bytes)
        end

        # ----------------------------------------------------------------
        # Test 2: Nested || operator (a || b || c pattern)
        # ----------------------------------------------------------------
        @testset "Nested || (triple condition)" begin
            @noinline function void_nested_or(state::VoidTestState, a::Int32, b::Int32, c::Int32)::Nothing
                if a > Int32(0) || b > Int32(0) || c > Int32(0)
                    state.value = Int32(1)
                end
                return nothing
            end

            bytes = compile(void_nested_or, (VoidTestState, Int32, Int32, Int32))
            @test length(bytes) > 0
            @test validate_wasm(bytes)
        end

        # ----------------------------------------------------------------
        # Test 3: Mixed && and || (a && (b || c) pattern)
        # ----------------------------------------------------------------
        @testset "Mixed && and ||" begin
            @noinline function void_mixed_and_or(state::VoidTestState, a::Int32, b::Int32, c::Int32)::Nothing
                if a > Int32(0) && (b > Int32(0) || c > Int32(0))
                    state.value = Int32(1)
                end
                return nothing
            end

            bytes = compile(void_mixed_and_or, (VoidTestState, Int32, Int32, Int32))
            @test length(bytes) > 0
            @test validate_wasm(bytes)
        end

        # ----------------------------------------------------------------
        # Test 4: Sequential if blocks
        # ----------------------------------------------------------------
        @testset "Sequential if blocks" begin
            @noinline function void_sequential_ifs(state::VoidTestState, a::Int32, b::Int32)::Nothing
                if a > Int32(0)
                    state.value = state.value + Int32(1)
                end
                if b > Int32(0)
                    state.value = state.value + Int32(10)
                end
                return nothing
            end

            bytes = compile(void_sequential_ifs, (VoidTestState, Int32, Int32))
            @test length(bytes) > 0
            @test validate_wasm(bytes)
        end

        # ----------------------------------------------------------------
        # Test 5: Three sequential if blocks
        # ----------------------------------------------------------------
        @testset "Three sequential if blocks" begin
            @noinline function void_three_ifs(state::VoidTestState, a::Int32, b::Int32, c::Int32)::Nothing
                if a > Int32(0)
                    state.value = state.value + Int32(1)
                end
                if b > Int32(0)
                    state.value = state.value + Int32(10)
                end
                if c > Int32(0)
                    state.value = state.value + Int32(100)
                end
                return nothing
            end

            bytes = compile(void_three_ifs, (VoidTestState, Int32, Int32, Int32))
            @test length(bytes) > 0
            @test validate_wasm(bytes)
        end

        # ----------------------------------------------------------------
        # Test 6: Early return in void function
        # ----------------------------------------------------------------
        @testset "Early return in void function" begin
            @noinline function void_early_return(state::VoidTestState, cond::Int32)::Nothing
                if cond > Int32(0)
                    return nothing
                end
                state.value = Int32(42)
                return nothing
            end

            bytes = compile(void_early_return, (VoidTestState, Int32))
            @test length(bytes) > 0
            @test validate_wasm(bytes)
        end

        # ----------------------------------------------------------------
        # Test 7: Early return with && condition
        # ----------------------------------------------------------------
        @testset "Early return with && condition" begin
            @noinline function void_early_return_and(state::VoidTestState, a::Int32, b::Int32)::Nothing
                if a > Int32(0) && b > Int32(0)
                    return nothing
                end
                state.value = Int32(99)
                return nothing
            end

            bytes = compile(void_early_return_and, (VoidTestState, Int32, Int32))
            @test length(bytes) > 0
            @test validate_wasm(bytes)
        end

        # ----------------------------------------------------------------
        # Test 8: Nested if-else in void function
        # ----------------------------------------------------------------
        @testset "Nested if-else in void function" begin
            @noinline function void_nested_if_else(state::VoidTestState, a::Int32, b::Int32)::Nothing
                if a > Int32(0)
                    if b > Int32(0)
                        state.value = Int32(1)
                    else
                        state.value = Int32(2)
                    end
                else
                    state.value = Int32(3)
                end
                return nothing
            end

            bytes = compile(void_nested_if_else, (VoidTestState, Int32, Int32))
            @test length(bytes) > 0
            @test validate_wasm(bytes)
        end

        # ----------------------------------------------------------------
        # Test 9: Quadruple && chain (winner checking pattern)
        # ----------------------------------------------------------------
        @testset "Quadruple && chain" begin
            @noinline function void_quad_and(state::VoidTestState, a::Int32, b::Int32, c::Int32, d::Int32)::Nothing
                if a == Int32(1) && b == Int32(1) && c == Int32(1) && d == Int32(1)
                    state.value = Int32(100)
                end
                return nothing
            end

            bytes = compile(void_quad_and, (VoidTestState, Int32, Int32, Int32, Int32))
            @test length(bytes) > 0
            @test validate_wasm(bytes)
        end

        # ----------------------------------------------------------------
        # Test 10: Complex TicTacToe-like winner checking pattern
        # ----------------------------------------------------------------
        @testset "TicTacToe winner pattern" begin
            @noinline function void_check_winner(state::VoidTestState, r1::Int32, r2::Int32, r3::Int32)::Nothing
                # Check if all three are equal and non-zero (like checking a row)
                if r1 != Int32(0) && r1 == r2 && r2 == r3
                    state.value = r1  # Winner found
                end
                return nothing
            end

            bytes = compile(void_check_winner, (VoidTestState, Int32, Int32, Int32))
            @test length(bytes) > 0
            @test validate_wasm(bytes)
        end

        # ----------------------------------------------------------------
        # Test 11: Multiple early returns
        # ----------------------------------------------------------------
        @testset "Multiple early returns" begin
            @noinline function void_multiple_returns(state::VoidTestState, code::Int32)::Nothing
                if code == Int32(1)
                    state.value = Int32(10)
                    return nothing
                end
                if code == Int32(2)
                    state.value = Int32(20)
                    return nothing
                end
                if code == Int32(3)
                    state.value = Int32(30)
                    return nothing
                end
                state.value = Int32(0)
                return nothing
            end

            bytes = compile(void_multiple_returns, (VoidTestState, Int32))
            @test length(bytes) > 0
            @test validate_wasm(bytes)
        end

        # ----------------------------------------------------------------
        # Test 12: If-else chain (switch-like pattern)
        # ----------------------------------------------------------------
        @testset "If-else chain" begin
            @noinline function void_if_else_chain(state::VoidTestState, x::Int32)::Nothing
                if x < Int32(0)
                    state.value = Int32(-1)
                elseif x == Int32(0)
                    state.value = Int32(0)
                elseif x < Int32(10)
                    state.value = Int32(1)
                else
                    state.value = Int32(2)
                end
                return nothing
            end

            bytes = compile(void_if_else_chain, (VoidTestState, Int32))
            @test length(bytes) > 0
            @test validate_wasm(bytes)
        end

        # ----------------------------------------------------------------
        # Test 13: Conditional with loop inside
        # ----------------------------------------------------------------
        @testset "Conditional with loop inside" begin
            @noinline function void_cond_with_loop(state::VoidTestState, n::Int32)::Nothing
                if n > Int32(0)
                    i = Int32(0)
                    while i < n
                        state.value = state.value + Int32(1)
                        i = i + Int32(1)
                    end
                end
                return nothing
            end

            bytes = compile(void_cond_with_loop, (VoidTestState, Int32))
            @test length(bytes) > 0
            @test validate_wasm(bytes)
        end

        # ----------------------------------------------------------------
        # Test 14: Pure void function (no side effects, just control flow)
        # ----------------------------------------------------------------
        @testset "Pure void with complex control flow" begin
            @noinline function void_pure_complex(a::Int32, b::Int32, c::Int32)::Nothing
                if a > Int32(0) && b > Int32(0)
                    if c > Int32(0)
                        # Do nothing
                    end
                elseif a > Int32(0) || b > Int32(0)
                    # Do nothing
                end
                return nothing
            end

            bytes = compile(void_pure_complex, (Int32, Int32, Int32))
            @test length(bytes) > 0
            @test validate_wasm(bytes)
        end

    end

    # ========================================================================
    # Phase 23: Union Types / Tagged Unions
    # ========================================================================
    @testset "Phase 23: Union Types" begin

        # Test 1: UnionInfo and TypeRegistry structures
        @testset "Union type registration" begin
            # Create a module and registry
            mod = WasmTarget.WasmModule()
            registry = WasmTarget.TypeRegistry()

            # Test needs_tagged_union function
            @test WasmTarget.needs_tagged_union(Union{Int32, Float64}) == true
            @test WasmTarget.needs_tagged_union(Union{Int32, String, Bool}) == true
            @test WasmTarget.needs_tagged_union(Union{Nothing, Int32}) == false

            # Test get_nullable_inner_type function
            @test WasmTarget.get_nullable_inner_type(Union{Nothing, Int32}) === Int32
            @test WasmTarget.get_nullable_inner_type(Union{Nothing, String}) === String
            @test WasmTarget.get_nullable_inner_type(Union{Int32, String}) === nothing

            # Test register_union_type!
            union_type = Union{Int32, Float64}
            info = WasmTarget.register_union_type!(mod, registry, union_type)
            @test info isa WasmTarget.UnionInfo
            @test info.julia_type === union_type
            @test length(info.variant_types) == 2
            @test Int32 in info.variant_types
            @test Float64 in info.variant_types
            @test haskey(info.tag_map, Int32)
            @test haskey(info.tag_map, Float64)

            # Test get_union_tag
            tag_int32 = WasmTarget.get_union_tag(info, Int32)
            tag_float64 = WasmTarget.get_union_tag(info, Float64)
            @test tag_int32 >= 0
            @test tag_float64 >= 0
            @test tag_int32 != tag_float64

            # Test union with Nothing
            union_with_nothing = Union{Nothing, Int32, String}
            info2 = WasmTarget.register_union_type!(mod, registry, union_with_nothing)
            @test length(info2.variant_types) == 3
            @test WasmTarget.get_union_tag(info2, Nothing) == Int32(0)  # Nothing always gets tag 0
        end

        # Test 2: Function parameter with union type
        @testset "Union parameter type" begin
            @noinline function process_union_value(x::Union{Int32, Float64})::Int32
                # This just returns a constant - we're testing type registration
                return Int32(1)
            end

            wasm_bytes = WasmTarget.compile(process_union_value, (Union{Int32, Float64},))
            @test length(wasm_bytes) > 0
            @test validate_wasm(wasm_bytes)
        end

        # Test 3: Triple union with Nothing as parameter
        @testset "Triple union parameter type" begin
            @noinline function triple_union_param(x::Union{Nothing, Int32, String})::Int32
                return Int32(0)
            end

            wasm_bytes = WasmTarget.compile(triple_union_param, (Union{Nothing, Int32, String},))
            @test length(wasm_bytes) > 0
            @test validate_wasm(wasm_bytes)
        end

        # Test 4: Julia-side concrete type resolution
        @testset "Concrete type resolution for unions" begin
            mod = WasmTarget.WasmModule()
            registry = WasmTarget.TypeRegistry()

            # Test that julia_to_wasm_type correctly handles union types
            union_type = Union{Int32, String}
            wasm_type = WasmTarget.julia_to_wasm_type(union_type)
            # Multi-variant unions should return a reference type (StructRef for now)
            @test wasm_type isa WasmTarget.RefType || wasm_type isa WasmTarget.NumType
        end

        # Test 5: Interpreter Value pattern - explicit tagged struct
        # This is the recommended pattern for runtime dynamic values
        mutable struct InterpValue
            tag::Int32       # 0 = nothing, 1 = int, 2 = float, 3 = bool
            int_val::Int64
            float_val::Float64
            bool_val::Int32  # 0 or 1
        end

        @testset "Interpreter value pattern" begin
            @noinline function make_int_value(x::Int64)::InterpValue
                return Base.inferencebarrier(InterpValue(Int32(1), x, Float64(0.0), Int32(0)))::InterpValue
            end

            @noinline function make_float_value(x::Float64)::InterpValue
                return Base.inferencebarrier(InterpValue(Int32(2), Int64(0), x, Int32(0)))::InterpValue
            end

            @noinline function is_int_value(v::InterpValue)::Bool
                return v.tag == Int32(1)
            end

            @noinline function get_int_value(v::InterpValue)::Int64
                return v.int_val
            end

            wasm1 = WasmTarget.compile(make_int_value, (Int64,))
            @test length(wasm1) > 0
            @test validate_wasm(wasm1)

            wasm2 = WasmTarget.compile(make_float_value, (Float64,))
            @test length(wasm2) > 0
            @test validate_wasm(wasm2)

            wasm3 = WasmTarget.compile(is_int_value, (InterpValue,))
            @test length(wasm3) > 0
            @test validate_wasm(wasm3)

            wasm4 = WasmTarget.compile(get_int_value, (InterpValue,))
            @test length(wasm4) > 0
            @test validate_wasm(wasm4)
        end

    end

    # ========================================================================
    # Phase 24: Advanced Recursion and Control Flow (BROWSER-013)
    # Tests for: mutual recursion, deep call stacks, complex control flow
    # Required for the interpreter's recursive eval() function
    # ========================================================================
    @testset "Phase 24: Advanced Recursion and Control Flow" begin

        @testset "Mutual recursion" begin
            # Compile both functions together to enable cross-calls
            wasm_bytes = WasmTarget.compile_multi([
                (is_even_mutual, (Int32,)),
                (is_odd_mutual, (Int32,))
            ])
            @test length(wasm_bytes) > 0

            # Test is_even
            @test run_wasm(wasm_bytes, "is_even_mutual", Int32(0)) == 1   # true
            @test run_wasm(wasm_bytes, "is_even_mutual", Int32(1)) == 0   # false
            @test run_wasm(wasm_bytes, "is_even_mutual", Int32(4)) == 1   # true
            @test run_wasm(wasm_bytes, "is_even_mutual", Int32(5)) == 0   # false
            @test run_wasm(wasm_bytes, "is_even_mutual", Int32(10)) == 1  # true

            # Test is_odd
            @test run_wasm(wasm_bytes, "is_odd_mutual", Int32(0)) == 0   # false
            @test run_wasm(wasm_bytes, "is_odd_mutual", Int32(1)) == 1   # true
            @test run_wasm(wasm_bytes, "is_odd_mutual", Int32(4)) == 0   # false
            @test run_wasm(wasm_bytes, "is_odd_mutual", Int32(5)) == 1   # true
        end

        @testset "Deep recursion (stack depth)" begin
            wasm_bytes = WasmTarget.compile(deep_recursion_test, (Int32, Int32))
            @test length(wasm_bytes) > 0

            # Test with increasing depths
            @test run_wasm(wasm_bytes, "deep_recursion_test", Int32(0), Int32(1)) == 1
            @test run_wasm(wasm_bytes, "deep_recursion_test", Int32(0), Int32(10)) == 10
            @test run_wasm(wasm_bytes, "deep_recursion_test", Int32(0), Int32(100)) == 100
            @test run_wasm(wasm_bytes, "deep_recursion_test", Int32(0), Int32(500)) == 500
            @test run_wasm(wasm_bytes, "deep_recursion_test", Int32(0), Int32(1000)) == 1000
        end

        @testset "Complex while loop with && condition" begin
            wasm_bytes = WasmTarget.compile(complex_while_test, (Int32,))
            @test length(wasm_bytes) > 0

            # Test various inputs
            @test run_wasm(wasm_bytes, "complex_while_test", Int32(5)) == 10   # 0+1+2+3+4 = 10
            @test run_wasm(wasm_bytes, "complex_while_test", Int32(10)) == 45  # 0+1+...+9 = 45
            @test run_wasm(wasm_bytes, "complex_while_test", Int32(20)) == 105 # stops when result >= 100
        end

        @testset "Nested conditionals" begin
            wasm_bytes = WasmTarget.compile(nested_cond_test, (Int32, Int32))
            @test length(wasm_bytes) > 0

            # Test all four branches
            @test run_wasm(wasm_bytes, "nested_cond_test", Int32(5), Int32(3)) == 8    # a>0, b>0: a+b
            @test run_wasm(wasm_bytes, "nested_cond_test", Int32(5), Int32(-3)) == 8   # a>0, b<=0: a-b
            @test run_wasm(wasm_bytes, "nested_cond_test", Int32(-5), Int32(3)) == 8   # a<=0, b>0: b-a
            @test run_wasm(wasm_bytes, "nested_cond_test", Int32(-5), Int32(-3)) == 15 # a<=0, b<=0: a*b
        end

        @testset "Multi-branch if-elseif-else" begin
            wasm_bytes = WasmTarget.compile(classify_number_test, (Int32,))
            @test length(wasm_bytes) > 0

            # Test all three branches
            @test run_wasm(wasm_bytes, "classify_number_test", Int32(-5)) == -1  # negative
            @test run_wasm(wasm_bytes, "classify_number_test", Int32(-1)) == -1  # negative
            @test run_wasm(wasm_bytes, "classify_number_test", Int32(0)) == 0    # zero
            @test run_wasm(wasm_bytes, "classify_number_test", Int32(1)) == 1    # positive
            @test run_wasm(wasm_bytes, "classify_number_test", Int32(100)) == 1  # positive
        end

    end

    # ========================================================================
    # Phase 25: Interpreter Tokenizer (BROWSER-020)
    # Tests for the Julia interpreter tokenizer that will be compiled to WASM
    # ========================================================================
    @testset "Phase 25: Interpreter Tokenizer" begin

        # Include the tokenizer module
        include("../src/Interpreter/Tokenizer.jl")

        @testset "Character classification (Int32 returns)" begin
            # Compile character classifiers
            wasm_bytes = WasmTarget.compile_multi([
                (is_digit, (Int32,)),
                (is_alpha, (Int32,)),
                (is_alnum, (Int32,)),
                (is_whitespace, (Int32,)),
                (is_newline, (Int32,))
            ])
            @test length(wasm_bytes) > 0

            # Test is_digit
            @test run_wasm(wasm_bytes, "is_digit", Int32(48)) == 1   # '0'
            @test run_wasm(wasm_bytes, "is_digit", Int32(57)) == 1   # '9'
            @test run_wasm(wasm_bytes, "is_digit", Int32(65)) == 0   # 'A'
            @test run_wasm(wasm_bytes, "is_digit", Int32(97)) == 0   # 'a'

            # Test is_alpha
            @test run_wasm(wasm_bytes, "is_alpha", Int32(97)) == 1   # 'a'
            @test run_wasm(wasm_bytes, "is_alpha", Int32(122)) == 1  # 'z'
            @test run_wasm(wasm_bytes, "is_alpha", Int32(65)) == 1   # 'A'
            @test run_wasm(wasm_bytes, "is_alpha", Int32(90)) == 1   # 'Z'
            @test run_wasm(wasm_bytes, "is_alpha", Int32(95)) == 1   # '_'
            @test run_wasm(wasm_bytes, "is_alpha", Int32(48)) == 0   # '0'

            # Test is_whitespace
            @test run_wasm(wasm_bytes, "is_whitespace", Int32(32)) == 1  # ' '
            @test run_wasm(wasm_bytes, "is_whitespace", Int32(9)) == 1   # '\t'
            @test run_wasm(wasm_bytes, "is_whitespace", Int32(13)) == 1  # '\r'
            @test run_wasm(wasm_bytes, "is_whitespace", Int32(10)) == 0  # '\n' - not whitespace in our lexer

            # Test is_newline
            @test run_wasm(wasm_bytes, "is_newline", Int32(10)) == 1  # '\n'
            @test run_wasm(wasm_bytes, "is_newline", Int32(13)) == 0  # '\r'
        end

        @testset "Tokenizer Julia-side functionality" begin
            # Test tokenize function in Julia (this tests the algorithm, not WASM)
            tokens = tokenize("x = 5", Int32(100))
            @test tokens.count == 4
            @test token_list_get(tokens, Int32(1)).type == TOK_IDENT
            @test token_list_get(tokens, Int32(2)).type == TOK_EQ
            @test token_list_get(tokens, Int32(3)).type == TOK_INT
            @test token_list_get(tokens, Int32(3)).int_value == 5
            @test token_list_get(tokens, Int32(4)).type == TOK_EOF

            # Test arithmetic
            tokens2 = tokenize("3 + 4 * 2", Int32(100))
            @test tokens2.count == 6
            @test token_list_get(tokens2, Int32(1)).type == TOK_INT
            @test token_list_get(tokens2, Int32(1)).int_value == 3
            @test token_list_get(tokens2, Int32(2)).type == TOK_PLUS
            @test token_list_get(tokens2, Int32(3)).type == TOK_INT
            @test token_list_get(tokens2, Int32(3)).int_value == 4
            @test token_list_get(tokens2, Int32(4)).type == TOK_STAR

            # Test keywords
            tokens3 = tokenize("if x end", Int32(100))
            @test token_list_get(tokens3, Int32(1)).type == TOK_KW_IF
            @test token_list_get(tokens3, Int32(2)).type == TOK_IDENT
            @test token_list_get(tokens3, Int32(3)).type == TOK_KW_END

            # Test comparison operators
            tokens4 = tokenize("a == b != c", Int32(100))
            @test token_list_get(tokens4, Int32(2)).type == TOK_EQ_EQ
            @test token_list_get(tokens4, Int32(4)).type == TOK_NE

            # Test float
            tokens5 = tokenize("3.14", Int32(100))
            @test token_list_get(tokens5, Int32(1)).type == TOK_FLOAT
            @test token_list_get(tokens5, Int32(1)).float_value ≈ Float32(3.14)

            # Test string
            tokens6 = tokenize("\"hello\"", Int32(100))
            @test token_list_get(tokens6, Int32(1)).type == TOK_STRING
        end

    end

    # ========================================================================
    # Phase 26: Interpreter Parser and AST (BROWSER-021)
    # Tests for the Julia interpreter parser that builds AST from tokens
    # ========================================================================
    @testset "Phase 26: Interpreter Parser and AST" begin

        # Include the parser module (tokenizer already included in Phase 25)
        include("../src/Interpreter/Parser.jl")

        @testset "Parser - Literal expressions" begin
            # Integer literal
            p1 = parser_new("42", Int32(100))
            ast1 = parse_expression(p1)
            @test ast1.kind == AST_INT_LIT
            @test ast1.int_value == Int32(42)

            # Float literal
            p2 = parser_new("3.14", Int32(100))
            ast2 = parse_expression(p2)
            @test ast2.kind == AST_FLOAT_LIT
            @test ast2.float_value ≈ Float32(3.14)

            # Boolean true
            p3 = parser_new("true", Int32(100))
            ast3 = parse_expression(p3)
            @test ast3.kind == AST_BOOL_LIT
            @test ast3.int_value == Int32(1)

            # Boolean false
            p4 = parser_new("false", Int32(100))
            ast4 = parse_expression(p4)
            @test ast4.kind == AST_BOOL_LIT
            @test ast4.int_value == Int32(0)

            # Nothing
            p5 = parser_new("nothing", Int32(100))
            ast5 = parse_expression(p5)
            @test ast5.kind == AST_NOTHING_LIT

            # Identifier
            p6 = parser_new("foo", Int32(100))
            ast6 = parse_expression(p6)
            @test ast6.kind == AST_IDENT
            @test ast6.str_start == Int32(1)
            @test ast6.str_length == Int32(3)
        end

        @testset "Parser - Binary expressions" begin
            # Addition
            p1 = parser_new("1 + 2", Int32(100))
            ast1 = parse_expression(p1)
            @test ast1.kind == AST_BINARY
            @test ast1.op == OP_ADD
            @test ast1.left.kind == AST_INT_LIT
            @test ast1.left.int_value == Int32(1)
            @test ast1.right.kind == AST_INT_LIT
            @test ast1.right.int_value == Int32(2)

            # Multiplication with precedence
            p2 = parser_new("1 + 2 * 3", Int32(100))
            ast2 = parse_expression(p2)
            @test ast2.kind == AST_BINARY
            @test ast2.op == OP_ADD
            @test ast2.left.int_value == Int32(1)
            @test ast2.right.kind == AST_BINARY
            @test ast2.right.op == OP_MUL

            # Comparison
            p3 = parser_new("x < 10", Int32(100))
            ast3 = parse_expression(p3)
            @test ast3.kind == AST_BINARY
            @test ast3.op == OP_LT

            # Equality
            p4 = parser_new("a == b", Int32(100))
            ast4 = parse_expression(p4)
            @test ast4.kind == AST_BINARY
            @test ast4.op == OP_EQ

            # Logical operators
            p5 = parser_new("x && y || z", Int32(100))
            ast5 = parse_expression(p5)
            @test ast5.kind == AST_BINARY
            @test ast5.op == OP_OR  # || has lower precedence
        end

        @testset "Parser - Unary expressions" begin
            # Negation
            p1 = parser_new("-5", Int32(100))
            ast1 = parse_expression(p1)
            @test ast1.kind == AST_UNARY
            @test ast1.op == OP_NEG
            @test ast1.left.kind == AST_INT_LIT
            @test ast1.left.int_value == Int32(5)

            # Not
            p2 = parser_new("not true", Int32(100))
            ast2 = parse_expression(p2)
            @test ast2.kind == AST_UNARY
            @test ast2.op == OP_NOT
        end

        @testset "Parser - Parenthesized expressions" begin
            # (1 + 2) * 3 - should compute 1+2 first
            p1 = parser_new("(1 + 2) * 3", Int32(100))
            ast1 = parse_expression(p1)
            @test ast1.kind == AST_BINARY
            @test ast1.op == OP_MUL
            @test ast1.left.kind == AST_BINARY
            @test ast1.left.op == OP_ADD
        end

        @testset "Parser - Function calls" begin
            # Single argument
            p1 = parser_new("foo(5)", Int32(100))
            ast1 = parse_expression(p1)
            @test ast1.kind == AST_CALL
            @test ast1.left.kind == AST_IDENT
            @test ast1.num_children == Int32(1)
            @test ast1.children[1].kind == AST_INT_LIT

            # Multiple arguments
            p2 = parser_new("bar(1, 2, 3)", Int32(100))
            ast2 = parse_expression(p2)
            @test ast2.kind == AST_CALL
            @test ast2.num_children == Int32(3)

            # No arguments
            p3 = parser_new("baz()", Int32(100))
            ast3 = parse_expression(p3)
            @test ast3.kind == AST_CALL
            @test ast3.num_children == Int32(0)
        end

        @testset "Parser - Assignment" begin
            p1 = parser_new("x = 5", Int32(100))
            parser_skip_terminators!(p1)
            ast1 = parse_statement(p1)
            @test ast1.kind == AST_ASSIGN
            @test ast1.left.kind == AST_IDENT
            @test ast1.right.kind == AST_INT_LIT
            @test ast1.right.int_value == Int32(5)
        end

        @testset "Parser - If statements" begin
            # Simple if
            p1 = parser_new("if x\n  y\nend", Int32(100))
            parser_skip_terminators!(p1)
            ast1 = parse_statement(p1)
            @test ast1.kind == AST_IF
            @test ast1.left.kind == AST_IDENT  # condition
            @test ast1.num_children >= Int32(1)  # then body

            # If-else
            p2 = parser_new("if x\n  1\nelse\n  2\nend", Int32(100))
            parser_skip_terminators!(p2)
            ast2 = parse_statement(p2)
            @test ast2.kind == AST_IF
            @test ast2.right !== nothing  # else branch
            @test ast2.right.kind == AST_BLOCK
        end

        @testset "Parser - While loops" begin
            p1 = parser_new("while x < 10\n  x = x + 1\nend", Int32(100))
            parser_skip_terminators!(p1)
            ast1 = parse_statement(p1)
            @test ast1.kind == AST_WHILE
            @test ast1.left.kind == AST_BINARY  # condition
            @test ast1.num_children >= Int32(1)  # body
        end

        @testset "Parser - For loops" begin
            p1 = parser_new("for i in range\n  x = i\nend", Int32(100))
            parser_skip_terminators!(p1)
            ast1 = parse_statement(p1)
            @test ast1.kind == AST_FOR
            @test ast1.left.kind == AST_IDENT  # iterator var
            @test ast1.right.kind == AST_IDENT  # iterable
        end

        @testset "Parser - Function definitions" begin
            p1 = parser_new("function add(a, b)\n  return a + b\nend", Int32(100))
            parser_skip_terminators!(p1)
            ast1 = parse_statement(p1)
            @test ast1.kind == AST_FUNC
            @test ast1.int_value == Int32(2)  # 2 parameters
            # Children: params + body
            @test ast1.num_children >= Int32(3)  # 2 params + 1 return stmt
        end

        @testset "Parser - Return statements" begin
            # Return with value
            p1 = parser_new("return 42", Int32(100))
            parser_skip_terminators!(p1)
            ast1 = parse_statement(p1)
            @test ast1.kind == AST_RETURN
            @test ast1.left !== nothing
            @test ast1.left.kind == AST_INT_LIT

            # Return without value
            p2 = parser_new("return\n", Int32(100))
            parser_skip_terminators!(p2)
            ast2 = parse_statement(p2)
            @test ast2.kind == AST_RETURN
            @test ast2.left === nothing
        end

        @testset "Parser - Full program" begin
            code = """
            x = 5
            y = 10
            z = x + y
            """
            p1 = parser_new(code, Int32(100))
            ast1 = parse_program(p1)
            @test ast1.kind == AST_PROGRAM
            @test ast1.num_children == Int32(3)
            @test ast1.children[1].kind == AST_ASSIGN
            @test ast1.children[2].kind == AST_ASSIGN
            @test ast1.children[3].kind == AST_ASSIGN
        end

        @testset "Parser - Complex program" begin
            code = """
            function factorial(n)
                if n <= 1
                    return 1
                else
                    return n * factorial(n - 1)
                end
            end
            result = factorial(5)
            """
            p1 = parser_new(code, Int32(200))
            ast1 = parse_program(p1)
            @test ast1.kind == AST_PROGRAM
            @test ast1.num_children == Int32(2)  # function def + assignment
            @test ast1.children[1].kind == AST_FUNC
            @test ast1.children[2].kind == AST_ASSIGN
        end

    end

    # ========================================================================
    # Phase 27: Interpreter Evaluator (BROWSER-022)
    # Tests for the Julia interpreter evaluator that executes AST nodes
    # ========================================================================
    @testset "Phase 27: Interpreter Evaluator" begin

        # Include the evaluator module (tokenizer and parser already included)
        include("../src/Interpreter/Evaluator.jl")

        @testset "Evaluator - Literal values" begin
            # Integer literal
            p1 = parser_new("42", Int32(100))
            ast1 = parse_expression(p1)
            env = env_new(Int32(100))
            (val1, _) = eval_node(ast1, "42", env)
            @test val1.tag == VAL_INT
            @test val1.int_val == Int32(42)

            # Float literal
            p2 = parser_new("3.14", Int32(100))
            ast2 = parse_expression(p2)
            (val2, _) = eval_node(ast2, "3.14", env)
            @test val2.tag == VAL_FLOAT
            @test val2.float_val ≈ Float32(3.14)

            # Boolean true
            p3 = parser_new("true", Int32(100))
            ast3 = parse_expression(p3)
            (val3, _) = eval_node(ast3, "true", env)
            @test val3.tag == VAL_BOOL
            @test val3.int_val == Int32(1)

            # Boolean false
            p4 = parser_new("false", Int32(100))
            ast4 = parse_expression(p4)
            (val4, _) = eval_node(ast4, "false", env)
            @test val4.tag == VAL_BOOL
            @test val4.int_val == Int32(0)

            # Nothing
            p5 = parser_new("nothing", Int32(100))
            ast5 = parse_expression(p5)
            (val5, _) = eval_node(ast5, "nothing", env)
            @test val5.tag == VAL_NOTHING
        end

        @testset "Evaluator - Arithmetic operations" begin
            env = env_new(Int32(100))

            # Addition
            p1 = parser_new("3 + 5", Int32(100))
            ast1 = parse_expression(p1)
            (val1, _) = eval_node(ast1, "3 + 5", env)
            @test val1.tag == VAL_INT
            @test val1.int_val == Int32(8)

            # Subtraction
            p2 = parser_new("10 - 4", Int32(100))
            ast2 = parse_expression(p2)
            (val2, _) = eval_node(ast2, "10 - 4", env)
            @test val2.tag == VAL_INT
            @test val2.int_val == Int32(6)

            # Multiplication
            p3 = parser_new("7 * 6", Int32(100))
            ast3 = parse_expression(p3)
            (val3, _) = eval_node(ast3, "7 * 6", env)
            @test val3.tag == VAL_INT
            @test val3.int_val == Int32(42)

            # Division
            p4 = parser_new("20 / 4", Int32(100))
            ast4 = parse_expression(p4)
            (val4, _) = eval_node(ast4, "20 / 4", env)
            @test val4.tag == VAL_INT
            @test val4.int_val == Int32(5)

            # Modulo
            p5 = parser_new("17 % 5", Int32(100))
            ast5 = parse_expression(p5)
            (val5, _) = eval_node(ast5, "17 % 5", env)
            @test val5.tag == VAL_INT
            @test val5.int_val == Int32(2)

            # Power
            p6 = parser_new("2 ^ 3", Int32(100))
            ast6 = parse_expression(p6)
            (val6, _) = eval_node(ast6, "2 ^ 3", env)
            @test val6.tag == VAL_INT
            @test val6.int_val == Int32(8)
        end

        @testset "Evaluator - Comparison operations" begin
            env = env_new(Int32(100))

            # Less than
            p1 = parser_new("3 < 5", Int32(100))
            ast1 = parse_expression(p1)
            (val1, _) = eval_node(ast1, "3 < 5", env)
            @test val1.tag == VAL_BOOL
            @test val1.int_val == Int32(1)  # true

            p2 = parser_new("5 < 3", Int32(100))
            ast2 = parse_expression(p2)
            (val2, _) = eval_node(ast2, "5 < 3", env)
            @test val2.int_val == Int32(0)  # false

            # Equality
            p3 = parser_new("5 == 5", Int32(100))
            ast3 = parse_expression(p3)
            (val3, _) = eval_node(ast3, "5 == 5", env)
            @test val3.int_val == Int32(1)

            p4 = parser_new("5 == 3", Int32(100))
            ast4 = parse_expression(p4)
            (val4, _) = eval_node(ast4, "5 == 3", env)
            @test val4.int_val == Int32(0)

            # Greater than or equal
            p5 = parser_new("5 >= 5", Int32(100))
            ast5 = parse_expression(p5)
            (val5, _) = eval_node(ast5, "5 >= 5", env)
            @test val5.int_val == Int32(1)
        end

        @testset "Evaluator - Logical operations" begin
            env = env_new(Int32(100))

            # AND with both true
            p1 = parser_new("true && true", Int32(100))
            ast1 = parse_expression(p1)
            (val1, _) = eval_node(ast1, "true && true", env)
            @test val1.tag == VAL_BOOL
            @test val1.int_val == Int32(1)

            # AND with one false (short-circuit)
            p2 = parser_new("false && true", Int32(100))
            ast2 = parse_expression(p2)
            (val2, _) = eval_node(ast2, "false && true", env)
            @test val2.int_val == Int32(0)

            # OR with one true (short-circuit)
            p3 = parser_new("true || false", Int32(100))
            ast3 = parse_expression(p3)
            (val3, _) = eval_node(ast3, "true || false", env)
            @test val3.int_val == Int32(1)

            # OR with both false
            p4 = parser_new("false || false", Int32(100))
            ast4 = parse_expression(p4)
            (val4, _) = eval_node(ast4, "false || false", env)
            @test val4.int_val == Int32(0)
        end

        @testset "Evaluator - Unary operations" begin
            env = env_new(Int32(100))

            # Negation
            p1 = parser_new("-5", Int32(100))
            ast1 = parse_expression(p1)
            (val1, _) = eval_node(ast1, "-5", env)
            @test val1.tag == VAL_INT
            @test val1.int_val == Int32(-5)

            # Not
            p2 = parser_new("not true", Int32(100))
            ast2 = parse_expression(p2)
            (val2, _) = eval_node(ast2, "not true", env)
            @test val2.tag == VAL_BOOL
            @test val2.int_val == Int32(0)
        end

        @testset "Evaluator - Variable assignment and lookup" begin
            env = env_new(Int32(100))
            source = "x = 42"

            # Parse and evaluate assignment
            p1 = parser_new(source, Int32(100))
            ast1 = parse_statement(p1)
            (val1, _) = eval_node(ast1, source, env)
            @test val1.tag == VAL_INT
            @test val1.int_val == Int32(42)

            # Check variable is stored
            x_val = env_get(env, "x")
            @test x_val.tag == VAL_INT
            @test x_val.int_val == Int32(42)

            # Variable lookup in expression
            source2 = "x + 8"
            p2 = parser_new(source2, Int32(100))
            ast2 = parse_expression(p2)
            (val2, _) = eval_node(ast2, source2, env)
            @test val2.tag == VAL_INT
            @test val2.int_val == Int32(50)
        end

        @testset "Evaluator - If statements" begin
            # If with true condition
            code1 = """
            x = 0
            if true
                x = 1
            end
            x
            """
            p1 = parser_new(code1, Int32(100))
            prog1 = parse_program(p1)
            output1 = eval_program(prog1, code1)
            @test contains(output1, "1")

            # If with false condition
            code2 = """
            x = 0
            if false
                x = 1
            end
            x
            """
            p2 = parser_new(code2, Int32(100))
            prog2 = parse_program(p2)
            output2 = eval_program(prog2, code2)
            @test contains(output2, "0")

            # If-else
            code3 = """
            x = 5
            if x > 10
                y = 1
            else
                y = 2
            end
            y
            """
            p3 = parser_new(code3, Int32(100))
            prog3 = parse_program(p3)
            output3 = eval_program(prog3, code3)
            @test contains(output3, "2")
        end

        @testset "Evaluator - While loops" begin
            code1 = """
            x = 0
            i = 0
            while i < 5
                x = x + i
                i = i + 1
            end
            x
            """
            p1 = parser_new(code1, Int32(100))
            prog1 = parse_program(p1)
            output1 = eval_program(prog1, code1)
            @test contains(output1, "10")  # 0+1+2+3+4 = 10
        end

        @testset "Evaluator - User-defined functions" begin
            code1 = """
            function add(a, b)
                return a + b
            end
            add(3, 5)
            """
            p1 = parser_new(code1, Int32(200))
            prog1 = parse_program(p1)
            output1 = eval_program(prog1, code1)
            @test contains(output1, "8")

            # Recursive function
            code2 = """
            function fact(n)
                if n <= 1
                    return 1
                else
                    return n * fact(n - 1)
                end
            end
            fact(5)
            """
            p2 = parser_new(code2, Int32(200))
            prog2 = parse_program(p2)
            output2 = eval_program(prog2, code2)
            @test contains(output2, "120")
        end

        @testset "Evaluator - Built-in functions" begin
            # println
            code1 = "println(42)"
            p1 = parser_new(code1, Int32(100))
            prog1 = parse_program(p1)
            clear_output()
            eval_program(prog1, code1)
            @test contains(get_output(), "42")

            # abs
            code2 = "abs(-5)"
            p2 = parser_new(code2, Int32(100))
            prog2 = parse_program(p2)
            output2 = eval_program(prog2, code2)
            @test contains(output2, "5")

            # min
            code3 = "min(3, 7)"
            p3 = parser_new(code3, Int32(100))
            prog3 = parse_program(p3)
            output3 = eval_program(prog3, code3)
            @test contains(output3, "3")

            # max
            code4 = "max(3, 7)"
            p4 = parser_new(code4, Int32(100))
            prog4 = parse_program(p4)
            output4 = eval_program(prog4, code4)
            @test contains(output4, "7")
        end

        @testset "Evaluator - Complex program" begin
            # FizzBuzz-like program
            code1 = """
            i = 1
            while i <= 3
                if i == 1
                    println(1)
                elseif i == 2
                    println(2)
                else
                    println(3)
                end
                i = i + 1
            end
            """
            p1 = parser_new(code1, Int32(200))
            prog1 = parse_program(p1)
            clear_output()
            eval_program(prog1, code1)
            output1 = get_output()
            @test contains(output1, "1")
            @test contains(output1, "2")
            @test contains(output1, "3")
        end

        @testset "Evaluator - The playground example" begin
            # x = 5; y = 3; println(x + y)
            code = "x = 5; y = 3; println(x + y)"
            p = parser_new(code, Int32(100))
            prog = parse_program(p)
            clear_output()
            eval_program(prog, code)
            output = get_output()
            @test contains(output, "8")
        end

    end

    # ========================================================================
    # Phase 28: Expr Tree-Walker Evaluator (PURE-012)
    # Tests for the ExprNode-based evaluator that compiles to WasmGC
    # ========================================================================
    @testset "Phase 28: Expr Tree-Walker Evaluator" begin

        @testset "Host-side eval_expr correctness" begin
            # Arithmetic
            @test eval_expr(:(1 + 2)) == 3
            @test eval_expr(:(10 - 3)) == 7
            @test eval_expr(:(4 * 5)) == 20
            @test eval_expr(:(2 + 3 * 4)) == 14  # (2 + (3*4)) = 14

            # Variable assignment and reference
            @test eval_expr(:(x = 5; x + 1)) == 6
            @test eval_expr(:(a = 10; b = 20; a + b)) == 30

            # Conditionals
            @test eval_expr(:(if true 1 else 2 end)) == 1
            @test eval_expr(:(if false 1 else 2 end)) == 2
            @test eval_expr(:(x = 0; if x 42 else 99 end)) == 99
        end

        @testset "eval_node compiles to WasmGC" begin
            wasm_bytes = WasmTarget.compile(eval_node, (Vector{ExprNode}, Int32, EvalEnv))
            @test length(wasm_bytes) > 0
            @test validate_wasm(wasm_bytes)
        end

    end

end
