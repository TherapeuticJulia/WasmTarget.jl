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

end
