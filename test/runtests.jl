using WasmTarget
using Test

include("utils.jl")

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

end
