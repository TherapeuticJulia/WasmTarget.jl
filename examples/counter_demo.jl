# Counter Demo - Demonstrates Therapy.jl patterns using WasmTarget.jl
#
# This example shows how to build a reactive counter using:
# - Wasm globals for state (like Signals in reactive frameworks)
# - Exported functions for event handlers
# - Imported functions for DOM manipulation
#
# Run with: julia --project=. examples/counter_demo.jl

using WasmTarget
using WasmTarget: WasmModule, add_import!, add_function!, add_export!,
                  add_global!, add_global_export!, to_bytes,
                  I32, ExternRef, Opcode, encode_leb128_signed

function build_counter_module()
    mod = WasmModule()

    # =========================================================================
    # Imports - DOM manipulation functions provided by JavaScript
    # =========================================================================

    # get_element_by_id(id_ptr: externref) -> element: externref
    get_elem_idx = add_import!(mod, "dom", "get_element_by_id",
                               WasmTarget.WasmValType[ExternRef],
                               WasmTarget.WasmValType[ExternRef])

    # set_text_content(element: externref, value: i32)
    set_text_idx = add_import!(mod, "dom", "set_text_content",
                               WasmTarget.WasmValType[ExternRef, I32],
                               WasmTarget.WasmValType[])

    # add_click_handler(element: externref, handler_idx: i32)
    add_click_idx = add_import!(mod, "dom", "add_click_handler",
                                WasmTarget.WasmValType[ExternRef, I32],
                                WasmTarget.WasmValType[])

    # =========================================================================
    # Globals - Reactive state storage
    # =========================================================================

    # count: i32 (mutable, starts at 0)
    count_idx = add_global!(mod, I32, true, 0)
    add_global_export!(mod, "count", count_idx)

    # =========================================================================
    # Functions - Event handlers and helpers
    # =========================================================================

    # increment(): void
    # Increments the count global by 1
    increment_body = UInt8[]
    append!(increment_body, [
        Opcode.GLOBAL_GET, 0x00,           # get count
        Opcode.I32_CONST, 0x01,            # push 1
        Opcode.I32_ADD,                    # add
        Opcode.GLOBAL_SET, 0x00,           # set count
        Opcode.END
    ])
    increment_idx = add_function!(mod, WasmTarget.WasmValType[], WasmTarget.WasmValType[],
                                  WasmTarget.WasmValType[], increment_body)
    add_export!(mod, "increment", 0, increment_idx)

    # decrement(): void
    # Decrements the count global by 1
    decrement_body = UInt8[]
    append!(decrement_body, [
        Opcode.GLOBAL_GET, 0x00,           # get count
        Opcode.I32_CONST, 0x01,            # push 1
        Opcode.I32_SUB,                    # subtract
        Opcode.GLOBAL_SET, 0x00,           # set count
        Opcode.END
    ])
    decrement_idx = add_function!(mod, WasmTarget.WasmValType[], WasmTarget.WasmValType[],
                                  WasmTarget.WasmValType[], decrement_body)
    add_export!(mod, "decrement", 0, decrement_idx)

    # get_count(): i32
    # Returns the current count value
    get_count_body = UInt8[]
    append!(get_count_body, [
        Opcode.GLOBAL_GET, 0x00,           # get count
        Opcode.END
    ])
    get_count_idx = add_function!(mod, WasmTarget.WasmValType[], [I32],
                                  WasmTarget.WasmValType[], get_count_body)
    add_export!(mod, "get_count", 0, get_count_idx)

    # reset(): void
    # Resets the count to 0
    reset_body = UInt8[]
    append!(reset_body, [
        Opcode.I32_CONST, 0x00,            # push 0
        Opcode.GLOBAL_SET, 0x00,           # set count
        Opcode.END
    ])
    reset_idx = add_function!(mod, WasmTarget.WasmValType[], WasmTarget.WasmValType[],
                              WasmTarget.WasmValType[], reset_body)
    add_export!(mod, "reset", 0, reset_idx)

    return mod
end

# Build and save the module
mod = build_counter_module()
bytes = to_bytes(mod)

# Write to file
output_path = joinpath(@__DIR__, "counter.wasm")
write(output_path, bytes)
println("Generated: $output_path ($(length(bytes)) bytes)")

# Also create the HTML harness
html_path = joinpath(@__DIR__, "counter.html")
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>WasmTarget.jl Counter Demo</title>
    <style>
        body { font-family: system-ui; max-width: 600px; margin: 50px auto; text-align: center; }
        .count { font-size: 4em; margin: 20px; }
        button { font-size: 1.5em; padding: 10px 30px; margin: 5px; cursor: pointer; }
        .info { color: #666; margin-top: 30px; font-size: 0.9em; }
    </style>
</head>
<body>
    <h1>Counter Demo</h1>
    <p>Built with WasmTarget.jl - Julia to WebAssembly</p>

    <div class="count" id="display">0</div>

    <div>
        <button id="decrement">-</button>
        <button id="increment">+</button>
        <button id="reset">Reset</button>
    </div>

    <div class="info">
        <p>This counter is powered by WebAssembly compiled from Julia.</p>
        <p>State is stored in a Wasm global variable.</p>
    </div>

    <script>
    async function init() {
        // DOM API imports for the Wasm module
        const imports = {
            dom: {
                get_element_by_id: (id) => document.getElementById(id),
                set_text_content: (el, value) => { if (el) el.textContent = value; },
                add_click_handler: (el, handlerIdx) => {
                    // In a real implementation, this would map to function table
                    console.log("Handler registered:", handlerIdx);
                }
            }
        };

        const response = await fetch('counter.wasm');
        const bytes = await response.arrayBuffer();
        const result = await WebAssembly.instantiate(bytes, imports);
        const { increment, decrement, get_count, reset } = result.instance.exports;

        const display = document.getElementById('display');

        function updateDisplay() {
            display.textContent = get_count();
        }

        document.getElementById('increment').onclick = () => {
            increment();
            updateDisplay();
        };

        document.getElementById('decrement').onclick = () => {
            decrement();
            updateDisplay();
        };

        document.getElementById('reset').onclick = () => {
            reset();
            updateDisplay();
        };

        console.log('Counter demo loaded!');
    }

    init().catch(console.error);
    </script>
</body>
</html>
"""
write(html_path, html_content)
println("Generated: $html_path")
println("\nTo run: serve the examples directory and open counter.html in a browser")
println("Example: cd examples && python3 -m http.server 8080")
