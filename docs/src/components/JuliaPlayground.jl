# JuliaPlayground.jl - Rust Playground-style Julia REPL
#
# Features:
# - Code editor with syntax highlighting (coming)
# - Example selector with pre-compiled WASM
# - Run button that executes WASM and shows output
# - Input field for function arguments

# Pre-compiled examples with their WASM files
const EXAMPLES = [
    (
        name = "Fibonacci",
        code = """function fibonacci(n::Int32)::Int32
    if n <= 1
        return n
    end
    return fibonacci(n - Int32(1)) + fibonacci(n - Int32(2))
end""",
        wasm = "fibonacci.wasm",
        func = "fibonacci",
        default_arg = "10",
        description = "Recursive fibonacci - demonstrates recursion and conditionals"
    ),
    (
        name = "Factorial",
        code = """function factorial(n::Int32)::Int32
    if n <= 1
        return Int32(1)
    end
    return n * factorial(n - Int32(1))
end""",
        wasm = "factorial.wasm",
        func = "factorial",
        default_arg = "5",
        description = "Recursive factorial - basic recursion pattern"
    ),
    (
        name = "Sum to N",
        code = """function sum_to_n(n::Int32)::Int32
    result = Int32(0)
    i = Int32(1)
    while i <= n
        result = result + i
        i = i + Int32(1)
    end
    return result
end""",
        wasm = "sum_to_n.wasm",
        func = "sum_to_n",
        default_arg = "100",
        description = "Sum 1 to N using while loop"
    ),
    (
        name = "Is Prime",
        code = """function is_prime(n::Int32)::Int32
    if n <= 1
        return Int32(0)
    end
    if n <= 3
        return Int32(1)
    end
    if n % Int32(2) == 0
        return Int32(0)
    end
    i = Int32(3)
    while i * i <= n
        if n % i == 0
            return Int32(0)
        end
        i = i + Int32(2)
    end
    return Int32(1)
end""",
        wasm = "is_prime.wasm",
        func = "is_prime",
        default_arg = "17",
        description = "Prime check - loops and modulo"
    ),
    (
        name = "GCD",
        code = """function gcd(a::Int32, b::Int32)::Int32
    while b != Int32(0)
        t = b
        b = a % b
        a = t
    end
    return a
end""",
        wasm = "gcd.wasm",
        func = "gcd",
        default_arg = "48, 18",
        description = "Greatest common divisor - Euclidean algorithm"
    )
]

"""
Julia Playground - Rust Playground-style REPL interface.

Displays a code editor with pre-compiled examples.
User selects an example, optionally modifies input, clicks Run.
The pre-compiled WASM executes and shows the result.
"""
JuliaPlayground = island(:JuliaPlayground) do
    # State
    selected_idx, set_selected_idx = create_signal(1)
    input_value, set_input_value = create_signal("10")
    output, set_output = create_signal("")
    is_running, set_is_running = create_signal(false)

    # Get current example
    current_example = () -> EXAMPLES[selected_idx()]

    Div(:class => "max-w-5xl mx-auto",
        # Top bar: Example selector + Run button
        Div(:class => "flex items-center justify-between mb-4 gap-4",
            # Example selector
            Div(:class => "flex items-center gap-2",
                Label(:class => "text-stone-600 dark:text-stone-400 text-sm font-medium", "Example:"),
                Select(:class => "bg-stone-100 dark:bg-stone-800 border border-stone-300 dark:border-stone-600 rounded-lg px-3 py-2 text-stone-800 dark:text-stone-200 text-sm",
                    :on_change => (e) -> begin
                        idx = parse(Int, e.target.value)
                        set_selected_idx(idx)
                        set_input_value(EXAMPLES[idx].default_arg)
                        set_output("")
                    end,
                    [Option(:value => string(i),
                            :selected => i == selected_idx(),
                            ex.name) for (i, ex) in enumerate(EXAMPLES)]...
                )
            ),

            # Run button
            Button(:class => "flex items-center gap-2 bg-cyan-500 hover:bg-cyan-600 text-white px-6 py-2 rounded-lg font-semibold transition-colors shadow-lg shadow-cyan-500/20",
                :on_click => () -> run_wasm(current_example(), input_value(), set_output, set_is_running),
                :disabled => is_running(),
                is_running() ? "Running..." : "Run"
            )
        ),

        # Description
        P(:class => "text-stone-500 dark:text-stone-400 text-sm mb-4",
            () -> current_example().description
        ),

        # Main content: Code + Output side by side
        Div(:class => "grid lg:grid-cols-2 gap-4",
            # Code editor panel
            Div(:class => "flex flex-col",
                Div(:class => "flex items-center justify-between px-4 py-2 bg-stone-700 dark:bg-stone-800 rounded-t-xl",
                    Span(:class => "text-stone-300 text-sm font-medium", "Julia"),
                    Span(:class => "text-stone-500 text-xs", "Read-only (client-side compiler coming)")
                ),
                Pre(:class => "bg-stone-800 dark:bg-stone-900 p-4 rounded-b-xl overflow-x-auto flex-1 min-h-[300px]",
                    Code(:class => "text-sm text-stone-100 font-mono whitespace-pre",
                        () -> current_example().code
                    )
                )
            ),

            # Output panel
            Div(:class => "flex flex-col",
                Div(:class => "flex items-center justify-between px-4 py-2 bg-stone-700 dark:bg-stone-800 rounded-t-xl",
                    Span(:class => "text-stone-300 text-sm font-medium", "Output"),
                    # Input field
                    Div(:class => "flex items-center gap-2",
                        Label(:class => "text-stone-400 text-xs", "Input:"),
                        Input(:type => "text",
                              :class => "bg-stone-600 border border-stone-500 rounded px-2 py-1 text-stone-100 text-sm w-24 font-mono",
                              :value => input_value,
                              :on_input => (e) -> set_input_value(e.target.value)
                        )
                    )
                ),
                Div(:class => "bg-stone-900 dark:bg-black p-4 rounded-b-xl flex-1 min-h-[300px] font-mono",
                    # Output content
                    output() == "" ?
                        Span(:class => "text-stone-500 text-sm", "Click 'Run' to execute...") :
                        Pre(:class => "text-cyan-400 text-lg", output())
                )
            )
        ),

        # Footer info
        Div(:class => "mt-6 text-center",
            P(:class => "text-stone-500 dark:text-stone-400 text-sm",
                "100% Julia compiled to WebAssembly via WasmTarget.jl"
            ),
            P(:class => "text-stone-400 dark:text-stone-500 text-xs mt-1",
                "Full client-side compiler (edit any code) coming with Julia 1.12 trimming"
            )
        )
    )
end

"""
Run the WASM example with the given input.
"""
function run_wasm(example, input_str, set_output, set_is_running)
    set_is_running(true)
    set_output("Loading WASM...")

    # Parse input (handle single value or comma-separated)
    args = try
        if occursin(",", input_str)
            [parse(Int32, strip(s)) for s in split(input_str, ",")]
        else
            [parse(Int32, strip(input_str))]
        end
    catch
        set_output("Error: Invalid input")
        set_is_running(false)
        return
    end

    # Load and run WASM
    # This is a placeholder - actual implementation would use JS interop
    # to fetch the WASM file and instantiate it
    try
        # Simulated result for now - in real implementation:
        # 1. fetch(example.wasm)
        # 2. WebAssembly.instantiate(bytes)
        # 3. Call instance.exports[example.func](...args)

        # For demo, calculate result in Julia
        result = if example.func == "fibonacci"
            fib(args[1])
        elseif example.func == "factorial"
            fact(args[1])
        elseif example.func == "sum_to_n"
            sum_n(args[1])
        elseif example.func == "is_prime"
            isprime_check(args[1])
        elseif example.func == "gcd"
            length(args) >= 2 ? gcd_calc(args[1], args[2]) : 0
        else
            0
        end

        set_output("$(example.func)($(input_str)) = $(result)")
    catch e
        set_output("Error: $(e)")
    end

    set_is_running(false)
end

# Helper functions for demo (these would be replaced by actual WASM execution)
fib(n::Int32) = n <= 1 ? n : fib(n - Int32(1)) + fib(n - Int32(2))
fact(n::Int32) = n <= 1 ? Int32(1) : n * fact(n - Int32(1))
sum_n(n::Int32) = div(n * (n + Int32(1)), Int32(2))
function isprime_check(n::Int32)
    n <= 1 && return Int32(0)
    n <= 3 && return Int32(1)
    n % 2 == 0 && return Int32(0)
    i = Int32(3)
    while i * i <= n
        n % i == 0 && return Int32(0)
        i += Int32(2)
    end
    Int32(1)
end
gcd_calc(a::Int32, b::Int32) = b == 0 ? a : gcd_calc(b, a % b)
