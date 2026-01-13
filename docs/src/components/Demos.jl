# Demos.jl - Interactive demo components for the Features page
#
# These are Therapy.jl islands that demonstrate pre-compiled WASM functionality.
# Each demo shows a specific Julia feature compiled to WebAssembly.

"""
Arithmetic demo - add, multiply, divide with two number inputs.
"""
ArithmeticDemo = island(:ArithmeticDemo) do
    a, set_a = create_signal(Int32(12))
    b, set_b = create_signal(Int32(5))

    # Results computed via pre-compiled WASM
    add_result = () -> a() + b()
    mul_result = () -> a() * b()
    div_result = () -> b() != 0 ? div(a(), b()) : Int32(0)

    Div(:class => "bg-white dark:bg-stone-700 rounded-xl p-6 w-full shadow-lg",
        # Inputs
        Div(:class => "flex gap-4 mb-4",
            Div(:class => "flex-1",
                Label(:class => "text-stone-500 dark:text-stone-400 text-xs block mb-1", "a"),
                Input(:type => "number",
                      :class => "w-full bg-stone-100 dark:bg-stone-600 border border-stone-300 dark:border-stone-500 rounded px-3 py-2 text-stone-800 dark:text-stone-100 font-mono",
                      :value => a,
                      :on_input => (e) -> set_a(parse(Int32, e.target.value))
                )
            ),
            Div(:class => "flex-1",
                Label(:class => "text-stone-500 dark:text-stone-400 text-xs block mb-1", "b"),
                Input(:type => "number",
                      :class => "w-full bg-stone-100 dark:bg-stone-600 border border-stone-300 dark:border-stone-500 rounded px-3 py-2 text-stone-800 dark:text-stone-100 font-mono",
                      :value => b,
                      :on_input => (e) -> set_b(parse(Int32, e.target.value))
                )
            )
        ),
        # Results
        Div(:class => "space-y-2 font-mono text-sm",
            Div(:class => "flex justify-between p-2 bg-stone-50 dark:bg-stone-600 rounded",
                Span(:class => "text-stone-500 dark:text-stone-400", "add(a, b)"),
                Span(:class => "text-cyan-500 font-bold", () -> string(add_result()))
            ),
            Div(:class => "flex justify-between p-2 bg-stone-50 dark:bg-stone-600 rounded",
                Span(:class => "text-stone-500 dark:text-stone-400", "multiply(a, b)"),
                Span(:class => "text-cyan-500 font-bold", () -> string(mul_result()))
            ),
            Div(:class => "flex justify-between p-2 bg-stone-50 dark:bg-stone-600 rounded",
                Span(:class => "text-stone-500 dark:text-stone-400", "divide(a, b)"),
                Span(:class => "text-cyan-500 font-bold", () -> string(div_result()))
            )
        )
    )
end

"""
Control flow demo - sign function with single number input.
"""
ControlFlowDemo = island(:ControlFlowDemo) do
    n, set_n = create_signal(Int32(0))

    sign_result = () -> begin
        val = n()
        if val > 0
            Int32(1)
        elseif val < 0
            Int32(-1)
        else
            Int32(0)
        end
    end

    Div(:class => "bg-white dark:bg-stone-700 rounded-xl p-6 w-full shadow-lg",
        # Input
        Div(:class => "mb-4",
            Label(:class => "text-stone-500 dark:text-stone-400 text-xs block mb-1", "n"),
            Input(:type => "number",
                  :class => "w-full bg-stone-100 dark:bg-stone-600 border border-stone-300 dark:border-stone-500 rounded px-3 py-2 text-stone-800 dark:text-stone-100 font-mono",
                  :value => n,
                  :on_input => (e) -> set_n(parse(Int32, e.target.value))
            )
        ),
        # Quick buttons
        Div(:class => "flex gap-2 mb-4",
            Button(:class => "flex-1 bg-stone-200 dark:bg-stone-600 hover:bg-stone-300 dark:hover:bg-stone-500 rounded py-1 text-sm text-stone-700 dark:text-stone-200",
                :on_click => () -> set_n(Int32(-5)), "-5"),
            Button(:class => "flex-1 bg-stone-200 dark:bg-stone-600 hover:bg-stone-300 dark:hover:bg-stone-500 rounded py-1 text-sm text-stone-700 dark:text-stone-200",
                :on_click => () -> set_n(Int32(0)), "0"),
            Button(:class => "flex-1 bg-stone-200 dark:bg-stone-600 hover:bg-stone-300 dark:hover:bg-stone-500 rounded py-1 text-sm text-stone-700 dark:text-stone-200",
                :on_click => () -> set_n(Int32(5)), "+5")
        ),
        # Result
        Div(:class => "flex justify-between p-3 bg-stone-50 dark:bg-stone-600 rounded font-mono",
            Span(:class => "text-stone-500 dark:text-stone-400", () -> "sign($(n()))"),
            Span(:class => "text-cyan-500 font-bold text-lg", () -> string(sign_result()))
        )
    )
end

"""
Recursion demo - factorial with single number input.
"""
RecursionDemo = island(:RecursionDemo) do
    n, set_n = create_signal(Int32(5))

    factorial_result = () -> begin
        function fact(x::Int32)::Int32
            x <= 1 ? Int32(1) : x * fact(x - Int32(1))
        end
        fact(n())
    end

    Div(:class => "bg-white dark:bg-stone-700 rounded-xl p-6 w-full shadow-lg",
        # Input
        Div(:class => "mb-4",
            Label(:class => "text-stone-500 dark:text-stone-400 text-xs block mb-1", "n (0-12)"),
            Input(:type => "number",
                  :class => "w-full bg-stone-100 dark:bg-stone-600 border border-stone-300 dark:border-stone-500 rounded px-3 py-2 text-stone-800 dark:text-stone-100 font-mono",
                  :value => n,
                  :min => "0",
                  :max => "12",
                  :on_input => (e) -> set_n(clamp(parse(Int32, e.target.value), Int32(0), Int32(12)))
            )
        ),
        # Quick buttons
        Div(:class => "flex gap-2 mb-4",
            [Button(:class => "flex-1 bg-stone-200 dark:bg-stone-600 hover:bg-stone-300 dark:hover:bg-stone-500 rounded py-1 text-sm text-stone-700 dark:text-stone-200",
                :on_click => () -> set_n(Int32(i)), string(i)) for i in 0:6]...
        ),
        # Result
        Div(:class => "flex justify-between p-3 bg-stone-50 dark:bg-stone-600 rounded font-mono",
            Span(:class => "text-stone-500 dark:text-stone-400", () -> "factorial($(n()))"),
            Span(:class => "text-cyan-500 font-bold text-lg", () -> string(factorial_result()))
        )
    )
end

"""
Loop demo - sum to n with single number input.
"""
LoopDemo = island(:LoopDemo) do
    n, set_n = create_signal(Int32(10))

    sum_result = () -> begin
        val = n()
        div(val * (val + Int32(1)), Int32(2))
    end

    Div(:class => "bg-white dark:bg-stone-700 rounded-xl p-6 w-full shadow-lg",
        # Input
        Div(:class => "mb-4",
            Label(:class => "text-stone-500 dark:text-stone-400 text-xs block mb-1", "n"),
            Input(:type => "number",
                  :class => "w-full bg-stone-100 dark:bg-stone-600 border border-stone-300 dark:border-stone-500 rounded px-3 py-2 text-stone-800 dark:text-stone-100 font-mono",
                  :value => n,
                  :on_input => (e) -> set_n(parse(Int32, e.target.value))
            )
        ),
        # Quick buttons
        Div(:class => "flex gap-2 mb-4",
            Button(:class => "flex-1 bg-stone-200 dark:bg-stone-600 hover:bg-stone-300 dark:hover:bg-stone-500 rounded py-1 text-sm text-stone-700 dark:text-stone-200",
                :on_click => () -> set_n(Int32(10)), "10"),
            Button(:class => "flex-1 bg-stone-200 dark:bg-stone-600 hover:bg-stone-300 dark:hover:bg-stone-500 rounded py-1 text-sm text-stone-700 dark:text-stone-200",
                :on_click => () -> set_n(Int32(100)), "100"),
            Button(:class => "flex-1 bg-stone-200 dark:bg-stone-600 hover:bg-stone-300 dark:hover:bg-stone-500 rounded py-1 text-sm text-stone-700 dark:text-stone-200",
                :on_click => () -> set_n(Int32(1000)), "1000")
        ),
        # Result
        Div(:class => "flex justify-between p-3 bg-stone-50 dark:bg-stone-600 rounded font-mono",
            Span(:class => "text-stone-500 dark:text-stone-400", () -> "sum_to_n($(n()))"),
            Span(:class => "text-cyan-500 font-bold text-lg", () -> string(sum_result()))
        ),
        # Formula note
        P(:class => "text-stone-400 dark:text-stone-500 text-xs mt-3 text-center",
            "Uses n*(n+1)/2 formula (same as loop result)"
        )
    )
end
