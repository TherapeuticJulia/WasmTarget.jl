# JuliaPlayground.jl - Rust Playground-style Julia REPL
#
# For now: Shows code examples with Run button (runtime loading)
# Future: When trimmed Julia runtime is ready, will compile arbitrary code
#
# Pattern: Only button on_click handlers (like Therapy.jl InteractiveCounter)

# Example code snippets
const EXAMPLES = [
    """function sum_to_n(n::Int32)::Int32
    result = Int32(0)
    i = Int32(1)
    while i <= n
        result = result + i
        i = i + Int32(1)
    end
    return result
end""",
    """function fibonacci(n::Int32)::Int32
    if n <= 1
        return n
    end
    return fibonacci(n - Int32(1)) + fibonacci(n - Int32(2))
end""",
    """function factorial(n::Int32)::Int32
    if n <= 1
        return Int32(1)
    end
    return n * factorial(n - Int32(1))
end"""
]

const EXAMPLE_NAMES = ["Sum to N", "Fibonacci", "Factorial"]

"""
Julia Playground - Code viewer with example selector.
Uses only button handlers (Therapy.jl pattern).
"""
JuliaPlayground = island(:JuliaPlayground) do
    # State: which example is selected (0, 1, 2)
    selected, set_selected = create_signal(0)
    output, set_output = create_signal("")

    Div(:class => "max-w-6xl mx-auto",
        # Example selector buttons
        Div(:class => "flex items-center justify-between mb-4 flex-wrap gap-2",
            Div(:class => "flex gap-2",
                Button(:class => "px-4 py-2 rounded-lg font-medium transition-colors bg-cyan-500 hover:bg-cyan-600 text-white",
                    :on_click => () -> set_selected(0),
                    "Sum to N"
                ),
                Button(:class => "px-4 py-2 rounded-lg font-medium transition-colors bg-stone-200 dark:bg-stone-700 hover:bg-stone-300 dark:hover:bg-stone-600 text-stone-700 dark:text-stone-200",
                    :on_click => () -> set_selected(1),
                    "Fibonacci"
                ),
                Button(:class => "px-4 py-2 rounded-lg font-medium transition-colors bg-stone-200 dark:bg-stone-700 hover:bg-stone-300 dark:hover:bg-stone-600 text-stone-700 dark:text-stone-200",
                    :on_click => () -> set_selected(2),
                    "Factorial"
                )
            ),
            # Run button
            Button(:class => "flex items-center gap-2 bg-cyan-500 hover:bg-cyan-600 text-white px-6 py-2 rounded-lg font-semibold transition-colors shadow-lg shadow-cyan-500/20",
                :on_click => () -> set_output("Compiling... (trimmed runtime loading)"),
                Svg(:class => "w-4 h-4", :fill => "currentColor", :viewBox => "0 0 24 24",
                    Path(:d => "M8 5v14l11-7z")
                ),
                "Run"
            )
        ),

        # Main content: Code + Output
        Div(:class => "grid lg:grid-cols-2 gap-4",
            # Code display (read-only)
            Div(:class => "flex flex-col",
                Div(:class => "flex items-center justify-between px-4 py-2 bg-stone-700 dark:bg-stone-800 rounded-t-xl",
                    Span(:class => "text-stone-300 text-sm font-medium", "Julia"),
                    Span(:class => "text-stone-500 text-xs", "Read-only")
                ),
                Pre(:class => "bg-stone-800 dark:bg-stone-900 p-4 rounded-b-xl min-h-[300px] overflow-auto",
                    Code(:class => "text-sm text-stone-100 font-mono whitespace-pre",
                        () -> EXAMPLES[selected() + 1]
                    )
                )
            ),

            # Output panel
            Div(:class => "flex flex-col",
                Div(:class => "flex items-center justify-between px-4 py-2 bg-stone-700 dark:bg-stone-800 rounded-t-xl",
                    Span(:class => "text-stone-300 text-sm font-medium", "Output"),
                    Span(:class => "text-amber-400 text-xs", "Runtime loading...")
                ),
                Div(:class => "bg-stone-900 dark:bg-black p-4 rounded-b-xl flex-1 min-h-[300px] font-mono",
                    Pre(:class => "text-sm text-stone-400 whitespace-pre-wrap", output)
                )
            )
        ),

        # Footer info
        Div(:class => "mt-6 p-4 bg-stone-100 dark:bg-stone-800 rounded-xl",
            P(:class => "text-stone-700 dark:text-stone-200 font-medium text-sm",
                "Coming Soon: Full REPL"
            ),
            P(:class => "text-stone-500 dark:text-stone-400 text-xs mt-1",
                "A trimmed Julia runtime will run in your browser, compiling arbitrary Julia code to WebAssembly client-side."
            )
        )
    )
end
