# Home page - Julia REPL Playground
#
# Rust Playground-style interface: code editor + run + output
# Uses pre-compiled WASM examples for now; full client-side compiler coming

function Index()
    Layout(
        Div(:class => "py-8",
            # Header
            Div(:class => "text-center mb-8",
                H1(:class => "text-4xl font-bold text-stone-800 dark:text-stone-100",
                    "Julia → WebAssembly"
                ),
                P(:class => "text-stone-500 dark:text-stone-400 mt-2",
                    "Write Julia. Compile to WASM. Run in the browser."
                )
            ),

            # Main Playground
            JuliaPlayground()
        )
    )
end

# Export the page component
Index
