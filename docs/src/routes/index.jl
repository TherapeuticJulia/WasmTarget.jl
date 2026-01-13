# Home page - WasmTarget.jl Documentation
#
# Features an interactive Wasm playground demonstrating Julia compiled to WebAssembly

function Index()
    Layout(
        # Hero Section
        Div(:class => "py-16 sm:py-24",
            Div(:class => "text-center",
                H1(:class => "text-4xl sm:text-6xl font-bold text-stone-800 dark:text-stone-100 tracking-tight",
                    "Compile Julia to",
                    Br(),
                    Span(:class => "text-cyan-500 dark:text-cyan-400", "WebAssembly")
                ),
                P(:class => "mt-6 text-xl text-stone-500 dark:text-stone-400 max-w-2xl mx-auto",
                    "WasmTarget.jl compiles Julia functions directly to WebAssembly binaries. Run Julia in the browser, Node.js, or any Wasm runtime with near-native performance."
                ),
                Div(:class => "mt-10 flex justify-center gap-4",
                    A(:href => "demo/",
                      :class => "bg-cyan-500 hover:bg-cyan-600 dark:bg-cyan-600 dark:hover:bg-cyan-500 text-white px-6 py-3 rounded-lg font-medium transition-colors shadow-lg shadow-cyan-500/20",
                      "View Examples"
                    ),
                    A(:href => "https://github.com/GroupTherapyOrg/WasmTarget.jl",
                      :class => "bg-stone-100 dark:bg-stone-800 text-stone-700 dark:text-stone-200 px-6 py-3 rounded-lg font-medium hover:bg-stone-200 dark:hover:bg-stone-700 transition-colors",
                      :target => "_blank",
                      "View on GitHub"
                    )
                )
            )
        ),

        # Interactive Playground Section
        Div(:class => "py-16 bg-gradient-to-r from-cyan-100 to-cyan-200 dark:from-cyan-950/30 dark:to-cyan-900/30 rounded-2xl shadow-xl mb-16",
            Div(:class => "text-center px-8 mb-8",
                H2(:class => "text-3xl font-bold mb-4 text-stone-800 dark:text-stone-100",
                    "Try It Live"
                ),
                P(:class => "text-stone-600 dark:text-stone-300 max-w-xl mx-auto",
                    "This playground is running Julia code compiled to WebAssembly. Every button click executes Wasm - no JavaScript, just Julia!"
                )
            ),
            WasmPlayground()
        ),

        # Feature Grid
        Div(:class => "py-16 bg-white dark:bg-stone-800 rounded-2xl shadow-sm transition-colors duration-200",
            H2(:class => "text-3xl font-bold text-center text-stone-800 dark:text-stone-100 mb-12",
                "Why WasmTarget.jl?"
            ),
            Div(:class => "grid md:grid-cols-3 gap-8 px-8",
                FeatureCard(
                    "Direct Compilation",
                    "No intermediate languages. Julia IR compiles directly to WebAssembly bytecode via Base.code_typed().",
                    "M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4"
                ),
                FeatureCard(
                    "WasmGC Support",
                    "Full support for WebAssembly GC proposal. Structs, arrays, and tuples work seamlessly.",
                    "M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"
                ),
                FeatureCard(
                    "JS Interop",
                    "Import JavaScript functions and export Wasm functions. Use externref for seamless JS object handling.",
                    "M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"
                )
            )
        ),

        # Code Example
        Div(:class => "py-16",
            H2(:class => "text-3xl font-bold text-center text-stone-800 dark:text-stone-100 mb-8",
                "Simple API"
            ),
            Div(:class => "bg-stone-800 dark:bg-stone-950 rounded-xl p-6 max-w-3xl mx-auto overflow-x-auto shadow-xl",
                Pre(:class => "text-sm text-stone-100",
                    Code(:class => "language-julia", """using WasmTarget

# Define a Julia function
function fibonacci(n::Int32)::Int32
    if n <= 1
        return n
    end
    return fibonacci(n - Int32(1)) + fibonacci(n - Int32(2))
end

# Compile to WebAssembly
wasm_bytes = compile(fibonacci, (Int32,))

# Save to file
write("fibonacci.wasm", wasm_bytes)

# Run in Node.js or browser!
# The function is exported and ready to use""")
                )
            ),
            P(:class => "text-center text-stone-500 dark:text-stone-400 mt-4 text-sm",
                "Supports recursion, loops, structs, arrays, strings, and more!"
            )
        ),

        # Supported Types Table
        Div(:class => "py-16 bg-white dark:bg-stone-800 rounded-2xl shadow-sm px-8",
            H2(:class => "text-3xl font-bold text-center text-stone-800 dark:text-stone-100 mb-8",
                "Supported Types"
            ),
            Div(:class => "max-w-2xl mx-auto overflow-x-auto",
                Table(:class => "w-full text-left",
                    Thead(:class => "border-b border-stone-200 dark:border-stone-700",
                        Tr(
                            Th(:class => "py-3 px-4 font-semibold text-stone-700 dark:text-stone-200", "Julia Type"),
                            Th(:class => "py-3 px-4 font-semibold text-stone-700 dark:text-stone-200", "Wasm Type"),
                            Th(:class => "py-3 px-4 font-semibold text-stone-700 dark:text-stone-200", "Notes")
                        )
                    ),
                    Tbody(:class => "text-stone-600 dark:text-stone-300",
                        TypeRow("Int32, UInt32", "i32", "Native"),
                        TypeRow("Int64, UInt64, Int", "i64", "Native"),
                        TypeRow("Float32", "f32", "Native"),
                        TypeRow("Float64", "f64", "Native"),
                        TypeRow("Bool", "i32", "0 or 1"),
                        TypeRow("String", "GC array", "i32 per char"),
                        TypeRow("struct", "GC struct", "Field-mapped"),
                        TypeRow("Tuple{...}", "GC struct", "Immutable"),
                        TypeRow("Vector{T}", "GC array", "Fixed-size")
                    )
                )
            )
        ),

        # CTA
        Div(:class => "py-16 text-center",
            H2(:class => "text-2xl font-bold text-stone-800 dark:text-stone-100 mb-4",
                "Ready to try?"
            ),
            P(:class => "text-stone-500 dark:text-stone-400 mb-8",
                "Check out the interactive demos or dive into the API reference."
            ),
            Div(:class => "flex justify-center gap-4",
                A(:href => "demo/",
                  :class => "bg-cyan-500 hover:bg-cyan-600 text-white px-6 py-3 rounded-lg font-medium transition-colors",
                  "Interactive Demos"
                ),
                A(:href => "api/",
                  :class => "bg-stone-200 dark:bg-stone-700 text-stone-700 dark:text-stone-200 px-6 py-3 rounded-lg font-medium hover:bg-stone-300 dark:hover:bg-stone-600 transition-colors",
                  "API Reference"
                )
            )
        )
    )
end

function FeatureCard(title, description, icon_path)
    Div(:class => "text-center p-6",
        Div(:class => "w-12 h-12 bg-cyan-100 dark:bg-cyan-950/30 rounded-lg flex items-center justify-center mx-auto mb-4",
            Svg(:class => "w-6 h-6 text-cyan-500 dark:text-cyan-400", :fill => "none", :viewBox => "0 0 24 24", :stroke => "currentColor", :stroke_width => "2",
                Path(:stroke_linecap => "round", :stroke_linejoin => "round", :d => icon_path)
            )
        ),
        H3(:class => "text-lg font-semibold text-stone-800 dark:text-stone-100 mb-2", title),
        P(:class => "text-stone-500 dark:text-stone-400", description)
    )
end

function TypeRow(julia_type, wasm_type, notes)
    Tr(:class => "border-b border-stone-100 dark:border-stone-700",
        Td(:class => "py-3 px-4 font-mono text-sm text-cyan-600 dark:text-cyan-400", julia_type),
        Td(:class => "py-3 px-4 font-mono text-sm", wasm_type),
        Td(:class => "py-3 px-4 text-sm", notes)
    )
end

# Export the page component
Index
