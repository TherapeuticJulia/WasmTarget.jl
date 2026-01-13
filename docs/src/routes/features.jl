# Features page - Supported Julia features with live WASM demos
#
# Each feature section shows:
# - Julia code
# - Interactive WASM demo
# - What's happening under the hood

function Features()
    Layout(
        # Header
        Div(:class => "py-8 text-center",
            H1(:class => "text-4xl font-bold text-stone-800 dark:text-stone-100 mb-4",
                "Supported Features"
            ),
            P(:class => "text-xl text-stone-500 dark:text-stone-400 max-w-2xl mx-auto",
                "Every demo below runs Julia code compiled to WebAssembly"
            )
        ),

        # Feature demos
        Div(:class => "space-y-16",
            # Arithmetic
            FeatureSection(
                "Integer Arithmetic",
                "Native i32/i64 operations compile to direct WASM instructions",
                """add(a::Int32, b::Int32)::Int32 = a + b
multiply(a::Int32, b::Int32)::Int32 = a * b
divide(a::Int32, b::Int32)::Int32 = div(a, b)""",
                ArithmeticDemo()
            ),

            # Control Flow
            FeatureSection(
                "Control Flow",
                "if/elseif/else compiles to WASM if/else blocks",
                """function sign(n::Int32)::Int32
    if n > 0
        return Int32(1)
    elseif n < 0
        return Int32(-1)
    else
        return Int32(0)
    end
end""",
                ControlFlowDemo()
            ),

            # Recursion
            FeatureSection(
                "Recursion",
                "Self-recursive calls compile to WASM call instructions",
                """function factorial(n::Int32)::Int32
    if n <= 1
        return Int32(1)
    end
    return n * factorial(n - Int32(1))
end""",
                RecursionDemo()
            )
        ),

        # Full feature list
        Div(:class => "py-16 mt-8 bg-white dark:bg-stone-800 rounded-2xl",
            H2(:class => "text-2xl font-bold text-center text-stone-800 dark:text-stone-100 mb-8",
                "Complete Feature List"
            ),
            Div(:class => "grid md:grid-cols-2 lg:grid-cols-3 gap-4 px-8 max-w-5xl mx-auto",
                # Supported
                FeatureItem("Integers", "i32, i64, u32, u64", true),
                FeatureItem("Floats", "f32, f64", true),
                FeatureItem("Arithmetic", "+, -, *, /, %, ^", true),
                FeatureItem("Comparisons", "==, !=, <, >, <=, >=", true),
                FeatureItem("Bitwise", "&, |, xor, <<, >>", true),
                FeatureItem("Booleans", "&&, ||, !", true),
                FeatureItem("if/else", "Conditionals", true),
                FeatureItem("while", "Loops", true),
                FeatureItem("for", "Range loops", true),
                FeatureItem("Recursion", "Self & mutual", true),
                FeatureItem("Structs", "WasmGC structs", true),
                FeatureItem("Tuples", "Immutable", true),
                FeatureItem("Vector{T}", "1D arrays", true),
                FeatureItem("Strings", "Concat, compare", true),
                FeatureItem("Closures", "Captured vars", true),
                FeatureItem("Exceptions", "try/catch/throw", true),
                FeatureItem("Union{Nothing,T}", "Optional types", true),
                FeatureItem("JS Interop", "externref, imports", true),

                # Coming soon
                FeatureItem("Matrix{T}", "Multi-dim arrays", false),
                FeatureItem("Full Dict", "Hash tables", false),
                FeatureItem("Varargs", "Variable args", false)
            )
        ),

        # CTA
        Div(:class => "py-12 text-center",
            A(:href => "./",
              :class => "bg-cyan-500 hover:bg-cyan-600 text-white px-8 py-3 rounded-lg font-semibold transition-colors",
              "Try the Playground"
            )
        )
    )
end

function FeatureSection(title, subtitle, code, demo)
    Div(:class => "max-w-5xl mx-auto",
        # Header
        H2(:class => "text-2xl font-bold text-stone-800 dark:text-stone-100 mb-2", title),
        P(:class => "text-stone-500 dark:text-stone-400 mb-6", subtitle),

        # Content: code + demo
        Div(:class => "grid lg:grid-cols-2 gap-6",
            # Code
            Pre(:class => "bg-stone-800 dark:bg-stone-900 rounded-xl p-4 overflow-x-auto",
                Code(:class => "text-sm text-stone-100 font-mono", code)
            ),

            # Demo
            Div(:class => "flex items-center justify-center",
                demo
            )
        )
    )
end

function FeatureItem(name, detail, supported)
    Div(:class => "flex items-center gap-3 p-3 rounded-lg " *
                  (supported ? "bg-stone-50 dark:bg-stone-700" : "bg-stone-100 dark:bg-stone-800 opacity-50"),
        Span(:class => supported ? "text-green-500 text-lg" : "text-stone-400 text-lg",
            supported ? "✓" : "○"
        ),
        Div(
            P(:class => "font-medium text-stone-800 dark:text-stone-100 text-sm", name),
            P(:class => "text-stone-500 dark:text-stone-400 text-xs", detail)
        )
    )
end

# Export
Features
