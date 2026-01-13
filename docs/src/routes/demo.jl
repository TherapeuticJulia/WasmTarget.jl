# Demo page - Interactive examples demonstrating WasmTarget.jl features
#
# Each example shows Julia code alongside an interactive Wasm-compiled demo

function Demo()
    Layout(
        # Header
        Div(:class => "py-12 text-center",
            H1(:class => "text-4xl font-bold text-stone-800 dark:text-stone-100 mb-4",
                "Interactive Demos"
            ),
            P(:class => "text-xl text-stone-500 dark:text-stone-400 max-w-2xl mx-auto",
                "Each demo below is running Julia code compiled to WebAssembly. Click the buttons to interact - all computation happens in Wasm!"
            )
        ),

        # Demo 1: Arithmetic
        DemoSection(
            "Integer Arithmetic",
            "Basic arithmetic operations compiled to WebAssembly. The +, -, *, / operations all translate directly to Wasm i64 instructions.",
            """# Integer arithmetic in WasmTarget.jl
function add(a::Int32, b::Int32)::Int32
    return a + b
end

function multiply(a::Int32, b::Int32)::Int32
    return a * b
end

# Compile to Wasm
wasm = compile_multi([
    (add, (Int32, Int32), "add"),
    (multiply, (Int32, Int32), "multiply")
])""",
            ArithmeticDemo(),
            "Try changing a and b, then click an operation!"
        ),

        # Demo 2: Control Flow
        DemoSection(
            "Control Flow",
            "If/else branches and comparisons work seamlessly. Ternary expressions compile to Wasm's select or if/else blocks.",
            """# Control flow in WasmTarget.jl
function sign(n::Int32)::Int32
    if n > 0
        return Int32(1)
    elseif n < 0
        return Int32(-1)
    else
        return Int32(0)
    end
end

function is_even(n::Int32)::Int32
    return n % 2 == 0 ? Int32(1) : Int32(0)
end""",
            ControlFlowDemo(),
            "Try positive, negative, and zero values!"
        ),

        # Demo 3: Recursion / Loops
        DemoSection(
            "Recursion & Loops",
            "WasmTarget.jl supports both recursive function calls and while/for loops. Recursion compiles to Wasm function calls.",
            """# Recursive factorial in WasmTarget.jl
function factorial(n::Int32)::Int32
    if n <= 1
        return Int32(1)
    end
    return n * factorial(n - Int32(1))
end

# Iterative version using while loop
function factorial_iter(n::Int32)::Int32
    result = Int32(1)
    while n > 1
        result = result * n
        n = n - Int32(1)
    end
    return result
end""",
            RecursionDemo(),
            "Click preset values to see factorial results!"
        ),

        # Features summary
        Div(:class => "py-16 bg-white dark:bg-stone-800 rounded-2xl shadow-sm mt-16",
            H2(:class => "text-2xl font-bold text-center text-stone-800 dark:text-stone-100 mb-8",
                "What's Supported"
            ),
            Div(:class => "grid md:grid-cols-2 lg:grid-cols-3 gap-6 px-8",
                FeatureItem("Numeric Types", "i32, i64, f32, f64", true),
                FeatureItem("Arithmetic", "+, -, *, /, %, ^", true),
                FeatureItem("Comparisons", "==, !=, <, >, <=, >=", true),
                FeatureItem("Bitwise Ops", "&, |, xor, <<, >>", true),
                FeatureItem("Control Flow", "if/else, while, for", true),
                FeatureItem("Recursion", "Direct & mutual", true),
                FeatureItem("Structs", "WasmGC structs", true),
                FeatureItem("Arrays", "WasmGC arrays", true),
                FeatureItem("Tuples", "Immutable structs", true),
                FeatureItem("Strings", "Concat, compare", true),
                FeatureItem("JS Interop", "externref, imports", true),
                FeatureItem("Multi-dispatch", "Type-based selection", true),
                FeatureItem("Closures", "Captured variables", true),
                FeatureItem("Exceptions", "try/catch/throw", true),
                FeatureItem("Async/Await", "Use callbacks", false)
            )
        ),

        # CTA
        Div(:class => "py-16 text-center",
            P(:class => "text-stone-500 dark:text-stone-400 mb-4",
                "Ready to compile your own Julia to Wasm?"
            ),
            A(:href => "https://github.com/GroupTherapyOrg/WasmTarget.jl",
              :class => "inline-block bg-cyan-500 hover:bg-cyan-600 text-white px-6 py-3 rounded-lg font-medium transition-colors",
              :target => "_blank",
              "Get Started on GitHub"
            )
        )
    )
end

"""
Demo section with code and interactive component.
"""
function DemoSection(title, description, code, demo_component, cta)
    Div(:class => "py-12 border-b border-stone-200 dark:border-stone-700",
        # Title and description
        Div(:class => "mb-8",
            H2(:class => "text-2xl font-bold text-stone-800 dark:text-stone-100 mb-2", title),
            P(:class => "text-stone-500 dark:text-stone-400", description)
        ),

        # Code and demo side by side
        Div(:class => "grid lg:grid-cols-2 gap-8",
            # Code block
            Div(:class => "bg-stone-800 dark:bg-stone-950 rounded-xl p-4 overflow-x-auto",
                Pre(:class => "text-sm text-stone-100",
                    Code(:class => "language-julia", code)
                )
            ),

            # Interactive demo
            Div(:class => "flex flex-col",
                demo_component,
                P(:class => "text-center text-cyan-500 dark:text-cyan-400 text-sm mt-4 font-medium",
                    cta
                )
            )
        )
    )
end

"""
Feature list item.
"""
function FeatureItem(name, detail, supported)
    Div(:class => "flex items-center gap-3 p-3 rounded-lg " * (supported ? "bg-stone-50 dark:bg-stone-700" : "bg-stone-100 dark:bg-stone-800 opacity-60"),
        # Checkmark or X
        Div(:class => supported ? "text-green-500" : "text-stone-400",
            Svg(:class => "w-5 h-5", :fill => "none", :viewBox => "0 0 24 24", :stroke => "currentColor", :stroke_width => "2",
                Path(:stroke_linecap => "round", :stroke_linejoin => "round",
                     :d => supported ? "M5 13l4 4L19 7" : "M6 18L18 6M6 6l12 12")
            )
        ),
        Div(
            P(:class => "font-medium text-stone-800 dark:text-stone-100 text-sm", name),
            P(:class => "text-stone-500 dark:text-stone-400 text-xs", detail)
        )
    )
end

# Export the page component
Demo
