# Guide page - Architecture and usage patterns for WasmTarget.jl
#
# Explains the dart2wasm-style architecture and two use cases

function Guide()
    Layout(
        # Header
        Div(:class => "py-12 text-center",
            H1(:class => "text-4xl font-bold text-stone-800 dark:text-stone-100 mb-4",
                "Getting Started"
            ),
            P(:class => "text-xl text-stone-500 dark:text-stone-400 max-w-2xl mx-auto",
                "Learn how WasmTarget.jl works and how to use it in your projects"
            )
        ),

        # Navigation
        Div(:class => "flex flex-wrap justify-center gap-3 mb-12",
            NavLink("#architecture", "Architecture"),
            NavLink("#quickstart", "Quick Start"),
            NavLink("#use-cases", "Use Cases"),
            NavLink("#roadmap", "Roadmap")
        ),

        # ========================================
        # Architecture Section
        # ========================================
        Div(:id => "architecture", :class => "py-8",
            SectionHeader("How It Works"),

            # dart2wasm comparison
            Div(:class => "bg-gradient-to-r from-cyan-50 to-cyan-100 dark:from-cyan-950/30 dark:to-cyan-900/30 rounded-2xl p-8 mb-8",
                H3(:class => "text-xl font-semibold text-stone-800 dark:text-stone-100 mb-4",
                    "dart2wasm-Style Architecture"
                ),
                P(:class => "text-stone-600 dark:text-stone-300 mb-6",
                    "WasmTarget.jl follows the same proven architecture as Dart's dart2wasm compiler. Instead of building a new compiler from scratch, we leverage Julia's existing compiler infrastructure."
                ),
                Div(:class => "grid md:grid-cols-2 gap-6",
                    # dart2wasm
                    Div(:class => "bg-white dark:bg-stone-800 rounded-xl p-6",
                        H4(:class => "font-semibold text-stone-800 dark:text-stone-100 mb-3", "dart2wasm"),
                        Pre(:class => "text-sm text-stone-600 dark:text-stone-300 font-mono",
                            "Dart source → CFE → Kernel IR → dart2wasm → WASM"
                        )
                    ),
                    # WasmTarget.jl
                    Div(:class => "bg-white dark:bg-stone-800 rounded-xl p-6",
                        H4(:class => "font-semibold text-cyan-600 dark:text-cyan-400 mb-3", "WasmTarget.jl"),
                        Pre(:class => "text-sm text-stone-600 dark:text-stone-300 font-mono",
                            "Julia source → Julia compiler → IR → WasmTarget.jl → WASM"
                        )
                    )
                ),
                P(:class => "text-stone-500 dark:text-stone-400 text-sm mt-4",
                    "Julia's compiler handles parsing, macro expansion, lowering, and type inference. We just do codegen."
                )
            ),

            # Key insight
            Div(:class => "bg-white dark:bg-stone-800 rounded-xl p-6 mb-8",
                H3(:class => "text-lg font-semibold text-stone-800 dark:text-stone-100 mb-4",
                    "Why This Matters"
                ),
                Ul(:class => "space-y-3",
                    Li(:class => "flex items-start gap-3",
                        Span(:class => "text-cyan-500 mt-1", "✓"),
                        Span(:class => "text-stone-600 dark:text-stone-300",
                            "Full Julia type inference - no explicit type annotations required"
                        )
                    ),
                    Li(:class => "flex items-start gap-3",
                        Span(:class => "text-cyan-500 mt-1", "✓"),
                        Span(:class => "text-stone-600 dark:text-stone-300",
                            "Every IR pattern we support expands what Julia code compiles"
                        )
                    ),
                    Li(:class => "flex items-start gap-3",
                        Span(:class => "text-cyan-500 mt-1", "✓"),
                        Span(:class => "text-stone-600 dark:text-stone-300",
                            "WasmGC handles memory management - no manual GC needed"
                        )
                    )
                )
            )
        ),

        # ========================================
        # Quick Start Section
        # ========================================
        Div(:id => "quickstart", :class => "py-8 border-t border-stone-200 dark:border-stone-700",
            SectionHeader("Quick Start"),

            # Installation
            Div(:class => "mb-8",
                H3(:class => "text-lg font-semibold text-stone-800 dark:text-stone-100 mb-4", "Installation"),
                Pre(:class => "bg-stone-900 dark:bg-stone-950 rounded-xl p-4 overflow-x-auto",
                    Code(:class => "text-sm text-stone-100", """# In Julia REPL
using Pkg
Pkg.add(url="https://github.com/GroupTherapyOrg/WasmTarget.jl")""")
                )
            ),

            # Basic usage
            Div(:class => "mb-8",
                H3(:class => "text-lg font-semibold text-stone-800 dark:text-stone-100 mb-4", "Basic Usage"),
                Pre(:class => "bg-stone-900 dark:bg-stone-950 rounded-xl p-4 overflow-x-auto",
                    Code(:class => "text-sm text-stone-100", """using WasmTarget

# Define your Julia function
function fibonacci(n::Int32)::Int32
    if n <= 1
        return n
    end
    return fibonacci(n - Int32(1)) + fibonacci(n - Int32(2))
end

# Compile to WebAssembly
wasm_bytes = compile(fibonacci, (Int32,))

# Save to file
write("fibonacci.wasm", wasm_bytes)""")
                )
            ),

            # Running the output
            Div(:class => "mb-8",
                H3(:class => "text-lg font-semibold text-stone-800 dark:text-stone-100 mb-4", "Running the Output"),

                # Node.js
                Div(:class => "mb-4",
                    H4(:class => "font-medium text-stone-700 dark:text-stone-200 mb-2", "Node.js"),
                    Pre(:class => "bg-stone-900 dark:bg-stone-950 rounded-xl p-4 overflow-x-auto",
                        Code(:class => "text-sm text-stone-100", """const fs = require('fs');
const bytes = fs.readFileSync('fibonacci.wasm');

WebAssembly.instantiate(bytes).then(({ instance }) => {
    console.log(instance.exports.fibonacci(10));  // 55
});""")
                    )
                ),

                # Browser
                Div(
                    H4(:class => "font-medium text-stone-700 dark:text-stone-200 mb-2", "Browser"),
                    Pre(:class => "bg-stone-900 dark:bg-stone-950 rounded-xl p-4 overflow-x-auto",
                        Code(:class => "text-sm text-stone-100", """<script>
fetch('fibonacci.wasm')
    .then(response => response.arrayBuffer())
    .then(bytes => WebAssembly.instantiate(bytes))
    .then(({ instance }) => {
        console.log(instance.exports.fibonacci(10));  // 55
    });
</script>""")
                    )
                )
            )
        ),

        # ========================================
        # Use Cases Section
        # ========================================
        Div(:id => "use-cases", :class => "py-8 border-t border-stone-200 dark:border-stone-700",
            SectionHeader("Two Use Cases"),

            Div(:class => "grid md:grid-cols-2 gap-8",
                # Build-time compilation
                Div(:class => "bg-white dark:bg-stone-800 rounded-xl p-6 shadow-sm",
                    Div(:class => "flex items-center gap-3 mb-4",
                        Div(:class => "w-10 h-10 bg-cyan-100 dark:bg-cyan-900/50 rounded-lg flex items-center justify-center",
                            Span(:class => "text-cyan-600 dark:text-cyan-400 font-bold", "1")
                        ),
                        H3(:class => "text-xl font-semibold text-stone-800 dark:text-stone-100",
                            "Build-Time Compilation"
                        )
                    ),
                    P(:class => "text-stone-600 dark:text-stone-300 mb-4",
                        "The main product. Compile Julia to WASM at build time, ship small binaries."
                    ),
                    Div(:class => "bg-stone-50 dark:bg-stone-900 rounded-lg p-4 font-mono text-sm text-stone-600 dark:text-stone-300",
                        Div("Dev machine: Julia → WasmTarget.jl → WASM"),
                        Div(:class => "text-cyan-600 dark:text-cyan-400 mt-2", "Deploy: Just the compiled WASM (KB-MB)"),
                        Div("Browser: Runs WASM - NO runtime shipped")
                    ),
                    P(:class => "text-stone-500 dark:text-stone-400 text-sm mt-4",
                        "Perfect for Therapy.jl apps, games, compute-heavy web apps."
                    )
                ),

                # Interactive REPL
                Div(:class => "bg-gradient-to-br from-cyan-50 to-cyan-100 dark:from-cyan-950/30 dark:to-cyan-900/30 rounded-xl p-6 shadow-sm border-2 border-cyan-200 dark:border-cyan-800",
                    Div(:class => "flex items-center gap-3 mb-4",
                        Div(:class => "w-10 h-10 bg-cyan-500 dark:bg-cyan-600 rounded-lg flex items-center justify-center",
                            Span(:class => "text-white font-bold", "2")
                        ),
                        H3(:class => "text-xl font-semibold text-stone-800 dark:text-stone-100",
                            "Interactive REPL"
                        ),
                        Span(:class => "text-xs bg-cyan-500 text-white px-2 py-1 rounded", "Coming Soon")
                    ),
                    P(:class => "text-stone-600 dark:text-stone-300 mb-4",
                        "Write arbitrary Julia in the browser. Compiler ships once, runs in WASM."
                    ),
                    Div(:class => "bg-white dark:bg-stone-800 rounded-lg p-4 font-mono text-sm text-stone-600 dark:text-stone-300",
                        Div("Ship ONCE: Trimmed Julia compiler (~2-5MB)"),
                        Div(:class => "text-cyan-600 dark:text-cyan-400 mt-2", "User types code → Compiler IN browser → Executes"),
                        Div("Like Rust Playground, but for Julia")
                    ),
                    P(:class => "text-stone-500 dark:text-stone-400 text-sm mt-4",
                        "Uses Julia 1.12 trimming. Not needed for normal apps."
                    )
                )
            )
        ),

        # ========================================
        # Roadmap Section
        # ========================================
        Div(:id => "roadmap", :class => "py-8 border-t border-stone-200 dark:border-stone-700",
            SectionHeader("Roadmap"),

            Div(:class => "space-y-6 max-w-3xl",
                RoadmapItem("Current", "Phase 1-3 Complete", [
                    "All basic types, structs, tuples, arrays",
                    "Control flow: if/else, loops, recursion",
                    "JS interop: externref, imports, exports",
                    "Closures, exceptions, union types",
                    "SimpleDict/StringDict hash tables",
                    "Julia 1.11 + 1.12 compatibility"
                ], true),

                RoadmapItem("Next", "Phase 4: Matrix Support", [
                    "Multi-dimensional arrays (Matrix{T})",
                    "Key differentiator - no other Julia-to-WASM compiler has this"
                ], false),

                RoadmapItem("Future", "Phase 5: Browser REPL", [
                    "Trimmed Julia runtime compiled to WASM",
                    "JuliaSyntax.jl + type inference in browser",
                    "Interactive Julia playground",
                    "This docs site will feature the REPL!"
                ], false)
            )
        ),

        # ========================================
        # Comparison Section
        # ========================================
        Div(:class => "py-8 border-t border-stone-200 dark:border-stone-700",
            SectionHeader("Comparison to Other Compilers"),

            Div(:class => "overflow-x-auto",
                Table(:class => "w-full",
                    Thead(:class => "bg-stone-100 dark:bg-stone-800",
                        Tr(
                            Th(:class => "py-3 px-4 text-left font-semibold text-stone-700 dark:text-stone-200", "Feature"),
                            Th(:class => "py-3 px-4 text-left font-semibold text-cyan-600 dark:text-cyan-400", "WasmTarget.jl"),
                            Th(:class => "py-3 px-4 text-left font-semibold text-stone-700 dark:text-stone-200", "WebAssemblyCompiler.jl"),
                            Th(:class => "py-3 px-4 text-left font-semibold text-stone-700 dark:text-stone-200", "StaticCompiler.jl")
                        )
                    ),
                    Tbody(:class => "divide-y divide-stone-200 dark:divide-stone-700 text-sm",
                        ComparisonRow("Memory model", "WasmGC", "WasmGC", "Linear"),
                        ComparisonRow("1D Arrays", "✓", "✓", "Limited"),
                        ComparisonRow("Multi-dim Arrays", "Coming", "✗", "✗"),
                        ComparisonRow("Exceptions", "✓", "✗", "✗"),
                        ComparisonRow("Closures", "✓", "✗", "✗"),
                        ComparisonRow("Union{Nothing,T}", "✓", "✗", "✗"),
                        ComparisonRow("Hash tables", "✓", "✓", "✗")
                    )
                )
            ),
            P(:class => "text-center text-stone-500 dark:text-stone-400 text-sm mt-4",
                "WasmTarget.jl advantages: Exceptions, Closures, Union types, Multi-dim arrays (coming)"
            )
        ),

        # Footer CTA
        Div(:class => "py-16 text-center",
            H2(:class => "text-2xl font-bold text-stone-800 dark:text-stone-100 mb-4",
                "Ready to try?"
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

# Helper Components

function NavLink(href, text)
    A(:href => href,
      :class => "px-4 py-2 bg-stone-100 dark:bg-stone-800 rounded-lg text-sm font-medium text-stone-700 dark:text-stone-200 hover:bg-cyan-100 dark:hover:bg-cyan-900/30 transition-colors",
      text)
end

function SectionHeader(title)
    H2(:class => "text-3xl font-bold text-stone-800 dark:text-stone-100 mb-6", title)
end

function RoadmapItem(status, title, items, current)
    Div(:class => "flex gap-4",
        # Status indicator
        Div(:class => "flex flex-col items-center",
            Div(:class => (current ? "bg-cyan-500" : "bg-stone-300 dark:bg-stone-600") * " w-4 h-4 rounded-full"),
            Div(:class => "w-0.5 h-full " * (current ? "bg-cyan-200 dark:bg-cyan-800" : "bg-stone-200 dark:bg-stone-700"))
        ),
        # Content
        Div(:class => "flex-1 pb-8",
            Div(:class => "flex items-center gap-2 mb-2",
                Span(:class => (current ? "text-cyan-600 dark:text-cyan-400" : "text-stone-500 dark:text-stone-400") * " text-sm font-medium", status),
                H3(:class => "font-semibold text-stone-800 dark:text-stone-100", title)
            ),
            Ul(:class => "space-y-1",
                [Li(:class => "text-stone-600 dark:text-stone-300 text-sm flex items-center gap-2",
                    Span(:class => current ? "text-green-500" : "text-stone-400", current ? "✓" : "○"),
                    item
                ) for item in items]...
            )
        )
    )
end

function ComparisonRow(feature, wasmtarget, wac, static)
    Tr(
        Td(:class => "py-2 px-4 text-stone-700 dark:text-stone-200", feature),
        Td(:class => "py-2 px-4 font-medium " * (wasmtarget == "✓" || wasmtarget == "Coming" ? "text-cyan-600 dark:text-cyan-400" : "text-stone-600 dark:text-stone-300"), wasmtarget),
        Td(:class => "py-2 px-4 " * (wac == "✓" ? "text-green-600 dark:text-green-400" : "text-stone-400"), wac),
        Td(:class => "py-2 px-4 " * (static == "✓" ? "text-green-600 dark:text-green-400" : "text-stone-400"), static)
    )
end

# Export the page component
Guide
