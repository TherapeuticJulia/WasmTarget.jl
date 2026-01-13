# API Reference page - Complete documentation for WasmTarget.jl
#
# Organized into sections: High-Level API, Low-Level Builder, Types, Opcodes

function Api()
    Layout(
        # Header
        Div(:class => "py-12 text-center",
            H1(:class => "text-4xl font-bold text-stone-800 dark:text-stone-100 mb-4",
                "API Reference"
            ),
            P(:class => "text-xl text-stone-500 dark:text-stone-400 max-w-2xl mx-auto",
                "Complete API documentation for WasmTarget.jl"
            )
        ),

        # Navigation
        Div(:class => "flex flex-wrap justify-center gap-3 mb-12",
            NavLink("#compile", "compile()"),
            NavLink("#compile-multi", "compile_multi()"),
            NavLink("#builder", "Low-Level Builder"),
            NavLink("#types", "Type Mappings"),
            NavLink("#features", "Supported Features")
        ),

        # ========================================
        # High-Level API Section
        # ========================================
        Div(:class => "py-8",
            SectionHeader("High-Level API"),
            P(:class => "text-stone-600 dark:text-stone-300 mb-8 max-w-3xl",
                "The high-level API provides simple functions to compile Julia code directly to WebAssembly bytes. This is the recommended way to use WasmTarget.jl."
            ),

            # compile()
            ApiFunction(
                "compile",
                "compile(f, arg_types; export_name=nothing) -> Vector{UInt8}",
                "Compile a single Julia function to WebAssembly bytes.",
                [
                    ("f", "Function", "The Julia function to compile"),
                    ("arg_types", "Tuple", "Tuple of argument types, e.g., (Int32, Int32)"),
                    ("export_name", "String", "Optional custom export name (default: function name)")
                ],
                """using WasmTarget

# Simple function
add(a::Int32, b::Int32)::Int32 = a + b
wasm = compile(add, (Int32, Int32))

# With custom export name
wasm = compile(add, (Int32, Int32); export_name="add_numbers")

# Save and use
write("add.wasm", wasm)
# Run with: node -e 'WebAssembly.instantiate(fs.readFileSync("add.wasm")).then(m => console.log(m.instance.exports.add(3, 4)))'"""
            ),

            # compile_multi()
            ApiFunction(
                "compile_multi",
                "compile_multi(functions; imports=[], globals=[]) -> Vector{UInt8}",
                "Compile multiple Julia functions into a single Wasm module. Functions can call each other.",
                [
                    ("functions", "Vector", "Array of (function, arg_types) or (function, arg_types, name) tuples"),
                    ("imports", "Vector", "Optional JS imports"),
                    ("globals", "Vector", "Optional Wasm globals")
                ],
                """using WasmTarget

# Multiple functions that call each other
square(x::Int32)::Int32 = x * x
sum_of_squares(a::Int32, b::Int32)::Int32 = square(a) + square(b)

wasm = compile_multi([
    (square, (Int32,)),
    (sum_of_squares, (Int32, Int32))
])

# With JS imports
wasm = compile_multi(
    [(my_func, (Int32,))],
    imports = [("console", "log", [I32], [])]
)"""
            )
        ),

        # ========================================
        # Low-Level Builder Section
        # ========================================
        Div(:id => "builder", :class => "py-8 border-t border-stone-200 dark:border-stone-700",
            SectionHeader("Low-Level Builder API"),
            P(:class => "text-stone-600 dark:text-stone-300 mb-8 max-w-3xl",
                "For advanced use cases, you can build Wasm modules manually using the builder API. This gives you full control over imports, exports, globals, tables, and memory."
            ),

            # WasmModule
            ApiFunction(
                "WasmModule",
                "WasmModule() -> WasmModule",
                "Create an empty Wasm module for manual construction.",
                [],
                """using WasmTarget

mod = WasmModule()
# Add types, imports, functions, etc.
bytes = to_bytes(mod)"""
            ),

            # add_function!
            ApiFunction(
                "add_function!",
                "add_function!(mod, param_types, result_types, locals, body) -> UInt32",
                "Add a function to the module. Returns the function index.",
                [
                    ("mod", "WasmModule", "The module to add to"),
                    ("param_types", "Vector{WasmValType}", "Parameter types"),
                    ("result_types", "Vector{WasmValType}", "Return types"),
                    ("locals", "Vector{WasmValType}", "Local variable types"),
                    ("body", "Vector{UInt8}", "Wasm bytecode for function body")
                ],
                """using WasmTarget

mod = WasmModule()

# Build function body manually
body = UInt8[
    Opcode.LOCAL_GET, 0x00,  # get first param
    Opcode.LOCAL_GET, 0x01,  # get second param
    Opcode.I32_ADD,          # add them
    Opcode.END               # end function
]

func_idx = add_function!(mod, [I32, I32], [I32], [], body)
add_export!(mod, "add", 0x00, func_idx)"""
            ),

            # add_import!
            ApiFunction(
                "add_import!",
                "add_import!(mod, module_name, func_name, param_types, result_types) -> UInt32",
                "Import a function from JavaScript. Returns the import index.",
                [
                    ("mod", "WasmModule", "The module"),
                    ("module_name", "String", "JS module name (e.g., \"console\")"),
                    ("func_name", "String", "JS function name (e.g., \"log\")"),
                    ("param_types", "Vector", "Parameter types"),
                    ("result_types", "Vector", "Return types")
                ],
                """using WasmTarget

mod = WasmModule()
log_idx = add_import!(mod, "console", "log", [I32], [])

# Use in a function:
# Opcode.CALL, log_idx"""
            ),

            # add_export!
            ApiFunction(
                "add_export!",
                "add_export!(mod, name, kind, idx)",
                "Export a function, global, table, or memory.",
                [
                    ("mod", "WasmModule", "The module"),
                    ("name", "String", "Export name visible to JS"),
                    ("kind", "UInt8", "Export kind (0x00=func, 0x01=table, 0x02=memory, 0x03=global)"),
                    ("idx", "UInt32", "Index of item to export")
                ],
                """add_export!(mod, "myFunc", 0x00, func_idx)  # Export function
add_export!(mod, "counter", 0x03, global_idx)  # Export global"""
            ),

            # add_global!
            ApiFunction(
                "add_global!",
                "add_global!(mod, valtype, mutable, init_value) -> UInt32",
                "Add a global variable. Returns the global index.",
                [
                    ("mod", "WasmModule", "The module"),
                    ("valtype", "WasmValType", "Type (I32, I64, F32, F64)"),
                    ("mutable", "Bool", "Whether the global can be modified"),
                    ("init_value", "Number", "Initial value")
                ],
                """counter_idx = add_global!(mod, I32, true, 0)   # Mutable i32
constant_idx = add_global!(mod, F64, false, 3.14)  # Immutable f64"""
            ),

            # to_bytes
            ApiFunction(
                "to_bytes",
                "to_bytes(mod) -> Vector{UInt8}",
                "Serialize a WasmModule to binary .wasm format.",
                [("mod", "WasmModule", "The module to serialize")],
                """bytes = to_bytes(mod)
write("output.wasm", bytes)"""
            )
        ),

        # ========================================
        # Type Mappings Section
        # ========================================
        Div(:id => "types", :class => "py-8 border-t border-stone-200 dark:border-stone-700",
            SectionHeader("Type Mappings"),
            P(:class => "text-stone-600 dark:text-stone-300 mb-8 max-w-3xl",
                "WasmTarget.jl automatically maps Julia types to WebAssembly types. Here's the complete mapping:"
            ),

            Div(:class => "overflow-x-auto",
                Table(:class => "w-full max-w-4xl mx-auto",
                    Thead(:class => "bg-stone-100 dark:bg-stone-800",
                        Tr(
                            Th(:class => "py-3 px-4 text-left font-semibold text-stone-700 dark:text-stone-200", "Julia Type"),
                            Th(:class => "py-3 px-4 text-left font-semibold text-stone-700 dark:text-stone-200", "Wasm Type"),
                            Th(:class => "py-3 px-4 text-left font-semibold text-stone-700 dark:text-stone-200", "Notes")
                        )
                    ),
                    Tbody(:class => "divide-y divide-stone-200 dark:divide-stone-700",
                        TypeMappingRow("Int32, UInt32", "i32", "Native 32-bit integer"),
                        TypeMappingRow("Int64, UInt64, Int", "i64", "Native 64-bit integer"),
                        TypeMappingRow("Float32", "f32", "32-bit IEEE float"),
                        TypeMappingRow("Float64", "f64", "64-bit IEEE float"),
                        TypeMappingRow("Bool", "i32", "0 or 1"),
                        TypeMappingRow("Char", "i32", "Unicode codepoint"),
                        TypeMappingRow("String", "WasmGC array (i32)", "Immutable, supports ==, length, *"),
                        TypeMappingRow("Vector{T}", "WasmGC array", "Mutable, T must be concrete"),
                        TypeMappingRow("struct Foo ... end", "WasmGC struct", "User-defined structs"),
                        TypeMappingRow("Tuple{A,B,...}", "WasmGC struct", "Immutable"),
                        TypeMappingRow("Union{Nothing,T}", "Tagged union", "Supports isa operator"),
                        TypeMappingRow("JSValue", "externref", "JavaScript object reference"),
                        TypeMappingRow("WasmGlobal{T,IDX}", "global", "Compile-time global access")
                    )
                )
            ),

            # Type constants
            Div(:class => "mt-12",
                H3(:class => "text-xl font-semibold text-stone-800 dark:text-stone-100 mb-4", "Type Constants"),
                P(:class => "text-stone-600 dark:text-stone-300 mb-4",
                    "Use these constants when building modules manually:"
                ),
                Div(:class => "flex flex-wrap gap-3",
                    TypeBadge("I32"),
                    TypeBadge("I64"),
                    TypeBadge("F32"),
                    TypeBadge("F64"),
                    TypeBadge("ExternRef"),
                    TypeBadge("FuncRef"),
                    TypeBadge("AnyRef")
                )
            )
        ),

        # ========================================
        # Supported Features Section
        # ========================================
        Div(:id => "features", :class => "py-8 border-t border-stone-200 dark:border-stone-700",
            SectionHeader("Supported Features"),
            P(:class => "text-stone-600 dark:text-stone-300 mb-8 max-w-3xl",
                "WasmTarget.jl supports a significant subset of Julia. Here's what works today:"
            ),

            Div(:class => "grid md:grid-cols-2 lg:grid-cols-3 gap-6",
                FeatureCategory("Control Flow", [
                    ("if/elseif/else", true),
                    ("while loops", true),
                    ("for loops (ranges)", true),
                    ("&& and || (short-circuit)", true),
                    ("try/catch/throw", true),
                    ("Recursion", true),
                    ("@goto/@label", false)
                ]),
                FeatureCategory("Functions", [
                    ("Regular functions", true),
                    ("Multiple functions", true),
                    ("Closures", true),
                    ("Multiple dispatch", true),
                    ("Varargs", false),
                    ("Keyword args", false)
                ]),
                FeatureCategory("Operators", [
                    ("Arithmetic (+, -, *, /, %)", true),
                    ("Comparison (==, <, >, etc)", true),
                    ("Logical (&&, ||, !)", true),
                    ("Bitwise (&, |, xor, <<, >>)", true),
                    ("Power (^)", true)
                ]),
                FeatureCategory("Data Structures", [
                    ("Structs", true),
                    ("Tuples", true),
                    ("Vector{T}", true),
                    ("Matrix{T}", false),
                    ("String ops", true),
                    ("SimpleDict/StringDict", true),
                    ("Full Dict", false)
                ]),
                FeatureCategory("JS Interop", [
                    ("externref (JSValue)", true),
                    ("Import JS functions", true),
                    ("Export Wasm functions", true),
                    ("Wasm globals", true),
                    ("Tables (funcref)", true),
                    ("Linear memory", true)
                ]),
                FeatureCategory("Advanced", [
                    ("Union{Nothing,T}", true),
                    ("Type inference", true),
                    ("Exception handling", true),
                    ("Data segments", true),
                    ("Generated functions", false),
                    ("FFI/ccall", false)
                ])
            )
        ),

        # Footer CTA
        Div(:class => "py-16 text-center border-t border-stone-200 dark:border-stone-700 mt-12",
            H2(:class => "text-2xl font-bold text-stone-800 dark:text-stone-100 mb-4",
                "Ready to build?"
            ),
            P(:class => "text-stone-500 dark:text-stone-400 mb-8",
                "Check out the interactive demos or explore the source code on GitHub."
            ),
            Div(:class => "flex justify-center gap-4",
                A(:href => "demo/",
                  :class => "bg-cyan-500 hover:bg-cyan-600 text-white px-6 py-3 rounded-lg font-medium transition-colors",
                  "Interactive Demos"
                ),
                A(:href => "https://github.com/GroupTherapyOrg/WasmTarget.jl",
                  :class => "bg-stone-200 dark:bg-stone-700 text-stone-700 dark:text-stone-200 px-6 py-3 rounded-lg font-medium hover:bg-stone-300 dark:hover:bg-stone-600 transition-colors",
                  :target => "_blank",
                  "View Source"
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
    H2(:class => "text-3xl font-bold text-stone-800 dark:text-stone-100 mb-4", title)
end

function ApiFunction(name, signature, description, params, example)
    Div(:id => lowercase(replace(name, "()" => "")), :class => "mb-12 bg-white dark:bg-stone-800 rounded-xl p-6 shadow-sm",
        # Function name and signature
        H3(:class => "text-xl font-bold text-cyan-600 dark:text-cyan-400 font-mono mb-2", name * "()"),
        Code(:class => "block bg-stone-100 dark:bg-stone-900 px-4 py-2 rounded-lg text-sm font-mono text-stone-700 dark:text-stone-300 mb-4", signature),
        P(:class => "text-stone-600 dark:text-stone-300 mb-4", description),

        # Parameters
        if !isempty(params)
            Div(:class => "mb-4",
                H4(:class => "font-semibold text-stone-800 dark:text-stone-100 mb-2", "Parameters"),
                Ul(:class => "space-y-2",
                    [Li(:class => "text-sm",
                        Span(:class => "font-mono text-cyan-600 dark:text-cyan-400", p[1]),
                        Span(:class => "text-stone-400 mx-2", ":"),
                        Span(:class => "text-stone-500 dark:text-stone-400 italic", p[2]),
                        Span(:class => "text-stone-400 mx-2", "-"),
                        Span(:class => "text-stone-600 dark:text-stone-300", p[3])
                    ) for p in params]...
                )
            )
        end,

        # Example
        Div(
            H4(:class => "font-semibold text-stone-800 dark:text-stone-100 mb-2", "Example"),
            Pre(:class => "bg-stone-900 dark:bg-stone-950 rounded-lg p-4 overflow-x-auto",
                Code(:class => "text-sm text-stone-100", example)
            )
        )
    )
end

function TypeMappingRow(julia, wasm, notes)
    Tr(
        Td(:class => "py-3 px-4 font-mono text-sm text-cyan-600 dark:text-cyan-400", julia),
        Td(:class => "py-3 px-4 font-mono text-sm text-stone-600 dark:text-stone-300", wasm),
        Td(:class => "py-3 px-4 text-sm text-stone-500 dark:text-stone-400", notes)
    )
end

function TypeBadge(name)
    Span(:class => "px-4 py-2 bg-stone-100 dark:bg-stone-700 rounded-lg font-mono text-sm text-stone-700 dark:text-stone-200", name)
end

function FeatureCategory(title, features)
    Div(:class => "bg-white dark:bg-stone-800 rounded-xl p-6 shadow-sm",
        H3(:class => "font-semibold text-stone-800 dark:text-stone-100 mb-4", title),
        Ul(:class => "space-y-2",
            [Li(:class => "flex items-center gap-2 text-sm",
                Span(:class => f[2] ? "text-green-500" : "text-stone-400",
                    f[2] ? "✓" : "○"
                ),
                Span(:class => f[2] ? "text-stone-700 dark:text-stone-200" : "text-stone-400 dark:text-stone-500",
                    f[1]
                )
            ) for f in features]...
        )
    )
end

# Export the page component
Api
