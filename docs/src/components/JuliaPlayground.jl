# JuliaPlayground.jl - True REPL experience
#
# NOT an island - uses client-side JS for textarea handling
# WASM compilation will be added when trimmed runtime is ready
#
# This is the correct architecture:
# - UI interactivity via vanilla JS (textarea, buttons)
# - WASM for the actual Julia compilation (when runtime ships)

const DEFAULT_CODE = """# Write any Julia code here!
function sum_to_n(n::Int32)::Int32
    result = Int32(0)
    i = Int32(1)
    while i <= n
        result = result + i
        i = i + Int32(1)
    end
    return result
end

# Try: sum_to_n(100)
"""

"""
Julia Playground - Full REPL experience with editable code.

This is a regular component (not an island) because:
1. Textarea input handling requires JS, not compiled WASM
2. The WASM part is the Julia compiler itself (coming with trimmed runtime)
3. UI interactivity and compiler are separate concerns
"""
function JuliaPlayground()
    Div(:class => "max-w-6xl mx-auto",
        # Top bar
        Div(:class => "flex items-center justify-between mb-4 flex-wrap gap-2",
            Span(:class => "text-stone-600 dark:text-stone-400 text-sm font-medium",
                "Julia Playground"
            ),
            # Run button
            Button(:id => "run-btn",
                   :class => "flex items-center gap-2 bg-cyan-500 hover:bg-cyan-600 text-white px-6 py-2 rounded-lg font-semibold transition-colors shadow-lg shadow-cyan-500/20",
                Svg(:class => "w-4 h-4", :fill => "currentColor", :viewBox => "0 0 24 24",
                    Path(:d => "M8 5v14l11-7z")
                ),
                "Run"
            )
        ),

        # Main content: Editor + Output
        Div(:class => "grid lg:grid-cols-2 gap-4",
            # Code editor
            Div(:class => "flex flex-col",
                Div(:class => "flex items-center justify-between px-4 py-2 bg-stone-700 dark:bg-stone-800 rounded-t-xl",
                    Span(:class => "text-stone-300 text-sm font-medium", "Julia"),
                    Span(:class => "text-stone-500 text-xs", "Edit your code")
                ),
                Textarea(:id => "code-editor",
                         :class => "bg-stone-800 dark:bg-stone-900 p-4 rounded-b-xl text-sm text-stone-100 font-mono min-h-[400px] w-full resize-y focus:outline-none focus:ring-2 focus:ring-cyan-500 border-0",
                         :spellcheck => "false",
                         :placeholder => "Write Julia code here...",
                         DEFAULT_CODE
                )
            ),

            # Output panel
            Div(:class => "flex flex-col",
                Div(:class => "flex items-center justify-between px-4 py-2 bg-stone-700 dark:bg-stone-800 rounded-t-xl",
                    Span(:class => "text-stone-300 text-sm font-medium", "Output"),
                    Span(:id => "status-indicator", :class => "text-amber-400 text-xs",
                        "Ready"
                    )
                ),
                Div(:id => "output-panel",
                    :class => "bg-stone-900 dark:bg-black p-4 rounded-b-xl flex-1 min-h-[400px] font-mono overflow-auto",
                    Pre(:id => "output-content",
                        :class => "text-sm text-stone-400 whitespace-pre-wrap",
                        "Click 'Run' to compile and execute your Julia code.\n\nThe trimmed Julia compiler will run entirely in your browser."
                    )
                )
            )
        ),

        # Footer
        Div(:class => "mt-6 p-4 bg-stone-100 dark:bg-stone-800 rounded-xl",
            Div(:class => "flex items-start gap-3",
                Div(:class => "flex-shrink-0 w-8 h-8 bg-cyan-500 rounded-full flex items-center justify-center",
                    Span(:class => "text-white text-sm font-bold", "?")
                ),
                Div(
                    P(:class => "text-stone-700 dark:text-stone-200 font-medium text-sm",
                        "How it works"
                    ),
                    P(:class => "text-stone-500 dark:text-stone-400 text-xs mt-1",
                        "When the trimmed Julia runtime loads, your code is parsed by JuliaSyntax, type-inferred, and compiled to WebAssembly by WasmTarget.jl - all client-side. No server required."
                    )
                )
            )
        ),

        # Client-side JavaScript for interactivity
        Script("""
            (function() {
                const editor = document.getElementById('code-editor');
                const runBtn = document.getElementById('run-btn');
                const output = document.getElementById('output-content');
                const status = document.getElementById('status-indicator');

                // Handle Run button click
                runBtn.addEventListener('click', function() {
                    const code = editor.value;
                    status.textContent = 'Compiling...';
                    status.className = 'text-cyan-400 text-xs';

                    // Show the code being compiled
                    output.innerHTML = '<span class="text-cyan-400">Compiling...</span>\\n\\n';
                    output.innerHTML += '<span class="text-stone-500">// Your code:</span>\\n';
                    output.innerHTML += '<span class="text-stone-300">' + escapeHtml(code.substring(0, 500)) + '</span>\\n\\n';

                    // Simulate compilation (replace with actual WASM call when runtime ready)
                    setTimeout(function() {
                        output.innerHTML += '<span class="text-amber-400">Trimmed Julia runtime loading...</span>\\n';
                        output.innerHTML += '<span class="text-stone-500">The browser-based compiler requires Julia 1.12 trimming.</span>\\n\\n';
                        output.innerHTML += '<span class="text-stone-500">Status: In development</span>\\n';
                        output.innerHTML += '<span class="text-stone-500">See: github.com/GroupTherapyOrg/WasmTarget.jl</span>';
                        status.textContent = 'Runtime loading...';
                        status.className = 'text-amber-400 text-xs';
                    }, 500);
                });

                // Helper to escape HTML
                function escapeHtml(text) {
                    const div = document.createElement('div');
                    div.textContent = text;
                    return div.innerHTML;
                }

                // Tab key inserts spaces in editor
                editor.addEventListener('keydown', function(e) {
                    if (e.key === 'Tab') {
                        e.preventDefault();
                        const start = this.selectionStart;
                        const end = this.selectionEnd;
                        this.value = this.value.substring(0, start) + '    ' + this.value.substring(end);
                        this.selectionStart = this.selectionEnd = start + 4;
                    }
                });
            })();
        """)
    )
end
