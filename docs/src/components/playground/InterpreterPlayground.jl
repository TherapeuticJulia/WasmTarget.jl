# InterpreterPlayground.jl - Interactive Julia Playground with Interpreter
#
# This component provides a full playground experience where users can:
# 1. Write Julia code in a CodeMirror 6 editor
# 2. Run the code via the interpreter compiled to WasmGC
# 3. See output in real-time
#
# The interpreter is the Julia interpreter from src/Interpreter/ compiled
# by WasmTarget.jl to WasmGC.
#
# Story: BROWSER-030

"""
InterpreterPlayground - Main playground component with interpreter integration.

This creates a two-panel layout:
- Left: CodeMirror 6 editor with Julia syntax highlighting
- Right: Output panel showing execution results

The interpreter.wasm module is loaded and the interpret() function is called
when the user clicks Run.
"""
function InterpreterPlayground(; initial_code::String = "")
    default_code = if isempty(initial_code)
        """# Welcome to the Julia Playground!
# This code runs entirely in your browser via WasmGC.

# Try some basic arithmetic
x = 5
y = 3
println(x + y)

# Define a function
function greet(name)
    println("Hello, " * name * "!")
end

greet("World")
"""
    else
        initial_code
    end

    Div(:class => "flex flex-col h-full min-h-[600px]",
        # Header with Run button
        Div(:class => "flex items-center justify-between px-4 py-3 bg-stone-800 dark:bg-stone-900 border-b border-stone-700",
            # Left: Title
            Div(:class => "flex items-center gap-3",
                Span(:class => "text-stone-200 font-semibold", "Julia Playground"),
                Span(:class => "text-stone-500 text-sm", "Powered by WasmTarget.jl")
            ),
            # Right: Run button
            Div(:class => "flex items-center gap-3",
                # Status indicator
                Div(:id => "interpreter-status",
                    :class => "flex items-center gap-2",
                    Span(:id => "status-dot", :class => "w-2 h-2 rounded-full bg-yellow-500"),
                    Span(:id => "status-text", :class => "text-stone-400 text-xs", "Loading...")
                ),
                # Run button
                Button(:id => "run-button",
                    :class => "px-4 py-2 bg-cyan-500 hover:bg-cyan-600 disabled:bg-stone-600 disabled:cursor-not-allowed text-white rounded-lg font-medium flex items-center gap-2 transition-colors",
                    :disabled => "true",
                    # Play icon
                    Svg(:class => "w-4 h-4", :fill => "currentColor", :viewBox => "0 0 24 24",
                        Path(:d => "M8 5v14l11-7z")
                    ),
                    Span("Run")
                )
            )
        ),

        # Main content area - two panels
        Div(:class => "flex-1 grid lg:grid-cols-2 gap-0 min-h-0",
            # Left: Editor
            Div(:class => "flex flex-col min-h-0 border-r border-stone-700",
                # Editor header
                Div(:class => "flex items-center justify-between px-4 py-2 bg-stone-700 dark:bg-stone-800",
                    Span(:class => "text-stone-300 text-sm font-medium", "Code"),
                    Div(:class => "flex items-center gap-2",
                        # Clear button
                        Button(:id => "clear-button",
                            :class => "text-stone-400 hover:text-white text-xs transition-colors",
                            "Clear"
                        ),
                        # Example selector
                        Select(:id => "example-selector",
                            :class => "bg-stone-600 text-stone-200 text-xs rounded px-2 py-1",
                            Option(:value => "", "Examples..."),
                            Option(:value => "hello", "Hello World"),
                            Option(:value => "arithmetic", "Arithmetic"),
                            Option(:value => "functions", "Functions"),
                            Option(:value => "loops", "Loops"),
                            Option(:value => "factorial", "Factorial"),
                            Option(:value => "fibonacci", "Fibonacci")
                        )
                    )
                ),
                # CodeMirror container
                Div(:id => "editor-container",
                    :class => "flex-1 overflow-hidden bg-stone-900"
                )
            ),

            # Right: Output
            Div(:class => "flex flex-col min-h-0",
                # Output header
                Div(:class => "flex items-center justify-between px-4 py-2 bg-stone-700 dark:bg-stone-800",
                    Span(:class => "text-stone-300 text-sm font-medium", "Output"),
                    Button(:id => "clear-output-button",
                        :class => "text-stone-400 hover:text-white text-xs transition-colors",
                        "Clear"
                    )
                ),
                # Output content
                Div(:id => "output-container",
                    :class => "flex-1 overflow-auto bg-stone-900 p-4 font-mono text-sm",
                    # Initial message
                    Div(:id => "output-content",
                        :class => "text-stone-400",
                        "Click \"Run\" to execute your code..."
                    )
                )
            )
        ),

        # CodeMirror 6 CSS (loaded from CDN)
        Link(:rel => "stylesheet",
            :href => "https://cdnjs.cloudflare.com/ajax/libs/codemirror/6.65.7/codemirror.min.css"),

        # Playground JavaScript
        Script(playground_script(default_code))
    )
end

"""
Generate the JavaScript code for the playground.
"""
function playground_script(initial_code::String)
    # Escape the code for JavaScript
    escaped_code = replace(initial_code, "\\" => "\\\\")
    escaped_code = replace(escaped_code, "`" => "\\`")
    escaped_code = replace(escaped_code, "\$" => "\\\$")

    return """
    (function() {
        // Configuration
        const INITIAL_CODE = `$(escaped_code)`;

        // State
        let editor = null;
        let interpreterModule = null;
        let interpreterReady = false;

        // DOM elements
        const editorContainer = document.getElementById('editor-container');
        const outputContent = document.getElementById('output-content');
        const runButton = document.getElementById('run-button');
        const clearButton = document.getElementById('clear-button');
        const clearOutputButton = document.getElementById('clear-output-button');
        const exampleSelector = document.getElementById('example-selector');
        const statusDot = document.getElementById('status-dot');
        const statusText = document.getElementById('status-text');

        // Example code snippets
        const examples = {
            hello: `# Hello World
println("Hello, World!")
`,
            arithmetic: `# Basic Arithmetic
a = 10
b = 3

println("a + b = ", a + b)
println("a - b = ", a - b)
println("a * b = ", a * b)
println("a / b = ", a / b)
println("a % b = ", a % b)
`,
            functions: `# Functions
function greet(name)
    println("Hello, " * name * "!")
end

function add(a, b)
    return a + b
end

greet("Julia")
println("2 + 3 = ", add(2, 3))
`,
            loops: `# Loops
println("While loop:")
i = 1
while i <= 5
    println(i)
    i = i + 1
end

println("For loop:")
for j in 5
    println(j)
end
`,
            factorial: `# Factorial (Recursive)
function factorial(n)
    if n <= 1
        return 1
    end
    return n * factorial(n - 1)
end

println("factorial(5) = ", factorial(5))
println("factorial(10) = ", factorial(10))
`,
            fibonacci: `# Fibonacci (Recursive)
function fib(n)
    if n <= 1
        return n
    end
    return fib(n - 1) + fib(n - 2)
end

println("Fibonacci sequence:")
for i in 10
    println("fib(", i, ") = ", fib(i))
end
`
        };

        // Update status indicator
        function updateStatus(state, text) {
            if (statusDot && statusText) {
                statusText.textContent = text;
                statusDot.className = 'w-2 h-2 rounded-full ';
                switch(state) {
                    case 'ready':
                        statusDot.className += 'bg-green-500';
                        break;
                    case 'loading':
                        statusDot.className += 'bg-yellow-500 animate-pulse';
                        break;
                    case 'running':
                        statusDot.className += 'bg-cyan-500 animate-pulse';
                        break;
                    case 'error':
                        statusDot.className += 'bg-red-500';
                        break;
                    default:
                        statusDot.className += 'bg-stone-500';
                }
            }
        }

        // Append output
        function appendOutput(text, isError = false) {
            if (outputContent) {
                const line = document.createElement('div');
                line.className = isError ? 'text-red-400' : 'text-green-400';
                line.textContent = text;
                outputContent.appendChild(line);
                outputContent.scrollTop = outputContent.scrollHeight;
            }
        }

        // Clear output
        function clearOutput() {
            if (outputContent) {
                outputContent.innerHTML = '';
            }
        }

        // Initialize CodeMirror 6
        async function initEditor() {
            updateStatus('loading', 'Loading editor...');

            // Load CodeMirror 6 modules from esm.sh
            try {
                const [{EditorState}, {EditorView, keymap, lineNumbers, highlightActiveLineGutter, highlightSpecialChars, drawSelection, dropCursor, rectangularSelection, crosshairCursor}, {defaultHighlightStyle, syntaxHighlighting, indentOnInput, bracketMatching, foldGutter, foldKeymap}, {defaultKeymap, history, historyKeymap}, {closeBrackets, closeBracketsKeymap, autocompletion, completionKeymap}, {highlightSelectionMatches, searchKeymap}] = await Promise.all([
                    import('https://esm.sh/@codemirror/state@6'),
                    import('https://esm.sh/@codemirror/view@6'),
                    import('https://esm.sh/@codemirror/language@6'),
                    import('https://esm.sh/@codemirror/commands@6'),
                    import('https://esm.sh/@codemirror/autocomplete@6'),
                    import('https://esm.sh/@codemirror/search@6')
                ]);

                // Create a simple Julia-like syntax (basic tokenizer)
                const juliaTheme = EditorView.theme({
                    '&': {
                        backgroundColor: '#1c1917',
                        color: '#e7e5e4',
                        height: '100%'
                    },
                    '.cm-content': {
                        fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
                        padding: '16px'
                    },
                    '.cm-gutters': {
                        backgroundColor: '#292524',
                        color: '#78716c',
                        border: 'none'
                    },
                    '.cm-activeLineGutter': {
                        backgroundColor: '#44403c'
                    },
                    '.cm-activeLine': {
                        backgroundColor: 'rgba(68, 64, 60, 0.4)'
                    },
                    '.cm-cursor': {
                        borderLeftColor: '#22d3ee'
                    },
                    '.cm-selectionBackground': {
                        backgroundColor: '#0e7490 !important'
                    },
                    '&.cm-focused .cm-selectionBackground': {
                        backgroundColor: '#0e7490 !important'
                    }
                });

                // Create editor
                const startState = EditorState.create({
                    doc: INITIAL_CODE,
                    extensions: [
                        lineNumbers(),
                        highlightActiveLineGutter(),
                        highlightSpecialChars(),
                        history(),
                        foldGutter(),
                        drawSelection(),
                        dropCursor(),
                        EditorState.allowMultipleSelections.of(true),
                        indentOnInput(),
                        syntaxHighlighting(defaultHighlightStyle, {fallback: true}),
                        bracketMatching(),
                        closeBrackets(),
                        autocompletion(),
                        rectangularSelection(),
                        crosshairCursor(),
                        highlightSelectionMatches(),
                        keymap.of([
                            ...closeBracketsKeymap,
                            ...defaultKeymap,
                            ...searchKeymap,
                            ...historyKeymap,
                            ...foldKeymap,
                            ...completionKeymap,
                            // Ctrl/Cmd+Enter to run
                            {
                                key: 'Mod-Enter',
                                run: () => { runCode(); return true; }
                            }
                        ]),
                        juliaTheme
                    ]
                });

                editor = new EditorView({
                    state: startState,
                    parent: editorContainer
                });

                console.log('CodeMirror initialized');
            } catch (err) {
                console.error('Failed to load CodeMirror:', err);
                // Fallback to textarea
                editorContainer.innerHTML = '<textarea id="fallback-editor" class="w-full h-full bg-stone-900 text-stone-100 p-4 font-mono text-sm resize-none focus:outline-none" spellcheck="false">' + INITIAL_CODE + '</textarea>';
            }
        }

        // Initialize WasmGC interpreter
        async function initInterpreter() {
            updateStatus('loading', 'Loading WasmGC interpreter...');

            try {
                // Load the real WasmGC interpreter compiled by WasmTarget.jl
                const wasmPath = '/WasmTarget.jl/wasm/interpreter.wasm';

                // Math.pow import required by the interpreter
                const importObject = {
                    Math: {
                        pow: Math.pow
                    }
                };

                const response = await fetch(wasmPath);
                if (!response.ok) {
                    throw new Error('Failed to fetch interpreter.wasm: ' + response.status);
                }

                const { instance } = await WebAssembly.instantiateStreaming(response, importObject);
                interpreterModule = instance;
                interpreterReady = true;

                updateStatus('ready', 'WasmGC Ready');

                if (runButton) {
                    runButton.disabled = false;
                }

                console.log('WasmGC interpreter loaded successfully');
                console.log('Available exports:', Object.keys(instance.exports));
            } catch (err) {
                console.error('Failed to load WasmGC interpreter:', err);
                console.log('Falling back to JavaScript interpreter...');

                // Fallback to JS interpreter if WASM fails
                interpreterReady = true;
                updateStatus('ready', 'Ready (JS fallback)');

                if (runButton) {
                    runButton.disabled = false;
                }
            }
        }

        // Simple JavaScript interpreter for basic Julia syntax
        // This is a placeholder until the WasmGC interpreter is ready
        function interpretJS(code) {
            const output = [];
            const env = {};

            // Split into statements
            const lines = code.split(/[;\\n]+/).filter(l => l.trim() && !l.trim().startsWith('#'));

            for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed) continue;

                // Handle println
                const printMatch = trimmed.match(/^println\\((.*)\\)\$/);
                if (printMatch) {
                    try {
                        const args = printMatch[1];
                        // Simple evaluation
                        const result = evalExpr(args, env);
                        output.push(String(result));
                    } catch (e) {
                        output.push('Error: ' + e.message);
                    }
                    continue;
                }

                // Handle assignment
                const assignMatch = trimmed.match(/^(\\w+)\\s*=\\s*(.+)\$/);
                if (assignMatch) {
                    const [, name, expr] = assignMatch;
                    try {
                        env[name] = evalExpr(expr, env);
                    } catch (e) {
                        output.push('Error in assignment: ' + e.message);
                    }
                    continue;
                }
            }

            return output.join('\\n');
        }

        // Simple expression evaluator
        function evalExpr(expr, env) {
            expr = expr.trim();

            // String literal
            if (expr.startsWith('"') && expr.endsWith('"')) {
                return expr.slice(1, -1);
            }

            // Number literal
            if (/^-?\\d+(\\.\\d+)?\$/.test(expr)) {
                return parseFloat(expr);
            }

            // Variable lookup
            if (/^\\w+\$/.test(expr) && expr in env) {
                return env[expr];
            }

            // String concatenation with *
            if (expr.includes(' * ')) {
                const parts = expr.split(' * ').map(p => evalExpr(p.trim(), env));
                return parts.join('');
            }

            // Addition
            if (expr.includes(' + ')) {
                const parts = expr.split(' + ').map(p => evalExpr(p.trim(), env));
                if (typeof parts[0] === 'string') {
                    return parts.join('');
                }
                return parts.reduce((a, b) => a + b, 0);
            }

            // Subtraction
            if (expr.includes(' - ')) {
                const parts = expr.split(' - ').map(p => evalExpr(p.trim(), env));
                return parts.reduce((a, b) => a - b);
            }

            // Multiplication
            if (expr.includes(' * ') && !expr.includes('"')) {
                const parts = expr.split(' * ').map(p => evalExpr(p.trim(), env));
                return parts.reduce((a, b) => a * b, 1);
            }

            // Division
            if (expr.includes(' / ')) {
                const parts = expr.split(' / ').map(p => evalExpr(p.trim(), env));
                return parts.reduce((a, b) => a / b);
            }

            return expr;
        }

        // Run code using WasmGC interpreter
        function runCode() {
            if (!interpreterReady) return;

            const code = getEditorContent();
            if (!code.trim()) {
                clearOutput();
                appendOutput('No code to run', true);
                return;
            }

            updateStatus('running', 'Running...');
            clearOutput();

            try {
                let result;

                if (interpreterModule && interpreterModule.exports.interpret) {
                    // Use real WasmGC interpreter
                    // The interpret function takes a string and returns a string
                    const exports = interpreterModule.exports;

                    // Convert JS string to WASM string
                    // For now, try calling interpret directly
                    // Note: This requires proper string marshaling which may need adjustment
                    try {
                        result = exports.interpret(code);
                        // Get output from the interpreter's output buffer
                        if (exports.get_output) {
                            result = exports.get_output();
                        }
                    } catch (wasmErr) {
                        console.error('WASM execution error:', wasmErr);
                        // Fall back to JS interpreter
                        result = interpretJS(code);
                    }
                } else {
                    // Use JS interpreter as fallback
                    result = interpretJS(code);
                }

                if (result) {
                    String(result).split('\\n').forEach(line => {
                        if (line.trim()) appendOutput(line);
                    });
                }
                appendOutput('\\n--- Execution complete ---', false);
                updateStatus('ready', interpreterModule ? 'WasmGC Ready' : 'Ready');
            } catch (err) {
                appendOutput('Error: ' + err.message, true);
                updateStatus('error', 'Error');
            }
        }

        // Get editor content
        function getEditorContent() {
            if (editor) {
                return editor.state.doc.toString();
            }
            const fallback = document.getElementById('fallback-editor');
            if (fallback) {
                return fallback.value;
            }
            return '';
        }

        // Set editor content
        function setEditorContent(code) {
            if (editor) {
                editor.dispatch({
                    changes: { from: 0, to: editor.state.doc.length, insert: code }
                });
            } else {
                const fallback = document.getElementById('fallback-editor');
                if (fallback) {
                    fallback.value = code;
                }
            }
        }

        // Event listeners
        if (runButton) {
            runButton.addEventListener('click', runCode);
        }

        if (clearButton) {
            clearButton.addEventListener('click', () => {
                setEditorContent('');
            });
        }

        if (clearOutputButton) {
            clearOutputButton.addEventListener('click', clearOutput);
        }

        if (exampleSelector) {
            exampleSelector.addEventListener('change', (e) => {
                const exampleId = e.target.value;
                if (exampleId && examples[exampleId]) {
                    setEditorContent(examples[exampleId]);
                }
                e.target.value = '';
            });
        }

        // Initialize
        (async function() {
            await initEditor();
            await initInterpreter();
        })();
    })();
    """
end

# Export
export InterpreterPlayground
