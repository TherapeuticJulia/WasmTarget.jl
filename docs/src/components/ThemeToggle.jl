# ThemeToggle.jl - An interactive island for dark/light mode switching
#
# This is a Therapy.jl island - an interactive component that gets compiled to Wasm.

"""
Theme toggle island - compiled to WebAssembly.

- `island()` marks this as an interactive component
- Signal value controls dark mode (0=light, 1=dark)
- :dark_mode prop binds signal to document.documentElement.classList
- Theme persists via localStorage
"""
ThemeToggle = island(:ThemeToggle) do
    # Create reactive state - 0 for light mode, 1 for dark mode
    # Use Int32 explicitly to avoid BigInt issues in JavaScript
    dark, set_dark = create_signal(Int32(0))

    # The :dark_mode prop tells the compiler to call set_dark_mode(value) when signal changes
    Div(:dark_mode => dark,
        Button(
            :class => "p-2 rounded-lg hover:bg-stone-100 dark:hover:bg-stone-700 transition-colors",
            :on_click => () -> set_dark(dark() == Int32(0) ? Int32(1) : Int32(0)),
            :title => "Toggle dark mode",
            # Sun/Moon icon
            Svg(:class => "w-5 h-5 text-stone-600 dark:text-stone-300",
                :fill => "none",
                :viewBox => "0 0 24 24",
                :stroke => "currentColor",
                :stroke_width => "2",
                Path(:stroke_linecap => "round",
                     :stroke_linejoin => "round",
                     :d => "M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z")
            )
        )
    )
end
