# Julia Interpreter Tokenizer - Written in WasmTarget-compatible Julia
# This tokenizer scans Julia source code and produces tokens.
#
# Design: Uses WasmTarget intrinsics for string operations:
# - str_char(s, i) - get character at index
# - str_len(s) - get string length
# - str_new(len) - create new string
# - str_setchar!(s, i, c) - set character at index
#
# Token types are represented as Int32 for efficient WASM execution.

export Token, TokenType, tokenize, token_type_name
export TOK_EOF, TOK_ERROR, TOK_INT, TOK_FLOAT, TOK_IDENT, TOK_STRING
export TOK_PLUS, TOK_MINUS, TOK_STAR, TOK_SLASH, TOK_PERCENT
export TOK_EQ, TOK_EQ_EQ, TOK_NE, TOK_LT, TOK_LE, TOK_GT, TOK_GE
export TOK_LPAREN, TOK_RPAREN, TOK_LBRACKET, TOK_RBRACKET, TOK_LBRACE, TOK_RBRACE
export TOK_COMMA, TOK_COLON, TOK_SEMICOLON, TOK_DOT, TOK_ARROW
export TOK_KW_IF, TOK_KW_ELSE, TOK_KW_ELSEIF, TOK_KW_END, TOK_KW_WHILE
export TOK_KW_FOR, TOK_KW_FUNCTION, TOK_KW_RETURN, TOK_KW_TRUE, TOK_KW_FALSE
export TOK_KW_AND, TOK_KW_OR, TOK_KW_NOT, TOK_KW_IN, TOK_KW_NOTHING, TOK_KW_LET
export TOK_NEWLINE, TOK_PIPE_PIPE, TOK_AMP_AMP

# ============================================================================
# Token Type Constants
# ============================================================================
const TOK_EOF       = Int32(0)   # End of file
const TOK_ERROR     = Int32(1)   # Error/invalid token
const TOK_INT       = Int32(2)   # Integer literal: 42
const TOK_FLOAT     = Int32(3)   # Float literal: 3.14
const TOK_IDENT     = Int32(4)   # Identifier: foo, bar
const TOK_STRING    = Int32(5)   # String literal: "hello"

# Operators
const TOK_PLUS      = Int32(10)  # +
const TOK_MINUS     = Int32(11)  # -
const TOK_STAR      = Int32(12)  # *
const TOK_SLASH     = Int32(13)  # /
const TOK_PERCENT   = Int32(14)  # %
const TOK_CARET     = Int32(15)  # ^

# Comparison
const TOK_EQ        = Int32(20)  # =
const TOK_EQ_EQ     = Int32(21)  # ==
const TOK_NE        = Int32(22)  # != or ≠
const TOK_LT        = Int32(23)  # <
const TOK_LE        = Int32(24)  # <=
const TOK_GT        = Int32(25)  # >
const TOK_GE        = Int32(26)  # >=

# Delimiters
const TOK_LPAREN    = Int32(30)  # (
const TOK_RPAREN    = Int32(31)  # )
const TOK_LBRACKET  = Int32(32)  # [
const TOK_RBRACKET  = Int32(33)  # ]
const TOK_LBRACE    = Int32(34)  # {
const TOK_RBRACE    = Int32(35)  # }
const TOK_COMMA     = Int32(36)  # ,
const TOK_COLON     = Int32(37)  # :
const TOK_SEMICOLON = Int32(38)  # ;
const TOK_DOT       = Int32(39)  # .
const TOK_ARROW     = Int32(40)  # ->
const TOK_NEWLINE   = Int32(41)  # \n

# Logical
const TOK_PIPE_PIPE = Int32(42)  # ||
const TOK_AMP_AMP   = Int32(43)  # &&

# Keywords (50-69)
const TOK_KW_IF       = Int32(50)
const TOK_KW_ELSE     = Int32(51)
const TOK_KW_ELSEIF   = Int32(52)
const TOK_KW_END      = Int32(53)
const TOK_KW_WHILE    = Int32(54)
const TOK_KW_FOR      = Int32(55)
const TOK_KW_FUNCTION = Int32(56)
const TOK_KW_RETURN   = Int32(57)
const TOK_KW_TRUE     = Int32(58)
const TOK_KW_FALSE    = Int32(59)
const TOK_KW_AND      = Int32(60)
const TOK_KW_OR       = Int32(61)
const TOK_KW_NOT      = Int32(62)
const TOK_KW_IN       = Int32(63)
const TOK_KW_NOTHING  = Int32(64)
const TOK_KW_LET      = Int32(65)

# ============================================================================
# Token Structure
# ============================================================================

"""
Token - Represents a single token in the source code.

Fields:
- type: Token type constant (TOK_*)
- int_value: For INT tokens, the numeric value
- float_value: For FLOAT tokens, the numeric value
- start: Start index in source (1-based)
- length: Length of token text
"""
mutable struct Token
    type::Int32       # Token type (TOK_* constant)
    int_value::Int32  # Integer value (for TOK_INT)
    float_value::Float32  # Float value (for TOK_FLOAT)
    start::Int32      # Start position in source (1-based)
    length::Int32     # Length of token text
end

"""
Create an empty/EOF token.
"""
@noinline function token_eof()::Token
    return Token(TOK_EOF, Int32(0), Float32(0.0), Int32(0), Int32(0))
end

"""
Create an error token.
"""
@noinline function token_error(pos::Int32)::Token
    return Token(TOK_ERROR, Int32(0), Float32(0.0), pos, Int32(1))
end

"""
Create a token with just type and position info.
"""
@noinline function token_simple(type::Int32, start::Int32, length::Int32)::Token
    return Token(type, Int32(0), Float32(0.0), start, length)
end

"""
Create an integer token.
"""
@noinline function token_int(value::Int32, start::Int32, length::Int32)::Token
    return Token(TOK_INT, value, Float32(0.0), start, length)
end

"""
Create a float token.
"""
@noinline function token_float(value::Float32, start::Int32, length::Int32)::Token
    return Token(TOK_FLOAT, Int32(0), value, start, length)
end

# ============================================================================
# Lexer State
# ============================================================================

"""
Lexer - Holds the lexer state for tokenization.
"""
mutable struct Lexer
    source::String    # Source code string
    pos::Int32        # Current position (1-based)
    len::Int32        # Length of source
end

"""
Create a new lexer for the given source code.
"""
@noinline function lexer_new(source::String)::Lexer
    return Lexer(source, Int32(1), str_len(source))
end

# ============================================================================
# Character Classification
# ============================================================================

"""
Check if character is a digit (0-9).
Returns Int32 (1 for true, 0 for false) to avoid Bool compilation issues.
"""
@noinline function is_digit(c::Int32)::Int32
    if c >= Int32(48)
        if c <= Int32(57)
            return Int32(1)
        end
    end
    return Int32(0)
end

"""
Check if character is a letter (a-z, A-Z) or underscore.
Returns Int32 (1 for true, 0 for false).
"""
@noinline function is_alpha(c::Int32)::Int32
    # Check lowercase a-z
    if c >= Int32(97)
        if c <= Int32(122)
            return Int32(1)
        end
    end
    # Check uppercase A-Z
    if c >= Int32(65)
        if c <= Int32(90)
            return Int32(1)
        end
    end
    # Check underscore
    if c == Int32(95)
        return Int32(1)
    end
    return Int32(0)
end

"""
Check if character is alphanumeric or underscore.
Returns Int32 (1 for true, 0 for false).
"""
@noinline function is_alnum(c::Int32)::Int32
    if is_alpha(c) == Int32(1)
        return Int32(1)
    end
    if is_digit(c) == Int32(1)
        return Int32(1)
    end
    return Int32(0)
end

"""
Check if character is whitespace (space, tab, carriage return).
Returns Int32 (1 for true, 0 for false).
"""
@noinline function is_whitespace(c::Int32)::Int32
    if c == Int32(32)   # ' '
        return Int32(1)
    end
    if c == Int32(9)    # '\t'
        return Int32(1)
    end
    if c == Int32(13)   # '\r'
        return Int32(1)
    end
    return Int32(0)
end

"""
Check if character is newline.
Returns Int32 (1 for true, 0 for false).
"""
@noinline function is_newline(c::Int32)::Int32
    if c == Int32(10)  # '\n'
        return Int32(1)
    end
    return Int32(0)
end

# ============================================================================
# Lexer Operations
# ============================================================================

"""
Get current character without advancing.
Returns -1 if at end of source.
"""
@noinline function lexer_peek(lex::Lexer)::Int32
    if lex.pos > lex.len
        return Int32(-1)  # EOF
    end
    return str_char(lex.source, lex.pos)
end

"""
Get character at offset from current position.
Returns -1 if out of bounds.
"""
@noinline function lexer_peek_at(lex::Lexer, offset::Int32)::Int32
    next_pos = lex.pos + offset
    if next_pos > lex.len
        return Int32(-1)  # EOF
    end
    if next_pos < Int32(1)
        return Int32(-1)  # Before start
    end
    return str_char(lex.source, next_pos)
end

"""
Advance position by one.
"""
@noinline function lexer_advance!(lex::Lexer)::Nothing
    if lex.pos <= lex.len
        lex.pos = lex.pos + Int32(1)
    end
    return nothing
end

"""
Skip whitespace (but not newlines).
"""
@noinline function lexer_skip_whitespace!(lex::Lexer)::Nothing
    while lex.pos <= lex.len
        c = str_char(lex.source, lex.pos)
        if is_whitespace(c) == Int32(0)
            return nothing
        end
        lex.pos = lex.pos + Int32(1)
    end
    return nothing
end

"""
Skip a line comment (starting with #).
"""
@noinline function lexer_skip_comment!(lex::Lexer)::Nothing
    # Skip until newline or EOF
    while lex.pos <= lex.len
        c = str_char(lex.source, lex.pos)
        if is_newline(c) == Int32(1)
            return nothing  # Don't skip the newline itself
        end
        lex.pos = lex.pos + Int32(1)
    end
    return nothing
end

# ============================================================================
# Number Scanning
# ============================================================================

"""
Scan an integer literal. Returns the token.
"""
@noinline function scan_integer(lex::Lexer)::Token
    start = lex.pos
    value = Int32(0)

    while lex.pos <= lex.len
        c = str_char(lex.source, lex.pos)
        if is_digit(c) == Int32(0)
            break
        end
        digit = c - Int32(48)  # Convert ASCII to digit
        value = value * Int32(10) + digit
        lex.pos = lex.pos + Int32(1)
    end

    length = lex.pos - start

    # Check for decimal point (float)
    if lex.pos <= lex.len
        c = str_char(lex.source, lex.pos)
        if c == Int32(46)  # '.'
            # Check next char is digit (not method call like 1.foo)
            if lex.pos + Int32(1) <= lex.len
                next = str_char(lex.source, lex.pos + Int32(1))
                if is_digit(next) == Int32(1)
                    return scan_float_after_dot(lex, start, value)
                end
            end
        end
    end

    return token_int(value, start, length)
end

"""
Continue scanning float after decimal point.
"""
@noinline function scan_float_after_dot(lex::Lexer, start::Int32, int_part::Int32)::Token
    # Skip the dot
    lex.pos = lex.pos + Int32(1)

    # Parse fractional part
    frac = Float32(0.0)
    divisor = Float32(10.0)

    while lex.pos <= lex.len
        c = str_char(lex.source, lex.pos)
        if is_digit(c) == Int32(0)
            break
        end
        digit = Float32(c - Int32(48))
        frac = frac + digit / divisor
        divisor = divisor * Float32(10.0)
        lex.pos = lex.pos + Int32(1)
    end

    length = lex.pos - start
    value = Float32(int_part) + frac

    return token_float(value, start, length)
end

# ============================================================================
# Identifier and Keyword Scanning
# ============================================================================

"""
Scan an identifier or keyword.
"""
@noinline function scan_identifier(lex::Lexer)::Token
    start = lex.pos

    # Scan the full identifier
    while lex.pos <= lex.len
        c = str_char(lex.source, lex.pos)
        if is_alnum(c) == Int32(0)
            break
        end
        lex.pos = lex.pos + Int32(1)
    end

    length = lex.pos - start

    # Check if it's a keyword
    kw_type = check_keyword(lex.source, start, length)

    if kw_type != TOK_IDENT
        return token_simple(kw_type, start, length)
    end

    return token_simple(TOK_IDENT, start, length)
end

"""
Check if identifier is a keyword. Returns keyword token type or TOK_IDENT.
"""
@noinline function check_keyword(source::String, start::Int32, length::Int32)::Int32
    # Check keywords by length for efficiency
    if length == Int32(2)
        return check_keyword_2(source, start)
    end
    if length == Int32(3)
        return check_keyword_3(source, start)
    end
    if length == Int32(4)
        return check_keyword_4(source, start)
    end
    if length == Int32(5)
        return check_keyword_5(source, start)
    end
    if length == Int32(6)
        return check_keyword_6(source, start)
    end
    if length == Int32(7)
        return check_keyword_7(source, start)
    end
    if length == Int32(8)
        return check_keyword_8(source, start)
    end
    return TOK_IDENT
end

# Length-2 keywords: if, in, or
@noinline function check_keyword_2(source::String, start::Int32)::Int32
    c1 = str_char(source, start)
    c2 = str_char(source, start + Int32(1))

    # "if"
    if c1 == Int32(105) && c2 == Int32(102)
        return TOK_KW_IF
    end
    # "in"
    if c1 == Int32(105) && c2 == Int32(110)
        return TOK_KW_IN
    end
    # "or"
    if c1 == Int32(111) && c2 == Int32(114)
        return TOK_KW_OR
    end

    return TOK_IDENT
end

# Length-3 keywords: end, for, let, not, and
@noinline function check_keyword_3(source::String, start::Int32)::Int32
    c1 = str_char(source, start)
    c2 = str_char(source, start + Int32(1))
    c3 = str_char(source, start + Int32(2))

    # "end"
    if c1 == Int32(101) && c2 == Int32(110) && c3 == Int32(100)
        return TOK_KW_END
    end
    # "for"
    if c1 == Int32(102) && c2 == Int32(111) && c3 == Int32(114)
        return TOK_KW_FOR
    end
    # "let"
    if c1 == Int32(108) && c2 == Int32(101) && c3 == Int32(116)
        return TOK_KW_LET
    end
    # "not"
    if c1 == Int32(110) && c2 == Int32(111) && c3 == Int32(116)
        return TOK_KW_NOT
    end
    # "and"
    if c1 == Int32(97) && c2 == Int32(110) && c3 == Int32(100)
        return TOK_KW_AND
    end

    return TOK_IDENT
end

# Length-4 keywords: else, true
@noinline function check_keyword_4(source::String, start::Int32)::Int32
    c1 = str_char(source, start)
    c2 = str_char(source, start + Int32(1))
    c3 = str_char(source, start + Int32(2))
    c4 = str_char(source, start + Int32(3))

    # "else"
    if c1 == Int32(101) && c2 == Int32(108) && c3 == Int32(115) && c4 == Int32(101)
        return TOK_KW_ELSE
    end
    # "true"
    if c1 == Int32(116) && c2 == Int32(114) && c3 == Int32(117) && c4 == Int32(101)
        return TOK_KW_TRUE
    end

    return TOK_IDENT
end

# Length-5 keywords: while, false
@noinline function check_keyword_5(source::String, start::Int32)::Int32
    c1 = str_char(source, start)
    c2 = str_char(source, start + Int32(1))
    c3 = str_char(source, start + Int32(2))
    c4 = str_char(source, start + Int32(3))
    c5 = str_char(source, start + Int32(4))

    # "while"
    if c1 == Int32(119) && c2 == Int32(104) && c3 == Int32(105) && c4 == Int32(108) && c5 == Int32(101)
        return TOK_KW_WHILE
    end
    # "false"
    if c1 == Int32(102) && c2 == Int32(97) && c3 == Int32(108) && c4 == Int32(115) && c5 == Int32(101)
        return TOK_KW_FALSE
    end

    return TOK_IDENT
end

# Length-6 keywords: elseif, return
@noinline function check_keyword_6(source::String, start::Int32)::Int32
    c1 = str_char(source, start)
    c2 = str_char(source, start + Int32(1))
    c3 = str_char(source, start + Int32(2))
    c4 = str_char(source, start + Int32(3))
    c5 = str_char(source, start + Int32(4))
    c6 = str_char(source, start + Int32(5))

    # "elseif"
    if c1 == Int32(101) && c2 == Int32(108) && c3 == Int32(115) && c4 == Int32(101) && c5 == Int32(105) && c6 == Int32(102)
        return TOK_KW_ELSEIF
    end
    # "return"
    if c1 == Int32(114) && c2 == Int32(101) && c3 == Int32(116) && c4 == Int32(117) && c5 == Int32(114) && c6 == Int32(110)
        return TOK_KW_RETURN
    end

    return TOK_IDENT
end

# Length-7 keywords: nothing
@noinline function check_keyword_7(source::String, start::Int32)::Int32
    c1 = str_char(source, start)
    c2 = str_char(source, start + Int32(1))
    c3 = str_char(source, start + Int32(2))
    c4 = str_char(source, start + Int32(3))
    c5 = str_char(source, start + Int32(4))
    c6 = str_char(source, start + Int32(5))
    c7 = str_char(source, start + Int32(6))

    # "nothing"
    if c1 == Int32(110) && c2 == Int32(111) && c3 == Int32(116) && c4 == Int32(104) && c5 == Int32(105) && c6 == Int32(110) && c7 == Int32(103)
        return TOK_KW_NOTHING
    end

    return TOK_IDENT
end

# Length-8 keywords: function
@noinline function check_keyword_8(source::String, start::Int32)::Int32
    c1 = str_char(source, start)
    c2 = str_char(source, start + Int32(1))
    c3 = str_char(source, start + Int32(2))
    c4 = str_char(source, start + Int32(3))
    c5 = str_char(source, start + Int32(4))
    c6 = str_char(source, start + Int32(5))
    c7 = str_char(source, start + Int32(6))
    c8 = str_char(source, start + Int32(7))

    # "function"
    if c1 == Int32(102) && c2 == Int32(117) && c3 == Int32(110) && c4 == Int32(99) && c5 == Int32(116) && c6 == Int32(105) && c7 == Int32(111) && c8 == Int32(110)
        return TOK_KW_FUNCTION
    end

    return TOK_IDENT
end

# ============================================================================
# String Scanning
# ============================================================================

"""
Scan a string literal (double-quoted).
"""
@noinline function scan_string(lex::Lexer)::Token
    start = lex.pos

    # Skip opening quote
    lex.pos = lex.pos + Int32(1)

    # Scan until closing quote or EOF
    while lex.pos <= lex.len
        c = str_char(lex.source, lex.pos)

        # Check for closing quote
        if c == Int32(34)  # '"'
            lex.pos = lex.pos + Int32(1)  # Skip closing quote
            length = lex.pos - start
            return token_simple(TOK_STRING, start, length)
        end

        # Check for escape sequence
        if c == Int32(92)  # '\'
            lex.pos = lex.pos + Int32(2)  # Skip backslash and next char
        else
            lex.pos = lex.pos + Int32(1)
        end
    end

    # Unterminated string
    return token_error(start)
end

# ============================================================================
# Main Tokenization
# ============================================================================

"""
Get the next token from the lexer.
"""
@noinline function lexer_next_token!(lex::Lexer)::Token
    # Skip whitespace (but not newlines)
    lexer_skip_whitespace!(lex)

    # Check for EOF
    if lex.pos > lex.len
        return token_eof()
    end

    c = str_char(lex.source, lex.pos)

    # Newline
    if is_newline(c) == Int32(1)
        start = lex.pos
        lex.pos = lex.pos + Int32(1)
        return token_simple(TOK_NEWLINE, start, Int32(1))
    end

    # Comment
    if c == Int32(35)  # '#'
        lexer_skip_comment!(lex)
        # After comment, get next token (recursive call is fine - bounded by newlines)
        return lexer_next_token!(lex)
    end

    # Number
    if is_digit(c) == Int32(1)
        return scan_integer(lex)
    end

    # Identifier or keyword
    if is_alpha(c) == Int32(1)
        return scan_identifier(lex)
    end

    # String literal
    if c == Int32(34)  # '"'
        return scan_string(lex)
    end

    # Operators and delimiters
    return scan_operator(lex)
end

"""
Scan an operator or delimiter.
"""
@noinline function scan_operator(lex::Lexer)::Token
    start = lex.pos
    c = str_char(lex.source, lex.pos)

    # Single character operators
    if c == Int32(43)  # '+'
        lex.pos = lex.pos + Int32(1)
        return token_simple(TOK_PLUS, start, Int32(1))
    end
    if c == Int32(42)  # '*'
        lex.pos = lex.pos + Int32(1)
        return token_simple(TOK_STAR, start, Int32(1))
    end
    if c == Int32(47)  # '/'
        lex.pos = lex.pos + Int32(1)
        return token_simple(TOK_SLASH, start, Int32(1))
    end
    if c == Int32(37)  # '%'
        lex.pos = lex.pos + Int32(1)
        return token_simple(TOK_PERCENT, start, Int32(1))
    end
    if c == Int32(94)  # '^'
        lex.pos = lex.pos + Int32(1)
        return token_simple(TOK_CARET, start, Int32(1))
    end
    if c == Int32(40)  # '('
        lex.pos = lex.pos + Int32(1)
        return token_simple(TOK_LPAREN, start, Int32(1))
    end
    if c == Int32(41)  # ')'
        lex.pos = lex.pos + Int32(1)
        return token_simple(TOK_RPAREN, start, Int32(1))
    end
    if c == Int32(91)  # '['
        lex.pos = lex.pos + Int32(1)
        return token_simple(TOK_LBRACKET, start, Int32(1))
    end
    if c == Int32(93)  # ']'
        lex.pos = lex.pos + Int32(1)
        return token_simple(TOK_RBRACKET, start, Int32(1))
    end
    if c == Int32(123)  # '{'
        lex.pos = lex.pos + Int32(1)
        return token_simple(TOK_LBRACE, start, Int32(1))
    end
    if c == Int32(125)  # '}'
        lex.pos = lex.pos + Int32(1)
        return token_simple(TOK_RBRACE, start, Int32(1))
    end
    if c == Int32(44)  # ','
        lex.pos = lex.pos + Int32(1)
        return token_simple(TOK_COMMA, start, Int32(1))
    end
    if c == Int32(58)  # ':'
        lex.pos = lex.pos + Int32(1)
        return token_simple(TOK_COLON, start, Int32(1))
    end
    if c == Int32(59)  # ';'
        lex.pos = lex.pos + Int32(1)
        return token_simple(TOK_SEMICOLON, start, Int32(1))
    end
    if c == Int32(46)  # '.'
        lex.pos = lex.pos + Int32(1)
        return token_simple(TOK_DOT, start, Int32(1))
    end

    # Two-character operators
    next = lexer_peek_at(lex, Int32(1))

    # '-' or '->'
    if c == Int32(45)  # '-'
        if next == Int32(62)  # '>'
            lex.pos = lex.pos + Int32(2)
            return token_simple(TOK_ARROW, start, Int32(2))
        end
        lex.pos = lex.pos + Int32(1)
        return token_simple(TOK_MINUS, start, Int32(1))
    end

    # '=' or '=='
    if c == Int32(61)  # '='
        if next == Int32(61)  # '='
            lex.pos = lex.pos + Int32(2)
            return token_simple(TOK_EQ_EQ, start, Int32(2))
        end
        lex.pos = lex.pos + Int32(1)
        return token_simple(TOK_EQ, start, Int32(1))
    end

    # '!' or '!='
    if c == Int32(33)  # '!'
        if next == Int32(61)  # '='
            lex.pos = lex.pos + Int32(2)
            return token_simple(TOK_NE, start, Int32(2))
        end
        # Just '!' is not handled yet - treat as error
        lex.pos = lex.pos + Int32(1)
        return token_error(start)
    end

    # '<' or '<='
    if c == Int32(60)  # '<'
        if next == Int32(61)  # '='
            lex.pos = lex.pos + Int32(2)
            return token_simple(TOK_LE, start, Int32(2))
        end
        lex.pos = lex.pos + Int32(1)
        return token_simple(TOK_LT, start, Int32(1))
    end

    # '>' or '>='
    if c == Int32(62)  # '>'
        if next == Int32(61)  # '='
            lex.pos = lex.pos + Int32(2)
            return token_simple(TOK_GE, start, Int32(2))
        end
        lex.pos = lex.pos + Int32(1)
        return token_simple(TOK_GT, start, Int32(1))
    end

    # '||'
    if c == Int32(124)  # '|'
        if next == Int32(124)  # '|'
            lex.pos = lex.pos + Int32(2)
            return token_simple(TOK_PIPE_PIPE, start, Int32(2))
        end
        # Single | not supported yet
        lex.pos = lex.pos + Int32(1)
        return token_error(start)
    end

    # '&&'
    if c == Int32(38)  # '&'
        if next == Int32(38)  # '&'
            lex.pos = lex.pos + Int32(2)
            return token_simple(TOK_AMP_AMP, start, Int32(2))
        end
        # Single & not supported yet
        lex.pos = lex.pos + Int32(1)
        return token_error(start)
    end

    # Unknown character
    lex.pos = lex.pos + Int32(1)
    return token_error(start)
end

# ============================================================================
# TokenList - Fixed-size token array for WASM
# ============================================================================

"""
TokenList - A fixed-size list of tokens.
"""
mutable struct TokenList
    tokens::Vector{Token}
    count::Int32
    capacity::Int32
end

"""
Create a new token list with given capacity.
"""
@noinline function token_list_new(capacity::Int32)::TokenList
    tokens = Vector{Token}(undef, capacity)
    # Initialize with EOF tokens
    i = Int32(1)
    while i <= capacity
        tokens[i] = token_eof()
        i = i + Int32(1)
    end
    return TokenList(tokens, Int32(0), capacity)
end

"""
Add a token to the list.
"""
@noinline function token_list_push!(list::TokenList, tok::Token)::Nothing
    if list.count < list.capacity
        list.count = list.count + Int32(1)
        list.tokens[list.count] = tok
    end
    return nothing
end

"""
Get token at index (1-based).
"""
@noinline function token_list_get(list::TokenList, index::Int32)::Token
    if index < Int32(1) || index > list.count
        return token_eof()
    end
    return list.tokens[index]
end

# ============================================================================
# Main Tokenize Function
# ============================================================================

"""
Tokenize a source string into a list of tokens.
"""
@noinline function tokenize(source::String, max_tokens::Int32)::TokenList
    lex = lexer_new(source)
    list = token_list_new(max_tokens)

    while list.count < max_tokens
        tok = lexer_next_token!(lex)
        token_list_push!(list, tok)

        # Stop at EOF
        if tok.type == TOK_EOF
            break
        end
    end

    return list
end

# ============================================================================
# Debug/Utility Functions (Julia-only, not for WASM)
# ============================================================================

"""
Get human-readable name for a token type.
"""
function token_type_name(type::Int32)::String
    type == TOK_EOF && return "EOF"
    type == TOK_ERROR && return "ERROR"
    type == TOK_INT && return "INT"
    type == TOK_FLOAT && return "FLOAT"
    type == TOK_IDENT && return "IDENT"
    type == TOK_STRING && return "STRING"
    type == TOK_PLUS && return "PLUS"
    type == TOK_MINUS && return "MINUS"
    type == TOK_STAR && return "STAR"
    type == TOK_SLASH && return "SLASH"
    type == TOK_PERCENT && return "PERCENT"
    type == TOK_CARET && return "CARET"
    type == TOK_EQ && return "EQ"
    type == TOK_EQ_EQ && return "EQ_EQ"
    type == TOK_NE && return "NE"
    type == TOK_LT && return "LT"
    type == TOK_LE && return "LE"
    type == TOK_GT && return "GT"
    type == TOK_GE && return "GE"
    type == TOK_LPAREN && return "LPAREN"
    type == TOK_RPAREN && return "RPAREN"
    type == TOK_LBRACKET && return "LBRACKET"
    type == TOK_RBRACKET && return "RBRACKET"
    type == TOK_LBRACE && return "LBRACE"
    type == TOK_RBRACE && return "RBRACE"
    type == TOK_COMMA && return "COMMA"
    type == TOK_COLON && return "COLON"
    type == TOK_SEMICOLON && return "SEMICOLON"
    type == TOK_DOT && return "DOT"
    type == TOK_ARROW && return "ARROW"
    type == TOK_NEWLINE && return "NEWLINE"
    type == TOK_PIPE_PIPE && return "OR"
    type == TOK_AMP_AMP && return "AND"
    type == TOK_KW_IF && return "IF"
    type == TOK_KW_ELSE && return "ELSE"
    type == TOK_KW_ELSEIF && return "ELSEIF"
    type == TOK_KW_END && return "END"
    type == TOK_KW_WHILE && return "WHILE"
    type == TOK_KW_FOR && return "FOR"
    type == TOK_KW_FUNCTION && return "FUNCTION"
    type == TOK_KW_RETURN && return "RETURN"
    type == TOK_KW_TRUE && return "TRUE"
    type == TOK_KW_FALSE && return "FALSE"
    type == TOK_KW_AND && return "KW_AND"
    type == TOK_KW_OR && return "KW_OR"
    type == TOK_KW_NOT && return "NOT"
    type == TOK_KW_IN && return "IN"
    type == TOK_KW_NOTHING && return "NOTHING"
    type == TOK_KW_LET && return "LET"
    return "UNKNOWN"
end
