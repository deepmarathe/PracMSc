class SimpleParser:
    def __init__(self, expr):
        # Tokenize the input expression, adding spaces around parentheses for easier splitting
        self.tokens = expr.replace('(', ' ( ').replace(')', ' ) ').split()
        self.pos = 0

    # The main parse method to start the parsing
    def parse(self):
        return self.expr()

    # Move to the next token
    def advance(self):
        self.pos += 1

    # Get the current token or None if at the end of the list
    def current_token(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    # Handle addition and subtraction
    def expr(self):
        result = self.term()  # Parse the first term
        while self.current_token() in ('+', '-'):
            if self.current_token() == '+':
                self.advance()
                result += self.term()  # Perform addition
            elif self.current_token() == '-':
                self.advance()
                result -= self.term()  # Perform subtraction
        return result

    # Handle multiplication and division
    def term(self):
        result = self.factor()  # Parse the first factor
        while self.current_token() in ('*', '/'):
            if self.current_token() == '*':
                self.advance()
                result *= self.factor()  # Perform multiplication
            elif self.current_token() == '/':
                self.advance()
                result /= self.factor()  # Perform division
        return result

    # Handle numbers and parentheses
    def factor(self):
        token = self.current_token()
        if token.isdigit():  # If it's a number
            self.advance()
            return int(token)
        elif token == '(':  # If it's a parenthesis
            self.advance()  # Skip the '('
            result = self.expr()  # Recursively parse the expression inside parentheses
            self.advance()  # Skip the ')'
            return result
        raise ValueError("Invalid syntax")  # Raise an error for any invalid token

# Example usage
if __name__ == "__main__":
    expr = "(3 + 5) * 2"  # You can change this to test other expressions

    parser = SimpleParser(expr)
    result = parser.parse()
    print(f"Result of '{expr}' is {result}")
    print('Deep Marathe - 53004230016')
