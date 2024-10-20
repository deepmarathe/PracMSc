class MathTutor:
    def __init__(self):
        # Define basic operations
        self.operations = {
            '+': lambda a, b: a + b,
            '-': lambda a, b: a - b,
            '*': lambda a, b: a * b,
            '/': lambda a, b: a / b if b != 0 else 'undefined'  # Handles division by zero
        }

    # Provide explanations for operations
    def explain_operation(self, operator):
        explanation = {
            '+': "Addition adds two numbers together.",
            '-': "Subtraction subtracts the second number from the first.",
            '*': "Multiplication gives the product of two numbers.",
            '/': "Division divides the first number by the second. Division by zero is undefined.",
        }
        return explanation.get(operator, "Invalid operation.")

    # Perform the specified operation
    def perform_operation(self, operator, a, b):
        if operator in self.operations:
            return self.operations[operator](a, b)
        else:
            return None

# Example usage:
if __name__ == "__main__":
    tutor = MathTutor()

    # Example operation
    operator = '/'  # You can test with '+', '-', '*', or '/'
    a, b = 10, 5

    # Display explanation and result
    print(tutor.explain_operation(operator))
    result = tutor.perform_operation(operator, a, b)
    print(f"Result of {a} {operator} {b} = {result}")

    print('Deep Marathe - 53004230016')
