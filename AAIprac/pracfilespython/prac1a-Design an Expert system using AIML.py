import aiml

# Create a Kernel instance
kernel = aiml.Kernel()

# Load AIML files
kernel.learn("C:\\Users\\deepm\\Deep\\msprac\\AAIJounral\\flu.aiml")

# Loop to interact with the expert system
print("Expert System for Identifying Flu Symptoms")
print("Type 'bye' to exit the conversation.")

while True:
    user_input = input("You: ")
    
    # Exit the conversation if the user types 'bye'
    if user_input.lower() == "bye":
        print("System: Goodbye! Stay healthy.")
        break

    # Get the system's response
    response = kernel.respond(user_input.upper())
    
    # Print the system's response
    print(f"System: {response}")

print('Deep Marathe -53004230016')
