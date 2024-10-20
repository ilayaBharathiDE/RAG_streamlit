import streamlit as st

# Simple Calculator Functions
def add(x, y):
    return x + y

def subtract(x, y):
    return x - y

def multiply(x, y):
    return x * y

def divide(x, y):
    if y != 0:
        return x / y
    else:
        return "Error! Division by zero."

# Streamlit Application
st.title("Dharani's Calculator")

# Input Fields
num1 = st.number_input("Enter first number", value=0.0)
num2 = st.number_input("Enter second number", value=0.0)

# Operation Selection
operation = st.selectbox("Select operation", ["Add", "Subtract", "Multiply", "Divide"])

# Calculate and Display Result
if st.button("Calculate"):
    if operation == "Add":
        result = add(num1, num2)
        st.write(f"The result of {num1} + {num2} is: {result}")
    elif operation == "Subtract":
        result = subtract(num1, num2)
        st.write(f"The result of {num1} - {num2} is: {result}")
    elif operation == "Multiply":
        result = multiply(num1, num2)
        st.write(f"The result of {num1} * {num2} is: {result}")
    elif operation == "Divide":
        result = divide(num1, num2)
        st.write(f"The result of {num1} / {num2} is: {result}")

# Run the app
if __name__ == "__main__":
    st.write("Welcome to the Streamlit Calculator!")
