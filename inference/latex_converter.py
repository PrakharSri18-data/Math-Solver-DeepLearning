# inference/latex_converter.py
# -------------------------------------------------
# Author : Prakhar Srivastava
# Date : 2026-03-10
# Description : This script contains a function to convert the predicted equations into LaTeX format for better readability.
# -------------------------------------------------


# =============================================
# to_latex Function
# ----------------------------------------
# This function takes a predicted equation as input, replaces certain characters with their LaTeX equivalents, & returns the equation in LaTeX format.
# It replaces '*' with '\\times' for multiplication & '/' with '\\div' for division, & wraps the equation in '$' symbols to indicate that it's a LaTeX math expression.
# =============================================
def to_latex(eq):

    # Replacing '*' with '\\times' for multiplication & '/' with '\\div' for division
    eq = eq.replace("*","\\times")
    eq = eq.replace("/","\\div")
 
    # Wrapping the equation in '$' symbols to indicate that it's a LaTeX math expression
    latex = f"${eq}$"

    return latex