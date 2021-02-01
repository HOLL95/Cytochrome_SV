import sympy
from sympy import symbols, Function, dsolve, Derivative, Eq
vars=["Cdl", "Ru", "t"]
cdl, ru, t=symbols((" ").join(vars))
i=Function("I")
dE=Function("dE")
eq=Eq(Derivative(i(t), t), (sympy.sin(t)/ru)-(i(t)/(ru*cdl)))
sol=dsolve(eq)
print(sol)
