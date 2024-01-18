import sympy


# processing generating function: inverse laplace
def inverse_laplace_sympy(equation, dummy_variable):
    return (
        sympy.integrals.transforms.inverse_laplace_transform(
            subequation / dummy_variable,
            dummy_variable,
            sympy.symbols("T", real=True, positive=True),
            noconds=True,
        )
        for subequation in equation
    )


def return_inverse_laplace_sympy(equation, dummy_variable, T):
    if dummy_variable is not None:
        return sympy.integrals.transforms.inverse_laplace_transform(
            equation / dummy_variable, dummy_variable, T, noconds=True
        )
    else:
        return equation
