import sage.all

# processing generating function: inverse laplace
def inverse_laplace(equation, dummy_variable):
    return (
        sage.all.inverse_laplace(
            subequation / dummy_variable,
            dummy_variable,
            sage.all.SR.var("T", domain="real"),
            algorithm="giac",
        )
        for subequation in equation
    )


def return_inverse_laplace(equation, dummy_variable):
    if dummy_variable is not None:
        return sage.all.inverse_laplace(
            equation / dummy_variable,
            dummy_variable,
            sage.all.SR.var("T", domain="real"),
            algorithm="giac",
        )
    else:
        return equation
