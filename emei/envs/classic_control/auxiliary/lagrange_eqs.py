from sympy import symbols, diff, sin, cos, simplify, nsimplify, solve, trigsimp
from sympy.physics.mechanics import dynamicsymbols
from sympy.solvers.solveset import linsolve
from IPython.display import display, Latex
from sympy.physics.vector.printing import vlatex


def eq_simplify(s):
    return simplify(trigsimp(s))


def cartpole(n):
    # constants
    g_acc = symbols("g", real=True)
    force = symbols("F", real=True)
    cart_m = symbols("M".format(n), real=True)
    pole_ms = symbols("m:{}".format(n), real=True)
    pole_lens = symbols(
        "l:{}".format(n), real=True
    )  # actually a half of the real lengths

    cart_x = dynamicsymbols("x")
    cart_x_dot = dynamicsymbols("x", 1)
    pole_thetas = dynamicsymbols(r"\theta_:{}".format(n))
    pole_thetas_dot = dynamicsymbols(r"\theta_:{}".format(n), 1)
    # initialization of kinetic energy and potential energy
    kin_energy = 1 / 2 * cart_m * cart_x_dot**2
    pot_energy = -force * cart_x

    pole_x_pos = cart_x
    pole_y_pos = 0
    vertical_theta = 0
    for i in range(n):
        vertical_theta += pole_thetas[i]
        pole_x_pos += pole_lens[i] * sin(vertical_theta)
        pole_y_pos += pole_lens[i] * cos(vertical_theta)
        inertia = 1 / 3 * pole_ms[i] * pole_lens[i] ** 2
        kin_energy += (
            1
            / 2
            * pole_ms[i]
            * (diff(pole_x_pos, "t") ** 2 + diff(pole_y_pos, "t") ** 2)
            + 1 / 2 * inertia * diff(vertical_theta, "t") ** 2
        )
        pot_energy += pole_ms[i] * g_acc * pole_lens[i] * cos(vertical_theta)
        pole_x_pos += pole_lens[i] * sin(vertical_theta)
        pole_y_pos += pole_lens[i] * cos(vertical_theta)
    # solve the Lagrange equation
    lagrange_func = kin_energy - pot_energy

    equations = []
    eq = diff(diff(lagrange_func, cart_x_dot), "t") - diff(lagrange_func, cart_x)
    equations.append(eq_simplify(eq))
    for i in range(n):
        eq = diff(diff(lagrange_func, pole_thetas_dot[i]), "t") - diff(
            lagrange_func, pole_thetas[i]
        )
        equations.append(eq_simplify(eq))
    # return the differential equations and the dynamic-symbols
    return equations, (cart_x, *pole_thetas)


def get_canonical_diff_equations(equations, dynamic_symbols):
    # calculate the second derivative of dynamic-symbols
    second_diff_symbols = [diff(diff(symbol, "t"), "t") for symbol in dynamic_symbols]
    # solve the linear equations
    solutions = solve(equations, second_diff_symbols, simplify=False, rational=False)
    diff_eqs = list([nsimplify(simplify(formula)) for formula in solutions.values()])
    return diff_eqs


if __name__ == "__main__":
    lag_eqs, dyna_syms = cartpole(1)
    for eq in lag_eqs:
        print(vlatex(nsimplify(simplify(eq))))
    diff_eqs = get_canonical_diff_equations(lag_eqs, dyna_syms)
    for eq in diff_eqs:
        print(nsimplify(simplify(eq)))
