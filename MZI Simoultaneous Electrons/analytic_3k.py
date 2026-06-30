import sympy as sp

# Define symbols
U, k_p, k_m = sp.symbols('U k_p k_m', complex=True)
p_1, p_2, p_3 = sp.symbols('p_1 p_2 p_3', real=True)
A_0, A_1, A_2, A_3 = sp.symbols('A_0 A_1 A_2 A_3', complex=True)


# Define the complex amplitude
term = (U * (k_p + k_m) + sp.exp(sp.I * p_1) * (k_p - k_m)) * (U * (k_p + k_m) + sp.exp(sp.I * p_2) * (k_p - k_m)) * (U * (k_p + k_m) + sp.exp(sp.I * p_3) * (k_p - k_m))

# Expand the term
expanded = sp.expand(term)

# Get coefficients and collect by U
coeff_kp3 = sp.collect(expanded.coeff(k_p**3), U)
coeff_kp2km = sp.collect(expanded.coeff(k_p**2 * k_m), U)
coeff_kpkm2 = sp.collect(expanded.coeff(k_p * k_m**2), U)
coeff_km3 = sp.collect(expanded.coeff(k_m**3), U)

# Print coefficients for specific terms
print(f"\nCoefficient of k_p^3: {coeff_kp3}")
print(f"Coefficient of k_p^2 * k_m: {coeff_kp2km}")
print(f"Coefficient of k_p * k_m^2: {coeff_kpkm2}")
print(f"Coefficient of k_m^3: {coeff_km3}")


# Replace U**i with A_i
coeff_kp3_new = coeff_kp3.subs([(U**3, A_3), (U**2, A_2), (U, A_1)])
coeff_kp2km_new = coeff_kp2km.subs([(U**3, A_3), (U**2, A_2), (U, A_1)])
coeff_kpkm2_new = coeff_kpkm2.subs([(U**3, A_3), (U**2, A_2), (U, A_1)])
coeff_km3_new = coeff_km3.subs([(U**3, A_3), (U**2, A_2), (U, A_1)])

# Define the amplitudes
coeffs = {
    'amplitude_3kp': coeff_kp3_new,
    'amplitude_2kp_km': coeff_kp2km_new,
    'amplitude_kp_2km': coeff_kpkm2_new,
    'amplitude_3km': coeff_km3_new
}

for name, expr in coeffs.items():
    print(f"\n▶ {name}")
    # Compute squared magnitude
    squared = sp.expand(expr * sp.conjugate(expr))
    squared = sp.simplify(squared)
    print("Squared magnitude:")
    print(f"${sp.latex(squared)}$")
