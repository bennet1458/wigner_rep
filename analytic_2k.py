import sympy as sp

# Define symbols
U, k_p, k_m = sp.symbols('U k_p k_m', complex=True)
p_1, p_2 = sp.symbols('p_1 p_2', real=True)
A_0, A_1, A_2 = sp.symbols('A_0 A_1 A_2', complex=True)


# Define the complex amplitude
term = (U * (k_p + k_m) + sp.exp(sp.I * p_1) * (k_p - k_m)) * (U * (k_p + k_m) + sp.exp(sp.I * p_2) * (k_p - k_m))

# Expand the term
expanded = sp.expand(term)

# Get coefficients and collect by U
coeff_kp2 = sp.collect(expanded.coeff(k_p**2), U)
coeff_kpkm = sp.collect(expanded.coeff(k_p * k_m), U)
coeff_km2 = sp.collect(expanded.coeff(k_m**2), U)

# Print coefficients for specific terms
print(f"\nCoefficient of k_p^2: {coeff_kp2}")
print(f"Coefficient of k_p * k_m: {coeff_kpkm}")
print(f"Coefficient of k_m^2: {coeff_km2}")

# Replace U**i with A_i
coeff_kp2_new = coeff_kp2.subs([(U**2, A_2), (U, A_1)])
coeff_kpkm_new = coeff_kpkm.subs([(U**2, A_2), (U, A_1)])
coeff_km2_new = coeff_km2.subs([(U**2, A_2), (U, A_1)])

# Define the amplitudes
coeffs = {
    'amplitude_2kp': coeff_kp2_new,
    'amplitude_kp_km': coeff_kpkm_new,
    'amplitude_kp_2km': coeff_km2_new
}

for name, expr in coeffs.items():
    print(f"\n▶ {name}")
    # Compute squared magnitude
    squared = sp.expand(expr * sp.conjugate(expr))
    squared = sp.simplify(squared)
    print("Squared magnitude:")
    print(f"${sp.latex(squared)}$")
