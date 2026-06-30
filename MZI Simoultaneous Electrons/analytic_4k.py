import sympy as sp

# Define symbols
U, k_p, k_m = sp.symbols('U k_p k_m', complex=True)
p_1, p_2, p_3, p_4 = sp.symbols('p_1 p_2 p_3 p_4', real=True)
A_0, A_1, A_2, A_3, A_4 = sp.symbols('A_0 A_1 A_2 A_3 A_4', complex=True)

# Define the complex amplitude
term = (U * (k_p + k_m) + sp.exp(sp.I * p_1) * (k_p - k_m)) * (U * (k_p + k_m) + sp.exp(sp.I * p_2) * (k_p - k_m)) * (U * (k_p + k_m) + sp.exp(sp.I * p_3) * (k_p - k_m)) * (U * (k_p + k_m) + sp.exp(sp.I * p_4) * (k_p - k_m))

# Expand the term
expanded = sp.expand(term)

# Get coefficients and collect by U
coeff_kp4 = sp.collect(expanded.coeff(k_p**4), U)
coeff_kp3km = sp.collect(expanded.coeff(k_p**3 * k_m), U)
coeff_kp2km2 = sp.collect(expanded.coeff(k_p**2 * k_m**2), U)
coeff_kpkm3 = sp.collect(expanded.coeff(k_p * k_m**3), U)
coeff_km4 = sp.collect(expanded.coeff(k_m**4), U)

# Print coefficients for specific terms
print(f"\nCoefficient of k_p^4: {coeff_kp4}")
print(f"Coefficient of k_p^3 * k_m: {coeff_kp3km}")
print(f"Coefficient of k_p^2 * k_m^2: {coeff_kp2km2}")
print(f"Coefficient of k_p * k_m^3: {coeff_kpkm3}")
print(f"Coefficient of k_m^4: {coeff_km4}")

# Replace U**i with A_i
coeff_kp4_new = coeff_kp4.subs([(U**4, A_4), (U**3, A_3), (U**2, A_2), (U, A_1)])
coeff_kp3km_new = coeff_kp3km.subs([(U**4, A_4), (U**3, A_3), (U**2, A_2), (U, A_1)])
coeff_kp2km2_new = coeff_kp2km2.subs([(U**4, A_4), (U**3, A_3), (U**2, A_2), (U, A_1)])
coeff_kpkm3_new = coeff_kpkm3.subs([(U**4,  A_4), (U**3, A_3), (U**2, A_2), (U, A_1)])
coeff_km4_new = coeff_km4.subs([(U**4, A_4), (U**3, A_3), (U**2, A_2), (U, A_1)])

# Define the amplitudes
coeffs = {
    'amplitude_4kp': coeff_kp4_new,
    'amplitude_3kp_km': coeff_kp3km_new,
    'amplitude_2kp_km2': coeff_kp2km2_new,
    'amplitude_kp_km3': coeff_kpkm3_new,
    'amplitude_4km': coeff_km4_new
}

for name, expr in coeffs.items():
    print(f"\n▶ {name}")
    # Compute squared magnitude
    squared = sp.expand(expr * sp.conjugate(expr))
    squared = sp.simplify(squared)
    print("Squared magnitude:")
    print(f"${sp.latex(squared)}$")
