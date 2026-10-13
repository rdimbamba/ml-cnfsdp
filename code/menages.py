import numpy as np, pandas as pd
def simuler_menages(n=2000, graine=0):
    """Ménages fictifs au format des biens DHS : 1 = le ménage possède le bien."""
    rng = np.random.default_rng(graine)
    urbain = rng.random(n) < 0.4
    richesse = rng.standard_normal(n) + 1.2 * urbain            # richesse latente (inobservée)
    sig = lambda t: 1 / (1 + np.exp(-t))
    biens = {  # (constante, effet richesse, effet urbain)
        "electricite": (-0.5, 2.0, 1.5), "eau_courante": (-1.5, 1.8, 1.5), "toilettes": (-1.0, 1.6, 0.8),
        "sol_fini": (-0.3, 1.8, 0.8), "radio": (0.3, 0.8, 0.0), "television": (-1.5, 2.2, 0.8),
        "refrigerateur": (-2.8, 2.2, 0.5), "telephone": (1.0, 1.5, 0.5), "moto": (-1.5, 0.8, -0.5),
        "voiture": (-4.5, 2.0, 0.3), "terre_agricole": (0.8, -0.3, -2.5), "betail": (0.0, -0.2, -2.0)}
    df = pd.DataFrame({b: (rng.random(n) < sig(a + r * richesse + u * urbain)).astype(int)
                       for b, (a, r, u) in biens.items()})
    return df, richesse, urbain
