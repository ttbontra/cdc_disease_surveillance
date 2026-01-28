import numpy as np
from cdc_platform.modeling.seir.model import SEIRParams, simulate_seir

def test_seir_conserves_population():
    pop = 1_000_000
    params = SEIRParams(beta=0.3, sigma=1/4, gamma=1/6)
    sim = simulate_seir(pop=pop, S0=pop-10, E0=5, I0=5, R0=0, params=params, days=30)
    total = sim["S"] + sim["E"] + sim["I"] + sim["R"]
    assert np.allclose(total, pop, atol=1e-3)
