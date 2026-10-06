import numpy as np

from core.experiments import ParConDecoherencia


def _propiedades_fisicas(rho):
    return (
        np.isclose(np.trace(rho).real, 1.0, atol=1e-10)
        and np.allclose(rho, rho.conj().T, atol=1e-10)
        and np.min(np.linalg.eigvalsh(rho)) >= -1e-10
    )


def test_decoherencia_conserva_estado_fisico():
    for gamma in np.linspace(0, 1, 11):
        par = ParConDecoherencia(seed=1)
        par.aplicar_represion(float(gamma))
        assert _propiedades_fisicas(par.rho)


def test_correlacion_teorica_decrece_con_gamma():
    valores = []
    for gamma in np.linspace(0, 1, 11):
        par = ParConDecoherencia(seed=1)
        par.aplicar_represion(float(gamma))
        valores.append(par.correlacion_teorica())
    assert np.all(np.diff(valores) <= 1e-10)
    assert np.isclose(valores[0], 1.0, atol=1e-10)


def test_desfase_total_elimina_entrelazamiento():
    par = ParConDecoherencia(seed=1)
    par.aplicar_represion(1.0)
    assert np.isclose(par.negatividad(), 0.0, atol=1e-10)
    assert np.isclose(par.concurrencia(), 0.0, atol=1e-10)
