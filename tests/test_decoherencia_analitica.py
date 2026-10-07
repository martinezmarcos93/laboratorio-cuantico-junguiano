import numpy as np

from core.experiments import ParConDecoherencia


def test_correlacion_teorica_sigue_la_prediccion_analitica():
    for gamma in np.linspace(0.0, 1.0, 11):
        par = ParConDecoherencia(seed=123)
        par.aplicar_represion(float(gamma))
        esperada = 1.0 - float(gamma) / 2.0
        assert np.isclose(par.correlacion_teorica(), esperada, atol=1e-12)


def test_correlacion_teorica_sin_ruido_es_uno():
    par = ParConDecoherencia(seed=123)
    assert np.isclose(par.correlacion_teorica(), 1.0, atol=1e-12)


def test_correlacion_teorica_con_decoherencia_total_es_un_medio():
    par = ParConDecoherencia(seed=123)
    par.aplicar_represion(1.0)
    assert np.isclose(par.correlacion_teorica(), 0.5, atol=1e-12)
