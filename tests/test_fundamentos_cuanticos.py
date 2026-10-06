import numpy as np

from core.experiments import Arquetipo, ParConDecoherencia
from analytics.qst import medir_base_z, medir_base_x, medir_base_y, tomografia_bloch, fidelidad_densidad


def test_arquetipo_normaliza_y_probabilidad():
    arq = Arquetipo(3, 4)
    assert np.isclose(abs(arq.alpha) ** 2 + abs(arq.beta) ** 2, 1.0)
    assert np.isclose(arq.prob_anima(), 9 / 25)


def test_rotacion_ry_preserva_norma():
    arq = Arquetipo(0.8, 0.6)
    rotado = arq.aplicar_rotacion(np.pi / 3)
    assert np.isclose(abs(rotado.alpha) ** 2 + abs(rotado.beta) ** 2, 1.0)


def test_bell_tiene_entrelazamiento_inicial():
    par = ParConDecoherencia(seed=7)
    assert np.isclose(par.negatividad(), 0.5, atol=1e-10)
    assert np.isclose(par.concurrencia(), 1.0, atol=1e-10)


def test_desfase_completo_elimina_entrelazamiento():
    par = ParConDecoherencia(seed=7)
    par.aplicar_represion(1.0)
    assert np.isclose(par.negatividad(), 0.0, atol=1e-10)
    assert np.isclose(par.concurrencia(), 0.0, atol=1e-10)


def test_tomografia_reconstruye_estado_real_aproximadamente():
    arq = Arquetipo(0.8, 0.6)
    obs_z = medir_base_z(arq, 5000, seed=10)
    obs_x = medir_base_x(arq, 5000, seed=11)
    obs_y = medir_base_y(arq, 5000, seed=12)
    resultado = tomografia_bloch(obs_z, obs_x, obs_y)

    rho_real = np.array([
        [abs(arq.alpha) ** 2, arq.alpha * np.conj(arq.beta)],
        [arq.beta * np.conj(arq.alpha), abs(arq.beta) ** 2],
    ], dtype=complex)

    assert fidelidad_densidad(resultado["rho_rec"], rho_real) > 0.98
