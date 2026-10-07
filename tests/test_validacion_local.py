"""Validación local: identidades analíticas, casos límite y reproducibilidad.

Los valores esperados se derivan a mano del canal y de los estados definidos en
docs/EXPERIMENTOS; ninguno se ajustó a la salida del código.
"""

import json
import warnings

import numpy as np
import pytest

from analytics.diagnostico import inferir_alpha
from analytics.events import DiarioIndividuacion
from analytics.qst import (
    fidelidad_densidad,
    medir_base_x,
    medir_base_y,
    medir_base_z,
    reconstruir_arquetipo,
    tomografia_bloch,
    tomografia_z,
)
from core.archetypes import RegistroCuantico
from core.experiments import Arquetipo, ParConDecoherencia
from core.interventions import (
    SesionTerapeutica,
    amplificacion,
    apertura_consciente,
    integracion_parcial,
    proyeccion,
)
from core.lindblad import ParConLindblad

XX = np.kron([[0, 1], [1, 0]], [[0, 1], [1, 0]])


def _fisico(rho, tol=1e-10):
    return (
        abs(np.trace(rho).real - 1) < tol
        and np.allclose(rho, rho.conj().T, atol=tol)
        and np.linalg.eigvalsh((rho + rho.conj().T) / 2).min() > -tol
    )


def _rho(arq):
    psi = np.array([arq.alpha, arq.beta], dtype=complex)
    return np.outer(psi, psi.conj())


# ───────────────────────── E001 ─────────────────────────

@pytest.mark.parametrize("alpha, beta, p_anima", [
    (3, 4, 9 / 25), (1, 1j, 0.5), (0.6, 0.8j, 0.36), (-2, 0, 1.0),
    (1e-200, 1e-200, 0.5), (1e200, 1e200, 0.5), (3e-200, 4e-200, 9 / 25),
])
def test_arquetipo_normaliza_amplitudes_reales_complejas_y_extremas(alpha, beta, p_anima):
    arq = Arquetipo(alpha, beta)
    assert np.isclose(abs(arq.alpha) ** 2 + abs(arq.beta) ** 2, 1.0)
    assert np.isclose(arq.prob_anima(), p_anima)


@pytest.mark.parametrize("alpha, beta", [(0, 0), (float("nan"), 1), (1, float("nan")), (float("inf"), 1), (1, complex("nanj"))])
def test_arquetipo_rechaza_amplitudes_no_fisicas(alpha, beta):
    with pytest.raises(ValueError):
        Arquetipo(alpha, beta)


def test_medicion_converge_a_la_probabilidad_teorica():
    arq = Arquetipo(0.8, 0.6, seed=5)
    n = 100_000
    frecuencia = np.mean([arq.medir() == 0 for _ in range(n)])
    assert abs(frecuencia - 0.64) < 4 * np.sqrt(0.64 * 0.36 / n)


def test_transformaciones_conservan_reproducibilidad_con_semilla():
    def secuencia(transformar):
        arq = transformar(Arquetipo(0.8, 0.6, seed=9))
        return [arq.medir() for _ in range(60)]

    for transformar in (
        lambda a: a.aplicar_rotacion(1.0),
        apertura_consciente,
        proyeccion,
        lambda a: amplificacion(a, polo=1, theta=0.4),
        lambda a: proyeccion(apertura_consciente(a)),
    ):
        assert secuencia(transformar) == secuencia(transformar)
    # Derivar un hijo no altera la secuencia del padre.
    padre_a, padre_b = Arquetipo(0.8, 0.6, seed=9), Arquetipo(0.8, 0.6, seed=9)
    padre_a.aplicar_rotacion(0.3)
    assert [padre_a.medir() for _ in range(60)] == [padre_b.medir() for _ in range(60)]


@pytest.mark.parametrize("theta", [0.0, 1e-9, np.pi / 3, np.pi, 2 * np.pi, 100.0, -0.7])
def test_ry_es_unitaria(theta):
    arq = Arquetipo(0.6, 0.8j)
    rotado = arq.aplicar_rotacion(theta)
    assert np.isclose(abs(rotado.alpha) ** 2 + abs(rotado.beta) ** 2, 1.0)
    # Ry(θ)·Ry(−θ) = I
    assert np.isclose(rotado.aplicar_rotacion(-theta).fidelidad(arq), 1.0)


def test_hadamard_es_involucion_y_no_siempre_equilibra():
    arq = Arquetipo(0.8, 0.6)
    assert np.isclose(apertura_consciente(apertura_consciente(arq)).fidelidad(arq), 1.0)
    assert np.isclose(apertura_consciente(Arquetipo(1, 0)).prob_anima(), 0.5)
    assert np.isclose(apertura_consciente(Arquetipo(1, 1)).prob_anima(), 1.0)


@pytest.mark.parametrize("alpha, beta", [(0.8, 0.6), (1 / np.sqrt(2), 1 / np.sqrt(2)), (0.6, 0.8)])
def test_amplificacion_aumenta_la_probabilidad_del_polo_elegido(alpha, beta):
    arq = Arquetipo(alpha, beta)
    theta = 0.3  # menor que la distancia angular a cualquiera de los polos
    assert amplificacion(arq, polo=0, theta=theta).prob_anima() > arq.prob_anima()
    assert amplificacion(arq, polo=1, theta=theta).prob_anima() < arq.prob_anima()
    with pytest.raises(ValueError):
        amplificacion(arq, polo=2)


def test_ry_pi_y_proyeccion_comparten_probabilidades_pero_no_estado():
    arq = Arquetipo(0.8, 0.6)
    girado, proyectado = integracion_parcial(arq, np.pi), proyeccion(arq)
    assert np.isclose(girado.prob_anima(), proyectado.prob_anima())
    assert np.isclose(girado.fidelidad(proyectado), (0.8**2 - 0.6**2) ** 2)


def test_bell_inicial():
    par = ParConDecoherencia(seed=7)
    assert _fisico(par.rho)
    assert np.isclose(par.negatividad(), 0.5)
    assert np.isclose(par.concurrencia(), 1.0)
    assert np.isclose(par.entropia_reducida(), 1.0)
    assert np.isclose(par.correlacion_teorica(), 1.0)
    assert np.isclose(np.trace(par.rho @ par.rho).real, 1.0)


# ───────────────────────── E002 / E003 ─────────────────────────

@pytest.mark.parametrize("gamma", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
def test_canal_de_desfase_coincide_con_la_solucion_analitica(gamma):
    par = ParConDecoherencia(seed=1)
    par.aplicar_represion(gamma)
    rho = par.rho
    assert _fisico(rho)
    assert np.isclose(abs(rho[0, 3]), (1 - gamma) / 2)             # coherencia
    assert np.isclose(np.trace(XX @ rho).real, 1 - gamma)           # ⟨X⊗X⟩
    assert np.isclose(par.correlacion_teorica(), 1 - gamma / 2)     # P(x1 = x2)
    assert np.isclose(par.negatividad(), (1 - gamma) / 2)
    assert np.isclose(par.concurrencia(), 1 - gamma)
    assert np.isclose(np.trace(rho @ rho).real, 1 - gamma + gamma**2 / 2)
    # E003: la entropía reducida no mide entrelazamiento de estados mixtos;
    # vale 1 bit para todo γ aunque el entrelazamiento desaparezca en γ = 1.
    assert np.isclose(par.entropia_reducida(), 1.0)


def test_metricas_de_entrelazamiento_son_monotonas_en_gamma():
    negatividades, concurrencias, correlaciones = [], [], []
    for gamma in np.linspace(0, 1, 41):
        par = ParConDecoherencia()
        par.aplicar_represion(float(gamma))
        negatividades.append(par.negatividad())
        concurrencias.append(par.concurrencia())
        correlaciones.append(par.correlacion_teorica())
    for serie in (negatividades, concurrencias, correlaciones):
        assert np.all(np.diff(serie) < 1e-12)


def test_desfase_total_deja_correlacion_clasica_sin_entrelazamiento():
    par = ParConDecoherencia()
    par.aplicar_represion(1.0)
    assert np.allclose(par.rho, np.diag([0.5, 0, 0, 0.5]))
    assert np.isclose(par.negatividad(), 0.0) and np.isclose(par.concurrencia(), 0.0)
    # Sigue habiendo correlación perfecta en base Z (estado clásicamente correlacionado).
    ZZ = np.kron(np.diag([1, -1]), np.diag([1, -1]))
    assert np.isclose(np.trace(ZZ @ par.rho).real, 1.0)
    assert np.isclose(par.correlacion_teorica(), 0.5)


def test_composicion_del_canal_de_desfase():
    par = ParConDecoherencia()
    par.aplicar_represion(0.3)
    par.aplicar_represion(0.5)
    assert np.isclose(abs(par.rho[0, 3]), (1 - 0.3) * (1 - 0.5) / 2)


@pytest.mark.parametrize("gamma", [0.0, 0.5, 1.0])
def test_monte_carlo_converge_a_la_prediccion(gamma):
    par = ParConDecoherencia(seed=3)
    par.aplicar_represion(gamma)
    n = 20_000
    observado = np.mean([(lambda r: r[0] == r[1])(par.medir_base_X()) for _ in range(n)])
    teorico = 1 - gamma / 2
    assert abs(observado - teorico) <= 4 * np.sqrt(max(teorico * (1 - teorico), 1e-12) / n)


def test_medicion_del_par_es_reproducible():
    a, b = ParConDecoherencia(seed=3), ParConDecoherencia(seed=3)
    a.aplicar_represion(0.3); b.aplicar_represion(0.3)
    assert [a.medir_base_X() for _ in range(100)] == [b.medir_base_X() for _ in range(100)]


@pytest.mark.parametrize("gamma", [-0.1, 1.1, float("inf")])
def test_gamma_fuera_de_rango(gamma):
    with pytest.raises(ValueError):
        ParConDecoherencia().aplicar_represion(gamma)


@pytest.mark.parametrize("gamma", [float("nan"), "0.5", None])
def test_gamma_no_numerico(gamma):
    with pytest.raises(TypeError):
        ParConDecoherencia().aplicar_represion(gamma)


@pytest.mark.parametrize("gamma", [np.float32(0.5), np.float64(0.5), np.int64(1), 1])
def test_gamma_acepta_reales_de_numpy(gamma):
    par = ParConDecoherencia()
    par.aplicar_represion(gamma)
    assert _fisico(par.rho)


# ───────────────────────── Lindblad ─────────────────────────

def _lindblad_analitico(g1, g2, t):
    rho = np.zeros((4, 4), dtype=complex)
    decaimiento = np.exp(-g1 * t)
    rho[0, 0] = 0.5
    rho[3, 3] = 0.5 * decaimiento
    rho[1, 1] = 0.5 * (1 - decaimiento)
    rho[0, 3] = rho[3, 0] = 0.5 * np.exp(-g1 * t / 2 - g2 * t)
    return rho


@pytest.mark.parametrize("g1", [0.0, 1e-6, 0.2, 1.0])
@pytest.mark.parametrize("g2", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("dt", [0.0, 1e-6, 0.1, 1.0, 10.0, 1000.0])
def test_lindblad_coincide_con_la_solucion_analitica(g1, g2, dt):
    for pasos in (1, 10, 1000):
        par = ParConLindblad(seed=1)
        par.aplicar_represion_lindblad(g1, g2, dt=dt, pasos=pasos)
        assert _fisico(par.rho)
        assert np.allclose(par.rho, _lindblad_analitico(g1, g2, dt), atol=1e-12)


def test_lindblad_es_un_semigrupo_e_independiente_de_pasos():
    a, b, c = ParConLindblad(), ParConLindblad(), ParConLindblad()
    a.aplicar_represion_lindblad(0.4, 0.7, dt=0.3); a.aplicar_represion_lindblad(0.4, 0.7, dt=0.7)
    b.aplicar_represion_lindblad(0.4, 0.7, dt=1.0, pasos=1)
    c.aplicar_represion_lindblad(0.4, 0.7, dt=1.0, pasos=5000)
    assert np.allclose(a.rho, b.rho, atol=1e-12)
    assert np.array_equal(b.rho, c.rho)


def test_lindblad_no_equivale_al_canal_de_kraus_con_la_misma_gamma():
    for gamma in (0.1, 0.5, 0.9):
        kraus, igual, ajustado = ParConDecoherencia(), ParConLindblad(), ParConLindblad()
        kraus.aplicar_represion(gamma)
        igual.aplicar_represion_lindblad(0.0, gamma)
        assert not np.allclose(kraus.rho, igual.rho)
        assert np.isclose(abs(igual.rho[0, 3]), np.exp(-gamma) / 2)
        # Coinciden con γ₂·dt = −ln(1 − γ).
        ajustado.aplicar_represion_lindblad(0.0, 1.0, dt=-np.log(1 - gamma))
        assert np.allclose(kraus.rho, ajustado.rho, atol=1e-12)


def test_lindblad_limite_de_tiempo_largo():
    par = ParConLindblad()
    par.aplicar_represion_lindblad(1.0, 0.0, dt=1000.0)
    assert np.allclose(par.rho, np.diag([0.5, 0.5, 0, 0]), atol=1e-9)
    assert np.isclose(par.negatividad(), 0.0, atol=1e-9)


@pytest.mark.parametrize("args, error", [
    ((-0.1, 0.0), ValueError), ((0.0, 1.1), ValueError), ((float("nan"), 0.0), TypeError),
    ((0.0, float("inf")), TypeError), (("a", 0.0), TypeError), ((0.5, 0.5, -1.0), ValueError),
    ((0.5, 0.5, float("nan")), TypeError), ((0.5, 0.5, 1.0, 0), ValueError),
    ((0.5, 0.5, 1.0, 2.5), ValueError), ((0.5, 0.5, 1.0, -3), ValueError),
])
def test_lindblad_rechaza_parametros_invalidos(args, error):
    with pytest.raises(error):
        ParConLindblad().aplicar_represion_lindblad(*args)


# ───────────────────────── E004: tomografía ─────────────────────────

ESTADOS = {
    "real": (0.8, 0.6),
    "base": (1, 0),
    "mas": (1, 1),
    "complejo": (0.6, 0.8j),
    "mas_i": (1, 1j),
    "menos_i": (1, -1j),
}


@pytest.mark.parametrize("nombre", list(ESTADOS))
def test_probabilidades_de_medicion_en_las_tres_bases(nombre):
    arq = Arquetipo(*ESTADOS[nombre])
    rho = _rho(arq)
    n = 40_000
    esperado = {
        "z": rho[0, 0].real,
        "x": 0.5 * (1 + 2 * rho[0, 1].real),          # (1 + ⟨X⟩)/2
        "y": 0.5 * (1 - 2 * rho[0, 1].imag),          # (1 + ⟨Y⟩)/2, ⟨Y⟩ = −2·Im ρ01
    }
    observado = {
        "z": np.mean(np.array(medir_base_z(arq, n, seed=1)) == 0),
        "x": np.mean(np.array(medir_base_x(arq, n, seed=2)) == 0),
        "y": np.mean(np.array(medir_base_y(arq, n, seed=3)) == 0),
    }
    for base in "zxy":
        p = esperado[base]
        assert abs(observado[base] - p) <= 4 * np.sqrt(max(p * (1 - p), 1e-12) / n) + 1e-12


@pytest.mark.parametrize("nombre", list(ESTADOS))
def test_tomografia_reconstruye_un_estado_fisico_y_converge(nombre):
    arq = Arquetipo(*ESTADOS[nombre])
    real = _rho(arq)
    errores, fidelidades = {}, {}
    for shots in (100, 1000, 10000):
        err, fid = [], []
        for s in range(20):
            rho = tomografia_bloch(
                medir_base_z(arq, shots, seed=3 * s),
                medir_base_x(arq, shots, seed=3 * s + 1),
                medir_base_y(arq, shots, seed=3 * s + 2),
            )["rho_rec"]
            assert _fisico(rho)
            err.append(np.linalg.norm(rho - real))
            fid.append(fidelidad_densidad(rho, real))
        errores[shots], fidelidades[shots] = np.mean(err), np.mean(fid)
    assert errores[100] > errores[1000] > errores[10000]
    assert fidelidades[100] < fidelidades[1000] < fidelidades[10000]
    assert fidelidades[10000] > 0.995
    # Escala Bernoulli: el error cae aproximadamente como 1/√shots.
    assert 0.5 < (errores[100] / errores[10000]) / 10 < 2.0


def test_tomografia_recupera_la_fase_compleja():
    arq = Arquetipo(0.6, 0.8j)
    resultado = tomografia_bloch(
        medir_base_z(arq, 20000, seed=1), medir_base_x(arq, 20000, seed=2), medir_base_y(arq, 20000, seed=3)
    )
    assert abs(resultado["ry"] - 0.96) < 0.03      # 2·Im(α*β)
    assert abs(resultado["rx"]) < 0.03
    reconstruido = reconstruir_arquetipo(resultado)
    assert reconstruido.fidelidad(arq) > 0.99
    assert len(medir_base_x(reconstruido, 10, seed=1)) == 10


def test_tomografia_es_reproducible_y_usa_semillas_independientes():
    arq = Arquetipo(1, 1)
    assert medir_base_x(arq, 50, seed=1) == medir_base_x(arq, 50, seed=1)
    assert medir_base_z(arq, 200, seed=1) != medir_base_z(arq, 200, seed=2)
    assert sum(medir_base_x(arq, 200, seed=4)) == 0          # P(+) = 1 para |+⟩
    assert 60 < sum(medir_base_z(arq, 200, seed=4)) < 140    # P(0) = 0.5


@pytest.mark.parametrize("args", [([], [0, 1], [0]), ([0, 1], [], [0]), ([0, 1], [0, 1], []), ([2, 3, 2], [0, 1], [0]), ([0, 1], [0, 0.5], None)])
def test_tomografia_rechaza_observaciones_invalidas(args):
    with pytest.raises(ValueError):
        tomografia_bloch(*args)


def test_tomografia_z_valida_y_estima():
    assert tomografia_z([0] * 64 + [1] * 36)["p_mle"] == 0.64
    for malas in ([], [0, 2]):
        with pytest.raises(ValueError):
            tomografia_z(malas)


def test_fidelidad_de_uhlmann_propiedades():
    a, b = _rho(Arquetipo(0.8, 0.6)), _rho(Arquetipo(0.6, -0.8))
    mixto = np.eye(2) / 2
    assert np.isclose(fidelidad_densidad(a, a), 1.0)
    assert np.isclose(fidelidad_densidad(a, b), 0.0, atol=1e-10)
    assert np.isclose(fidelidad_densidad(a, mixto), 0.5)
    assert np.isclose(fidelidad_densidad(a, mixto), fidelidad_densidad(mixto, a))


# ───────────────────────── Analítica ─────────────────────────

def test_inferencia_bayesiana_posterior_beta():
    r = inferir_alpha([0] * 64 + [1] * 36)
    assert np.isclose(r["p_media"], 65 / 102, atol=1e-4)      # Beta(65, 37)
    assert np.isclose(r["p_moda"], 0.64, atol=1e-4)
    assert r["IC_95_p"][0] < 0.64 < r["IC_95_p"][1]
    for malas in ([], [0, 1, 2], [0.3]):
        with pytest.raises(ValueError):
            inferir_alpha(malas)
    with pytest.raises(ValueError):
        inferir_alpha([0, 1], prior_a=0)


def test_intervalo_de_credibilidad_tiene_cobertura_nominal():
    rng = np.random.default_rng(0)
    p, n, repeticiones = 0.64, 200, 1500
    dentro = 0
    for _ in range(repeticiones):
        ic = inferir_alpha(list((rng.random(n) >= p).astype(int)))["IC_95_p"]
        dentro += ic[0] <= p <= ic[1]
    assert abs(dentro / repeticiones - 0.95) < 0.02


def test_registro_cuantico_es_reproducible_y_sus_indices_estan_acotados():
    a, b = RegistroCuantico(seed=5), RegistroCuantico(seed=5)
    assert [a.medir_todo() for _ in range(20)] == [b.medir_todo() for _ in range(20)]
    grafo = a.grafo_sincronicidad()
    assert np.allclose(grafo, grafo.T) and np.allclose(np.diag(grafo), 1.0)
    assert 0.0 <= grafo.min() and grafo.max() <= 1.0 + 1e-12
    assert 0.0 <= a.indice_individuacion() <= 1.0
    assert 0.0 <= a.tension_yo_sombra() <= 1.0


def test_sesion_serializa_estados_reales_y_complejos():
    sesion = SesionTerapeutica(Arquetipo(0.9, 0.436))
    for nombre in ("apertura_consciente", "integracion_parcial", "proyeccion"):
        sesion.aplicar(nombre)
    datos = json.loads(sesion.exportar_json())
    assert len(datos["historial"]) == 4
    assert isinstance(datos["historial"][0]["alpha"], float)
    compleja = SesionTerapeutica(Arquetipo(0.6, 0.8j))
    compleja.aplicar("proyeccion")
    assert json.loads(compleja.exportar_json())["historial"][0]["beta"] == [0.0, 0.8]
    with pytest.raises(ValueError):
        sesion.aplicar("intervencion_inexistente")
    with pytest.raises(TypeError):
        sesion.aplicar("proyeccion", theta=1.0)


def test_diario_persiste_y_reproduce_eventos(tmp_path):
    ruta = tmp_path / "diario.jsonl"
    diario = DiarioIndividuacion(str(ruta))
    for i in range(3):
        sesion = SesionTerapeutica(Arquetipo(0.9, 0.436))
        sesion.aplicar("integracion_parcial", theta=0.2 * (i + 1))
        diario.registrar(sesion, "paciente_a")
    otra = SesionTerapeutica(Arquetipo(0.5, 0.866))
    otra.aplicar("proyeccion")
    diario.registrar(otra, "paciente_b")

    releido = DiarioIndividuacion(str(ruta))
    assert len(releido.replay()) == 4
    assert releido.pacientes() == ["paciente_a", "paciente_b"]
    assert len(releido.trayectoria("paciente_a")) == 3
    assert releido.ultima_sesion("inexistente") is None
    assert releido.estadisticas_longitudinales("paciente_a")["sesiones"] == 3
    assert sum(1 for _ in ruta.open(encoding="utf-8")) == 4      # sólo se agrega


def test_no_hay_warnings_numericos_en_el_recorrido_basico():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        par = ParConLindblad(seed=1)
        par.aplicar_represion_lindblad(0.5, 0.5)
        par.metricas()
        ParConDecoherencia(seed=1).aplicar_represion(1.0)
        Arquetipo(1, 0).entropia_shannon()
