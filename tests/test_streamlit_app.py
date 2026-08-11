"""Comprobación de que la app de Streamlit se ejecuta de principio a fin.

No prueba la apariencia, sino que el script corre sin excepciones y que el aviso
de backend caído aparece: un fallo de sintaxis o un uso incorrecto de la API de
Streamlit solo se manifiesta al ejecutar el script, no al importarlo.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "frontend" / "streamlit_app.py"

#: Puerto reservado al servicio "discard": nada escucha ahí, así que la llamada
#: de salud falla al instante y sin salir de la máquina.
UNREACHABLE_API = "http://127.0.0.1:9/api/v1"


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> AppTest:
    monkeypatch.setenv("RAG_API_URL", UNREACHABLE_API)
    import streamlit as st

    st.cache_resource.clear()  # el cliente se cachea entre ejecuciones
    return AppTest.from_file(str(APP), default_timeout=30)


def test_the_app_runs_without_raising(app: AppTest) -> None:
    app.run()

    assert not app.exception


def test_the_title_and_purpose_are_shown(app: AppTest) -> None:
    app.run()

    assert "Asistente RAG de Nicho" in app.title[0].value
    assert "citando documento y página" in app.caption[0].value


def test_a_down_backend_is_reported_instead_of_crashing(app: AppTest) -> None:
    app.run()

    assert any("API no disponible" in error.value for error in app.error)


def test_the_question_box_is_available(app: AppTest) -> None:
    app.run()

    assert app.chat_input
