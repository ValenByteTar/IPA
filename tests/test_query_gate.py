"""Tests for the deterministic query gate (chat retrieval gating)."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ipa.agent.query_gate import classify_message, is_imperative  # noqa: E402


def test_identity_questions_skip_retrieval():
    assert classify_message("Comentame, como te sentis?") == "identity"
    assert classify_message("¿Cómo te sentís?") == "identity"
    assert classify_message("¿Quién sos?") == "identity"
    assert classify_message("¿Qué podés hacer?") == "identity"
    assert classify_message("¿Tenés sentimientos?") == "identity"
    assert classify_message("¿Sos un robot?") == "identity"
    assert classify_message("¿Quién te creó?") == "identity"
    assert classify_message("Contame de vos") == "identity"


def test_greetings_skip_retrieval():
    assert classify_message("hola") == "greeting"
    assert classify_message("Gracias!") == "greeting"
    assert classify_message("dale") == "greeting"


def test_commands_skip_retrieval():
    assert classify_message("investiga sobre RAG agéntico") == "command"
    assert classify_message("corre la ingesta") == "command"
    assert classify_message("cual es el status del sistema?") == "command"


def test_knowledge_questions_run_retrieval():
    assert classify_message("Que sabes de GPT6 Astra?") == "knowledge"
    assert classify_message("Podes generar mapas de estudio?") == "knowledge"
    assert classify_message("¿Qué es retrieval augmented generation?") == "knowledge"
    assert classify_message("Quiero aprender sobre Agentic RAG") == "knowledge"


def test_identity_wins_over_other_patterns():
    # identity phrase embedded in a longer message
    assert classify_message("hola, como te sentis hoy?") == "identity"


def test_accents_and_case_normalized():
    assert classify_message("CÓMO TE SENTÍS?") == "identity"
    assert classify_message("¿Quién SOS?") == "identity"


def test_imperative_orders_detected():
    assert is_imperative("Profundiza en el que consideres más interesante")
    assert is_imperative("Analizá el documento completo")
    assert is_imperative("Leé el documento y contame")
    assert is_imperative("Compará los dos enfoques")
    assert is_imperative("buscá más detalles técnicos")


def test_non_imperative_not_flagged():
    assert not is_imperative("Que sabes de GPT6 Astra?")
    assert not is_imperative("¿Cómo te sentís?")
    assert not is_imperative("hola")


def test_imperative_still_knowledge_kind():
    # Imperatives run retrieval (they need corpus context) — they are
    # knowledge-kind with the deep flag.
    assert classify_message("Profundiza en Functionary") == "knowledge"
    assert is_imperative("Profundiza en Functionary")


def test_memory_questions_route_to_recall():
    assert classify_message("que sabes de mi?") == "memory"
    assert classify_message("¿Qué sabés de mí?") == "memory"
    assert classify_message("te acordás qué hablamos ayer?") == "memory"
    assert classify_message("cuales son mis intereses?") == "memory"
    assert classify_message("lo que hablamos de RAG") == "memory"


def test_memory_does_not_steal_knowledge():
    # Questions ABOUT topics stay knowledge — memory is only for the
    # user profile / past sessions / agent's own memory.
    assert classify_message("que sabes de GPT6 Astra?") == "knowledge"
    assert classify_message("mis dudas sobre embeddings") == "knowledge"
    assert classify_message("quiero aprender sobre agentes") == "knowledge"
