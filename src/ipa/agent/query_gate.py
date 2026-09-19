"""Deterministic query gate for the chat surface.

Classifies an incoming user message BEFORE auto-retrieval so that
identity/social questions do not trigger a corpus search (the agent
answers those from its identity, not from evidence). One clear
responsibility: classify. No retrieval, no LLM, no state.
"""
from __future__ import annotations

import re
import unicodedata


def _normalize(text: str) -> str:
    """Lowercase and strip accents so matching is robust to accents/typos."""
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


# Social openers with no content to retrieve (anchored at start).
_GREETING = re.compile(
    r"^(hola|buenas|hey|hi|hello|gracias|ok|si|no|dale|bueno|chau|adios|listo)\b"
)

# Explicit tool / system commands: the tool loop handles them, not retrieval.
_COMMAND = re.compile(
    r"(tarea|task|status|estado|sistema|corre|ejecuta|inicia|"
    r"investig|busca.*web|research|pipeline|ingesta)"
)

# Self-referential questions: the answer lives in the identity/persona,
# not in the corpus. Retrieving for these wastes seconds and pollutes
# the prompt with irrelevant context.
_IDENTITY = re.compile(
    r"\b("
    r"(como\s+)?te\s+(sentis|sientes|va)|"
    r"como\s+(estas|andas|te\s+encuentras|te\s+sentis)|"
    r"que\s+tal\s+(estas|andas)|"
    r"quien\s+(sos|eres)|que\s+(sos|eres)|"
    r"cual\s+es\s+tu\s+nombre|como\s+te\s+llamas|"
    r"que\s+(podes|puedes)\s+hacer|tus\s+capacidades|"
    r"(contame|hablame|decime)\s+de\s+vos|sobre\s+vos|acerca\s+de\s+vos|"
    r"tenes\s+(sentimientos|emociones|conciencia|opinion)|"
    r"sentis\s+(emociones|algo|dolor)|te\s+sientes|"
    r"estas\s+vivo|sos\s+humano|"
    r"sos\s+una?\s+(ia|ai|robot|maquina|persona|llm|modelo)|"
    r"quien\s+te\s+(creo|creaste|programo|hizo)|de\s+donde\s+venis"
    r")\b",
)


# Memory questions: about the user profile, past conversations, or the
# agent's own memory. These resolve via recall_memory (the agentic
# corpus), not via document-corpus retrieval.
_MEMORY = re.compile(
    r"\b("
    r"que\s+(sabes|tenes|recordas)\s+(de|sobre)\s+mi\b|"
    r"que\s+sabes\s+de\s+mi\b|"
    r"te\s+(acordas|acordaste|acuerdas)|recordas\s+(que|cuando|lo|la)|"
    r"que\s+(hablamos|charlamos|vimos|conversamos)|"
    r"lo\s+que\s+(hablamos|charlamos|te\s+(dije|pedi|cont[eé]))|"
    r"mi\s+(perfil|modelo)|mis\s+(intereses|objetivos|goals|preferencias)|"
    r"que\s+sabes\s+de\s+mi\s+perfil|quien\s+soy\s+yo|"
    r"la\s+(ultima|última)\s+vez\s+que\s+(hablamos|charlamos)|"
    r"ayer\s+(te|hablamos|charlamos)|"
    r"tu\s+memoria|que\s+recordas\b"
    r")\b",
)


# Imperative work orders: the user is COMMANDING multi-step work
# ("profundizá", "analizá", "leé el documento"). The reactive 9B tends
# to narrate intent and ask permission instead of executing — these
# queries get an explicit "execute now" instruction injected.
_IMPERATIVE = re.compile(
    r"\b("
    r"profundiza|profundiz|analiza|analiz|compara|compar|"
    r"le[eé]\s+(el|los|la|las)\s|le[eé]lo|le[eé]la|"
    r"investiga\s+(m[aá]s|más|eso|esto|el|la|los|las)|"
    r"busc[aá]\s+(m[aá]s|más|eso|esto)|"
    r"detall(a|e|ame)\s+(m[aá]s|más|el|la|los|las)|"
    r"expand[ií]|desarrolla\s+(el|la|los|las|m[aá]s|más)|"
    r"seg[uú]n\s+el\s+documento|del\s+documento\s+completo"
    r")\b",
)


def classify_message(message: str) -> str:
    """Classify a chat message for the retrieval gate.

    Returns:
        "identity"  — question about the agent itself: skip retrieval,
                      answer from the system prompt persona.
        "greeting"  — social opener with no content: skip retrieval.
        "command"   — explicit tool/system command: tool loop handles it.
        "memory"    — question about the user/past sessions: resolve via
                      recall_memory, not the document corpus.
        "knowledge" — everything else: auto-retrieval runs.
    """
    norm = _normalize(message.strip())
    if _IDENTITY.search(norm):
        return "identity"
    if re.match(r"^(hola|buenas|hey|hi|hello|gracias|ok|si|no|dale|bueno|chau|adios|listo)\b", norm):
        return "greeting"
    if _MEMORY.search(norm):
        return "memory"
    if _COMMAND.search(norm):
        return "command"
    return "knowledge"


def is_imperative(message: str) -> bool:
    """True if the message is a direct work order needing execution,
    not permission-asking. Used to inject an 'execute now' instruction."""
    return bool(_IMPERATIVE.search(_normalize(message.strip())))


# ── Research narration net (chat general) ──────────────────────────────
# El 9B narra la investigación sin emitir el marcador [TOOL:] (bug del
# 2026-09-19: 5 narraciones, 0 tool_calls grabados). Estos patrones
# detectan (a) pedido/aceptación del usuario y (b) claim falso del modelo,
# para que la red de seguridad dispare research_topic de verdad.
# Stems parciales a propósito: las citas llegan truncadas por la UI
# («…busqu…», «nvestiguemos…» — pierden letras al cortar).

# El usuario pidió investigar explícitamente.
RESEARCH_ASK_RE = re.compile(
    r"investig|busc[aá].*web|buscar.*web|research|busc[aá].*internet",
    re.IGNORECASE,
)

# La cita contiene lenguaje de investigación/roadmap (truncada o no).
# "nvestig" sin la i: la truncación corta palabras por delante
# («nvestiguemos…») y "investig" queda cubierto como substring.
RESEARCH_CITE_RE = re.compile(
    r"nvestig|busqu|research|roadmap|fuentes",
    re.IGNORECASE,
)

# Aceptación genérica ("dale", "avancemos", "perfecto") — solo cuenta si
# el assistant ofreció investigar en el turno previo (ver OFFER_RE).
RESEARCH_ACCEPT_RE = re.compile(
    r"\b(?:dale|avancemos|adelante|perfecto|ambas|interesante|acepto|"
    r"vamos|ok|s[ií])\b",
    re.IGNORECASE,
)

# El turno assistant previo ofreció investigar / armar roadmap.
RESEARCH_OFFER_RE = re.compile(
    r"investig|roadmap|búsqueda|research|fuentes",
    re.IGNORECASE,
)

# El modelo AFIRMA que una investigación corre/arrancó sin emitir la tool.
# Frases observadas en producción + variantes; deliberadamente no cubre
# menciones condicionales ("una investigación web mostraría…").
RESEARCH_CLAIM_RE = re.compile(
    r"investigaci[oó]n.*activa|está ejecutándose|en curso|"
    r"voy a investigar|inici[eé] la búsqueda|búsqueda.*activa|"
    r"investigando ahora|estoy investigando|"
    r"ejecutar una investigaci[oó]n|ejecuto una investigaci[oó]n|"
    r"acepto la investigaci[oó]n|lanc[eé] la investigaci[oó]n|"
    r"lanzo la (?:investigaci[oó]n|búsqueda)|"
    r"proces\w*\s+la solicitud|procesa (?:las )?fuentes|"
    r"mientras (?:el sistema )?procesa",
    re.IGNORECASE,
)


__all__ = [
    "classify_message", "is_imperative",
    "RESEARCH_ASK_RE", "RESEARCH_CITE_RE", "RESEARCH_ACCEPT_RE",
    "RESEARCH_OFFER_RE", "RESEARCH_CLAIM_RE",
]
