"""Heuristic transcript-based clip ranking; not an engagement prediction."""
import re
import unicodedata

HOOKS = ("voce sabia", "o segredo", "a verdade", "ninguem", "presta atencao", "o problema", "eu descobri", "voce precisa", "nao cometa", "como", "por que")
EMOTION = ("vida", "familia", "dinheiro", "medo", "sonho", "erro", "fracasso", "coragem", "disciplina", "sucesso", "verdade", "resultado", "aprendi")
CLOSURE = ("por isso", "entao", "essa e a", "a licao", "no final", "e assim", "foi ai")
BAD_OPEN = ("e ai", "mas ai", "entao assim", "voltando", "como eu disse")
BAD_END = ("porque", "mas", "entao", "quando", "que", "se")


def plain(text):
    normalized = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in normalized if unicodedata.category(c) != "Mn")


def rank_highlights(subtitles, duration, minimum=30, maximum=55, limit=None):
    """Return chronologically ordered, disjoint highlights of variable count."""
    if duration <= 1:
        return []
    rows = []
    for row in subtitles:
        try:
            second, caption = float(row["time"]), re.sub(r"\s+", " ", str(row["text"])).strip()
        except (KeyError, ValueError, TypeError):
            continue
        if caption and 0 <= second < duration:
            if not rows or caption != rows[-1][1] or second - rows[-1][0] > 3:
                rows.append((second, caption))
    rows.sort()
    if not rows:
        return []
    low = max(8, min(int(minimum), int(duration)))
    high = max(low, min(int(maximum), int(duration)))
    options = []
    for i, (start, opening) in enumerate(rows):
        if start + low > duration + 1:
            continue
        candidates = [j for j in range(i, len(rows)) if low - 4 <= rows[j][0] - start <= high]
        if not candidates:
            continue
        complete = [j for j in candidates if rows[j][1].endswith((".", "!", "?"))]
        j = complete[-1] if complete else candidates[-1]
        end = min(duration, start + high, max(start + low, rows[j][0] + 2))
        lines = [text for _, text in rows[i:j + 1]]
        transcript = " ".join(lines)
        words = len(transcript.split())
        if words < 20 or words / max(1, end - start) < 0.55:
            continue
        first = plain(" ".join(lines[:2]))
        final = plain(" ".join(lines[-2:]))
        full = plain(transcript)
        hook = any(word in first for word in HOOKS) or "?" in " ".join(lines[:2])
        emotion = sum(word in full for word in EMOTION)
        closure = any(word in final for word in CLOSURE) or lines[-1].endswith((".", "!", "?"))
        bad_open = any(first.startswith(word) for word in BAD_OPEN)
        bad_end = any(final.rstrip(" .!?").endswith(" " + word) for word in BAD_END)
        score = 35 + (24 if hook else 0) + min(18, emotion * 6)
        score += 14 if closure else -10
        score += 7 if 0.9 <= words / max(1, end - start) <= 3.8 else 0
        score -= 22 if bad_open else 0
        score -= 18 if bad_end else 0
        score = max(0, min(100, score))
        if score < 62:
            continue
        reasons = []
        if hook:
            reasons.append("Gancho forte na abertura")
        if emotion:
            reasons.append("Tema emocional ou compartilhável")
        if closure:
            reasons.append("Final com conclusão")
        if bad_open or bad_end:
            reasons.append("Contexto pode estar incompleto")
        options.append({"start": int(start), "end": int(end), "duration": max(1, int(end)-int(start)), "score": score, "reason": "; ".join(reasons), "opening": plain(opening)})
    options.sort(key=lambda row: (-row["score"], row["start"]))
    selected = []
    for candidate in options:
        if any(candidate["start"] < row["end"] and row["start"] < candidate["end"] for row in selected):
            continue
        if candidate["opening"] in [row["opening"] for row in selected]:
            continue
        selected.append(candidate)
        if limit is not None and len(selected) >= max(1, int(limit)):
            break
    selected.sort(key=lambda row: row["start"])
    for row in selected:
        row.pop("opening")
    return selected
