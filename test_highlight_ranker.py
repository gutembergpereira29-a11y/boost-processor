import unittest
from highlight_ranker import rank_highlights


def passage(start, first):
    rows = [{"time": start, "text": first}]
    speech = [
        "Isso mudou a minha vida e a minha familia.",
        "Eu aprendi como a disciplina muda o resultado de cada pessoa.",
        "A verdade e que todo mundo precisa de coragem e respeito.",
        "Eu cometi um erro, mas descobri que podemos melhorar.",
        "Por isso, nunca desista dos seus sonhos e objetivos."
    ]
    for idx, text in enumerate(speech):
        rows.append({"time": start + 7 * (idx + 1), "text": text})
    return rows


class RankerTests(unittest.TestCase):
    def test_empty_transcript_returns_no_claimed_highlights(self):
        self.assertEqual(rank_highlights([], 120), [])

    def test_selects_multiple_independent_candidates_without_fixed_five(self):
        rows = []
        for start in (0, 95, 200, 320, 440, 560):
            rows += passage(start, "Voce sabia o segredo para vencer o medo?")
        found = rank_highlights(rows, 650, 25, 46)
        self.assertGreater(len(found), 5)
        self.assertTrue(all(a["end"] <= b["start"] for a, b in zip(found, found[1:])))
        self.assertTrue(all(0 <= c["score"] <= 100 for c in found))

    def test_limits_only_when_requested(self):
        rows = passage(5, "Voce sabia como isso funciona?") + passage(130, "A verdade sobre o sucesso!")
        self.assertEqual(len(rank_highlights(rows, 250, 25, 46, limit=1)), 1)

    def test_does_not_create_poor_contextless_moments(self):
        rows = [{"time": t, "text": "e ai e ai mas ai enfim sem sentido"} for t in range(0, 120, 5)]
        self.assertEqual(rank_highlights(rows, 150, 25, 46), [])


if __name__ == "__main__":
    unittest.main()
