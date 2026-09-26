from phone.tts import PhraseChunker, speakable


def stream(chunker, text, size=3):
    phrases = []
    for i in range(0, len(text), size):
        phrases += chunker.feed(text[i:i + size])
    return phrases + chunker.finish()


def test_first_phrase_breaks_early_then_at_sentences():
    phrases = stream(PhraseChunker(), "Well hello there, sailor, where did you come from? I was just thinking about you. Bye")
    assert phrases == ["Well hello there,", "sailor, where did you come from?",
                       "I was just thinking about you.", "Bye"]


def test_short_leading_clause_is_not_split_off():
    assert PhraseChunker().feed("Oh, it's you! ") == ["Oh, it's you!"]


def test_abbreviations_do_not_end_sentences():
    phrases = stream(PhraseChunker(), "Ask Mr. Smith and Dr. Jones about it. Then call me.")
    assert phrases == ["Ask Mr. Smith and Dr. Jones about it.", "Then call me."]


def test_long_run_on_text_is_split_at_a_space():
    chunker = PhraseChunker()
    chunker.first = False
    phrases = chunker.feed("word " * 60)
    assert phrases and all(len(p) <= PhraseChunker.MAX for p in phrases)


def test_finish_resets_for_the_next_reply():
    chunker = PhraseChunker()
    chunker.feed("First reply is going on, and on")
    chunker.finish()
    assert chunker.first
    assert chunker.feed("Second one, with a clause") == []  # short first clause waits


def test_speakable_strips_markup_emoji_and_stage_directions():
    assert speakable("*laughs* Oh **honey** (winks) you're a riot 😂 [sighs]") == "Oh honey you're a riot"
    assert speakable("Cześć, jak się masz?") == "Cześć, jak się masz?"
