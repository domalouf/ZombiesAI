"""Learning from real gameplay: recording it, ingesting it, labelling it, and cloning it.

`PLAN.md` milestone M6. Two ways in, one way out:

* **Recorded with input logging** (`recorder`, `capture`, `win32_input`, `inputs`) -- frames and the human's
  own mouse counts and key transitions, quantized into spec actions. The best labels there are, and the only
  source that can train the inverse dynamics model.
* **Video nobody logged** (`video`) -- any recording of the game, decoded into policy frames and segmented
  into continuous clips, then labelled by the inverse dynamics model (`idm`) from the pixels alone.

Both land in the clip store (`clips`), and `bc` trains one policy over whatever mix of them you have.
"""
