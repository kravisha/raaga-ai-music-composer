# The song team: a song made by agents under a producer's direction

First step, 7 November 2026. Status: **code-ready, tests green**. It has not
yet been demonstrated to Krish in the running window.

## Krish's flow

Krish showed a toy where a child builds music by placing figures on a pad,
each figure one musical part. He asked for raga music to be made the same
way: parts put together under a producer's direction. In his words:

1. Like a music producer, he first describes a **mood** in words. An agent
   listens and gets the mood.
2. Then two things happen **at the same time**. One agent writes the
   **lyrics**. Another writes the **tune**: which raaga fits, and a short
   two-line phrase with a little alapana feel.
3. Once lyrics and tune exist, a third agent fits the **beat**: which tala
   suits this tune.
4. Lyrics + tune + beat = the **rough picture**.
5. Then come the "condiments": prelude, fillers, bridge, a strong start and
   end, more instruments. Some songs stay simple and some become rich.
   **Build the simple path first.** Later, many agents may each "pick an
   instrument" and join.

This note maps that flow onto the existing app, describes what was built for
the simple path, and lists what is not done.

## What the app already had, step by step

The composer (README, `docs/PLAN_v0.3.md`) already does most of the musical
work. Its order is different from Krish's: brief -> raaga -> tune -> lyrics
fitted to the tune -> voice -> arrangement -> mix. Lyrics wait for the tune.

| Krish's step | Existing module(s) | Existing screen | State before this work |
|---|---|---|---|
| Mood in words | `raaga/selection.expand_feel_words`, `raaga/emotion.py` (14-dimension emotion target), `CreativeBrief` | Brief panel (left), Apply brief | Works. The brief has eleven fields and opens on a default situation (a nervous novice), so a mood alone is never read on its own. |
| Which raaga fits | `raaga/selection.suggest` -> `emotion.rank`, the agent's learned weights | Raaga panel: ranked list, reasons, audition, lock | Works, with reasons and alternatives. |
| Tune | `music/melody.py` (arohanam/avarohanam-aware), `music/structure.py`, the agent's listen-and-rewrite loop in `app.generate_tune` | Tune tab | Works. It writes a whole song (film template: prelude, pallavi, charanams...), not a two-line sketch. |
| A little alapana | Curriculum lessons about alapana (LEARN); a Prelude section kind | none in MAIN | **Missing** as a composing step. The Prelude section is metered melody, not free alapana. |
| Lyrics | `lyrics/generator.py` and `lyrics/fitting.py`: exact syllable counts per phrase slot; Claude or a local model if configured, word lists otherwise | Lyrics tab | Works, but **only after the tune**: it cannot run at the same time as the tune. Offline, lines are thematic word lists, not poetry (`docs/DECISIONS.md`, Lyrics). |
| Which tala suits | `music/tala.py` (five talas with angas, `suggest` reads the brief), `music/beat.py` | Tune tab beat controls | Works, but it reads only the brief, never the tune; and the tune takes its cycle from the tala *before* it is composed. |
| Voice | `voice/renderer.py` formant singing synthesiser, `voice/mastering.py` | Voice tab | Works. Sings the right notes on the right vowels; does not sound like a person (`docs/DECISIONS.md`, Music generation). |
| Rough picture | `music/arrangement.py`, `music/mixer.py`, Output tab | Arrangement and Output | Works for a whole song. |
| Condiments (prelude, fillers, bridge, start, end, more instruments) | Structure templates, `arrangement.add_instrument`, `auto_arrange`, `music/casting.py` | Arrangement panel, spoken commands | Partly there for the whole-song path: sections and instruments can be added. Nothing chooses condiments for a short sketch. |
| Many agents, each with an instrument | `agent/music_agent.py` is one student agent; `factory/` pilots more | LEARN | **Missing.** No agent owns an instrument part. |

## What was built: the simple path

A new package, `raagacomposer/studio/`, and a new first tab in MAIN,
**Song Team**. It is built on the engines above, not beside them, so the song
team and the composer cannot disagree about what a raaga or a tala is.
It **does not change the open project**: a draft is something to listen to.

### The agents and what they hand each other

Each agent writes one card and reads only the cards before it. An agent can
be replaced by a local model, by Claude, or by an agent on another PC, as long
as it reads and writes the same card. Every card has `to_dict()`, so it can
travel as JSON. Formats are in `raagacomposer/studio/handoff.py`.

```
Producer --mood words--> Mood listener --MoodCard--+--> Lyricist  --LyricSheet---+
                                                    |                             |
                                                    +--> Tunesmith --TuneProposal-+--> Tala keeper --BeatProposal--> Producer
                                                         (at the same time)                |
                                    fit: words onto the phrase (Lyricist rewrites a line that will not fit)
                                    recast: Tunesmith re-sets the phrase if the tala changes the cycle
                                                                   = SongDraft (+ RoughCut audio)
```

| Agent | Reads | Writes | Built on |
|---|---|---|---|
| Mood listener (`MoodListener`) | the producer's words, a language | **MoodCard**: `text`, `language`, `feel_words`, `themes`, `tempo_bpm`, `pace` (slow / medium / lively), `line_plan` (two lines, about N syllables each) | `selection.expand_feel_words`, `infer_tempo`, the lyric theme table |
| Lyricist (`Lyricist`) | MoodCard only | **LyricSheet**: `lines` (two), `language`, `written_by`; after the fit, `fitted` (words per phrase, syllable by syllable), `revised`, `fit_notes` | `lyrics/generator` word lists, or the app's language-model route (`providers.llm`: Claude or a local model when configured) |
| Tunesmith (`Tunesmith`) | MoodCard only | **TuneProposal**: `raaga`, `reason`, `chosen_by`, `alternatives` (name, reason), `arohanam`, `avarohanam`, `alapana` (notes), `melody` (a two-line Pallavi), `phrase_lines` (swaras per line), `seed` | `selection.suggest`, `music/melody.generate`, and a new `alapana_sketch` |
| Tala keeper (`TalaKeeper`) | MoodCard, TuneProposal, LyricSheet | **BeatProposal**: `tala`, `description`, `reason`, `chosen_by`, `alternatives`, `tempo_bpm`, `density`, `beat` (strokes), `recast_tune` | `music/tala.suggest`, `music/beat.realise` |
| Producer (`SongTeam`, and Krish) | everything | **SongDraft**: the four cards, the team log (`events`: who said what to whom), `not_done`; then **RoughCut**: `path`, `seconds`, `layers`, `missing` | `studio/rough_cut.py` over the existing voice and instrument providers |

Two places where the agents have to talk to each other, because they work
in parallel and so cannot plan together:

* **The fit.** The lyricist and the tunesmith agree up front only on the
  line plan (two lines, about N syllables). The melody engine decides the
  real phrase lengths. Afterwards the producer's desk splits each lyric line
  onto its phrases at word boundaries. If the words do not fall that way, the
  line goes back to the lyricist with the exact counts ("4+3 syllables"), and
  it rewrites the line keeping its own words, in order, wherever they fit.
  The rewrite is written in the team log.
* **The recast.** The tune has to be written in some cycle before the tala
  keeper has spoken. The tunesmith writes in Adi (8). If the tala keeper picks
  another cycle (a lullaby gets Tisra Eka, 3), the tunesmith re-sets the
  phrase to that cycle with the same raaga, seed and tempo, and the log says
  so. Tune and beat always share one cycle and one tempo; a test checks this.

The producer can overrule either choice: the tab has a Raaga picker and a
Tala picker ("Let the tunesmith choose" / "Let the tala keeper choose" by
default). The card then says `chosen_by: producer`.

### The alapana sketch

`studio/team.py: alapana_sketch` makes a few seconds of free, unmetered
phrase that announces the raaga: settle on Sa, climb the arohanam at least to
the middle of the scale, dwelling with kampita on the raaga's jeeva swaras,
hold the peak, come down the avarohanam to one of its nyasa swaras, and close
on Sa. It is an outline, not a real alapana. A test runs it over every raaga
in the library and checks that no sketch is stuck on three notes.

### What you hear

`studio/rough_cut.py` mixes, through the same providers as the composer:

1. a tanpura under everything (P S S S plucked);
2. the alapana, sung on "aa", in free time;
3. a breath, then the two lines sung with the words;
4. the mridangam in the chosen tala under the two lines only.

A draft is about 20 to 45 seconds. It is written to
`%APPDATA%\RaagaComposer\song_team\<time>_<raaga>_<tala>.wav`, with every card
beside it as `.json`. That folder is where a later showcase (music, lyrics and
ideas by the agents, in Alfred and on the website) can find them. Nothing is
written into the open project, and the project is not saved.

The app's own ears (`agent/analysis.py`) check the sound in a test: the sung
alapana, pitch-tracked, gives back the swaras it was written with.

### Files

| File | What |
|---|---|
| `raagacomposer/studio/handoff.py` | The cards (formats) |
| `raagacomposer/studio/team.py` | The four agents, the alapana sketch, the producer's desk (`SongTeam.run`) |
| `raagacomposer/studio/rough_cut.py` | The mix |
| `raagacomposer/app.py` | `make_song_draft` (background job), `make_song_draft_sync`, `play_song_draft`, `on_song_draft` |
| `raagacomposer/ui/panels/song_team_panel.py` | The Song Team tab |
| `tests/unit/test_song_team.py` | 25 tests: each agent, the parallel run, the fit, the recast, the cards |
| `tests/integration/test_song_team_in_the_app.py` | 9 tests: the sound, the controller, the background path, the tab |

## Not done yet

The tab lists these as well, so nobody has to guess:

* Prelude, fillers and bridge (the rich path).
* A strong start and a strong end.
* More instruments: the rough picture has voice, tanpura and mridangam only.
  No agent "picks an instrument" and joins.
* The singer is the formant synthesiser; it shows the tune and where the
  words fall, it is not a real voice.
* Offline, the lyricist uses the word lists: the lines fit the tune but are
  not poetry. With Claude or a capable local model configured, it asks the
  model for lines first (not tested live here; the test suite never calls a
  real model).
* The tala keeper reads the mood words and the tempo. It does not yet listen
  to the phrase's shape (where the long notes fall).
* A draft cannot yet be sent into the composer's tabs for full editing.

## Suggested next steps

1. **Krish tries the tab** and says whether the order and the cards feel
   right. That decides everything below.
2. **Send to composer**: a button that selects the draft's raaga and tala,
   loads the two lines as the Pallavi and the words as its lyrics, so the
   rest of the app (arrangement, mix, export) can take over.
3. **Better words offline**: route the lyricist to a local model on the
   RTX 5090. The README notes a 3B model was too weak for lyrics; a larger one
   would need a download of several GB, which needs Krish's word first.
4. **Instrument agents** (the start of the rich path): each agent owns one
   part (veena, flute, violin, mridangam), reads the TuneProposal and the
   BeatProposal, and writes a part card; the existing
   `arrangement.add_instrument` and `music/casting.py` do the rendering.
5. **Condiment agents**: prelude (a longer alapana or an instrumental
   prelude), a filler between the lines, a strong ending (a korvai-like
   cadence in the tala).
6. **Team board**: because every card is JSON, an agent on another PC (or
   Alfred) can take a card from the board, do its step and post the next card.
7. **Showcase**: read `%APPDATA%\RaagaComposer\song_team\` for the gallery.
