RAAGA AI MUSIC COMPOSER - STAGE 1 KNOWLEDGE PACK
Version: 1.0
Scope: 72 Sampurna Melakarta ragas only
Purpose: teach the AI swaras -> Melakarta grammar -> scale character -> brief-to-raga ranking.

FILES
00_STAGE1_README.txt
01_SWARA_AND_EMOTION_BASE_MAP.txt
02_MELAKARTA_MAP_01_24.txt
03_MELAKARTA_MAP_25_48.txt
04_MELAKARTA_MAP_49_72.txt
05_BRIEF_TO_RAGA_SELECTION_ENGINE.txt
06_CLAUDE_HANDOFF_AND_ACCEPTANCE_TESTS.txt

CRITICAL DISTINCTION
[GRAMMAR] = music-theory fact used as a hard rule.
[HEURISTIC] = useful starter model for emotion/raga selection. It is NOT a claim that a swara or raga has one fixed emotion.

STAGE 1 LEARNING ORDER
1. Learn swara names and swarasthanas.
2. Hear/play each swara relative to Sa.
3. Learn all 72 Melakarta Arohana/Avarohana.
4. Learn the 6 valid R-G blocks, 2 Madhyamas, and 6 valid D-N blocks.
5. Use the block-character map to create an initial emotional profile for each Melakarta.
6. Parse Title + Situation + Mood + Feel into an emotion target.
7. Rank and return a LIST of suitable ragas, with reasons.
8. On user selection, play Arohana then Avarohana accurately.
9. Later in Stage 1, generate short constrained melodic candidates and learn from user preference.
10. Stage 2 begins only after this foundation: lyrics for a selected/locked tune.

MELAKARTA HARD RULES
- 72 parent scales.
- Seven swaras occur in both ascent and descent.
- Same swaras in Arohana and Avarohana.
- Order is krama: S R G M P D N S and reverse in descent.
- S and P are fixed.
- M is M1 or M2.
- Valid R-G pairs: R1G1, R1G2, R1G3, R2G2, R2G3, R3G3.
- Valid D-N pairs: D1N1, D1N2, D1N3, D2N2, D2N3, D3N3.
- 2 x 6 x 6 = 72.
- 1-36 use M1; 37-72 use M2.

IMPORTANT MUSIC LIMITATION
A scale alone does not fully define lived raga identity. Gamaka, characteristic prayoga, note emphasis, register, tempo, and phrase grammar matter. Therefore this pack is a Stage-1 parent-scale engine, not the final Carnatic-musicianship model.

RESEARCH ANCHORS USED TO CHECK THE PACK
- Standard 72-Melakarta/Govindacharya scheme and krama-sampurna rules.
- Carnatic swarasthana references showing 16 functional names occupying 12 principal pitch positions.
- Experimental music-emotion research indicating that tonic-interval structure correlates with broad emotional valence, while tempo/rhythm strongly affect arousal.

DESIGN PRINCIPLE
Do not let the AI say "I know the raga" merely because it can recite a scale. Make knowledge testable: identify -> play -> compare -> rank -> explain -> receive trainer feedback -> update learned weights.
