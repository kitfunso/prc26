# prc26

Entry for the [PRC Data Challenge 2026](https://prc-data-challenge-2026.netlify.app/):
predict taxi-out time at 10 major European airports, scored on RMSE.

Team `merry-octopus`. Deadline 11 Oct 2026.

- `NOTES.md`: challenge rules, data layout, current leaderboard, open questions.
- `probe.py`: day-1 schema dump and the leak check.
- `lb.py`: pulls the live leaderboard (the site's own is broken).

## Running

    cp .env.example .env   # fill in from the MinIO console
    set -a; . ./.env; set +a
    python probe.py

## Data

Weather features (lane W1) come from the ASOS/METAR archive of the Iowa
Environmental Mesonet, Iowa State University (public domain; attribution
appreciated per https://mesonet.agron.iastate.edu/disclaimer.php), pulled
2026-09-05 with `python -m taxi.weather fetch` into `data/weather/2026-09-05/`;
the per-file sha256 in its `manifest.json` pins the pull. `data/` is not
committed.

Licensed GPLv3, as the challenge requires of prize-eligible entries.
