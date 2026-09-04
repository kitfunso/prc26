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

Licensed GPLv3, as the challenge requires of prize-eligible entries.
