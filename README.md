# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/BrotherMus/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/BrotherMus/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, two days after Grand Theft Auto VI's launch, every PS5 trophy's Exophase earn rate one week in, as five
quantiles per trophy, scored by pinball loss on the logit scale.

How it works (`main.py`, Python 3.13, standard library only):

- **Dev floor.** The launch-night page tracked a few hundred review / early-access accounts that already held almost
  every trophy. Their holdings stay in the counts, so each rare trophy sits on a floor of a handful of players that is
  diluted as players tracked grow. The model subtracts it, forecasts the real share, and adds the floor back diluted
  by the forecast players tracked.
- **Log-time logistic growth.** If progress is play speed x time and speeds are log-logistic across players, the real
  share of a milestone is linear in logit against ln(hours since launch). Each trophy's slope is fitted over Exophase's
  distinct refreshes (recency-weighted), then shrunk toward a pooled slope-vs-rarity line fitted across the
  well-measured trophies, damped, and extrapolated to the target.
- **Players tracked** grow on a damped log-log trend.
- **Honest spread.** A logit-scale band that widens with the horizon and the slope's uncertainty; trophies still on
  the floor get a band skewed upward (they can only take off), and trophies nobody holds yet are pinned near the
  0.01% bottom of the score's clamp.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=8jskxc5q&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
