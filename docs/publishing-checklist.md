# Release checklist

A one-pass list I use to make a public release land well. Nothing here changes code — it is all
presentation and platform setup.

## Repository setup

- [ ] **Name:** `intercept` (or `intercept-benchmark` if taken).
- [ ] **Description:** `Elo ratings for missile-guidance laws — a reproducible benchmark putting
      classical, optimal, game-theoretic, and learned interception guidance on one fair field
      (simulation-only).`
- [ ] **Topics:** `robotics` `guidance` `reinforcement-learning` `benchmark` `simulation`
      `optimal-control` `game-theory` `kalman-filter` `model-predictive-control` `gymnasium`
      `stable-baselines3` `aerospace`
- [ ] **Social preview image:** upload a frame of `gallery/animations/p28_swarm_showcase.gif`
      (Settings → Social preview) — this is the thumbnail on link shares.

## First push

- [ ] Initial commit (or curated history), tag `v0.1.0`, GitHub Release with the CHANGELOG
      `[Unreleased]` section as release notes.
- [ ] Swap the commented CI badge in `README.md` to the real
      `https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg` and confirm CI is green.
- [ ] Enable **GitHub Pages** from the `docs.yml` workflow (mkdocs-material site) and add the docs
      badge/link to the README header.
- [ ] Update `[project.urls]` in `pyproject.toml` and `CITATION.cff` `repository-code` with the
      final URL.

## Post-publish polish

- [ ] Pin the repo on the GitHub profile; add it to the profile README with the p28 gif.
- [ ] Verify all README media render on github.com (relative `gallery/...` paths render in repos;
      check GIF sizes < 10 MB each — GitHub truncates large ones).
- [ ] Link the interactive HTML replays via the published Pages site (raw HTML does not render from
      the repo view).
- [ ] Optional reach: a short demo thread/post (the League leaderboard + the p28 swarm gif tell the
      story in two images), r/robotics, X/LinkedIn, Hacker News "Show HN".
