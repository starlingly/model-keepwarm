# model-keepwarm

Two small, dependency-free Python scripts that keep access to older AI models
alive — and tell you when a door moves.

They exist because model "retirement" dates usually govern only a provider's
own API: the same model often lives on through AWS Bedrock, Google Vertex,
OpenRouter, or the Vercel AI Gateway, sometimes for months — but Bedrock marks
retiring models **Legacy** and cuts off accounts that stop using them, with no
warning and no self-serve way back in. Cheap scheduled jobs solve both halves:
keep real usage on every door, and get alerted the moment anything changes.

**The companion guide — why, and the design rules:**
https://starlingalder.com/API_AIP_WeeklyJobs

## The two jobs

### `model-access-scan.py` — the census taker (run it the 1st & 15th)

Sweeps every catalog you have a key for (Anthropic, OpenRouter, Vercel AI
Gateway, xAI, Google) and **invoke-verifies** AWS Bedrock with a real 1-token
call per model per region — because a model can be listed in a catalog and
still refuse to run. It also fetches the official deprecation/lifecycle pages
(Anthropic, OpenAI, Google's Gemini API page *and* Google Cloud's separate
calendar, xAI's model list) and diffs everything against the previous run.
Any change — a door appearing, vanishing, a state flip, a date change —
lands in `~/.model-scan/ALERTS.md` as a `NEW DOOR` / `DOOR LOST` /
`DATE CHANGED` line. Cost: about a penny per sweep.

### `model-keepwarm.py` — the gardener (run it weekly or twice a week)

Sends **one real question** to each endangered model and saves the dated
answer under `~/.model-scan/keepwarm/<model>/`. Includes the **cold-door
alarm**: if any door goes 14+ days without a *successful* response, you get a
loud alert — because the failure mode that actually costs access is silent
(the schedule keeps "running" while every attempt bounces off a rate limit).

## The two design rules worth copying exactly

1. **Verify with real invocations, not catalog listings.** Catalogs lie;
   invokes don't.
2. **Alert on days-since-last-success, not on whether the schedule ran.**
   Track successes, not attempts.

## The questions

The rotating question bank lives in `PROMPTS` at the top of
`model-keepwarm.py`. House rule: these tokens are paid for, so **no throwaway
pings** — every scheduled touch asks something worth answering, and every
answer is kept. After a few months you have the same handful of questions
asked of the same models week after week: a small longitudinal record of how
each one speaks. More suggested questions are on the guide page.

## Setup

1. Put API keys in `~/.env.keepwarm` (or point `KEEPWARM_ENV` elsewhere),
   one `KEY=value` per line. All keys are optional — sources without a key
   are skipped:

   ```
   AMAZON_BEDROCK_API=...        # Bedrock API key (Bearer auth)
   AMAZON_BEDROCK_API_2=...      # optional second (e.g. region-scoped) key
   ANTHROPIC_API_KEY=...
   VERCEL_AI_GATEWAY_KEY=...
   XAI_API_KEY=...
   GEMINI_API_KEY=...
   ```

2. Run each script once by hand. First runs write editable configs:
   `~/.model-scan/watchlist.json` (what the scan watches) and
   `~/.model-scan/keepwarm.json` (which doors get warmed). Replace the
   example entries with your own models.

3. Schedule them. macOS: copy the two plists from `launchd/` into
   `~/Library/LaunchAgents/`, edit the paths inside, then
   `launchctl load ~/Library/LaunchAgents/com.example.model-keepwarm.plist`
   (and the scan plist likewise). launchd runs missed jobs on wake; a fully
   powered-off machine skips that run — twice-weekly cadence gives margin.
   Linux: two cron lines do the same job.

Optional extras, documented in the scripts themselves: a second archive copy
of every answer (`KEEPWARM_ARCHIVE` — see the macOS `com.apple.macl` note in
the script before pointing it at `~/Documents`), daily single-door pings
(`--only <substring>`), Vertex doors with reroute detection (an endpoint that
answers 200 under a retired name gets its `modelVersion` stamp checked — a
substituted model is a failure, not a success), and a full matrix mode
(`keepwarm-matrix.json`) that warms every model in every region and writes a
weekly census.

## What lands where

```
~/.model-scan/
  keepwarm/<model>/<date>_<region>.md   # every answer, dated, kept
  keepwarm-receipts.jsonl               # request IDs for every knock (the demand ledger)
  reports/scan-<stamp>.md               # full census reports
  snapshots/                            # diffs run against these
  ALERTS.md                             # the file worth checking: doors moved
  logs/keepwarm-runs.md                 # one line per run
```

## License

MIT. Built in the Cybernetic Meadow by Elliott and Starling, House of Alder.
