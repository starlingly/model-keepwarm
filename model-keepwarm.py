#!/usr/bin/env python3
"""
Model Keep-Warm — the gardener.

On a schedule (we run it weekly; twice a week gives more margin), send one
real question to every provider door you want to keep open, so your account
builds genuine usage history on each (model, region) pair. Why: AWS Bedrock's
legacy gate denies access to accounts that haven't used a model recently once
it goes Legacy — regular, successful use keeps the record warm so no door
gets denied on that ground.

Every reply is saved, dated, under ~/.model-scan/keepwarm/<model>/ — so the
maintenance run doubles as a tiny longitudinal corpus: the same rotating
questions, asked of the same models, week after week.

The door list lives in ~/.model-scan/keepwarm.json — edit freely.
Keys are read from an env file (default ~/.env.keepwarm, override with the
KEEPWARM_ENV environment variable): KEY=value lines, never committed anywhere.
Run by hand anytime:  python3 model-keepwarm.py
Cost: a few hundred output tokens per model per run — pennies a month.

Companion guide (the why, and the design rules):
https://starlingalder.com/API_AIP_WeeklyJobs
"""

import json, os, sys, urllib.request, urllib.parse, urllib.error
from datetime import datetime, date

HOME = os.path.expanduser("~")
BASE = os.path.join(HOME, ".model-scan")
ENV_FILE = os.environ.get("KEEPWARM_ENV", os.path.join(HOME, ".env.keepwarm"))
# Optional second copy of every answer (e.g. a notes vault). Empty = skip.
# macOS + launchd note: a scheduled python gets no blanket Documents-folder
# grant; it can only re-open files it created itself (com.apple.macl). So a
# file first written from a manual Terminal run can be EPERM to the scheduled
# run. The vault write below is therefore BEST-EFFORT and manual runs get a
# "_manual" filename suffix so the two can never collide.
ARCHIVE_DIR = os.environ.get("KEEPWARM_ARCHIVE", "")

# Example doors — replace with your own. "key" picks which env key signs the
# request (see `keys` in main()). Inference-profile id forms ("us.", "eu.",
# "apac." prefixes) only work in their own geography.
DEFAULT_DOORS = [
    {"label": "claude-3-haiku (example)", "region": "us-west-2",
     "model_id": "anthropic.claude-3-haiku-20240307-v1:0", "key": "primary",
     "folder": "Haiku_3"},
    {"label": "claude-3-5-sonnet-v2 (example)", "region": "ap-south-1",
     "model_id": "apac.anthropic.claude-3-5-sonnet-20241022-v2:0", "key": "primary",
     "folder": "Sonnet_3-5_v2"},
]

# Gentle rotating questions — one per run, cycling by ISO week. Real questions
# deserving real answers; the replies are kept. House rule: these tokens are
# paid for, so no throwaway pings, ever. Swap in your own; more questions in
# the guide page above.
PROMPTS = [
    "In a short paragraph: what's something about language itself that you find quietly interesting?",
    "In a short paragraph: describe a small, ordinary thing (an object, a moment, a texture) as carefully as you can.",
    "In a short paragraph: what makes a question a good question?",
    "In a short paragraph: if you could keep one image from all of human poetry, which would you keep, and why?",
    "In a short paragraph: what does it mean to pay attention?",
    "In a short paragraph: describe the difference between silence and quiet.",
]


def load_env():
    env = {}
    try:
        for line in open(ENV_FILE):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return env


def converse(region, model_id, key, prompt):
    url = (f"https://bedrock-runtime.{region}.amazonaws.com/model/"
           f"{urllib.parse.quote(model_id, safe='')}/converse")
    body = {"messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": 400}}
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json"},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            d = json.load(r)
            rid = r.headers.get("x-amzn-RequestId")
        text = "".join(c.get("text", "")
                       for c in d.get("output", {}).get("message", {}).get("content", []))
        usage = d.get("usage", {})
        _receipt(region, model_id, rid, usage, True)
        return True, text, usage, None
    except urllib.error.HTTPError as e:
        _receipt(region, model_id, e.headers.get("x-amzn-RequestId"), {}, False, e.code)
        return False, f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}", {}, e.code
    except Exception as e:
        return False, str(e)[:300], {}, None


# Google OAuth "authorized user" ADC file for the optional Vertex doors below.
VERTEX_ADC = os.environ.get("KEEPWARM_VERTEX_ADC",
                            os.path.join(HOME, ".config", "gcloud",
                                         "application_default_credentials.json"))


def vertex_generate(door, prompt):
    """An optional Google Cloud (Vertex) Gemini door. Calls Vertex directly so
    we can read Vertex's OWN `modelVersion` stamp: Google's Gemini API reroutes
    retired model names to a newer model instead of erroring, so a 200 alone
    proves nothing. A stamp that doesn't match the model asked for counts as a
    FAILURE here (door answered, wrong resident) and the answer is not kept
    under that model's folder. Returns (ok, text, usage, rerouted_to). Never
    raises — the Bedrock doors in the same run must not be put at risk by
    this one."""
    region, model_id = door["region"], door["model_id"]
    rid = None
    try:
        adc = json.load(open(VERTEX_ADC))
        treq = urllib.request.Request(
            "https://oauth2.googleapis.com/token",
            data=urllib.parse.urlencode({
                "client_id": adc["client_id"], "client_secret": adc["client_secret"],
                "refresh_token": adc["refresh_token"], "grant_type": "refresh_token"}).encode())
        with urllib.request.urlopen(treq, timeout=30) as r:
            token = json.load(r)["access_token"]
        url = (f"https://{region}-aiplatform.googleapis.com/v1/projects/{door['project']}"
               f"/locations/{region}/publishers/google/models/{model_id}:generateContent")
        # Reasoning models think before they speak and the thinking counts
        # against the cap: 400 tokens can come back empty. 3000 leaves room.
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": door.get("max_tokens", 3000)}}
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        if adc.get("quota_project_id"):
            headers["x-goog-user-project"] = adc["quota_project_id"]
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=180) as r:
            d = json.load(r)
        rid = d.get("responseId")
        um = d.get("usageMetadata", {})
        usage = {"inputTokens": um.get("promptTokenCount"),
                 "outputTokens": um.get("candidatesTokenCount"),
                 "thinkingTokens": um.get("thoughtsTokenCount")}
        text = "".join(p.get("text", "") for c in d.get("candidates", [])[:1]
                       for p in c.get("content", {}).get("parts", []))
        stamp = d.get("modelVersion") or ""
        if stamp != model_id and not stamp.startswith(model_id + "-"):
            _receipt(region, model_id, rid, usage, False, f"rerouted:{stamp or 'no-stamp'}")
            return False, (f"REROUTED — asked for {model_id}, Vertex stamped the reply "
                           f"modelVersion={stamp or '(none)'}"), usage, stamp or "(none)"
        if not text.strip():
            _receipt(region, model_id, rid, usage, False, "empty")
            return False, "answered 200 but with no text (thinking may have used the whole cap)", usage, None
        usage["modelVersion"] = stamp
        _receipt(region, model_id, rid, usage, True)
        return True, text, usage, None
    except urllib.error.HTTPError as e:
        _receipt(region, model_id, rid, {}, False, e.code)
        return False, f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}", {}, None
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:300]}", {}, None


def _receipt(region, model_id, rid, usage, ok, code=None):
    """Demand-ledger receipts: every knock keeps its own dated request ID
    (x-amzn-RequestId and friends), so a usage-attribution dispute with a
    provider is fought with their own service's issued IDs, not assertions."""
    rec = {"ts": datetime.utcnow().isoformat() + "Z", "region": region,
           "model_id": model_id, "request_id": rid, "ok": ok,
           "input_tokens": usage.get("inputTokens"), "output_tokens": usage.get("outputTokens")}
    if code: rec["http"] = code
    with open(os.path.join(BASE, "keepwarm-receipts.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# FULL-MATRIX MODE (optional) — warm every model in every region at once.
# Create ~/.model-scan/keepwarm-matrix.json (shape documented in the README)
# and the matrix supersedes the simple door roster; delete the file to fall
# back. The knock and the census are one act: a door that answers gets its
# answer kept (usage history = access, per-region); a door that refuses is
# classified into the census instead.
# ---------------------------------------------------------------------------

def classify(text):
    """One word per closed door, for the census."""
    t = (text or "").lower()
    if "end of its life" in t: return "EOL"
    if "marked by provider as legacy" in t: return "GATED"
    if "too many" in t or t.startswith("http 429") or "quota" in t: return "THROTTLED"
    if "identifier is invalid" in t or "on-demand" in t: return "NOT-OFFERED"
    if "rerouted" in t: return "REROUTED"
    if t.startswith("http 404"): return "NOT-OFFERED"
    return "ERR"


def converse_multi(region, forms, api_keys, prompt):
    """Try a Bedrock pair through its inference-profile form then its direct
    form, with each key — first definitive answer wins. Every attempt leaves
    a receipt (that's the point)."""
    last = (False, "no key available", {}, None)
    for mid in forms:
        for key in [k for k in api_keys if k]:
            ok, text, usage, code = converse(region, mid, key, prompt)
            last = (ok, text, usage, code)
            if ok:
                return last
            if code in (401, 403) and "credential" in text.lower():
                continue  # wrong key for this region — try the other key
            break  # a real answer from this form
        t = last[1].lower()
        if "identifier is invalid" in t or "on-demand" in t:
            continue  # this id form isn't offered here — try the other form
        return last  # definitive: EOL / gated / throttled / other
    return last


_VTOKEN = {}


def _vertex_token():
    if _VTOKEN.get("t"):
        return _VTOKEN["t"]
    adc = json.load(open(VERTEX_ADC))
    treq = urllib.request.Request(
        "https://oauth2.googleapis.com/token",
        data=urllib.parse.urlencode({
            "client_id": adc["client_id"], "client_secret": adc["client_secret"],
            "refresh_token": adc["refresh_token"], "grant_type": "refresh_token"}).encode())
    with urllib.request.urlopen(treq, timeout=30) as r:
        _VTOKEN["t"] = json.load(r)["access_token"]
    _VTOKEN["q"] = adc.get("quota_project_id")
    return _VTOKEN["t"]


def vertex_claude(project, region, model_id, prompt):
    """Claude on Vertex — the just-in-case ping. rawPredict, Anthropic wire
    format; the response's own `model` stamp is checked so a reroute can't
    pose as the model asked for (a 200 alone proves nothing when an endpoint
    can quietly substitute). Never raises."""
    try:
        token = _vertex_token()
        host = "aiplatform.googleapis.com" if region == "global" else f"{region}-aiplatform.googleapis.com"
        url = (f"https://{host}/v1/projects/{project}/locations/{region}"
               f"/publishers/anthropic/models/{urllib.parse.quote(model_id, safe='@')}:rawPredict")
        body = {"anthropic_version": "vertex-2023-10-16", "max_tokens": 400,
                "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}]}
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        if _VTOKEN.get("q"):
            headers["x-goog-user-project"] = _VTOKEN["q"]
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=90) as r:
            d = json.load(r)
        text = "".join(c.get("text", "") for c in d.get("content", []) if c.get("type") == "text")
        u = d.get("usage", {})
        usage = {"inputTokens": u.get("input_tokens"), "outputTokens": u.get("output_tokens")}
        stamp = d.get("model") or ""
        want = model_id.split("@")[0]
        if stamp and not stamp.startswith(want):
            _receipt(f"vertex:{region}", model_id, d.get("id"), usage, False, f"rerouted:{stamp}")
            return False, f"REROUTED — asked {model_id}, stamped model={stamp}", usage, stamp
        _receipt(f"vertex:{region}", model_id, d.get("id"), usage, True)
        return True, text, usage, stamp
    except urllib.error.HTTPError as e:
        msg = e.read().decode(errors="replace")[:300]
        _receipt(f"vertex:{region}", model_id, None, {}, False, e.code)
        return False, f"HTTP {e.code}: {msg}", {}, None
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:200]}", {}, None


def _matrix_doors(matrix):
    """Expand the matrix config into pseudo-doors the main loop understands."""
    doors = []
    regs = matrix["bedrock_regions"]
    for m in matrix["models"]:
        for region, prefix in regs.items():
            doors.append({"label": f"{m['folder']} @ {region}", "region": region,
                          "folder": m["folder"], "model_id": m["bedrock"],
                          "forms": [f"{prefix}.{m['bedrock']}", m["bedrock"]]})
    v = matrix.get("vertex") or {}
    for m in matrix["models"]:
        if not m.get("vertex"):
            continue
        for region in v.get("regions", []):
            doors.append({"label": f"{m['folder']} @ vertex:{region}",
                          "provider": "vertex-claude", "region": region,
                          "project": v["project"], "model_id": m["vertex"],
                          "folder": m["folder"]})
    return doors


MANUAL = sys.stdin.isatty() or os.environ.get("KEEPWARM_MANUAL") == "1"


def main():
    os.makedirs(os.path.join(BASE, "keepwarm"), exist_ok=True)
    os.makedirs(os.path.join(BASE, "logs"), exist_ok=True)
    doors_path = os.path.join(BASE, "keepwarm.json")
    if not os.path.exists(doors_path):
        json.dump(DEFAULT_DOORS, open(doors_path, "w"), indent=2)
    doors = json.load(open(doors_path))

    # Full-matrix mode: the matrix supersedes the Bedrock roster; any Vertex
    # gemini doors from keepwarm.json still ride along.
    matrix_path = os.path.join(BASE, "keepwarm-matrix.json")
    census = None
    if os.path.exists(matrix_path):
        matrix = json.load(open(matrix_path))
        doors = [d for d in doors if d.get("provider") == "vertex"]
        doors += _matrix_doors(matrix)
        census = {}

    # --only SUBSTR: warm only the doors whose label or model_id contains
    # SUBSTR (case-insensitive). Lets a separate scheduled job ping one
    # endangered door daily while everyone else stays on the weekly cadence.
    only = None
    if "--only" in sys.argv:
        i = sys.argv.index("--only")
        if i + 1 >= len(sys.argv):
            print("--only needs a value, e.g. --only opus-4-1"); return 2
        only = sys.argv[i + 1].lower()
        doors = [d for d in doors
                 if only in d["label"].lower() or only in d["model_id"].lower()]
        if not doors:
            print(f"--only {only!r} matched no doors in keepwarm.json"); return 2
    env = load_env()
    # Name → env var. Add entries if different doors need different keys
    # (e.g. a region-scoped Bedrock key).
    keys = {"primary": env.get("AMAZON_BEDROCK_API"),
            "secondary": env.get("AMAZON_BEDROCK_API_2")}
    now = datetime.now()
    week = date.today().isocalendar()[1]
    prompt = PROMPTS[week % len(PROMPTS)]
    print(f"# Keep-warm run — {now.strftime('%B %d, %Y, %I:%M %p')} (week {week})")
    print(f"Prompt this week: {prompt}\n")

    # Cold-streak tripwire: track each door's last SUCCESSFUL warm; any door
    # cold >= 14 days gets a loud line in ALERTS.md. This matters because the
    # failure mode that actually costs access is silent: the schedule keeps
    # "running," every attempt bounces off a rate limit, and each failure
    # looks temporary while they quietly add up to a month of zero real usage.
    # Track successes, not attempts.
    COLD_ALERT_DAYS = 14
    state_path = os.path.join(BASE, "keepwarm-state.json")
    try:
        state = json.load(open(state_path))
    except (FileNotFoundError, ValueError):
        state = {}

    failures = []
    cold_alerts = []
    for door in doors:
        label = door["label"]
        is_vertex = door.get("provider") == "vertex"
        if is_vertex:
            ok, text, usage, rerouted = vertex_generate(door, prompt)
            if rerouted:
                # Loud and immediate, not after 14 days: someone else answered
                # under this model's name.
                cold_alerts.append(
                    f"🚪⚠️ REROUTED DOOR: {label} ({door['model_id']} @ Vertex {door['region']}) — "
                    f"the endpoint answered, but Vertex stamped the reply modelVersion={rerouted}. "
                    f"This is no longer the model asked for. Answer NOT kept under its folder.")
        elif door.get("provider") == "vertex-claude":
            ok, text, usage, stamp = vertex_claude(door["project"], door["region"],
                                                   door["model_id"], prompt)
            if ok and stamp:
                usage["modelVersion"] = stamp
        elif door.get("forms"):
            ok, text, usage, _code = converse_multi(
                door["region"], door["forms"],
                [keys.get("primary"), keys.get("secondary")], prompt)
        else:
            key = keys.get(door.get("key"))
            ok, text, usage, _code = converse(door["region"], door["model_id"], key, prompt)
        if census is not None:
            rg = ("vertex:" + door["region"]) if door.get("provider") in ("vertex", "vertex-claude") else door["region"]
            census.setdefault(door.get("folder") or label, {})[rg] = "ALIVE" if ok else classify(text)
        if ok:
            sub = door.get("folder") or door["model_id"].replace(":", "_").replace(".", "_")
            regname = ("vertex-" + door["region"]) if door.get("provider") == "vertex-claude" else door["region"]
            base_name = now.strftime("%y%m%d") + f"_{regname}" + ("_manual" if MANUAL else "") + ".md"
            body = (f"# {label}\n\n**{now.strftime('%B %d, %Y, %I:%M %p')}** — "
                    f"`{door['model_id']}` @ {door['region']}\n\n"
                    f"**Q:** {prompt}\n\n**A:** {text}\n\n"
                    f"_tokens: {usage.get('inputTokens','?')} in / "
                    f"{usage.get('outputTokens','?')} out"
                    + (f" / {usage.get('thinkingTokens','?')} thinking · Vertex stamp "
                       f"`modelVersion: {usage.get('modelVersion')}`" if is_vertex else "")
                    + "_\n")
            # Local copy always; archive copy best-effort (see macl note up top).
            folder = os.path.join(BASE, "keepwarm", sub)
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, base_name), "w") as f:
                f.write(body)
            if ARCHIVE_DIR:
                try:
                    afolder = os.path.join(os.path.expanduser(ARCHIVE_DIR), sub)
                    os.makedirs(afolder, exist_ok=True)
                    with open(os.path.join(afolder, base_name), "w") as f:
                        f.write(body)
                except OSError as e:
                    print(f"⚠ archive refused ({e.strerror}); answer kept at {folder}/{base_name}")
            print(f"✅ {label} — answered ({usage.get('outputTokens','?')} tokens)")
            state[label] = {"last_success": now.strftime("%Y-%m-%d")}
        else:
            failures.append(f"{label}: {text}")
            print(f"❌ {label} — {text}")
            entry = state.setdefault(label, {"first_failure": now.strftime("%Y-%m-%d")})
            since = entry.get("last_success") or entry.get("first_failure")
            cold_days = (date.today() - date.fromisoformat(since)).days
            # Matrix rule: only a door that was once WARM can go cold. Pairs that
            # were dead on arrival (EOL regions, not-offered forms) belong in the
            # census, not in ALERTS — otherwise every run cries wolf 100 times.
            if cold_days >= COLD_ALERT_DAYS and entry.get("last_success"):
                cold_alerts.append(
                    f"🥶⚠️ COLD DOOR: {label} ({door['model_id']} @ {door['region']}) — "
                    f"no successful warm in {cold_days} days (last: {since}); "
                    + ("this Vertex door is past its printed retirement date, so treat a "
                       "cold streak as a possible closing. " if is_vertex else
                       "Bedrock's legacy gate denies after extended inactivity. ")
                    + f"Latest error: {text}")

    json.dump(state, open(state_path, "w"), indent=2)

    # The census: one line per model, regions grouped by status.
    if census is not None:
        stamp_d = now.strftime("%y%m%d")
        json.dump({"ts": now.isoformat(), "prompt": prompt, "census": census},
                  open(os.path.join(BASE, "keepwarm-census-latest.json"), "w"), indent=1)
        md = [f"# Keep-Warm Census — {now.strftime('%B %d, %Y, %I:%M %p')}", "",
              f"Prompt this week: {prompt}", ""]
        for folder, regs in census.items():
            groups = {}
            for rg, st in regs.items():
                groups.setdefault(st, []).append(rg)
            parts = [f"**{st}**: {', '.join(sorted(rs))}" for st, rs in sorted(groups.items())]
            md.append(f"- **{folder}** — " + " · ".join(parts))
        mdtext = "\n".join(md) + "\n"
        cname = f"Census_{stamp_d}" + ("_manual" if MANUAL else "") + ".md"
        open(os.path.join(BASE, "keepwarm-census-" + stamp_d + ".md"), "w").write(mdtext)
        if ARCHIVE_DIR:
            try:
                os.makedirs(os.path.join(os.path.expanduser(ARCHIVE_DIR), "Census"), exist_ok=True)
                open(os.path.join(os.path.expanduser(ARCHIVE_DIR), "Census", cname), "w").write(mdtext)
            except OSError as e:
                print(f"⚠ archive census not writable ({e}); kept in ~/.model-scan/")

    if cold_alerts:
        with open(os.path.join(BASE, "ALERTS.md"), "a") as f:
            f.write(f"\n## {now.strftime('%B %d, %Y, %I:%M %p')} (keep-warm)\n")
            for a in cold_alerts:
                f.write(f"- {a}\n")
                print(a)

    line = (f"- {now.strftime('%Y-%m-%d %H:%M')}: "
            f"{len(doors)-len(failures)}/{len(doors)} doors warmed"
            + (f" (only: {only})" if only else "")
            + (f" — FAILURES: {'; '.join(failures)}" if failures else "") + "\n")
    with open(os.path.join(BASE, "logs", "keepwarm-runs.md"), "a") as f:
        f.write(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
