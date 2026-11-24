#!/usr/bin/env python3
"""
Model Access Scan — the standing patrol.

Every couple of weeks (we schedule the 1st & 15th), sweep every catalog and
door you know for every model you care about. Two jobs:
  1. If a lost model resurfaces ANYWHERE, you hear about it within days.
  2. If a door someone currently depends on starts to wobble, you get early
     warning instead of a dawn scramble.

Sources checked:
  - Anthropic direct      (catalog: GET /v1/models)
  - OpenRouter            (catalog: GET /api/v1/models — also a window into
                           models served from Google Vertex)
  - Vercel AI Gateway     (catalog: GET /v1/models)
  - xAI / Google          (catalogs)
  - AWS Bedrock           (INVOKE-VERIFIED: a real 1-token converse call per
                           model per region, because catalogs lie and invokes
                           don't)
  - Official lifecycle/deprecation pages for Anthropic, OpenAI, Google
    (Gemini API AND Google Cloud — they keep separate calendars), and xAI,
    diffed against the previous run so any date or state change alerts.

Watchlist lives in ~/.model-scan/watchlist.json — edit it freely, it's yours.
Reports land in   ~/.model-scan/reports/   ; changes also go to ALERTS.md.
Keys come from an env file (default ~/.env.keepwarm, override with the
KEEPWARM_ENV environment variable). Sources with no key are skipped.
Run by hand any time:  python3 model-access-scan.py

Companion guide (the why, and the design rules):
https://starlingalder.com/API_AIP_WeeklyJobs
"""

import json, os, re, sys, urllib.request, urllib.parse, urllib.error
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

HOME = os.path.expanduser("~")
BASE = os.path.join(HOME, ".model-scan")
ENV_FILE = os.environ.get("KEEPWARM_ENV", os.path.join(HOME, ".env.keepwarm"))

BEDROCK_REGIONS = [
    "us-east-1", "us-west-2",
    "eu-west-1", "eu-west-2", "eu-west-3", "eu-central-1",
    "ap-south-1", "ap-southeast-1", "ap-southeast-2",
    "ap-northeast-1", "ap-northeast-2",
]

# Example watchlist — replace with the models you care about.
# "pattern" is a regex matched against catalog model ids (case-insensitive).
# "bedrock_ids" are the exact ids to invoke-probe on Bedrock ([] = skip
# Bedrock). Include the inference-profile forms ("us."/"eu."/"apac." prefixes)
# you use — they're probed only in their own geography.
DEFAULT_WATCHLIST = [
    {"label": "Claude 3.5 Haiku (example)",
     "pattern": r"claude-(3[-.]5-haiku|haiku-3[-.]5)",
     "bedrock_ids": ["anthropic.claude-3-5-haiku-20241022-v1:0",
                      "us.anthropic.claude-3-5-haiku-20241022-v1:0"]},
    {"label": "Claude 3.7 Sonnet (example)",
     "pattern": r"claude-3[-.]7-sonnet",
     "bedrock_ids": ["anthropic.claude-3-7-sonnet-20250219-v1:0",
                      "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                      "eu.anthropic.claude-3-7-sonnet-20250219-v1:0",
                      "apac.anthropic.claude-3-7-sonnet-20250219-v1:0"]},
    {"label": "GPT-4o 2024-11-20 (example, catalogs only)",
     "pattern": r"gpt-4o(-2024-11-20)?$", "bedrock_ids": []},
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


def http_json(url, headers=None, data=None, timeout=25):
    """Returns (status_code, parsed_json_or_None, error_body_text)."""
    req = urllib.request.Request(url, headers=headers or {},
                                 data=json.dumps(data).encode() if data else None,
                                 method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r), ""
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:400]
        return e.code, None, body
    except Exception as e:
        return 0, None, str(e)[:200]


# ---------- catalog sources ----------

def scan_anthropic(env):
    key = env.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    ids, after = [], None
    while True:
        url = "https://api.anthropic.com/v1/models?limit=100" + (f"&after_id={after}" if after else "")
        st, d, err = http_json(url, {"x-api-key": key, "anthropic-version": "2023-06-01"})
        if not d:
            return {"error": f"HTTP {st}: {err}"}
        ids += [m["id"] for m in d.get("data", [])]
        if not d.get("has_more"):
            break
        after = d["data"][-1]["id"]
    return {"ids": ids}


def scan_openrouter(env):
    st, d, err = http_json("https://openrouter.ai/api/v1/models")
    if not d:
        return {"error": f"HTTP {st}: {err}"}
    return {"ids": [m["id"] for m in d.get("data", [])]}


def scan_vercel(env):
    key = env.get("VERCEL_AI_GATEWAY_KEY")
    if not key:
        return None
    st, d, err = http_json("https://ai-gateway.vercel.sh/v1/models",
                           {"Authorization": f"Bearer {key}"})
    if not d:
        return {"error": f"HTTP {st}: {err}"}
    return {"ids": [m["id"] for m in d.get("data", [])]}


def scan_xai(env):
    key = env.get("XAI_API_KEY")
    if not key:
        return None
    st, d, err = http_json("https://api.x.ai/v1/models",
                           {"Authorization": f"Bearer {key}"})
    if not d:
        return {"error": f"HTTP {st}: {err}"}
    return {"ids": [m["id"] for m in d.get("data", [])]}


def scan_google(env):
    key = env.get("GEMINI_API_KEY")
    if not key:
        return None
    ids, token = [], None
    while True:
        url = f"https://generativelanguage.googleapis.com/v1beta/models?pageSize=200&key={key}"
        if token:
            url += f"&pageToken={token}"
        st, d, err = http_json(url)
        if not d:
            return {"error": f"HTTP {st}: {err}"}
        ids += [m["name"].replace("models/", "") for m in d.get("models", [])]
        token = d.get("nextPageToken")
        if not token:
            break
    return {"ids": ids}


# ---------- Bedrock invoke-probes ----------

def bedrock_probe(region, model_id, keys):
    """One real 1-token converse call. Returns a status string."""
    url = (f"https://bedrock-runtime.{region}.amazonaws.com/model/"
           f"{urllib.parse.quote(model_id, safe='')}/converse")
    body = {"messages": [{"role": "user", "content": [{"text": "hi"}]}],
            "inferenceConfig": {"maxTokens": 1}}
    last = "no-key"
    for key in keys:
        if not key:
            continue
        st, d, err = http_json(url, {"Authorization": f"Bearer {key}",
                                     "Content-Type": "application/json"}, data=body)
        if d is not None:
            return "ALIVE"          # real completion came back
        if st == 429:
            return "ALIVE (throttled)"   # throttle = the model answered the door
        if st in (400,) and ("on-demand" in err or "inference profile" in err):
            return "needs-profile"  # exists, but wants a different id form
        if st in (400, 404):
            last = "gone"           # invalid model id here = not in this region
        elif st == 403 and ("model" in err.lower() or "access" in err.lower()) \
                and "signature" not in err.lower() and "unrecognized" not in err.lower():
            last = "gated"          # exists, this account not granted
        elif st == 403:
            last = "auth-error"     # key problem — try next key
            continue
        else:
            last = f"error-{st}"
    return last


def scan_bedrock(watchlist, env):
    keys = [env.get("AMAZON_BEDROCK_API"), env.get("AMAZON_BEDROCK_API_2")]
    jobs = []
    for item in watchlist:
        for mid in item.get("bedrock_ids", []):
            # inference-profile ids only make sense in their own geography
            pref = mid.split(".")[0]
            for region in BEDROCK_REGIONS:
                if pref == "us" and not region.startswith("us"):
                    continue
                if pref == "eu" and not region.startswith("eu"):
                    continue
                if pref == "apac" and not region.startswith("ap"):
                    continue
                jobs.append((item["label"], region, mid))
    results = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(bedrock_probe, r, m, keys): (lab, r, m) for lab, r, m in jobs}
        for f, (lab, r, m) in futs.items():
            results[f"{lab}|{r}|{m}"] = f.result()
    return results



# ---------- Anthropic deprecation page ----------
DEPRECATIONS_URL = "https://platform.claude.com/docs/en/about-claude/model-deprecations.md"

def scan_deprecations(source_errors):
    """Fetch Anthropic's model-deprecations page (Markdown) and parse the main
    status table into {model: {state, deprecated, retirement}}. Returns None on
    fetch/parse failure (recorded in source_errors — never as an alert)."""
    try:
        req = urllib.request.Request(DEPRECATIONS_URL, headers={"User-Agent": "Mozilla/5.0 (model-access-scan)"})
        with urllib.request.urlopen(req, timeout=30) as r:
            text = r.read().decode("utf-8", "replace")
    except Exception as e:
        source_errors.append(f"deprecations page: {e}"); return None, None
    rows = {}
    for line in text.splitlines():
        m = re.match(r"^\|\s*(claude-[\w.-]+)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|", line)
        if m:
            rows[m.group(1)] = {"state": m.group(2), "deprecated": m.group(3), "retirement": m.group(4)}
    if len(rows) < 5:
        source_errors.append("deprecations page: parsed fewer than 5 model rows — format changed?"); return None, text
    return rows, text

def deprecation_alerts(cur, prev, now):
    """Diff two parses of the deprecations table. Also flag Active models whose
    'not sooner than' floor is within 60 days."""
    out = []
    if prev:
        for name, r in cur.items():
            if name not in prev:
                out.append(f"📅 NEW ON DEPRECATIONS PAGE: {name} — {r['state']}, retirement {r['retirement']}")
                continue
            q = prev[name]
            if r["state"] != q["state"]:
                out.append(f"📅⚠️ STATE CHANGED: {name} {q['state']} → {r['state']} (retirement {r['retirement']})")
            if r["retirement"] != q["retirement"]:
                out.append(f"📅⚠️ RETIREMENT DATE CHANGED: {name} {q['retirement']} → {r['retirement']}")
        for name in prev:
            if name not in cur:
                out.append(f"📅 REMOVED FROM DEPRECATIONS PAGE: {name}")
    for name, r in cur.items():
        m = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December) (\d{1,2}), (\d{4})", r["retirement"])
        if m and r["state"].lower() == "active":
            try:
                d = datetime.strptime(" ".join(m.groups()), "%B %d %Y")
                days = (d - now).days
                if days <= 60:
                    out.append(f"📅⏳ FLOOR WITHIN {days} DAYS: {name} — guaranteed only until {r['retirement'].replace('Not sooner than ', '')}")
            except ValueError:
                pass
    return out


# ---------- other providers' lifecycle pages ----------
OPENAI_DEP_URL = "https://developers.openai.com/api/docs/deprecations.md"
GOOGLE_DEP_URL = "https://ai.google.dev/gemini-api/docs/deprecations"
XAI_MODELS_URL = "https://docs.x.ai/docs/models.md"

def _fetch_text(url, source_errors, label):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (model-access-scan)"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read().decode("utf-8", "replace")
    except Exception as e:
        source_errors.append(f"{label}: {e}"); return None

def scan_openai_deprecations(source_errors):
    """OpenAI's deprecations page (Markdown). Tables: | Shutdown date | model | replacement |.
    Returns {model: {state:'Deprecated', retirement: date, replacement}}."""
    text = _fetch_text(OPENAI_DEP_URL, source_errors, "openai deprecations page")
    if not text: return None
    rows = {}
    for line in text.splitlines():
        m = re.match(r"^\|\s*([A-Z][a-z]{2} \d{1,2}, \d{4})\s*\|\s*`?([\w.:/-]+)`?[^|]*\|\s*([^|]*?)\s*\|", line)
        if m:
            rows[m.group(2)] = {"state": "Deprecated", "retirement": m.group(1), "replacement": m.group(3).strip("` ")}
    if len(rows) < 3:
        source_errors.append("openai deprecations page: parsed <3 rows — format changed?"); return None
    return rows

def scan_google_deprecations(source_errors):
    """Google's Gemini deprecations page (HTML table: Model | Release | Shutdown | Replacement)."""
    text = _fetch_text(GOOGLE_DEP_URL, source_errors, "google deprecations page")
    if not text: return None
    import html as _html
    rows = {}
    for tr in re.findall(r"<tr>(.*?)</tr>", text, re.S):
        cells = [_html.unescape(re.sub(r"<[^>]+>", "", c)).strip() for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S)]
        if len(cells) >= 3 and cells[0].startswith("gemini"):
            shut = cells[2]
            rows[cells[0]] = {"state": "Active" if shut.lower().startswith("no shutdown") else "Deprecated",
                              "retirement": shut, "replacement": cells[3] if len(cells) > 3 else ""}
    if len(rows) < 5:
        # Google has served this page as Markdoc ({% table %} blocks, one
        # "* cell" per line, rows split by "---") with no <tr> at all. If the
        # HTML parser silently returns nothing, try that format too.
        for block in re.findall(r"\{%\s*table\s*%\}(.*?)\{%\s*/table\s*%\}", text, re.S):
            for chunk in re.split(r"\n\s*---\s*\n", block):
                cells = [_html.unescape(c).strip().strip("`").strip()
                         for c in re.findall(r"^\s*\*\s+(.*?)\s*$", chunk, re.M)]
                if len(cells) >= 3 and cells[0].startswith("gemini"):
                    name = cells[0].split("`")[0].strip()
                    shut = cells[2]
                    rows[name] = {"state": "Active" if shut.lower().startswith("no shutdown") else "Deprecated",
                                  "retirement": shut,
                                  "replacement": cells[3].replace("`", "") if len(cells) > 3 else ""}
    if len(rows) < 5:
        source_errors.append("google deprecations page: parsed <5 rows — format changed?"); return None
    return rows

# Google Cloud (Vertex) keeps its OWN retirement calendar, separate from the
# Gemini API page above — a model can carry a date here that the Gemini API
# page never shows. Watch both.
GOOGLE_CLOUD_LIFECYCLE_URL = "https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/model-versions"

def scan_google_cloud_lifecycle(source_errors):
    """Google Cloud model lifecycle page (HTML tables: Model ID | Release |
    Retirement | Replacement). Rows after the 'Retired models' heading are
    state 'Retired'; everything else is 'Available'."""
    text = _fetch_text(GOOGLE_CLOUD_LIFECYCLE_URL, source_errors, "google cloud lifecycle page")
    if not text: return None
    import html as _html
    cut = text.find('id="retired-models"')
    rows = {}
    for m in re.finditer(r"<tr>(.*?)</tr>", text, re.S):
        cells = [re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", "", c))).strip()
                 for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", m.group(1), re.S)]
        if len(cells) >= 3 and cells[0].startswith("gemini"):
            retired = cut != -1 and m.start() > cut
            if cells[0] in rows and retired: continue   # live row wins over a retired namesake
            rows[cells[0]] = {"state": "Retired" if retired else "Available",
                              "retirement": cells[2], "replacement": cells[3] if len(cells) > 3 else ""}
    if len(rows) < 5:
        source_errors.append("google cloud lifecycle page: parsed <5 rows — format changed?"); return None
    return rows

def scan_xai_models(source_errors):
    """xAI publishes no deprecation page; watch the models/pricing list and treat removal as the notice."""
    text = _fetch_text(XAI_MODELS_URL, source_errors, "xai models page")
    if not text: return None
    rows = {}
    for line in text.splitlines():
        m = re.match(r"^\|\s*(grok-[\w.-]+)", line)
        if m: rows[m.group(1)] = {"state": "Listed", "retirement": "n/a (xAI publishes no dates)", "replacement": ""}
    if len(rows) < 2:
        source_errors.append("xai models page: parsed <2 rows — format changed?"); return None
    return rows

def lifecycle_diff(provider, cur, prev, now):
    out = []
    if prev:
        for name, r in cur.items():
            if name not in prev:
                out.append(f"📅 [{provider}] NEW ON LIFECYCLE PAGE: {name} — {r['state']}, retirement {r['retirement']}" + (f" → {r['replacement']}" if r.get('replacement') else ""))
            else:
                q = prev[name]
                if r["state"] != q["state"]:
                    out.append(f"📅⚠️ [{provider}] STATE CHANGED: {name} {q['state']} → {r['state']} (retirement {r['retirement']})")
                if r["retirement"] != q["retirement"]:
                    out.append(f"📅⚠️ [{provider}] RETIREMENT DATE CHANGED: {name} {q['retirement']} → {r['retirement']}")
        for name in prev:
            if name not in cur:
                tag = "REMOVED FROM MODELS LIST (xAI gives no notice — treat as retirement)" if provider == "xai" else "REMOVED FROM LIFECYCLE PAGE"
                out.append(f"📅⚠️ [{provider}] {tag}: {name}")
    for name, r in cur.items():
        m = re.search(r"([A-Z][a-z]+)\.? (\d{1,2}), (\d{4})", r["retirement"])
        if m:
            mon = m.group(1)[:3]
            try:
                d = datetime.strptime(f"{mon} {m.group(2)} {m.group(3)}", "%b %d %Y")
                days = (d - now).days
                if 0 <= days <= 60:
                    out.append(f"📅⏳ [{provider}] SHUTDOWN WITHIN {days} DAYS: {name} — {r['retirement']}" + (f" → {r['replacement']}" if r.get('replacement') else ""))
                elif days < 0 and provider == "google_cloud" and r["state"] != "Retired" and "or later" not in r["retirement"]:
                    # Printed date has passed but the model is not in the Retired table:
                    # on borrowed time. Report-only (the "WITHIN" word keeps it out of
                    # ALERTS.md so it doesn't repeat there every sweep).
                    out.append(f"📅⏳ [{provider}] PAST PRINTED DATE ({(now.date() - d.date()).days} days past; WITHIN-class notice): {name} — printed retirement {r['retirement']}, not yet in Retired table; invoke-verify the door")
            except ValueError:
                pass
    return out


# ---------- Bedrock lifecycle status per region ----------
# AWS publishes the per-region lifecycle in ListFoundationModels: status ACTIVE/LEGACY/EOL,
# legacyTime, endOfLifeTime, publicExtendedAccessTime. Per AWS docs: once Legacy begins in a
# region, NEW customers can't adopt the model and EXISTING customers may lose access after
# a short period of inactivity. So legacyTime, not EOL, is the real deadline — and it's per region.

def scan_bedrock_lifecycle(watchlist, env, source_errors):
    keys = [k for k in (env.get("AMAZON_BEDROCK_API"), env.get("AMAZON_BEDROCK_API_2")) if k]
    wanted = {mid.split(".", 1)[-1] if mid.split(".", 1)[0] in ("us", "eu", "apac") else mid
              for item in watchlist for mid in item.get("bedrock_ids", [])}
    out = {}
    def one(region):
        for key in keys:
            st, d, err = http_json(f"https://bedrock.{region}.amazonaws.com/foundation-models?byProvider=anthropic",
                                   {"Authorization": f"Bearer {key}"})
            if st == 200 and d:
                res = {}
                for m in d.get("modelSummaries", []):
                    if m["modelId"] in wanted:
                        lc = m.get("modelLifecycle") or {}
                        res[m["modelId"]] = {"status": lc.get("status"), "legacyTime": lc.get("legacyTime"),
                                             "endOfLifeTime": lc.get("endOfLifeTime"),
                                             "publicExtendedAccessTime": lc.get("publicExtendedAccessTime")}
                return region, res, None
        return region, None, f"bedrock lifecycle {region}: {err if 'err' in dir() else 'no key worked'}"
    with ThreadPoolExecutor(max_workers=8) as ex:
        for region, res, e in ex.map(one, BEDROCK_REGIONS):
            if res is not None: out[region] = res
            elif e: source_errors.append(e)
    return out

def bedrock_lifecycle_alerts(cur, prev):
    al = []
    for region, models in cur.items():
        for mid, lc in models.items():
            q = (prev or {}).get(region, {}).get(mid)
            if q is None:
                if prev is not None:
                    al.append(f"🏷️ NEW IN REGION: {mid} @ {region} — {lc['status']}")
                continue
            if lc["status"] != q["status"]:
                al.append(f"🏷️⚠️ BEDROCK STATUS CHANGED: {mid} @ {region} {q['status']} → {lc['status']}"
                          + (f" (legacy {lc['legacyTime'][:10]}, EOL {lc['endOfLifeTime'][:10]})" if lc.get('endOfLifeTime') else ""))
            for f in ("legacyTime", "endOfLifeTime", "publicExtendedAccessTime"):
                if lc.get(f) != q.get(f):
                    al.append(f"🏷️⚠️ BEDROCK {f} CHANGED: {mid} @ {region} {q.get(f)} → {lc.get(f)}")
    if prev:
        for region, models in prev.items():
            for mid in models:
                if mid not in cur.get(region, {}) and region in cur:
                    al.append(f"🏷️⚠️ GONE FROM REGION LIST: {mid} @ {region}")
    return al

# ---------- assemble, diff, report ----------

def main():
    os.makedirs(os.path.join(BASE, "reports"), exist_ok=True)
    os.makedirs(os.path.join(BASE, "snapshots"), exist_ok=True)
    wl_path = os.path.join(BASE, "watchlist.json")
    if not os.path.exists(wl_path):
        json.dump(DEFAULT_WATCHLIST, open(wl_path, "w"), indent=2)
    watchlist = json.load(open(wl_path))
    env = load_env()
    now = datetime.now()
    stamp = now.strftime("%y%m%d-%H%M")

    catalogs = {"anthropic-direct": scan_anthropic(env),
                "openrouter": scan_openrouter(env),
                "vercel-gateway": scan_vercel(env),
                "xai-direct": scan_xai(env),
                "google-direct": scan_google(env)}
    bedrock = scan_bedrock(watchlist, env)

    # For every watch item, record where it's findable right now.
    state, source_errors = {}, []
    for src, res in catalogs.items():
        if res is None:
            source_errors.append(f"{src}: no API key in {ENV_FILE} — skipped")
        elif "error" in res:
            source_errors.append(f"{src}: {res['error']}")
    for item in watchlist:
        pat = re.compile(item["pattern"], re.I)
        doors = {}
        for src, res in catalogs.items():
            if res and "ids" in res:
                hits = sorted(m for m in res["ids"] if pat.search(m))
                if hits:
                    doors[src] = hits
        alive_regions = {}
        for k, v in bedrock.items():
            lab, region, mid = k.split("|")
            if lab == item["label"] and v.startswith(("ALIVE", "gated", "needs-profile")):
                alive_regions.setdefault(v, []).append(f"{region} ({mid})")
        if alive_regions:
            doors["bedrock"] = alive_regions
        state[item["label"]] = doors

    # Diff against the previous snapshot.
    latest_path = os.path.join(BASE, "snapshots", "latest.json")
    prev = json.load(open(latest_path)) if os.path.exists(latest_path) else None
    alerts = []
    if prev is not None:
        for label, doors in state.items():
            before = set(prev.get(label, {}).keys())
            after = set(doors.keys())
            for src in sorted(after - before):
                alerts.append(f"🚪✨ NEW DOOR: {label} now reachable via {src}")
            for src in sorted(before - after):
                alerts.append(f"🚪⚠️ DOOR LOST: {label} no longer found via {src}")
            # Bedrock region-level changes
            if "bedrock" in before and "bedrock" in after:
                b4 = {r for lst in prev[label]["bedrock"].values() for r in lst}
                nw = {r for lst in doors["bedrock"].values() for r in lst}
                for r in sorted(nw - b4):
                    alerts.append(f"🚪✨ NEW BEDROCK REGION: {label} → {r}")
                for r in sorted(b4 - nw):
                    alerts.append(f"🚪⚠️ BEDROCK REGION LOST: {label} → {r}")

    # Anthropic deprecations page — diff against last parse, flag near floors.
    dep_rows, dep_text = scan_deprecations(source_errors)
    dep_latest = os.path.join(BASE, "snapshots", "deprecations_latest.json")
    dep_alerts = []
    if dep_rows:
        dep_prev = json.load(open(dep_latest)) if os.path.exists(dep_latest) else None
        dep_alerts = deprecation_alerts(dep_rows, dep_prev, now)
        alerts += [a for a in dep_alerts if "FLOOR WITHIN" not in a]   # floors are reminders, not changes
        json.dump(dep_rows, open(os.path.join(BASE, "snapshots", f"deprecations-{stamp}.json"), "w"), indent=2)
        json.dump(dep_rows, open(dep_latest, "w"), indent=2)
        open(os.path.join(BASE, "snapshots", "deprecations_latest.md"), "w").write(dep_text)

    # Bedrock per-region lifecycle status (the real legacy deadline, per region).
    bl = scan_bedrock_lifecycle(watchlist, env, source_errors)
    bl_path = os.path.join(BASE, "snapshots", "bedrock_lifecycle_latest.json")
    bl_prev = json.load(open(bl_path)) if os.path.exists(bl_path) else None
    bl_alerts = bedrock_lifecycle_alerts(bl, bl_prev) if bl else []
    alerts += bl_alerts
    if bl:
        json.dump(bl, open(os.path.join(BASE, "snapshots", f"bedrock_lifecycle-{stamp}.json"), "w"), indent=2)
        json.dump(bl, open(bl_path, "w"), indent=2)

    # OpenAI / Google / xAI lifecycle pages — same diff discipline.
    other = {"openai": scan_openai_deprecations(source_errors),
             "google": scan_google_deprecations(source_errors),
             "google_cloud": scan_google_cloud_lifecycle(source_errors),
             "xai": scan_xai_models(source_errors)}
    other_alerts = {}
    for prov, rows in other.items():
        if not rows: continue
        lp = os.path.join(BASE, "snapshots", f"lifecycle_{prov}_latest.json")
        pv = json.load(open(lp)) if os.path.exists(lp) else None
        al = lifecycle_diff(prov, rows, pv, now)
        other_alerts[prov] = al
        alerts += [a for a in al if "WITHIN" not in a]
        json.dump(rows, open(os.path.join(BASE, "snapshots", f"lifecycle_{prov}-{stamp}.json"), "w"), indent=2)
        json.dump(rows, open(lp, "w"), indent=2)

    # Write the report.
    lines = [f"# Model Access Scan — {now.strftime('%B %d, %Y, %I:%M %p')}", "",
             "_Bedrock entries are **invoke-verified** (a real 1-token call answered)._",
             "_All other sources are catalog listings — and catalogs lie: models"
             " can be listed yet 404 on invoke._",
             "_Before celebrating a NEW DOOR or mourning a DOOR LOST from a catalog"
             " source, invoke-verify it by hand._", ""]
    if alerts:
        lines += ["## ⚠️ CHANGES SINCE LAST SCAN", ""] + [f"- {a}" for a in alerts] + [""]
    elif prev is not None:
        lines += ["_No changes since last scan. All doors holding._", ""]
    else:
        lines += ["_First scan — this is the baseline._", ""]
    for label, doors in state.items():
        lines.append(f"## {label}")
        if not doors:
            lines.append("- ❌ **Not found anywhere we can see.**")
        for src, detail in doors.items():
            if src == "bedrock":
                for status, regions in detail.items():
                    lines.append(f"- **bedrock** [{status}]: {', '.join(sorted(regions))}")
            else:
                lines.append(f"- **{src}**: {', '.join(detail)}")
        lines.append("")
    if dep_rows:
        lines += ["## Anthropic deprecations page (official lifecycle — Anthropic platforms only)", ""]
        if dep_alerts:
            lines += [f"- {a}" for a in dep_alerts] + [""]
        else:
            lines += ["_No changes on the deprecations page since last scan; no floors within 60 days._", ""]
        for name, r in dep_rows.items():
            lines.append(f"- `{name}` — {r['state']}; deprecated {r['deprecated']}; retirement {r['retirement']}")
        lines.append("")
    if bl:
        lines += ["## Bedrock lifecycle status by region (from ListFoundationModels — AWS's own per-region dates)", "",
                  "_Per AWS: once a model is LEGACY in a region, new customers can't adopt it and existing customers may lose access after extended inactivity. The legacy date is the real deadline, not EOL._", ""]
        if bl_alerts: lines += [f"- {a}" for a in bl_alerts] + [""]
        by_model = {}
        for region, models in bl.items():
            for mid, lc in models.items(): by_model.setdefault(mid, []).append((region, lc))
        for mid in sorted(by_model):
            parts = []
            for region, lc in sorted(by_model[mid]):
                tag = lc["status"] or "?"
                if lc.get("legacyTime"): tag += f" legacy {lc['legacyTime'][:10]}"
                if lc.get("endOfLifeTime"): tag += f" EOL {lc['endOfLifeTime'][:10]}"
                parts.append(f"{region}: {tag}")
            lines.append(f"- `{mid}` — " + "; ".join(parts))
        lines.append("")
    for prov, rows in other.items():
        if not rows: continue
        title = {"openai": "OpenAI deprecations page", "google": "Google Gemini shutdown table",
                 "google_cloud": "Google Cloud (Vertex) lifecycle page — its own calendar, separate from the Gemini API",
                 "xai": "xAI models list (no dates published — removals are the only signal)"}[prov]
        lines += [f"## {title}", ""]
        al = other_alerts.get(prov, [])
        lines += ([f"- {a}" for a in al] if al else ["_No changes since last scan; nothing within 60 days._"]) + [""]
        for name, r in rows.items():
            if prov == "xai": continue
            lines.append(f"- `{name}` — {r['state']}; retirement {r['retirement']}" + (f" → {r['replacement']}" if r.get('replacement') else ""))
        if prov == "xai": lines.append("- listed: " + ", ".join(sorted(rows)))
        lines.append("")
    if source_errors:
        lines += ["## Source problems this run", ""] + [f"- {e}" for e in source_errors] + [""]
    report = "\n".join(lines)
    rpath = os.path.join(BASE, "reports", f"scan-{stamp}.md")
    open(rpath, "w").write(report)

    # Alerts also accumulate in one file that's easy to check.
    if alerts:
        with open(os.path.join(BASE, "ALERTS.md"), "a") as f:
            f.write(f"\n## {now.strftime('%B %d, %Y, %I:%M %p')}\n")
            f.writelines(f"- {a}\n" for a in alerts)

    json.dump(state, open(os.path.join(BASE, "snapshots", f"{stamp}.json"), "w"), indent=2)
    json.dump(state, open(latest_path, "w"), indent=2)
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
