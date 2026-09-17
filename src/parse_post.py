import json
import re
import glob
import os
import csv
from collections import Counter
from datetime import datetime, timezone

POSTS_DIR = "info"
INFLUENCERS_FILE = "influencers.txt"
OUTPUT_TYPE_SAMPLES = "type_samples.json"
OUTPUT_SAMPLE_CSV = "parsed_sample.csv"
OUTPUT_REPORT = "dataset_report.txt"


def load_influencers(path):
    influencers = {}
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < 5:
            continue
        username, category, followers, followees, posts = parts[:5]
        try:
            influencers[username] = {
                "category": category,
                "followers": int(followers),
                "followees": int(followees),
                "posts_count": int(posts),
            }
        except ValueError:
            continue
    return influencers


def parse_post(filepath):
    with open(filepath, encoding="utf-8") as f:
        d = json.load(f)

    caption_edges = d.get("edge_media_to_caption", {}).get("edges", [])
    caption = caption_edges[0]["node"]["text"] if caption_edges else ""
    hashtags = re.findall(r"#\w+", caption)

    typename = d.get("__typename")
    media_type = {
        "GraphImage": "image",
        "GraphVideo": "video",
        "GraphSidecar": "carousel",
    }.get(typename, typename)

    ts = d.get("taken_at_timestamp")
    timestamp = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None

    sidecar_children = d.get("edge_sidecar_to_children", {}).get("edges", [])

    return {
        "post_id": d.get("id"),
        "shortcode": d.get("shortcode"),
        "username": d.get("owner", {}).get("username"),
        "caption": caption,
        "caption_length": len(caption),
        "hashtags": hashtags,
        "hashtag_count": len(hashtags),
        "likes": d.get("edge_media_preview_like", {}).get("count"),
        "comments": d.get("edge_media_to_parent_comment", {}).get("count"),
        "comments_disabled": d.get("comments_disabled", False),
        "timestamp": timestamp,
        "media_type": media_type,
        "typename": typename,
        "is_sponsored": bool(d.get("edge_media_to_sponsor_user", {}).get("edges")),
        "usertag_count": len(d.get("edge_media_to_tagged_user", {}).get("edges", [])),
        "has_location": d.get("location") is not None,
        "video_view_count": d.get("video_view_count"),
        "sidecar_child_count": len(sidecar_children) if sidecar_children else None,
        "raw_top_level_keys": list(d.keys()),
    }


def describe(name, vals, lines):
    if not vals:
        lines.append(f"{name}: no data")
        return
    v = sorted(vals)
    n = len(v)
    msg = f"{name}: n={n} min={v[0]} max={v[-1]} mean={sum(v)/n:.2f} median={v[n//2]}"
    print(msg)
    lines.append(msg)


def main():
    lines = []

    def log(msg):
        print(msg)
        lines.append(msg)

    influencers = load_influencers(INFLUENCERS_FILE)
    log(f"Loaded {len(influencers)} influencers from {INFLUENCERS_FILE}")

    files = glob.glob(os.path.join(POSTS_DIR, "*.info"))
    log(f"Found {len(files)} post files in {POSTS_DIR}")

    type_counter = Counter()
    empty_files = []
    corrupt_files = []
    error_files = []
    type_samples = {}
    parsed_rows = []
    matched_users = set()
    unmatched_users = set()

    report_every = max(1, len(files) // 10)

    for i, f in enumerate(files):
        if i % report_every == 0:
            print(f"...processed {i}/{len(files)}")

        if os.path.getsize(f) == 0:
            empty_files.append(f)
            continue

        try:
            row = parse_post(f)
        except json.JSONDecodeError as e:
            corrupt_files.append((f, str(e)))
            continue
        except Exception as e:
            error_files.append((f, str(e)))
            continue

        t = row["typename"]
        type_counter[t] += 1

        if t not in type_samples:
            with open(f, encoding="utf-8") as fh:
                type_samples[t] = json.load(fh)

        username = row["username"]
        if username in influencers:
            matched_users.add(username)
            row["category"] = influencers[username]["category"]
            row["followers"] = influencers[username]["followers"]
        else:
            unmatched_users.add(username)
            row["category"] = None
            row["followers"] = None

        parsed_rows.append(row)

    log("")
    log(f"Successfully parsed: {len(parsed_rows)}")
    log(f"Empty files (0 bytes): {len(empty_files)}")
    log(f"Corrupt/truncated JSON: {len(corrupt_files)}")
    log(f"Other parse errors: {len(error_files)}")

    bad_total = len(empty_files) + len(corrupt_files)
    if files and bad_total / len(files) > 0.01:
        log(f"WARNING: {bad_total} of {len(files)} files ({bad_total/len(files)*100:.1f}%) are empty or corrupt.")
        log("This usually means the split-zip extraction was incomplete (a missing .zXX part).")
        log("Re-check all archive parts before trusting these results.")

    log(f"__typename breakdown: {dict(type_counter)}")
    log(f"Users matched to influencers.txt: {len(matched_users)}")
    log(f"Users NOT matched (no follower/category data): {len(unmatched_users)}")

    timestamps = [r["timestamp"] for r in parsed_rows if r["timestamp"]]
    if timestamps:
        log(f"Date range: {min(timestamps)} to {max(timestamps)}")

    likes = [r["likes"] for r in parsed_rows if r["likes"] is not None]
    comments = [r["comments"] for r in parsed_rows if r["comments"] is not None]
    hashtag_counts = [r["hashtag_count"] for r in parsed_rows]
    caption_lengths = [r["caption_length"] for r in parsed_rows]

    log("")
    describe("Likes", likes, lines)
    describe("Comments", comments, lines)
    describe("Hashtag count", hashtag_counts, lines)
    describe("Caption length", caption_lengths, lines)

    posts_per_user = Counter(r["username"] for r in parsed_rows)
    describe("Posts per user", list(posts_per_user.values()), lines)

    category_counts = Counter(r["category"] for r in parsed_rows if r["category"])
    log(f"\nCategory breakdown: {dict(category_counts)}")

    sponsored_count = sum(1 for r in parsed_rows if r["is_sponsored"])
    log(f"Sponsored posts: {sponsored_count} ({sponsored_count/max(1,len(parsed_rows))*100:.1f}%)")

    missing_caption = sum(1 for r in parsed_rows if not r["caption"])
    log(f"Empty captions: {missing_caption} ({missing_caption/max(1,len(parsed_rows))*100:.1f}%)")

    video_rows = [r for r in parsed_rows if r["typename"] == "GraphVideo"]
    if video_rows:
        vv = [r["video_view_count"] for r in video_rows if r["video_view_count"] is not None]
        log(f"\nGraphVideo posts: {len(video_rows)}, with video_view_count present: {len(vv)}")
        describe("video_view_count", vv, lines)

    sidecar_rows = [r for r in parsed_rows if r["typename"] == "GraphSidecar"]
    if sidecar_rows:
        sc = [r["sidecar_child_count"] for r in sidecar_rows if r["sidecar_child_count"]]
        log(f"\nGraphSidecar posts: {len(sidecar_rows)}, with children count present: {len(sc)}")
        describe("sidecar_child_count", sc, lines)

    with open(OUTPUT_TYPE_SAMPLES, "w", encoding="utf-8") as f:
        json.dump(type_samples, f, ensure_ascii=False, indent=2)
    log(f"\nSaved one full raw example per __typename to {OUTPUT_TYPE_SAMPLES}")

    sample_size = min(3000, len(parsed_rows))
    with open(OUTPUT_SAMPLE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "post_id", "username", "category", "followers", "media_type",
            "likes", "comments", "hashtag_count", "caption_length",
            "is_sponsored", "usertag_count", "timestamp",
        ])
        for r in parsed_rows[:sample_size]:
            writer.writerow([
                r["post_id"], r["username"], r["category"], r["followers"],
                r["media_type"], r["likes"], r["comments"], r["hashtag_count"],
                r["caption_length"], r["is_sponsored"], r["usertag_count"],
                r["timestamp"],
            ])
    log(f"Saved {sample_size}-row parsed sample to {OUTPUT_SAMPLE_CSV}")

    if empty_files:
        log(f"\nFirst 5 empty files:")
        for f in empty_files[:5]:
            log(f"  {f}")

    if corrupt_files:
        log(f"\nFirst 5 corrupt files:")
        for f, e in corrupt_files[:5]:
            log(f"  {f}: {e}")

    if error_files:
        log(f"\nFirst 5 other parse errors:")
        for f, e in error_files[:5]:
            log(f"  {f}: {e}")

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"\nFull report saved to {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()