import json
import time
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/raw/visual_genome/halloc_500_image_manifest.json"
IMAGE_ROOT = ROOT / "data/images/visual_genome"
REPORT = ROOT / "data/raw/visual_genome/halloc_500_download_report.json"

manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

success = []
failed = []

for i, item in enumerate(manifest, 1):
    image_id = str(item["image_id"])
    url = item["url"]

    # Keep the directory structure encoded in the official URL.
    subdir = "VG_100K_2" if "/VG_100K_2/" in url else "VG_100K"
    destination = IMAGE_ROOT / subdir / f"{image_id}.jpg"
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size > 0:
        success.append(image_id)
        continue

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "hallu-lvlm-research/1.0"},
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            content_type = response.headers.get("Content-Type", "")
            data = response.read()

        if not data or not content_type.lower().startswith("image/"):
            raise ValueError(
                f"Unexpected response: content_type={content_type!r}, "
                f"bytes={len(data)}"
            )

        # Avoid leaving a partial file if a write fails.
        temporary = destination.with_suffix(".jpg.part")
        temporary.write_bytes(data)
        temporary.replace(destination)
        success.append(image_id)

    except Exception as exc:
        failed.append({
            "image_id": image_id,
            "url": url,
            "error": str(exc),
        })

    if i % 25 == 0:
        print(f"Processed {i}/{len(manifest)}; downloaded or present: "
              f"{len(success)}; failed: {len(failed)}")

    time.sleep(0.05)

report = {
    "manifest_count": len(manifest),
    "success_count": len(success),
    "failure_count": len(failed),
    "successful_image_ids": success,
    "failures": failed,
}
REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\nDownload finished.")
print("Success:", len(success))
print("Failed:", len(failed))
print("Report:", REPORT)
