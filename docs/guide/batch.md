# Batch processing

Every step is a function with explicit parameters, so a series of images is
processed with an ordinary loop. Keep the parameters in one place and save
them with the results for reproducibility:

```python
import json
from pathlib import Path

import numpy as np
import polarmap as pm

params = dict(min_distance=30, threshold_rel=0.2, exclude_border=5, box=10,
              image_mode="HAADF")

out = Path("results")
out.mkdir(exist_ok=True)
summary = []
for path in sorted(Path("data").glob("*.dm4")):
    image, sampling = pm.load_image(path)
    seeds = pm.detect_peaks(image, params["min_distance"], image_mode=params["image_mode"],
                            threshold_rel=params["threshold_rel"],
                            exclude_border=params["exclude_border"])
    A = pm.refine_gaussian(image, seeds, box=params["box"], image_mode=params["image_mode"])
    v1, v2 = pm.estimate_lattice_vectors(A)
    field = pm.measure_cage_displacement(image, A, v1, v2, image_mode=params["image_mode"])
    field.to_csv(out / f"{path.stem}_displacements.csv", sampling=sampling)
    desc = pm.describe_displacements(field, sampling)
    summary.append({"file": path.name, "n": desc["n"],
                    "median_pm": desc["median_pm"],
                    "circular_mean_deg": desc["circular_mean_deg"]})

(out / "parameters.json").write_text(json.dumps(
    dict(params, polarmap=pm.__version__), indent=2))
(out / "summary.json").write_text(json.dumps(summary, indent=2))
```

Recording `pm.__version__` with the results identifies the exact code version;
together with a tagged release (and its Zenodo DOI) this makes the analysis
reproducible.

For many large images, the per-column Gaussian fits dominate the run time; the
loop parallelises naturally over files, e.g. with
`concurrent.futures.ProcessPoolExecutor`.
