# Raw data

This directory is ignored by Git. Files are downloaded from the exact sources
in `configs/data-manifest.yaml`, written via `.partial` files, and verified before
being renamed into place. Never manually edit a verified raw artifact.

For the driving application run:

```bash
.venv/bin/autodrive-fly download --dataset malecns
.venv/bin/autodrive-fly verify --dataset malecns
make prepare
```
