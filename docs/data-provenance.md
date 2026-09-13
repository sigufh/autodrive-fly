# Data provenance

## MaleCNS v1.0

Official source: <https://male-cns.janelia.org/download/>. License: CC BY 4.0.
The frozen paper count is 166,691 neurons, while the current v1.0 annotation table
contains 166,700 records with a neuron `superclass`. The downloadable table has
211,577 total segments. All 166,700 current release records with a superclass are
retained; this is not the same as filtering only strict `status == Traced`.

URLs, byte counts and hashes are pinned in `configs/data-manifest.yaml`. The full
flat weight table has 151,856,684 source rows. Its GCS MD5 is
`f30e9dcca25cfd021bf1e7b3d975599e`; SHA-256 is
`e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1`.

The induced canonical graph contains:

- 166,700 nodes;
- 25,582,938 directed edges;
- contact-count sum 124,177,617;
- 217 isolated nodes at minimum confidence 0.5;
- 140,638 nodes with real soma or alternate-soma coordinates.

The other 26,062 canonical nodes stay in computation but receive no invented 3D
position. Detailed morphologies are downloaded on demand from the official
unsharded Neuroglancer skeleton collection and cached locally. Raw data, processed
graphs and skeleton caches do not enter Git.

## Derived retinal map

Most photoreceptors have no direct optic-hex coordinate in the annotation table.
For each R1–R6 cell, the project computes a contact-weighted mean coordinate from
its position-annotated postsynaptic optic-lobe partners. This maps 3,344 receptors.
The derived cache is local and reproducible from the pinned source files.
