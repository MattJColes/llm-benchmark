## ADDED Requirements

### Requirement: Deterministic owned VLM assets
The system SHALL store synthetic floor plan, chart, architecture-diagram, and table generators with committed seeds and manifests; it SHALL store owned photo manifests with stripped EXIF and no unmanifested test images.

#### Scenario: Regenerating a chart asset
- **WHEN** a committed chart generator is run with its manifest seed
- **THEN** it SHALL generate the pinned-dimension asset and ground-truth values represented in that manifest

### Requirement: Manifest-defined VLM scoring
Every VLM question SHALL declare its expected answer, scoring mode, and applicable tolerance in a manifest, and the suite SHALL score exact, numeric, set, and structural responses in code.

#### Scenario: Numeric chart answer
- **WHEN** a numeric VLM response falls within the manifest tolerance
- **THEN** the suite SHALL score that question correct

#### Scenario: Missing manifest entry
- **WHEN** an image has no manifest entry
- **THEN** the suite SHALL exclude it from evaluation

### Requirement: Constrained scoreable outputs
The VLM suite SHALL use grammar-constrained JSON for scoreable responses and SHALL record the runner/image preprocessing evidence needed to compare vision results across backends.

#### Scenario: Cross-backend vision comparison
- **WHEN** a report compares vision accuracy across backends
- **THEN** it SHALL include only runs whose preflight image-token-count check passed
