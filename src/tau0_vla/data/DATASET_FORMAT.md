# Dataset format

What `tau0_vla.data` requires of a **LeRobot v3.0** dataset. Other versions are
not read — convert v2.1 with LeRobot's own tooling first.

Each section states the requirement, then gives an illustrative value so you
have something concrete to compare your own dataset against. The example values
are never the requirement; the prose is.

## Directory layout

```
<dataset_root>/
├── meta/
│   ├── info.json          ← the contract: features, counts, annotations
│   ├── stats.json
│   ├── tasks.parquet
│   ├── episodes/
│   └── episodes_stats/
├── data/
│   └── chunk-000/
│       ├── file-000.parquet    ← per-frame state, action, indices, flags
│       └── ...
└── videos/
    ├── observation.images.top_head/
    │   └── chunk-000/file-000.mp4
    ├── observation.images.hand_left/
    └── observation.images.hand_right/
```

`data_path` and `video_path` in `info.json` are format strings that spell this
out, so the actual names are yours to choose:

```json
"data_path":  "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
"video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
```

## `meta/info.json`

### Required scalars

| key | example | meaning |
|---|---|---|
| `codebase_version` | `"v3.0"` | must be `v3.0` |
| `robot_type` | `"a2d"` | free-form; your adapter decides what it means |
| `total_episodes` | `349` | |
| `total_frames` | `1829431` | |
| `fps` | `30` | frame rate; time windows are specified in seconds and converted using this |
| `chunks_size` | `1000` | episodes per chunk directory |

### `features`

Declares every column and video stream. The example dataset has 20 entries; the
ones this pipeline reads:

| feature | dtype | shape | note |
|---|---|---|---|
| `observation.state` | float32 | `[93]` | **no column names** — see below |
| `action` | float32 | `[41]` | likewise |
| `observation.images.<camera>` | video | `[H, W, 3]` | one entry per camera |

The rest (`episode_index`, `frame_index`, `index`, `task_index`, `timestamp`,
and per-frame boolean flags) are LeRobot bookkeeping and are read by LeRobot
itself.

Video entries carry an `info` block with `video.height` / `video.width` /
`video.fps` / `video.codec`. The example uses hevc at 30 fps, `400x640` for the
head camera and `480x848` for the wrists.

**The important property: `observation.state` and `action` are bare float
vectors with no per-column names.** Nothing in the file says which of the 93
state columns is the left elbow. That mapping lives in your adapter's registry
entry — see [`src/tau0_vla/adapters/README.md`](../adapters/README.md). It is the single biggest
thing to understand before adding a robot.

## Annotations

Three optional tracks, all keyed by episode index **as a string**
(`"0"`, `"1"`, …). Which ones you need depends on your Robot Config's
`prompt_source` and `frame_filter`.

### `instruction_segments` — per-subtask instructions ("l3")

A list per episode. Each entry labels a frame range with a natural-language
instruction:

```json
{
  "track": "default",
  "instruction": "Move the right arm to pick up the object ...",
  "start_frame_index": 0,
  "end_frame_index": 266,
  "origin_instruction": "..."
}
```

Used two ways, independently:

- `PromptSource.from_label("instruction_segments")` draws the per-sample prompt
  from whichever segment contains the anchor frame.
- `FrameFilter(positive=["l3"])` restricts training anchors to frames covered by
  a segment. In the example this keeps 1,758,462 of 1,829,431 frames (96.12%)
  across 1,044 segments.

`end_frame_index` is exclusive.

### `key_frame` — sub-task intervals and error spans ("l2", `error_frame`)

Per episode, either a list of items, or a dict of named tracks each holding a
list. The example uses the dict form with tracks `single` and `dual`; all tracks
are flattened together.

```json
{
  "track": "default",
  "frame_type_name": "SubTask Frame",
  "start": 0,
  "end": 431,
  "comment": "",
  "frame_detail": {"comment": "Pick up the object", "is_result_succeed": null}
}
```

`frame_type_name` selects what the item means. Matching **normalises first**:
every non-alphanumeric character is stripped and the rest lowercased, so
`"SubTask Frame"`, `"Sub-Task Frame"` and `"subtaskframe"` are the same thing.

| normalised | meaning | filter label |
|---|---|---|
| `subtaskframe`, `taskframe` (+ numbered variants) | a sub-task interval | `l2` |
| `errorframe`, `error` (+ variants) | a bad span to exclude | `error_frame` |

An error span can also be marked with a truthy `error_frame` field on the item
instead of via `frame_type_name`. Both forms are recognised.

Sub-task text is read from `frame_detail.comment` (several other key names are
accepted). If an episode has no `l2` track, `l2` falls back to `l3` rather than
emptying the episode.

### `high_level_instruction` — one instruction per episode ("l1")

A single string per episode, e.g. `"Pick up the object"`. Available to
`PromptSource`, unused by the shipped example.

## Minimal example

The smallest `info.json` that this pipeline will train on — one camera, joint
control, l3 prompts:

```json
{
  "codebase_version": "v3.0",
  "robot_type": "my_robot",
  "total_episodes": 2,
  "total_frames": 1200,
  "fps": 30,
  "chunks_size": 1000,
  "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
  "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
  "features": {
    "observation.state": {"dtype": "float32", "shape": [16]},
    "action":            {"dtype": "float32", "shape": [16]},
    "observation.images.head": {
      "dtype": "video", "shape": [480, 640, 3],
      "info": {"video.height": 480, "video.width": 640,
               "video.fps": 30, "video.codec": "hevc"}
    },
    "episode_index": {"dtype": "int64", "shape": [1]},
    "frame_index":   {"dtype": "int64", "shape": [1]},
    "index":         {"dtype": "int64", "shape": [1]},
    "task_index":    {"dtype": "int64", "shape": [1]},
    "timestamp":     {"dtype": "float32", "shape": [1]}
  },
  "instruction_segments": {
    "0": [{"track": "default", "instruction": "pick up the cup",
           "start_frame_index": 0, "end_frame_index": 600}],
    "1": [{"track": "default", "instruction": "put down the cup",
           "start_frame_index": 0, "end_frame_index": 600}]
  }
}
```

With no `key_frame`, drop `l2` and `error_frame` from your `FrameFilter`.

## Limits

- **Version.** LeRobot v3.0 only. Convert v2.1 first; this repository ships no
  converter.
- **Column meaning.** It comes from the adapter's registry entry, not from the
  dataset. Per-column names on `observation.state` are ignored if present.
- **Filter labels.** `FrameFilter` accepts the three annotation tracks above
  (`l1`, `l2`, `l3`) and `error_frame`. Any other label raises `unknown label`.
