# Dataset format

What `tau0_vla.data` requires of a **LeRobot v3.0** dataset. Other versions are
not read; convert v2.1 with LeRobot's own tooling first.

Each section states the requirement, then gives an illustrative value. Example
values are never the requirement.

## Directory layout

```text
<dataset_root>/
├── meta/
│   ├── info.json
│   ├── stats.json
│   ├── tasks.parquet
│   ├── episodes/
│   └── episodes_stats/
├── data/
│   └── chunk-000/
│       ├── file-000.parquet
│       └── ...
└── videos/
    ├── observation.images.head/
    │   └── chunk-000/file-000.mp4
    └── ...
```

`data_path` and `video_path` in `info.json` define the exact paths:

```json
"data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
"video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
```

## `meta/info.json`

### Required scalars

| key | example | meaning |
|---|---|---|
| `codebase_version` | `"v3.0"` | must be `v3.0` |
| `robot_type` | `"my_robot"` | free-form; the adapter defines its meaning |
| `total_episodes` | `2` | number of episodes |
| `total_frames` | `1200` | number of frames |
| `fps` | `30` | time offsets are converted using this rate |
| `chunks_size` | `1000` | episodes per chunk directory |

### Features and field descriptions

`features` declares every flat column and video stream. The pipeline reads:

| feature | dtype | shape | requirement |
|---|---|---|---|
| `observation.state` | float32 | `[D_state]` | flat state plus `field_descriptions` |
| `action` | float32 | `[D_action]` | flat action plus `field_descriptions` |
| `observation.images.<camera>` | video | `[H, W, 3]` | one entry per configured camera |

LeRobot bookkeeping fields such as `episode_index`, `frame_index`, `index`,
`task_index`, `timestamp`, and per-frame flags are read by LeRobot itself.

Video features carry an `info` block with height, width, fps, and codec. Codec
and resolution are dataset-specific; the config decides which streams and
transforms are used.

State and action are flat vectors at runtime. Each feature must include a
`field_descriptions` object that maps semantic field names to native `indices`;
`dimensions` must equal the number of indices:

```json
"field_descriptions": {
  "state/joint/position": {
    "description": "left arm followed by right arm",
    "dimensions": 14,
    "indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
  },
  "state/gripper/position": {
    "description": "left then right",
    "dimensions": 2,
    "indices": [14, 15]
  }
}
```

A component adapter's `repack["state"]["semantic"]` and
`repack["action"]["semantic"]` refer to these names. A unified adapter
additionally uses `dim_registry.json` to split native columns into left/right
groups and scatter them into fixed 40D slots. The field descriptions, registry,
and raw vector widths must agree. See the
[adapter contract](../adapters/README.md).

## Annotations

Annotation maps use episode indices as strings (`"0"`, `"1"`, ...). The tracks
needed depend on `prompt_source` and `frame_filter`.

### `instruction_segments` — frame-level subtask instructions (`l3`)

Each episode contains a list of labelled frame ranges:

```json
{
  "track": "default",
  "instruction": "Move the right arm to pick up the object",
  "start_frame_index": 0,
  "end_frame_index": 266,
  "origin_instruction": "..."
}
```

Used independently by:

- `PromptSource.from_label(source="instruction_segments")`, which resolves the
  segment containing the anchor frame;
- `FrameFilter(positive=["l3"])`, which keeps anchors covered by a segment and
  keeps the action-chunk tail inside that segment.

`end_frame_index` is exclusive.

### `key_frame` — subtask and error intervals (`l2`, `error_frame`)

An episode may contain a list, or a dictionary of named tracks whose lists are
flattened:

```json
{
  "track": "default",
  "frame_type_name": "SubTask Frame",
  "start": 0,
  "end": 431,
  "comment": "",
  "frame_detail": {
    "comment": "Pick up the object",
    "is_result_succeed": null
  }
}
```

`frame_type_name` is normalized by removing non-alphanumeric characters and
lowercasing:

| normalized value | meaning | filter label |
|---|---|---|
| `subtaskframe`, `taskframe` and numbered variants | subtask interval | `l2` |
| `errorframe`, `error` and variants | invalid interval | `error_frame` |

An item can also set a truthy `error_frame` field. Subtask text is normally read
from `frame_detail.comment`. If an episode has no `l2` track, `l2` falls back to
`l3`.

### `high_level_instruction` — episode instruction (`l1`)

One string per episode, for example `"Pick up the object"`. It is available to
`PromptSource`; `l1` is not a valid `FrameFilter` label.

## Minimal example

This is a minimal one-camera, 16D joint-control `info.json`. The matching
adapter uses arm columns `0:14`, gripper columns `14:16`, and a registry to
split left/right sides for a unified route.

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
    "observation.state": {
      "dtype": "float32",
      "shape": [16],
      "field_descriptions": {
        "state/joint/position": {
          "dimensions": 14,
          "indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        },
        "state/gripper/position": {
          "dimensions": 2,
          "indices": [14, 15]
        }
      }
    },
    "action": {
      "dtype": "float32",
      "shape": [16],
      "field_descriptions": {
        "action/joint/position": {
          "dimensions": 14,
          "indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        },
        "action/gripper/position": {
          "dimensions": 2,
          "indices": [14, 15]
        }
      }
    },
    "observation.images.head": {
      "dtype": "video",
      "shape": [480, 640, 3],
      "info": {
        "video.height": 480,
        "video.width": 640,
        "video.fps": 30,
        "video.codec": "h264"
      }
    },
    "episode_index": {"dtype": "int64", "shape": [1]},
    "frame_index": {"dtype": "int64", "shape": [1]},
    "index": {"dtype": "int64", "shape": [1]},
    "task_index": {"dtype": "int64", "shape": [1]},
    "timestamp": {"dtype": "float32", "shape": [1]}
  },
  "instruction_segments": {
    "0": [{
      "track": "default",
      "instruction": "pick up the cup",
      "start_frame_index": 0,
      "end_frame_index": 600
    }],
    "1": [{
      "track": "default",
      "instruction": "put down the cup",
      "start_frame_index": 0,
      "end_frame_index": 600
    }]
  }
}
```

With no `key_frame`, use only `l3` as a positive frame filter and no negative
filter.

## Limits

- **Version:** LeRobot v3.0 only.
- **Field descriptions:** both state and action semantic maps are required.
- **Unified mapping:** a unified route also needs a matching registry entry.
- **Filter labels:** positive frame filters accept `l2` and `l3`; negative
  filters accept `error_frame`.
