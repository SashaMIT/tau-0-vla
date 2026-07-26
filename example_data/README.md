# Example data

This directory contains a small, self-contained subset extracted from
[AgiBot World Alpha](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha)
and converted to the LeRobot v3.0 layout used by tau-0-vla.

The subset contains one G1 dual-arm task, **Strike the gong**:

- 25 episodes
- 9,469 frames at 30 FPS
- three 640×480 camera streams: head, left hand, and right hand
- a 163D observation state and a 36D action
- episode-level and segment-level language instructions

The `.annexb` conversion intermediates from the source directory are not
included; the data loader reads the three `.mp4` files referenced by
`meta/info.json`.

## License and attribution

This `example_data/` directory is derived from AgiBot World Alpha and is
licensed separately from the repository code under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
It is provided for non-commercial use and derivatives must be shared under the
same license. The Apache-2.0 license at the repository root does **not** apply to
these data files.

Please cite the original project:

```bibtex
@misc{contributors2024agibotworldrepo,
  title        = {AgiBot World Colosseum},
  author       = {AgiBot World Colosseum contributors},
  howpublished = {\url{https://github.com/OpenDriveLab/AgiBot-World}},
  year         = {2024}
}
```
