# Hazard label review queue

Auto-labeled from scenario name: **0 rows**.
Needing manual review: **61 rows** across **11 videos**.

Open each video below, then for each track confirm or correct the suggested label and paste the final value into the `true_label` column of `data/labels/hazard_labels.csv`.

Priority order within a video: `approaching` (safety-critical) , then crossings, then everything else.


## `IMG_4972` (7 tracks)

Demo video: [`outputs/annotated_videos/IMG_4972_hazards.mp4`](outputs/annotated_videos/IMG_4972_hazards.mp4)

| track_id | class | suggested | n_frames | frame range | evidence (truncated) |
|---|---|---|---|---|---|
| `person_2` | person | **approaching** | 17 | 1-17 | person_2 classified as approaching because bbox grew by 0.79 and approach_score=1.00; cent... |
| `person_1` | person | **approaching** | 15 | 0-14 | person_1 classified as approaching because bbox grew by 21.98 and approach_score=1.00; cen... |
| `bicycle_1` | bicycle | **approaching** | 6 | 9-14 | bicycle_1 classified as approaching because bbox grew by 11.42 and approach_score=1.00; ce... |
| `motorcycle_1` | motorcycle | **uncertain** | 2 | 7-8 | motorcycle_1 classified as uncertain (short_track) because track is only 2 frames long (mi... |
| `person_3` | person | **uncertain** | 2 | 14-17 | person_3 classified as uncertain (short_track) because track is only 2 frames long (min=3)... |
| `skateboard_1` | skateboard | **uncertain** | 2 | 9-10 | skateboard_1 classified as uncertain (short_track) because track is only 2 frames long (mi... |
| `motorcycle_2` | motorcycle | **uncertain** | 1 | 7-7 | motorcycle_2 classified as uncertain (short_track) because track is only 1 frame long (min... |

## `IMG_4973` (6 tracks)

Demo video: [`outputs/annotated_videos/IMG_4973_hazards.mp4`](outputs/annotated_videos/IMG_4973_hazards.mp4)

| track_id | class | suggested | n_frames | frame range | evidence (truncated) |
|---|---|---|---|---|---|
| `person_3` | person | **approaching** | 20 | 7-28 | person_3 classified as approaching because bbox grew by 31.91 and approach_score=1.00; cen... |
| `person_4` | person | **approaching** | 11 | 14-28 | person_4 classified as approaching because bbox grew by 2.27 and approach_score=0.75; flow... |
| `person_5` | person | **moving_away** | 5 | 23-27 | person_5 classified as moving_away because bbox shrank: growth_ratio=-0.23 below -0.15. |
| `person_1` | person | **static** | 1 | 0-0 | person_1 classified as static because low motion (disp_norm=0.000, flow_mag=0.00, frame_di... |
| `person_2` | person | **uncertain** | 9 | 0-9 | person_2 classified as uncertain (low_signal) because motion was detected (flow_mag=14.38,... |
| `bicycle_1` | bicycle | **uncertain** | 1 | 27-27 | bicycle_1 classified as uncertain (short_track) because track is only 1 frame long (min=3)... |

## `IMG_4974` (4 tracks)

Demo video: [`outputs/annotated_videos/IMG_4974_hazards.mp4`](outputs/annotated_videos/IMG_4974_hazards.mp4)

| track_id | class | suggested | n_frames | frame range | evidence (truncated) |
|---|---|---|---|---|---|
| `person_3` | person | **approaching** | 16 | 0-15 | person_3 classified as approaching because bbox grew by 2.67 and approach_score=1.00; cent... |
| `person_4` | person | **approaching** | 15 | 0-15 | person_4 classified as approaching because bbox grew by 1.32 and approach_score=0.75; flow... |
| `person_1` | person | **approaching** | 8 | 0-7 | person_1 classified as approaching because bbox grew by 1.80 and approach_score=0.75; flow... |
| `person_2` | person | **approaching** | 8 | 0-8 | person_2 classified as approaching because bbox grew by 1.93 and approach_score=0.75; flow... |

## `IMG_4975` (5 tracks)

Demo video: [`outputs/annotated_videos/IMG_4975_hazards.mp4`](outputs/annotated_videos/IMG_4975_hazards.mp4)

| track_id | class | suggested | n_frames | frame range | evidence (truncated) |
|---|---|---|---|---|---|
| `person_1` | person | **approaching** | 20 | 0-19 | person_1 classified as approaching because bbox grew by 5.89 and approach_score=0.75; flow... |
| `person_2` | person | **approaching** | 15 | 0-14 | person_2 classified as approaching because bbox grew by 19.96 and approach_score=0.75; flo... |
| `bicycle_1` | bicycle | **approaching** | 6 | 10-15 | bicycle_1 classified as approaching because bbox grew by 12.10 and approach_score=1.00; ce... |
| `person_3` | person | **approaching** | 3 | 14-16 | person_3 classified as approaching because bbox grew by 3.30 and approach_score=1.00; cent... |
| `bicycle_2` | bicycle | **uncertain** | 2 | 15-16 | bicycle_2 classified as uncertain (short_track) because track is only 2 frames long (min=3... |

## `IMG_4976` (6 tracks)

Demo video: [`outputs/annotated_videos/IMG_4976_hazards.mp4`](outputs/annotated_videos/IMG_4976_hazards.mp4)

| track_id | class | suggested | n_frames | frame range | evidence (truncated) |
|---|---|---|---|---|---|
| `person_2` | person | **approaching** | 15 | 1-17 | person_2 classified as approaching because bbox grew by 2.19 and approach_score=0.75; flow... |
| `person_1` | person | **approaching** | 14 | 0-16 | person_1 classified as approaching because bbox grew by 231.01 and approach_score=1.00; ce... |
| `person_3` | person | **approaching** | 10 | 7-19 | person_3 classified as approaching because bbox grew by 0.53 and approach_score=0.75; flow... |
| `person_4` | person | **moving_away** | 3 | 15-18 | person_4 classified as moving_away because bbox shrank: growth_ratio=-0.65 below -0.15. |
| `bicycle_1` | bicycle | **uncertain** | 2 | 15-16 | bicycle_1 classified as uncertain (short_track) because track is only 2 frames long (min=3... |
| `bicycle_2` | bicycle | **uncertain** | 1 | 17-17 | bicycle_2 classified as uncertain (short_track) because track is only 1 frame long (min=3)... |

## `IMG_4977` (5 tracks)

Demo video: [`outputs/annotated_videos/IMG_4977_hazards.mp4`](outputs/annotated_videos/IMG_4977_hazards.mp4)

| track_id | class | suggested | n_frames | frame range | evidence (truncated) |
|---|---|---|---|---|---|
| `bicycle_1` | bicycle | **uncertain** | 1 | 14-14 | bicycle_1 classified as uncertain (short_track) because track is only 1 frame long (min=3)... |
| `bicycle_2` | bicycle | **uncertain** | 1 | 15-15 | bicycle_2 classified as uncertain (short_track) because track is only 1 frame long (min=3)... |
| `bicycle_3` | bicycle | **uncertain** | 1 | 15-15 | bicycle_3 classified as uncertain (short_track) because track is only 1 frame long (min=3)... |
| `bicycle_4` | bicycle | **uncertain** | 1 | 15-15 | bicycle_4 classified as uncertain (short_track) because track is only 1 frame long (min=3)... |
| `person_1` | person | **uncertain** | 1 | 15-15 | person_1 classified as uncertain (short_track) because track is only 1 frame long (min=3),... |

## `IMG_4978` (1 tracks)

Demo video: [`outputs/annotated_videos/IMG_4978_hazards.mp4`](outputs/annotated_videos/IMG_4978_hazards.mp4)

| track_id | class | suggested | n_frames | frame range | evidence (truncated) |
|---|---|---|---|---|---|
| `person_1` | person | **approaching** | 5 | 7-11 | person_1 classified as approaching because bbox grew by 1.72 and approach_score=1.00; cent... |

## `IMG_4979` (2 tracks)

Demo video: [`outputs/annotated_videos/IMG_4979_hazards.mp4`](outputs/annotated_videos/IMG_4979_hazards.mp4)

| track_id | class | suggested | n_frames | frame range | evidence (truncated) |
|---|---|---|---|---|---|
| `person_1` | person | **uncertain** | 1 | 7-7 | person_1 classified as uncertain (short_track) because track is only 1 frame long (min=3),... |
| `person_2` | person | **uncertain** | 1 | 8-8 | person_2 classified as uncertain (short_track) because track is only 1 frame long (min=3),... |

## `IMG_4980` (2 tracks)

Demo video: [`outputs/annotated_videos/IMG_4980_hazards.mp4`](outputs/annotated_videos/IMG_4980_hazards.mp4)

| track_id | class | suggested | n_frames | frame range | evidence (truncated) |
|---|---|---|---|---|---|
| `person_1` | person | **crossing_left_to_right** | 20 | 0-19 | person_1 classified as crossing_left_to_right because dx=145.8px (15.2% of width) with bbo... |
| `person_2` | person | **crossing_right_to_left** | 12 | 7-19 | person_2 classified as crossing_right_to_left because dx=-522.0px (-54.4% of width) with b... |

## `IMG_4981` (7 tracks)

Demo video: [`outputs/annotated_videos/IMG_4981_hazards.mp4`](outputs/annotated_videos/IMG_4981_hazards.mp4)

| track_id | class | suggested | n_frames | frame range | evidence (truncated) |
|---|---|---|---|---|---|
| `person_2` | person | **approaching** | 18 | 2-19 | person_2 classified as approaching because bbox grew by 16.23 and approach_score=1.00; cen... |
| `person_3` | person | **approaching** | 11 | 9-20 | person_3 classified as approaching because bbox grew by 1.92 and approach_score=1.00; cent... |
| `bicycle_1` | bicycle | **approaching** | 4 | 17-20 | bicycle_1 classified as approaching because bbox grew by 6.78 and approach_score=1.00; cen... |
| `bicycle_2` | bicycle | **crossing_right_to_left** | 3 | 21-24 | bicycle_2 classified as crossing_right_to_left because dx=-210.5px (-21.9% of width) with ... |
| `person_4` | person | **crossing_right_to_left** | 3 | 22-24 | person_4 classified as crossing_right_to_left because dx=-144.5px (-15.1% of width) with b... |
| `motorcycle_1` | motorcycle | **uncertain** | 2 | 23-24 | motorcycle_1 classified as uncertain (short_track) because track is only 2 frames long (mi... |
| `person_1` | person | **uncertain** | 1 | 1-1 | person_1 classified as uncertain (short_track) because track is only 1 frame long (min=3),... |

## `IMG_4982` (16 tracks)

Demo video: [`outputs/annotated_videos/IMG_4982_hazards.mp4`](outputs/annotated_videos/IMG_4982_hazards.mp4)

| track_id | class | suggested | n_frames | frame range | evidence (truncated) |
|---|---|---|---|---|---|
| `person_3` | person | **approaching** | 5 | 7-11 | person_3 classified as approaching because bbox grew by 0.92 and approach_score=0.75; flow... |
| `person_4` | person | **approaching** | 4 | 16-19 | person_4 classified as approaching because bbox grew by 4.06 and approach_score=1.00; cent... |
| `person_7` | person | **crossing_left_to_right** | 6 | 26-34 | person_7 classified as crossing_left_to_right because dx=225.9px (23.5% of width) with bbo... |
| `bicycle_3` | bicycle | **uncertain** | 2 | 32-33 | bicycle_3 classified as uncertain (short_track) because track is only 2 frames long (min=3... |
| `motorcycle_1` | motorcycle | **uncertain** | 2 | 30-31 | motorcycle_1 classified as uncertain (short_track) because track is only 2 frames long (mi... |
| `person_10` | person | **uncertain** | 2 | 37-38 | person_10 classified as uncertain (short_track) because track is only 2 frames long (min=3... |
| `bicycle_1` | bicycle | **uncertain** | 1 | 5-5 | bicycle_1 classified as uncertain (short_track) because track is only 1 frame long (min=3)... |
| `bicycle_2` | bicycle | **uncertain** | 1 | 5-5 | bicycle_2 classified as uncertain (short_track) because track is only 1 frame long (min=3)... |
| `bicycle_4` | bicycle | **uncertain** | 1 | 34-34 | bicycle_4 classified as uncertain (short_track) because track is only 1 frame long (min=3)... |
| `person_1` | person | **uncertain** | 1 | 6-6 | person_1 classified as uncertain (short_track) because track is only 1 frame long (min=3),... |
| `person_2` | person | **uncertain** | 1 | 6-6 | person_2 classified as uncertain (short_track) because track is only 1 frame long (min=3),... |
| `person_5` | person | **uncertain** | 1 | 17-17 | person_5 classified as uncertain (short_track) because track is only 1 frame long (min=3),... |
| `person_6` | person | **uncertain** | 1 | 18-18 | person_6 classified as uncertain (short_track) because track is only 1 frame long (min=3),... |
| `person_8` | person | **uncertain** | 1 | 27-27 | person_8 classified as uncertain (short_track) because track is only 1 frame long (min=3),... |
| `person_9` | person | **uncertain** | 1 | 34-34 | person_9 classified as uncertain (short_track) because track is only 1 frame long (min=3),... |
| `skateboard_1` | skateboard | **uncertain** | 1 | 16-16 | skateboard_1 classified as uncertain (short_track) because track is only 1 frame long (min... |

---

When done, re-run:

```
python scripts/run_week3_pipeline.py --all
```

Evaluation metrics will appear in `outputs/evaluation/`.
