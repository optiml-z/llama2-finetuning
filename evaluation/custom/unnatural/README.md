# Task-name

### Paper

Title: `Unnatural Instructions: Tuning Language Models with (Almost) No Human Labor`

Abstract: `https://arxiv.org/abs/2212.09689`

It contains the full 240,670 Unnatural Instructions (instruction-input-output triplets) examples. It was constructed by expanding the core data with automatically generated instruction paraphrases.

Homepage: `https://github.com/orhonovich/unnatural-instructions/?tab=readme-ov-file`


### Citation

```
@misc{honovich2022unnatural,
      title={Unnatural Instructions: Tuning Language Models with (Almost) No Human Labor}, 
      author={Or Honovich and Thomas Scialom and Omer Levy and Timo Schick},
      year={2022},
      eprint={2212.09689},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

### Groups and Tasks


#### Tasks

* `task_name`: `unnatural`

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
