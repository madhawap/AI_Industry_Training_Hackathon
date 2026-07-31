"""Pipeline stages. Every stage is a pure file-in / file-out step:

    ingest -> curate -> render -> train -> predict -> evaluate -> select -> export

Because each stage only reads files written by earlier stages, any single stage
can be re-run in isolation. In particular `predict` (slow, GPU) is separate from
`evaluate` (fast, CPU) so grader definitions can churn without regenerating.
"""

from ftpipe.stages import (  # noqa: F401
    curate,
    evaluate,
    export,
    ingest,
    predict,
    render,
    select,
    train,
)

ORDER = ["ingest", "curate", "render", "train", "predict", "evaluate", "select", "export"]

RUNNERS = {
    "ingest": ingest.run,
    "curate": curate.run,
    "render": render.run,
    "train": train.run,
    "predict": predict.run,
    "evaluate": evaluate.run,
    "select": select.run,
    "export": export.run,
}
