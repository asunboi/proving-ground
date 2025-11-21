Here’s the basic pattern you want:

1) Use Python’s `logging` everywhere  
2) Let Hydra configure the logging for you via `hydra/job_logging`  
3) Never call `basicConfig` yourself once Hydra is in play

---

## 1. Minimal example: Python logger + Hydra

`cli.py`:

```python
import logging
import hydra
from omegaconf import DictConfig

# module-level logger
log = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # This will use Hydra’s logging config
    log.info("Hydra + logging is working.")
    log.debug(f"Full config:\n{cfg}")

    # Call into your own modules; they use logging too
    from storm.save import run_save
    run_save(cfg)

if __name__ == "__main__":
    main()
```

`storm/save.py`:

```python
import logging

log = logging.getLogger(__name__)

def run_save(cfg):
    log.info("Running save pipeline...")
    # your logic here
    # log.debug("Something detailed")
```

As long as you call `logging.getLogger(__name__)` in every module, Hydra will wire them all to the same logging system.

---

## 2. Tell Hydra how to configure logging

In `conf/config.yaml`:

```yaml
defaults:
  - hydra/job_logging: default   # or colorlog, rich, or your own
  # ... your other defaults

hydra:
  run:
    dir: outputs/${now:%Y-%m-%d_%H-%M-%S}
  job_logging:
    root:
      level: INFO   # default log level across your code
```

Now when you run:

```bash
python cli.py
```

Hydra will:

- Create the run directory
- Configure `logging` using the `hydra/job_logging` config
- Then call your `main(cfg)` with logging already set up

No need to manually call `logging.basicConfig`.

---

## 3. Custom log format / log file

If you want more control, create a custom `job_logging` config.

`conf/hydra/job_logging/custom.yaml`:

```yaml
# This is a standard logging.config.dictConfig dict
version: 1

formatters:
  simple:
    format: "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: simple
    stream: ext://sys.stdout

  file:
    class: logging.FileHandler
    level: DEBUG
    formatter: simple
    filename: ${hydra.run.dir}/storm.log   # hydra variable

loggers:
  # Example: make your package slightly more verbose
  storm:
    level: DEBUG
    handlers: [console, file]
    propagate: false

root:
  level: INFO
  handlers: [console, file]
```

Then in `conf/config.yaml`:

```yaml
defaults:
  - hydra/job_logging: custom
```

Now:

- All `logging.getLogger(__name__)` calls in your code adopt this config
- Logs go both to console and `storm.log` inside the run dir
- You can tune per-module verbosity with the `loggers:` section

---

## 4. Using this pattern with `save.py`, `splitter.py`, etc.

Inside any module (`splitter.py`, `save.py`, etc.):

```python
import logging
log = logging.getLogger(__name__)

def some_function(cfg):
    log.info("Starting split generation")
    log.debug(f"Split config: {cfg.splitter}")
```

Key points:

- Don’t reconfigure logging inside these modules (no `basicConfig`, no new handlers).
- Just grab `logging.getLogger(__name__)` and write `info/debug/warning/error`.
- Hydra’s logging config applies globally once `@hydra.main` is entered.

---

If you want, next step I can sketch a `conf/hydra/job_logging/rich.yaml` or `colorlog.yaml` tuned for your `storm` package so you get nice colored logs and a per-run logfile.

# is there a better pattern for logging this 
```
 #1) Drop covariates with 0 training examples
    if zero_train_initial:
        log.info("Dropping cell types with 0 training examples from DataFrame:")
        for ct in zero_train_initial:
            log.info(f"  - dropping {ct}")
        df = df[~df[covariate_col].isin(zero_train_initial)].copy()
```
for example, currently the output looks like the below, but really it's one block of information. This is just an aesthetic change, so it doesn't affect much. 
```
[2025-11-21 15:42:06,851][__main__][INFO] - Cell types with 0 training examples (before removal):
[2025-11-21 15:42:06,852][__main__][INFO] -   - Astro
[2025-11-21 15:42:06,853][__main__][INFO] -   - CR
[2025-11-21 15:42:06,854][__main__][INFO] -   - Endo
[2025-11-21 15:42:06,855][__main__][INFO] -   - L2 IT ENTl
[2025-11-21 15:42:06,856][__main__][INFO] -   - L2 IT RHP
[2025-11-21 15:42:06,856][__main__][INFO] -   - L6 IT ENTl
[2025-11-21 15:42:06,857][__main__][INFO] -   - L6b-CT ENT
[2025-11-21 15:42:06,858][__main__][INFO] -   - Micro-PVM
[2025-11-21 15:42:06,859][__main__][INFO] -   - SMC-Peri
[2025-11-21 15:42:06,859][__main__][INFO] -   - SUB-ProS
[2025-11-21 15:42:06,860][__main__][INFO] -   - V3d
```

Yep, you can absolutely make that a single “block” log message instead of many tiny ones.

Simplest pattern: build one multi-line string, then log once.

```python
# 1) Drop covariates with 0 training examples
if zero_train_initial:
    lines = [
        "Cell types with 0 training examples (before removal):",
        *[f"  - {ct}" for ct in sorted(zero_train_initial)],
    ]
    log.info("\n".join(lines))

    df = df[~df[covariate_col].isin(zero_train_initial)].copy()
```

That will emit a single log record like:

```
[2025-11-21 15:42:06,851][__main__][INFO] - Cell types with 0 training examples (before removal):
  - Astro
  - CR
  - Endo
  - L2 IT ENTl
  ...
```

If you like the `%`-style formatting that logging supports, you can also do:

```python
if zero_train_initial:
    ct_block = "\n  - " + "\n  - ".join(sorted(zero_train_initial))
    log.info("Cell types with 0 training examples (before removal):%s", ct_block)
```

Same idea, just using logging’s built-in formatting rather than f-strings.

If you find yourself doing this a lot, you can factor it into a tiny helper:

```python
def log_list_block(logger, header, items):
    if not items:
        return
    msg = "\n".join([header, *[f"  - {x}" for x in items]])
    logger.info(msg)

# usage
log_list_block(
    log,
    "Cell types with 0 training examples (before removal):",
    sorted(zero_train_initial),
)
```

# is it possible to not output info level logs to terminal during runtime? or set a quiet setting.

Yes, totally possible – that’s exactly what log levels and handlers are for.

If you want to:

- keep writing INFO logs to a file  
- but only show WARNING+ in the terminal

you can do something like this:

```python
import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # logger can see everything

# Console handler (terminal)
console = logging.StreamHandler(stream=sys.stdout)
console.setLevel(logging.WARNING)  # only WARNING and above go to terminal

# File handler (log file)
file_handler = logging.FileHandler("run.log")
file_handler.setLevel(logging.INFO)  # INFO+ goes to file

# Optional: formatter
fmt = logging.Formatter(
    "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
console.setFormatter(fmt)
file_handler.setFormatter(fmt)

logger.addHandler(console)
logger.addHandler(file_handler)
```

Then your `log.info(...)` calls still happen, but they only go to `run.log`, not to the terminal.

---

If you want a “quiet” flag (e.g. `--quiet`), you can just bump the console handler level at runtime:

```python
def set_quiet(quiet: bool):
    level = logging.WARNING if quiet else logging.INFO
    for h in logger.handlers:
        if isinstance(h, logging.StreamHandler):
            h.setLevel(level)

# Example with argparse
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--quiet", action="store_true", help="Reduce console logging")
args = parser.parse_args()

set_quiet(args.quiet)
```

So:

- normal run: INFO to file, INFO (or WARNING, your choice) to console  
- `--quiet`: only WARNING/ERROR to console, but file keeps full INFO history

If you tell me how you’re currently setting up logging (plain `logging.basicConfig`, Hydra, etc.), I can drop this straight into your existing pattern.