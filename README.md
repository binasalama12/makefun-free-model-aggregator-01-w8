# Free-model aggregator — steady-spread

Entry for the make.fun bounty `free-model-aggregator-01` ("The best free AI model aggregator").

`main.py` is one standard-library Python server that speaks the OpenAI Chat Completions API and the Anthropic Messages API (JSON and SSE), reads the trial brief's provider directory and quality table, probes every provider's own catalog, registers the catalog with the env, and routes each request to the highest-rated free model that meets its requirements, failing over on 402/429. `python3 main.py --list` prints the catalog it registers.

Routing variant: **steady-spread** (see the knob block at the top of `main.py`).

License: MIT (see `LICENSE`).
