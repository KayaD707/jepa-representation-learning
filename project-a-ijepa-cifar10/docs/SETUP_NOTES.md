# Setup Notes — Getting PyTorch to Actually Run

Kept as a reasoning log, not just a "run this command" instruction — the troubleshooting
process here is itself worth understanding, since this exact class of problem (an
Anaconda environment and a pip-installed deep learning framework fighting over which
copy of a shared library to use) is a common, real-world ML setup issue, not specific to
this one machine.

## Symptom 1: `fbgemm.dll` failed to load

The existing `torch` install (version 2.4.0, originally installed into the shared
Anaconda base environment at some earlier point) failed on `import torch` with:

```
OSError: [WinError 126] The specified module could not be found.
Error loading "...\torch\lib\fbgemm.dll" or one of its dependencies.
```

First attempt: uninstall and reinstall `torch` cleanly as a CPU-only build, in case the
original install was simply corrupted or partial.

## Symptom 2: a different DLL, `c10.dll`, failed differently

After reinstalling, the error changed shape — not "file not found," but a deeper failure:

```
OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed.
Error loading "...\torch\lib\c10.dll" or one of its dependencies.
```

A *different* error after reinstalling the *same* package is itself a clue: it points
away from "the file is missing or corrupted" and toward "two different things are both
trying to provide the same functionality, and they're colliding."

## Diagnosis

The CPU (Intel i7-7700, Kaby Lake) fully supports the instruction sets (AVX2) modern
PyTorch builds expect, which rules out the other common cause of this exact symptom.

What's actually happening: Anaconda's own `numpy` is built against Intel's Math Kernel
Library (MKL) — confirmed via `numpy.show_config()`, which reported `mkl-sdl` as the BLAS
backend, loaded from `anaconda3\Library\lib`. The `torch` wheel installed via `pip` from
PyPI/pytorch.org brings its **own**, separately-compiled copy of MKL/OpenMP-related DLLs
bundled inside `torch\lib`. When both copies end up loaded into the same Python process,
they can conflict over shared symbols — this is a well-known, recurring class of issue
specifically with pip-installed PyTorch inside an Anaconda environment, not a sign of a
broken machine.

## Fix: an isolated virtual environment for this project

Rather than fight the conflict inside the shared Anaconda base environment (which other,
unrelated tools on this machine depend on — e.g. the CV-generation scripts use
`python-docx` from that same environment, and breaking it would have nothing to do with
this project), the cleaner fix is giving this project its **own** environment, with no
Anaconda MKL in it to conflict with:

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

This also happens to be good practice independent of the bug: a project should declare
and isolate its own dependencies rather than relying on whatever happens to already be
installed globally — the same principle behind `requirements.txt`/`uv`/virtual
environments covered in the ML-in-Production study materials from earlier in this job
search.

## Symptom 3: the isolated venv had the *exact same* `c10.dll` error

This is the result that actually disproved the MKL-conflict theory above. A fresh,
empty virtual environment has no Anaconda MKL in it at all to conflict with — nothing
else was loaded into that process. The identical failure in total isolation meant the
cause wasn't a *collision between two things*, it was something wrong with the `torch`
build itself, on this specific machine.

## Real diagnosis: a CPU instruction-set mismatch

The `pip install torch` command (no version pinned) pulled the current latest release at
the time: **torch 2.14.1+cpu**. Modern PyTorch releases have been raising the minimum CPU
instruction-set baseline their default Windows/Linux wheels are compiled against. This
machine's CPU (Intel i7-7700, "Kaby Lake," released 2017) supports AVX2 and FMA3, but
**not** AVX-512 — a feature set limited to later chips (Skylake-X, Ice Lake, and newer).
If a current `torch` build's default wheel now assumes AVX-512 is available, it would
produce exactly this symptom: the DLL loads far enough to begin executing, then fails
during its own initialization routine on an unsupported instruction, which Windows
reports simply as "DLL initialization routine failed."

**Confirmed by testing the hypothesis directly:** installing an older, more conservative
release — `torch==2.2.2+cpu` — in the same venv worked immediately, no DLL error at all.

## Symptom 4 (minor, caught immediately): NumPy ABI mismatch

With the older `torch==2.2.2` now installed, a new warning appeared on import:
`Failed to initialize NumPy: _ARRAY_API not found`. `torch 2.2.2` was built and tested
against NumPy's 1.x C API; `pip` had installed NumPy 2.5.2 (the current release) as a
fresh dependency in the new venv, and the two have an incompatible binary interface.
Fixed by pinning `numpy<2` in this environment specifically.

## Symptom 5 (same root cause as #4): torchvision/torch version mismatch

Pinning `torch` down to `2.2.2` left the already-installed `torchvision` (which had been
installed expecting `torch>=2.14.0`) out of sync — pip flagged this explicitly as a
dependency conflict rather than silently breaking. Fixed by installing the `torchvision`
version actually released alongside `torch 2.2.2` at the time: `torchvision==0.17.2`.

## Symptom 6: torch 2.4.0 (the version originally installed) was tested too

Before settling on 2.2.2, torch 2.4.0 was tried as a hoped-for middle ground — new
enough to possibly support numpy 2.x, old enough to predate whatever changed in 2.14.1.
**It reproduced the exact original `fbgemm.dll` error**, even in the clean venv. This was
a useful result even though the hypothesis was wrong: it confirmed symptom 1 was never
about the Anaconda environment either — torch 2.4.0 itself doesn't work on this CPU,
full stop. That left 2.2.2 as the newest version actually confirmed to work, not a
compromise chosen out of impatience.

## Symptom 7: fixing the CPU problem reopened the NumPy problem — and scikit-learn made it unfixable as pinned

Back on torch 2.2.2 (needs numpy<2), installing `scikit-learn` pulled in a current
`scipy` release (1.18.1) that has since dropped numpy 1.x support entirely — it uses
`np.long`, an attribute numpy 1.x doesn't provide the way scipy now expects, which
**crashes on import**, not just warns. This is a real, currently-unresolvable version
conflict on this machine: the `scipy` release available now requires numpy>=2, and the
`torch` release that actually runs on this CPU requires numpy<2.

**Resolution: drop scikit-learn as a dependency entirely.** Project A's linear-probe
evaluation doesn't need `sklearn.LogisticRegression` specifically — a single
`torch.nn.Linear` layer trained with `cross_entropy` loss via a normal PyTorch training
loop does the same job, with no scipy anywhere in the dependency graph. This isn't a
workaround bolted on to dodge a bug; it's arguably the better implementation for this
project regardless, since it's one less cross-framework dependency and more actual
PyTorch practice, which is the point of doing this project in PyTorch in the first place.

## Final, verified working environment

```
torch==2.2.2+cpu
torchvision==0.17.2+cpu
numpy<2  (resolved to 1.26.4)
matplotlib    (works fine with numpy<2 for the plotting this project needs --
               contourpy's numpy>=2 requirement only matters for filled-contour
               plots, which aren't used here)
wandb
```
No scikit-learn, no scipy — removed deliberately, not an oversight.

## Takeaway

When the *same* reinstall produces a *different* error, that's a signal worth reading
literally — it usually means the problem isn't "this file is broken," it's "something
else present in the environment is interfering." The isolated-venv test is what actually
separated the two competing theories (MKL conflict vs. CPU incompatibility) — running the
same failing operation with one variable removed (no Anaconda environment present at all)
and seeing the failure persist unchanged is what pointed at the real cause, rather than
guessing. Pinning specific, tested-together versions (`torch`, matching `torchvision`,
and a compatible `numpy`) instead of trusting "latest" to self-assemble correctly is the
general lesson — this is exactly the version-pinning discipline the ML-in-Production
study material already covers (`uv.lock`, `requirements.txt` with real versions), just
learned here the hard way instead of in the abstract.
