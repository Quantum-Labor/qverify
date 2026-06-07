# Deploying the Hugging Face Space

The live demo at
[huggingface.co/spaces/Laborator/qverify](https://huggingface.co/spaces/Laborator/qverify)
is a mirror of the Space files kept in this repository. The mirror is produced
by the GitHub Actions workflow
[`.github/workflows/deploy-space.yml`](../.github/workflows/deploy-space.yml),
which copies `space/`, `assets/`, and `benchmarks/` from `main` to the Space repo.

Source of truth lives here; the Space repo is a deploy target and should not be
edited directly.

## What gets deployed

| Source in this repo | Destination at the Space root |
| --- | --- |
| `space/` (app.py, safety.py, README.md, requirements.txt, Dockerfile, ...) | repo root |
| `assets/` | `assets/` |
| `benchmarks/` | `benchmarks/` |

The Space's `.gitattributes` (its Git-LFS config) is **not** overwritten by the
sync; it exists only on the Hugging Face side.

## One-time setup: the `HF_TOKEN` secret

The workflow needs a Hugging Face token with write access to the Space.

1. Mint the token at
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens):
   - Click **New token**.
   - Type/scope: **Write** (a fine-grained token also works if it is granted
     write access to the `Laborator/qverify` Space).
   - Copy the token value (`hf_...`); it is shown only once.
2. Add it as a GitHub repository secret:
   - Go to
     [github.com/Quantum-Labor/qverify/settings/secrets/actions](https://github.com/Quantum-Labor/qverify/settings/secrets/actions).
   - Click **New repository secret**.
   - Name: `HF_TOKEN`. Value: the token from step 1.
   - Save.

The workflow fails fast with a clear message if `HF_TOKEN` is missing.

## When it runs

- **Automatically** on every push to `main` that touches `space/**`,
  `assets/**`, or `benchmarks/**` (the `paths` filter). Pushes that change only
  `qverify/`, `tests/`, or docs do not trigger a deploy.
- **Manually** at any time: GitHub repo -> **Actions** tab -> **Deploy to HF
  Space** -> **Run workflow** -> pick `main` -> **Run workflow**.

If the sync produces no file changes, the run logs `no changes to deploy` and
exits successfully without creating an empty commit.

## Rollback

The deploy is a pure mirror, so rolling back the Space means rolling back the
source:

1. Revert the offending commit on `main`:
   ```bash
   git revert <bad-commit-sha>
   git push origin main
   ```
2. The revert touches `space/` (or `assets/`/`benchmarks/`), so the workflow
   fires again and redeploys the previous-good state.

For an emergency manual deploy without waiting for CI, reproduce the workflow
steps locally: clone the Space, `rsync` from this repo using the rules above,
then commit and push to the Space's `main`.

## Manual deploy (reference)

```bash
git clone https://huggingface.co/spaces/Laborator/qverify hf-space
cd hf-space && git pull --rebase
rsync -av --delete \
  --exclude='.git/' --exclude='__pycache__/' \
  --exclude='assets/' --exclude='benchmarks/' --exclude='.gitattributes' \
  /path/to/qverify/space/ ./
rsync -av --delete /path/to/qverify/assets/ ./assets/
rsync -av --delete /path/to/qverify/benchmarks/ ./benchmarks/
git add -A && git commit -m "deploy: sync from qverify main" && git push origin main
```
