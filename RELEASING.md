# Releasing tsugi-kpool

`tsugi-kpool` is published to [PyPI](https://pypi.org/project/tsugi-kpool/) by
the `.github/workflows/release.yml` workflow. Distribution is **tokenless**: it
uses PyPI [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) over
OpenID Connect (OIDC), so there is no API token or secret stored in the
repository or the workflow.

A GitHub Release is the published distribution surface. Cutting a release (or
pushing a `v*` tag) triggers the workflow, which builds the sdist + wheel and
uploads them to PyPI.

## One-time maintainer prerequisites (account-bound, do these once)

These cannot be automated in the repository because they are bound to the PyPI
project account, not the code. Do them once before the first release.

### 1. Register the Trusted Publisher on PyPI

On PyPI, configure GitHub Actions as a trusted publisher for the `tsugi-kpool`
project. The values must match this repository exactly:

- Go to the project's **Settings -> Publishing** page
  (`https://pypi.org/manage/project/tsugi-kpool/settings/publishing/`). For the
  very first release of a brand-new project, use **Your account -> Publishing ->
  Add a pending publisher** instead, with the same field values plus the project
  name `tsugi-kpool`.
- Add a new **GitHub** trusted publisher with:
  - **Owner:** `tsugiai`
  - **Repository name:** `tsugi-kpool`
  - **Workflow name:** `release.yml`
  - **Environment name:** `pypi`

The environment name must be `pypi` to match the `environment:` block on the
`publish` job in `release.yml`.

### 2. (Recommended) Protect the `pypi` GitHub environment

In the GitHub repo **Settings -> Environments**, create an environment named
`pypi` and optionally add a required-reviewer protection rule so a human must
approve before any publish runs. The workflow already references this
environment; adding protection makes publishing a gated step.

## Cutting a release

Version is set in `pyproject.toml` (`[project] version`). It stays `0.1.0`
until a deliberate bump.

1. Bump the version in `pyproject.toml` if this release changes the package
   (skip if only re-tagging the same version is intended; PyPI rejects a
   re-upload of an existing version).
2. Commit the version bump and merge it to `main`.
3. Create a GitHub Release whose tag matches the version with a `v` prefix
   (for example, tag `v0.1.0` for version `0.1.0`):

   ```bash
   gh release create v0.1.0 --title "v0.1.0" --notes "Release notes here"
   ```

   Publishing the release fires `release: published`. Alternatively, pushing a
   `v*` tag (`git tag v0.1.0 && git push origin v0.1.0`) also triggers the
   workflow via the `push: tags` event.
4. Watch the **Release** workflow in the Actions tab. The `build` job produces
   the sdist + wheel and runs `twine check`; the `publish` job uploads them to
   PyPI via OIDC. No token entry is required.
5. Confirm the new version appears at
   <https://pypi.org/project/tsugi-kpool/>.

## Notes

- The publish job requests `id-token: write` (for the OIDC token) and
  `contents: read` only. No other permissions are granted, and no secrets are
  read.
- The workflow does not use `pull_request_target` and is not triggered by
  forked pull requests, so the publish path cannot be reached by untrusted
  contributors.
- `pypa/gh-action-pypi-publish` is pinned to a release tag in the workflow;
  bump that pin deliberately when adopting a newer version of the action.
